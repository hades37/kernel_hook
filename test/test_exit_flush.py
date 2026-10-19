#!/usr/bin/env python3
"""Exercise the real preload DSO and verify its process-exit CSV flush."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile
import sys


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "test" / "msprof_exit_flush_probe.cpp"
FAKE_API = ROOT / "test" / "fake_msprof_api.cpp"
EXPECTED_OPS_HEADER = [
    "thread_id", "timestamp", "op_name", "op_type", "task_type", "task_type_name",
    "block_dim", "op_flag", "input_num", "input_formats", "input_dtypes",
    "input_shapes", "output_num", "output_formats", "output_dtypes", "output_shapes",
]


def read_rows(path: Path, expected_header: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise AssertionError(f"expected CSV was not written at process exit: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != expected_header:
            raise AssertionError(f"unexpected CSV header in {path}: {reader.fieldnames!r}")
        rows = list(reader)
    if not rows:
        raise AssertionError(f"CSV has no records: {path}")
    return rows


def compile_one(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"compile failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def run(hook_library: Path, cxx: str) -> None:
    hook_library = hook_library.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="msprof-exit-flush-") as temporary:
        work = Path(temporary)
        output = work / "profile-output"
        output.mkdir()
        fake_library = work / "libfake_msprof_api.so"
        probe = work / "msprof_exit_flush_probe"

        compile_one([
            cxx, "-std=c++11", "-Wall", "-Wextra", "-fPIC", "-shared",
            f"-I{ROOT / 'include'}", str(FAKE_API), "-o", str(fake_library),
        ], work)
        compile_one([
            cxx, "-std=c++11", "-Wall", "-Wextra", f"-I{ROOT / 'include'}",
            str(PROBE), f"-L{work}", f"-Wl,-rpath,{work}", "-lfake_msprof_api",
            "-o", str(probe),
        ], work)

        env = os.environ.copy()
        env["LD_PRELOAD"] = str(hook_library)
        env["LD_LIBRARY_PATH"] = str(work) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        env["MSPROF_HOOK_OUT"] = str(output)
        env.pop("MSPROF_HOOK_VERBOSE", None)
        result = subprocess.run([str(probe)], cwd=work, env=env,
                                text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise AssertionError(
                f"probe exited {result.returncode}; stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        if "MSPROF_EXIT_FLUSH_PROBE_OK records=1 hash_names=2 reports=3" not in result.stdout:
            raise AssertionError(f"probe did not reach its clean-exit marker: {result.stdout!r}")

        files = {path.name for path in output.iterdir() if path.is_file()}
        if files != {"ops.csv", "launch.csv", "hash_dic.csv"}:
            raise AssertionError(f"unexpected files after DSO exit flush: {sorted(files)!r}")
        ops = read_rows(output / "ops.csv", EXPECTED_OPS_HEADER)
        launches = read_rows(
            output / "launch.csv",
            ["thread_id", "op_name", "begin_ns", "end_ns", "duration_ns"],
        )
        hashes = read_rows(output / "hash_dic.csv", ["hash", "string"])

        if len(ops) != 1:
            raise AssertionError(f"expected one op row after exit, got {len(ops)}")
        row = ops[0]
        expected = {
            "thread_id": "47", "timestamp": "987654321",
            "op_name": "ExitFlushProbeOp", "op_type": "ExitFlushProbeType",
            "task_type": "2", "task_type_name": "AI_VECTOR_CORE", "block_dim": "7",
            "input_num": "1", "input_formats": "ND", "input_dtypes": "FLOAT",
            "input_shapes": "2,3", "output_num": "1", "output_formats": "ND",
            "output_dtypes": "INT32", "output_shapes": "2,3",
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise AssertionError(f"ops.csv {field}={row.get(field)!r}, expected {value!r}")

        if launches != [{
            "thread_id": "47", "op_name": "ExitFlushProbeOp", "begin_ns": "1000",
            "end_ns": "1037", "duration_ns": "37",
        }]:
            raise AssertionError(f"launch.csv record differs: {launches!r}")
        if {item["string"] for item in hashes} != {"ExitFlushProbeOp", "ExitFlushProbeType"}:
            raise AssertionError(f"hash dictionary entries did not survive exit: {hashes!r}")
        if list(work.glob("*.csv")):
            raise AssertionError("CSV files were written outside MSPROF_HOOK_OUT")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hook-library", required=True, type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()
    try:
        run(args.hook_library, args.cxx)
    except (AssertionError, OSError, subprocess.SubprocessError) as error:
        print(f"test_exit_flush: FAIL: {error}", file=sys.stderr)
        return 1
    print("test_exit_flush: PASS: forwarded Msprof records survive DSO exit in all CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
