#!/usr/bin/env python3
"""Contract test for opt-in raw Msprof report auditing through the real DSO.

This compiles a CPU-only fake Msprof provider and a caller, then runs that
caller with the requested hook library in LD_PRELOAD. It does not require CANN
or an NPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "include"
FAKE_API = ROOT / "test" / "raw_report_fake_msprof_api.cpp"
PROBE = ROOT / "test" / "raw_report_contract_probe.cpp"
TARGET_OP = "kda_gate_chunk_cumsum_vector_kernel"
TARGET_TYPE = "RawReportTargetType"
EXPECTED_CSV_NAMES = {"ops.csv", "launch.csv", "hash_dic.csv"}
EXPECTED_OPS_HEADER = [
    "thread_id", "timestamp", "op_name", "op_type", "task_type", "task_type_name",
    "block_dim", "op_flag", "input_num", "input_formats", "input_dtypes",
    "input_shapes", "output_num", "output_formats", "output_dtypes", "output_shapes",
]


def compile_one(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"compile failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def build_probe(work: Path, cxx: str) -> tuple[Path, Path]:
    fake_library = work / "libraw_report_fake_msprof_api.so"
    probe = work / "raw_report_contract_probe"
    compile_one([
        cxx, "-std=c++11", "-Wall", "-Wextra", "-fPIC", "-shared",
        f"-I{INCLUDE}", str(FAKE_API), "-o", str(fake_library),
    ], work)
    compile_one([
        cxx, "-std=c++11", "-Wall", "-Wextra", f"-I{INCLUDE}", str(PROBE),
        f"-L{work}", f"-Wl,-rpath,{work}", "-lraw_report_fake_msprof_api", "-o", str(probe),
    ], work)
    return fake_library, probe


def process_env(hook_library: Path, fake_library: Path, output: Path,
                raw_enabled: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(hook_library)
    env["LD_LIBRARY_PATH"] = str(fake_library.parent) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env["MSPROF_HOOK_OUT"] = str(output)
    env.pop("MSPROF_HOOK_VERBOSE", None)
    env.pop("MSPROF_HOOK_RAW_OP", None)
    env.pop("MSPROF_HOOK_RAW_OP_HASH", None)
    if raw_enabled:
        env["MSPROF_HOOK_RAW_OP"] = TARGET_OP
    return env


def run_probe(probe: Path, hook_library: Path, fake_library: Path,
              output: Path, raw_enabled: bool, fail_raw_io: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_reports.jsonl"
    if fail_raw_io:
        raw_path.mkdir()
    result = subprocess.run(
        [str(probe)], cwd=output, env=process_env(hook_library, fake_library, output, raw_enabled),
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"probe exited {result.returncode}; stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if "RAW_REPORT_CONTRACT_PROBE_OK calls=10 hashes=3 target_hash=" not in result.stdout:
        raise AssertionError(f"probe did not reach its forwarding marker: {result.stdout!r}")


def require_csvs(output: Path, expect_raw: bool) -> dict[str, bytes]:
    regular_files = {path.name for path in output.iterdir() if path.is_file()}
    expected = EXPECTED_CSV_NAMES | ({"raw_reports.jsonl"} if expect_raw else set())
    if regular_files != expected:
        raise AssertionError(f"unexpected output files: expected {sorted(expected)}, got {sorted(regular_files)}")
    if not expect_raw and (output / "raw_reports.jsonl").is_file():
        raise AssertionError("raw-report capture file was created while the audit was disabled")
    contents = {name: (output / name).read_bytes() for name in EXPECTED_CSV_NAMES}
    header = contents["ops.csv"].splitlines()[0].decode("utf-8").split(",")
    if header != EXPECTED_OPS_HEADER:
        raise AssertionError(f"ops.csv header changed: {header!r}")
    if contents["launch.csv"].splitlines()[0] != b"thread_id,op_name,begin_ns,end_ns,duration_ns":
        raise AssertionError("launch.csv header changed")
    if contents["hash_dic.csv"].splitlines()[0] != b"hash,string":
        raise AssertionError("hash_dic.csv header changed")
    return contents


def load_raw(output: Path) -> list[dict[str, Any]]:
    raw_path = output / "raw_reports.jsonl"
    if not raw_path.is_file():
        raise AssertionError(f"raw audit did not create {raw_path}")
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AssertionError(f"invalid JSONL at line {line_no}: {error}") from error
        if not isinstance(value, dict):
            raise AssertionError(f"JSONL line {line_no} is not an object")
        records.append(value)
    return records


def check_raw_contract(records: list[dict[str, Any]]) -> None:
    expected_events = [
        "hash_registration", "api", "compact_basic",
        "additional_tensor_info", "additional_tensor_info", "additional_tensor_info",
        "additional_tensor_info", "additional_tensor_info", "additional_tensor_info",
    ]
    events = [record.get("event") for record in records]
    if events != expected_events:
        raise AssertionError(f"unexpected raw events or ordering: {events!r}")
    seqs = [record.get("event_seq") for record in records]
    if any(not isinstance(seq, int) for seq in seqs) or seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
        raise AssertionError(f"event_seq is not strictly increasing: {seqs!r}")
    if any(record.get("op_name") != TARGET_OP for record in records):
        raise AssertionError("raw filter included a non-target name/hash")

    registration, api, compact = records[:3]
    if registration.get("op_hash") != api.get("item_id") or api.get("op_name") != TARGET_OP:
        raise AssertionError("registered target hash and raw API itemId do not agree")
    if (api.get("thread_id"), api.get("level"), api.get("type"), api.get("reserve"),
        api.get("begin_time"), api.get("end_time")) != (731, 10000, 5, 0xA1B2C3D4, 1200, 1250):
        raise AssertionError(f"raw API metadata changed: {api!r}")
    expected_type_hash = 14695981039346656037
    for byte in TARGET_TYPE.encode("utf-8"):
        expected_type_hash ^= byte
        expected_type_hash = (expected_type_hash * 1099511628211) & ((1 << 64) - 1)
    if (compact.get("timestamp"), compact.get("data_len"), compact.get("op_hash"),
        compact.get("op_type_hash"), compact.get("task_type"), compact.get("block_dim"),
        compact.get("op_flag")) != (
            1001, 28, api.get("item_id"), expected_type_hash, 2, 48, 0xF00D1234
        ):
        raise AssertionError(f"raw CompactInfo key/length changed: {compact!r}")

    additions = records[3:]
    timestamps = [record.get("timestamp") for record in additions]
    if timestamps != [2001, 2002, 2003, 2004, 3001, 3003]:
        raise AssertionError(f"raw additional timestamps were joined/reordered: {timestamps!r}")
    if any(record.get("is_batch") for record in additions[:4]):
        raise AssertionError("single AdditionalInfo was marked as a batch item")
    if [(r.get("batch_index"), r.get("batch_count")) for r in additions[4:]] != [(0, 3), (2, 3)]:
        raise AssertionError("batch item indices/count do not preserve original batch positions")

    valid, zero, data_len_zero, partial, batch_first, batch_last = additions
    if valid.get("tensor_num") != 1 or valid.get("decoded_tensor_count") != 1:
        raise AssertionError(f"single tensor payload was not captured: {valid!r}")
    slots = valid.get("raw_tensor_slots")
    if not isinstance(slots, list) or len(slots) != 5:
        raise AssertionError(f"raw tensor slots are not the fixed five-slot ABI: {slots!r}")
    slot = slots[0]
    if (slot.get("tensor_type"), slot.get("format"), slot.get("dtype"),
        slot.get("shape_raw")) != (0, 0x12340000, 0xABCDEF00, [1, 0, 3, 0, 5, 0, 7, 0]):
        raise AssertionError(f"raw dtype/format/eight-dimensional shape were remapped: {slot!r}")
    if valid.get("truncated") is not False or valid.get("legacy_accept") is not True:
        raise AssertionError(f"valid full TensorInfo was incorrectly rejected/truncated: {valid!r}")

    if zero.get("tensor_num") != 0 or zero.get("decoded_tensor_count") != 0:
        raise AssertionError(f"tensorNum=0 was not retained: {zero!r}")
    if data_len_zero.get("data_len") != 0 or data_len_zero.get("tensor_num") != 4:
        raise AssertionError(f"short declared dataLen hid available payload bytes: {data_len_zero!r}")
    if data_len_zero.get("decoded_tensor_count") != 4 or data_len_zero.get("legacy_accept") is not False:
        raise AssertionError(f"dataLen-zero raw/legacy acceptance distinction was lost: {data_len_zero!r}")
    if data_len_zero.get("legacy_reject_reason") != "data_len_short":
        raise AssertionError(f"unexpected short-dataLen legacy rejection: {data_len_zero!r}")
    if data_len_zero.get("truncated") is not False:
        raise AssertionError(f"available raw payload was mislabeled as input-truncated: {data_len_zero!r}")

    if partial.get("truncated") is not True or partial.get("decoded_tensor_count") != 0:
        raise AssertionError(f"partial input buffer was not bounded/reported truncated: {partial!r}")
    if partial.get("legacy_accept") is not False or not partial.get("truncation_reason"):
        raise AssertionError(f"partial input did not retain truncation/rejection detail: {partial!r}")
    if partial.get("input_length", 0) >= valid.get("input_length", 0):
        raise AssertionError("partial record did not preserve its shorter original callback length")
    for batch_item in (batch_first, batch_last):
        if batch_item.get("is_batch") is not True or batch_item.get("decoded_tensor_count") not in (1, 2):
            raise AssertionError(f"matching batch TensorInfo item missing: {batch_item!r}")

    if any(record.get("timestamp") in (1002, 2005, 3002, 4001) for record in records):
        raise AssertionError("raw target filter recorded a non-target item")


def run(hook_library: Path, cxx: str) -> None:
    hook_library = hook_library.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="msprof-raw-report-") as temporary:
        work = Path(temporary)
        fake_library, probe = build_probe(work, cxx)

        disabled = work / "disabled"
        disabled.mkdir()
        run_probe(probe, hook_library, fake_library, disabled, raw_enabled=False)
        disabled_csv = require_csvs(disabled, expect_raw=False)

        enabled = work / "enabled"
        enabled.mkdir()
        run_probe(probe, hook_library, fake_library, enabled, raw_enabled=True)
        enabled_csv = require_csvs(enabled, expect_raw=True)
        if enabled_csv != disabled_csv:
            changed = sorted(name for name in EXPECTED_CSV_NAMES if enabled_csv[name] != disabled_csv[name])
            raise AssertionError(f"opt-in raw audit changed legacy CSV bytes: {changed}")
        check_raw_contract(load_raw(enabled))

        failed_io = work / "raw-io-failure"
        failed_io.mkdir()
        run_probe(probe, hook_library, fake_library, failed_io,
                  raw_enabled=True, fail_raw_io=True)
        failed_io_csv = require_csvs(failed_io, expect_raw=False)
        if failed_io_csv != disabled_csv:
            raise AssertionError("raw audit I/O failure changed legacy CSV output")
        if not (failed_io / "raw_reports.jsonl").is_dir():
            raise AssertionError("I/O-failure fixture did not retain its blocking directory")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hook-library", required=True, type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()
    try:
        run(args.hook_library, args.cxx)
    except (AssertionError, OSError, subprocess.SubprocessError) as error:
        print(f"test_raw_report_contract: FAIL: {error}", file=sys.stderr)
        return 1
    print("test_raw_report_contract: PASS: opt-in raw reports preserve forwarding and legacy CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
