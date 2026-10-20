#!/usr/bin/env python3
"""Check that ops.csv dtype labels follow the installed CANN GE DataType ABI.

The test builds the real hook and a CPU-only Msprof provider/caller. The caller
uses enum values from graph/c_types.h and sends them through the public Msprof
report entry points. It also checks the overlapping ACL values, so no device or
CANN runtime library is required.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "include"
FAKE_API = ROOT / "test" / "fake_msprof_api.cpp"
HOOK_SOURCE = ROOT / "src" / "msprof_hook.cpp"
EXPECTED_OPS_HEADER = [
    "thread_id", "timestamp", "op_name", "op_type", "task_type", "task_type_name",
    "block_dim", "op_flag", "input_num", "input_formats", "input_dtypes",
    "input_shapes", "output_num", "output_formats", "output_dtypes", "output_shapes",
]
EXPECTED_LAUNCH_HEADER = ["thread_id", "op_name", "begin_ns", "end_ns", "duration_ns"]
EXPECTED_HASH_HEADER = ["hash", "string"]

EXPECTED_DTYPE_NAMES = {
    0: "FLOAT",
    1: "FLOAT16",
    2: "INT8",
    3: "INT32",
    4: "UINT8",
    6: "INT16",
    7: "UINT16",
    8: "UINT32",
    9: "INT64",
    10: "UINT64",
    11: "DOUBLE",
    12: "BOOL",
    13: "STRING",
    14: "DUAL_SUB_INT8",
    15: "DUAL_SUB_UINT8",
    16: "COMPLEX64",
    17: "COMPLEX128",
    18: "QINT8",
    19: "QINT16",
    20: "QINT32",
    21: "QUINT8",
    22: "QUINT16",
    23: "RESOURCE",
    24: "STRING_REF",
    25: "DUAL",
    26: "VARIANT",
    27: "BF16",
    28: "UNDEFINED",
    29: "INT4",
    30: "UINT1",
    31: "INT2",
    32: "UINT2",
    33: "COMPLEX32",
    34: "HIFLOAT8",
    35: "FLOAT8_E5M2",
    36: "FLOAT8_E4M3FN",
    37: "FLOAT8_E8M0",
    38: "FLOAT6_E3M2",
    39: "FLOAT6_E2M3",
    40: "FLOAT4_E2M1",
    41: "FLOAT4_E1M2",
    42: "HIFLOAT4",
}


PROBE_SOURCE = r'''#include "profiling/aprof_pub.h"
#include <acl/acl_base.h>
#include <graph/c_types.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
constexpr char kOpName[] = "DTypeContractOp";
constexpr char kOpType[] = "DTypeContractType";
constexpr uint32_t kThreadId = 47;
constexpr uint64_t kTimestamp = UINT64_C(987654321);

// aclDataType is the subset exposed by ACL. Msprof tensor dataType carries
// ge::DataType values, represented by C_DataType in this public C header.
#define ASSERT_ACL_GE(name) \
    static_assert(ACL_##name == C_DT_##name, #name " differs between ACL and GE")
ASSERT_ACL_GE(FLOAT);
ASSERT_ACL_GE(FLOAT16);
ASSERT_ACL_GE(INT8);
ASSERT_ACL_GE(INT32);
ASSERT_ACL_GE(UINT8);
ASSERT_ACL_GE(INT16);
ASSERT_ACL_GE(UINT16);
ASSERT_ACL_GE(UINT32);
ASSERT_ACL_GE(INT64);
ASSERT_ACL_GE(UINT64);
ASSERT_ACL_GE(DOUBLE);
ASSERT_ACL_GE(BOOL);
ASSERT_ACL_GE(STRING);
ASSERT_ACL_GE(COMPLEX64);
ASSERT_ACL_GE(COMPLEX128);
ASSERT_ACL_GE(BF16);
ASSERT_ACL_GE(INT4);
ASSERT_ACL_GE(UINT1);
ASSERT_ACL_GE(COMPLEX32);
ASSERT_ACL_GE(HIFLOAT8);
ASSERT_ACL_GE(FLOAT8_E5M2);
ASSERT_ACL_GE(FLOAT8_E4M3FN);
ASSERT_ACL_GE(FLOAT8_E8M0);
ASSERT_ACL_GE(FLOAT6_E3M2);
ASSERT_ACL_GE(FLOAT6_E2M3);
ASSERT_ACL_GE(FLOAT4_E2M1);
ASSERT_ACL_GE(FLOAT4_E1M2);
#undef ASSERT_ACL_GE
static_assert(ACL_DT_UNDEFINED == -1, "ACL undefined sentinel changed");
static_assert(C_DT_UNDEFINED == 28, "GE UNDEFINED value changed");

// Report every GE value from 0 through C_DT_MAX, including the reserved 5,
// plus ACL's -1 sentinel and an out-of-range value. C_DT_MAX is a boundary,
// not a data type, and must remain UNKNOWN.
const uint32_t kDtypes[] = {
    static_cast<uint32_t>(ACL_DT_UNDEFINED),
    static_cast<uint32_t>(C_DT_FLOAT), static_cast<uint32_t>(C_DT_FLOAT16),
    static_cast<uint32_t>(C_DT_INT8), static_cast<uint32_t>(C_DT_INT32),
    static_cast<uint32_t>(C_DT_UINT8), 5,
    static_cast<uint32_t>(C_DT_INT16), static_cast<uint32_t>(C_DT_UINT16),
    static_cast<uint32_t>(C_DT_UINT32), static_cast<uint32_t>(C_DT_INT64),
    static_cast<uint32_t>(C_DT_UINT64), static_cast<uint32_t>(C_DT_DOUBLE),
    static_cast<uint32_t>(C_DT_BOOL), static_cast<uint32_t>(C_DT_STRING),
    static_cast<uint32_t>(C_DT_DUAL_SUB_INT8), static_cast<uint32_t>(C_DT_DUAL_SUB_UINT8),
    static_cast<uint32_t>(C_DT_COMPLEX64), static_cast<uint32_t>(C_DT_COMPLEX128),
    static_cast<uint32_t>(C_DT_QINT8), static_cast<uint32_t>(C_DT_QINT16),
    static_cast<uint32_t>(C_DT_QINT32), static_cast<uint32_t>(C_DT_QUINT8),
    static_cast<uint32_t>(C_DT_QUINT16), static_cast<uint32_t>(C_DT_RESOURCE),
    static_cast<uint32_t>(C_DT_STRING_REF), static_cast<uint32_t>(C_DT_DUAL),
    static_cast<uint32_t>(C_DT_VARIANT), static_cast<uint32_t>(C_DT_BF16),
    static_cast<uint32_t>(C_DT_UNDEFINED), static_cast<uint32_t>(C_DT_INT4),
    static_cast<uint32_t>(C_DT_UINT1), static_cast<uint32_t>(C_DT_INT2),
    static_cast<uint32_t>(C_DT_UINT2), static_cast<uint32_t>(C_DT_COMPLEX32),
    static_cast<uint32_t>(C_DT_HIFLOAT8), static_cast<uint32_t>(C_DT_FLOAT8_E5M2),
    static_cast<uint32_t>(C_DT_FLOAT8_E4M3FN), static_cast<uint32_t>(C_DT_FLOAT8_E8M0),
    static_cast<uint32_t>(C_DT_FLOAT6_E3M2), static_cast<uint32_t>(C_DT_FLOAT6_E2M3),
    static_cast<uint32_t>(C_DT_FLOAT4_E2M1), static_cast<uint32_t>(C_DT_FLOAT4_E1M2),
    static_cast<uint32_t>(C_DT_HIFLOAT4), static_cast<uint32_t>(C_DT_MAX),
    UINT32_C(1000),
};

int Fail(const char *message)
{
    std::fprintf(stderr, "dtype_contract_probe: %s\n", message);
    return 1;
}

bool ReportNode(uint64_t op_hash, uint64_t type_hash)
{
    MsprofApi api{};
    api.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    api.level = MSPROF_REPORT_NODE_LEVEL;
    api.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    api.threadId = kThreadId;
    api.beginTime = 1000;
    api.endTime = 1037;
    api.itemId = op_hash;
    if (MsprofReportApi(0, &api) != 101) return false;

    MsprofCompactInfo compact{};
    compact.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    compact.level = MSPROF_REPORT_NODE_LEVEL;
    compact.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    compact.threadId = kThreadId;
    compact.dataLen = sizeof(MsprofNodeBasicInfo);
    compact.timeStamp = kTimestamp;
    compact.data.nodeBasicInfo.opName = op_hash;
    compact.data.nodeBasicInfo.opType = type_hash;
    compact.data.nodeBasicInfo.taskType = 2;
    compact.data.nodeBasicInfo.blockDim = 7;
    if (MsprofReportCompactInfo(0, &compact, sizeof(compact)) != 102) return false;

    const size_t dtype_count = sizeof(kDtypes) / sizeof(kDtypes[0]);
    for (size_t offset = 0; offset < dtype_count; offset += MSPROF_GE_TENSOR_DATA_NUM) {
        MsprofAdditionalInfo additional{};
        additional.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
        additional.level = MSPROF_REPORT_NODE_LEVEL;
        additional.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
        additional.threadId = kThreadId;
        additional.timeStamp = kTimestamp;
        auto *tensor_info = reinterpret_cast<MsprofTensorInfo *>(additional.data);
        tensor_info->opName = op_hash;
        const size_t count = (dtype_count - offset < MSPROF_GE_TENSOR_DATA_NUM)
            ? dtype_count - offset : MSPROF_GE_TENSOR_DATA_NUM;
        tensor_info->tensorNum = static_cast<uint32_t>(count);
        for (size_t index = 0; index < count; ++index) {
            auto &tensor = tensor_info->tensorData[index];
            tensor.tensorType = MSPROF_GE_TENSOR_TYPE_INPUT;
            tensor.format = 2;
            tensor.dataType = kDtypes[offset + index];
            tensor.shape[0] = 1;
        }
        additional.dataLen = sizeof(MsprofTensorInfo);
        if (MsprofReportAdditionalInfo(0, &additional, sizeof(additional)) != 103) {
            return false;
        }
    }

    std::printf("DTYPE_VALUES=");
    for (size_t index = 0; index < dtype_count; ++index) {
        std::printf("%s%u", index == 0 ? "" : ",", kDtypes[index]);
    }
    std::printf("\nDTYPE_CONTRACT_PROBE_OK count=%zu\n", dtype_count);
    return true;
}
}  // namespace

int main()
{
    const uint64_t op_hash = MsprofGetHashId(kOpName, std::strlen(kOpName));
    const uint64_t type_hash = MsprofStr2Id(kOpType, std::strlen(kOpType));
    if (op_hash == 0 || type_hash == 0) return Fail("hash API returned zero");
    if (!ReportNode(op_hash, type_hash)) return Fail("Msprof report API failed");
    return 0;
}
'''


def compile_one(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"compile failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def build_artifacts(work: Path, cxx: str, sdk_include: Path,
                    probe_source: str = PROBE_SOURCE) -> tuple[Path, Path, Path]:
    fake_library = work / "libdtype_fake_msprof_api.so"
    hook_library = work / "libmsprof_hook_dtype_contract.so"
    probe_file = work / "contract_probe.cpp"
    probe = work / "dtype_contract_probe"
    probe_file.write_text(probe_source, encoding="utf-8")

    compile_one([
        cxx, "-std=c++11", "-Wall", "-Wextra", "-fPIC", "-shared",
        f"-I{INCLUDE}", str(FAKE_API), "-o", str(fake_library),
    ], work)
    compile_one([
        cxx, "-std=c++11", "-Wall", "-Wextra", "-fPIC", "-shared",
        f"-I{INCLUDE}", str(HOOK_SOURCE), "-ldl", "-o", str(hook_library),
    ], work)
    compile_one([
        cxx, "-std=c++11", "-Wall", "-Wextra", f"-I{INCLUDE}", f"-I{sdk_include}",
        str(probe_file), f"-L{work}", f"-Wl,-rpath,{work}", "-ldtype_fake_msprof_api",
        "-o", str(probe),
    ], work)
    return hook_library, fake_library, probe


def run_probe(probe: Path, hook_library: Path, fake_library: Path,
              output: Path) -> str:
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(hook_library)
    env["LD_LIBRARY_PATH"] = str(fake_library.parent) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env["MSPROF_HOOK_OUT"] = str(output)
    env.pop("MSPROF_HOOK_VERBOSE", None)
    env.pop("MSPROF_HOOK_RAW_OP", None)
    env.pop("MSPROF_HOOK_RAW_OP_HASH", None)
    result = subprocess.run(
        [str(probe)], cwd=output, env=env, text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"probe exited {result.returncode}; stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if "DTYPE_CONTRACT_PROBE_OK count=46" not in result.stdout:
        raise AssertionError(f"probe did not report all dtype values: {result.stdout!r}")
    return result.stdout


def read_csv(path: Path, expected_header: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise AssertionError(f"expected CSV was not written: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != expected_header:
            raise AssertionError(f"unexpected CSV header in {path.name}: {reader.fieldnames!r}")
        rows = list(reader)
    return rows


def check_results(output: Path, stdout: str) -> None:
    observed_line = next(
        (line for line in stdout.splitlines() if line.startswith("DTYPE_VALUES=")), None
    )
    if observed_line is None:
        raise AssertionError("probe did not expose the SDK enum values it reported")
    observed_values = [int(value) for value in observed_line.partition("=")[2].split(",")]
    expected_values = [0xFFFFFFFF, *range(44), 1000]
    if observed_values != expected_values:
        raise AssertionError(
            f"installed GE DataType values changed: expected {expected_values!r}, "
            f"got {observed_values!r}"
        )

    actual_files = {path.name for path in output.iterdir() if path.is_file()}
    if actual_files != {"ops.csv", "launch.csv", "hash_dic.csv"}:
        raise AssertionError(f"unexpected CSV output files: {sorted(actual_files)!r}")
    ops = read_csv(output / "ops.csv", EXPECTED_OPS_HEADER)
    launches = read_csv(output / "launch.csv", EXPECTED_LAUNCH_HEADER)
    hashes = read_csv(output / "hash_dic.csv", EXPECTED_HASH_HEADER)
    if len(ops) != 1:
        raise AssertionError(f"expected one ops.csv row, got {len(ops)}")

    expected_labels = ["UNKNOWN"]  # ACL_DT_UNDEFINED is -1 in the unsigned report field.
    expected_labels.extend(EXPECTED_DTYPE_NAMES.get(value, "UNKNOWN") for value in range(44))
    # Value 43 (C_DT_MAX) is already covered by range(44); out-of-range values
    # retain the existing UNKNOWN fallback.
    expected_labels.append("UNKNOWN")
    row = ops[0]
    expected_fields = {
        "thread_id": "47",
        "timestamp": "987654321",
        "op_name": "DTypeContractOp",
        "op_type": "DTypeContractType",
        "task_type": "2",
        "task_type_name": "AI_VECTOR_CORE",
        "block_dim": "7",
        "input_num": "46",
        "input_formats": ";".join(["ND"] * 46),
        "input_dtypes": ";".join(expected_labels),
        "input_shapes": ";".join(["1"] * 46),
        "output_num": "0",
        "output_formats": "N/A",
        "output_dtypes": "N/A",
        "output_shapes": "N/A",
    }
    for field, expected in expected_fields.items():
        if row.get(field) != expected:
            raise AssertionError(
                f"ops.csv {field} differs from the SDK contract; got {row.get(field)!r}"
            )

    if launches != [{
        "thread_id": "47", "op_name": "DTypeContractOp", "begin_ns": "1000",
        "end_ns": "1037", "duration_ns": "37",
    }]:
        raise AssertionError(f"launch.csv schema or legacy row changed: {launches!r}")
    if {item["string"] for item in hashes} != {"DTypeContractOp", "DTypeContractType"}:
        raise AssertionError(f"hash_dic.csv schema or names changed: {hashes!r}")


def resolve_sdk_include(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    roots = [
        os.environ.get(name)
        for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "CANN_HOME", "ASCEND_HOME")
    ]
    for value in roots:
        if not value:
            continue
        root = Path(value)
        for include in (root / "include", root / "x86_64-linux" / "include"):
            if ((include / "acl" / "acl_base.h").is_file() and
                    (include / "graph" / "c_types.h").is_file()):
                return include
    raise AssertionError(
        "CANN SDK headers were not found; set ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME "
        "or pass --sdk-include-dir (contains acl/acl_base.h and graph/c_types.h)"
    )


def run(cxx: str, sdk_include: Path) -> None:
    required_headers = (sdk_include / "acl" / "acl_base.h",
                        sdk_include / "graph" / "c_types.h")
    if any(not header.is_file() for header in required_headers):
        raise AssertionError(
            f"SDK include dir must contain acl/acl_base.h and graph/c_types.h: {sdk_include}"
        )
    with tempfile.TemporaryDirectory(prefix="msprof-dtype-contract-") as temporary:
        work = Path(temporary)
        hook_library, fake_library, probe = build_artifacts(work, cxx, sdk_include)
        output = work / "profile-output"
        output.mkdir()
        stdout = run_probe(probe, hook_library, fake_library, output)
        check_results(output, stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--sdk-include-dir", "--acl-include-dir", dest="sdk_include_dir",
                        type=Path, help="CANN SDK include directory containing ACL and GE headers")
    args = parser.parse_args()
    try:
        run(args.cxx, resolve_sdk_include(args.sdk_include_dir))
    except (AssertionError, OSError, subprocess.SubprocessError) as error:
        print(f"test_dtype_contract: FAIL: {error}", file=sys.stderr)
        return 1
    print("test_dtype_contract: PASS: ops.csv dtype labels match CANN GE DataType")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
