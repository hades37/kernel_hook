#!/usr/bin/env python3
"""Check ops.csv format labels against the installed CANN GE C_Format ABI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    from test_dtype_contract import (
        EXPECTED_HASH_HEADER,
        EXPECTED_LAUNCH_HEADER,
        EXPECTED_OPS_HEADER,
        build_artifacts,
        read_csv,
        resolve_sdk_include,
    )
except ImportError:
    from .test_dtype_contract import (
        EXPECTED_HASH_HEADER,
        EXPECTED_LAUNCH_HEADER,
        EXPECTED_OPS_HEADER,
        build_artifacts,
        read_csv,
        resolve_sdk_include,
    )


ROOT = Path(__file__).resolve().parents[1]
TARGET_OP = "FormatContractOp"
EXPECTED_FORMAT_COUNT = 61

EXPECTED_FORMAT_NAMES = {
    0: "NCHW",
    1: "NHWC",
    2: "ND",
    3: "NC1HWC0",
    4: "FRACTAL_Z",
    5: "NC1C0HWPAD",
    6: "NHWC1C0",
    7: "FSR_NCHW",
    8: "FRACTAL_DECONV",
    9: "C1HWNC0",
    10: "FRACTAL_DECONV_TRANSPOSE",
    11: "FRACTAL_DECONV_SP_STRIDE_TRANS",
    12: "NC1HWC0_C04",
    13: "FRACTAL_Z_C04",
    14: "CHWN",
    15: "FRACTAL_DECONV_SP_STRIDE8_TRANS",
    16: "HWCN",
    17: "NC1KHKWHWC0",
    18: "BN_WEIGHT",
    19: "FILTER_HWCK",
    20: "HASHTABLE_LOOKUP_LOOKUPS",
    21: "HASHTABLE_LOOKUP_KEYS",
    22: "HASHTABLE_LOOKUP_VALUE",
    23: "HASHTABLE_LOOKUP_OUTPUT",
    24: "HASHTABLE_LOOKUP_HITS",
    25: "C1HWNCoC0",
    26: "MD",
    27: "NDHWC",
    28: "FRACTAL_ZZ",
    29: "FRACTAL_NZ",
    30: "NCDHW",
    31: "DHWCN",
    32: "NDC1HWC0",
    33: "FRACTAL_Z_3D",
    34: "CN",
    35: "NC",
    36: "DHWNC",
    37: "FRACTAL_Z_3D_TRANSPOSE",
    38: "FRACTAL_ZN_LSTM",
    39: "FRACTAL_Z_G",
    41: "ALL",
    42: "NULL",
    43: "ND_RNN_BIAS",
    44: "FRACTAL_ZN_RNN",
    45: "NYUV",
    46: "NYUV_A",
    47: "NCL",
    48: "FRACTAL_Z_WINO",
    49: "C1HWC0",
    50: "FRACTAL_NZ_C0_16",
    51: "FRACTAL_NZ_C0_32",
    52: "FRACTAL_NZ_C0_2",
    53: "FRACTAL_NZ_C0_4",
    54: "FRACTAL_NZ_C0_8",
}


PROBE_SOURCE = r'''#include "profiling/aprof_pub.h"
#include <graph/c_types.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
constexpr char kOpName[] = "FormatContractOp";
constexpr char kOpType[] = "FormatContractType";
constexpr uint32_t kThreadId = 53;
constexpr uint64_t kTimestamp = UINT64_C(123456789);

static_assert(C_FORMAT_HWCN == 16, "CANN HWCN format value changed");
static_assert(C_FORMAT_NDHWC == 27, "CANN NDHWC format value changed");
static_assert(C_FORMAT_FRACTAL_NZ == 29, "CANN FRACTAL_NZ format value changed");
static_assert(C_FORMAT_RESERVED == 40, "CANN reserved format value changed");
static_assert(C_FORMAT_END == 55, "CANN format end boundary changed");
static_assert(C_FORMAT_MAX == 0xff, "CANN format max boundary changed");

// Named entries come from CANN's C_Format enum. 40 is RESERVED, 55 is END,
// and MAX is a range sentinel; these are exercised as labels too.
const uint32_t kFormats[] = {
    C_FORMAT_NCHW, C_FORMAT_NHWC, C_FORMAT_ND, C_FORMAT_NC1HWC0, C_FORMAT_FRACTAL_Z,
    C_FORMAT_NC1C0HWPAD, C_FORMAT_NHWC1C0, C_FORMAT_FSR_NCHW, C_FORMAT_FRACTAL_DECONV,
    C_FORMAT_C1HWNC0, C_FORMAT_FRACTAL_DECONV_TRANSPOSE,
    C_FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS, C_FORMAT_NC1HWC0_C04,
    C_FORMAT_FRACTAL_Z_C04, C_FORMAT_CHWN, C_FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS,
    C_FORMAT_HWCN, C_FORMAT_NC1KHKWHWC0, C_FORMAT_BN_WEIGHT, C_FORMAT_FILTER_HWCK,
    C_FORMAT_HASHTABLE_LOOKUP_LOOKUPS, C_FORMAT_HASHTABLE_LOOKUP_KEYS,
    C_FORMAT_HASHTABLE_LOOKUP_VALUE, C_FORMAT_HASHTABLE_LOOKUP_OUTPUT,
    C_FORMAT_HASHTABLE_LOOKUP_HITS, C_FORMAT_C1HWNCoC0, C_FORMAT_MD, C_FORMAT_NDHWC,
    C_FORMAT_FRACTAL_ZZ, C_FORMAT_FRACTAL_NZ, C_FORMAT_NCDHW, C_FORMAT_DHWCN,
    C_FORMAT_NDC1HWC0, C_FORMAT_FRACTAL_Z_3D, C_FORMAT_CN, C_FORMAT_NC, C_FORMAT_DHWNC,
    C_FORMAT_FRACTAL_Z_3D_TRANSPOSE, C_FORMAT_FRACTAL_ZN_LSTM, C_FORMAT_FRACTAL_Z_G,
    C_FORMAT_RESERVED, C_FORMAT_ALL, C_FORMAT_NULL, C_FORMAT_ND_RNN_BIAS,
    C_FORMAT_FRACTAL_ZN_RNN, C_FORMAT_NYUV, C_FORMAT_NYUV_A, C_FORMAT_NCL,
    C_FORMAT_FRACTAL_Z_WINO, C_FORMAT_C1HWC0, C_FORMAT_FRACTAL_NZ_C0_16,
    C_FORMAT_FRACTAL_NZ_C0_32, C_FORMAT_FRACTAL_NZ_C0_2, C_FORMAT_FRACTAL_NZ_C0_4,
    C_FORMAT_FRACTAL_NZ_C0_8, C_FORMAT_END, C_FORMAT_MAX, 56,
    UINT32_C(0x12340000) | C_FORMAT_HWCN,
    UINT32_C(0xABCD0000) | C_FORMAT_NDHWC,
    UINT32_C(0xFACE0000) | C_FORMAT_MAX,
};

int Fail(const char *message)
{
    std::fprintf(stderr, "format_contract_probe: %s\n", message);
    return 1;
}

bool ReportNode(uint64_t op_hash, uint64_t type_hash)
{
    MsprofApi api{};
    api.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    api.level = MSPROF_REPORT_NODE_LEVEL;
    api.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    api.threadId = kThreadId;
    api.beginTime = 2000;
    api.endTime = 2050;
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
    compact.data.nodeBasicInfo.blockDim = 9;
    if (MsprofReportCompactInfo(0, &compact, sizeof(compact)) != 102) return false;

    const size_t format_count = sizeof(kFormats) / sizeof(kFormats[0]);
    for (size_t offset = 0; offset < format_count; offset += MSPROF_GE_TENSOR_DATA_NUM) {
        MsprofAdditionalInfo additional{};
        additional.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
        additional.level = MSPROF_REPORT_NODE_LEVEL;
        additional.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
        additional.threadId = kThreadId;
        additional.timeStamp = kTimestamp;
        auto *tensor_info = reinterpret_cast<MsprofTensorInfo *>(additional.data);
        tensor_info->opName = op_hash;
        const size_t count = (format_count - offset < MSPROF_GE_TENSOR_DATA_NUM)
            ? format_count - offset : MSPROF_GE_TENSOR_DATA_NUM;
        tensor_info->tensorNum = static_cast<uint32_t>(count);
        for (size_t index = 0; index < count; ++index) {
            auto &tensor = tensor_info->tensorData[index];
            tensor.tensorType = MSPROF_GE_TENSOR_TYPE_INPUT;
            tensor.format = kFormats[offset + index];
            tensor.dataType = 0;
            tensor.shape[0] = 1;
        }
        additional.dataLen = sizeof(MsprofTensorInfo);
        if (MsprofReportAdditionalInfo(0, &additional, sizeof(additional)) != 103) {
            return false;
        }
    }

    std::printf("FORMAT_VALUES=");
    for (size_t index = 0; index < format_count; ++index) {
        std::printf("%s%u", index == 0 ? "" : ",", kFormats[index]);
    }
    std::printf("\nFORMAT_CONTRACT_PROBE_OK count=%zu\n", format_count);
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


def run_probe(probe: Path, hook_library: Path, fake_library: Path,
              output: Path) -> tuple[str, list[dict[str, object]]]:
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(hook_library)
    env["LD_LIBRARY_PATH"] = str(fake_library.parent) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env["MSPROF_HOOK_OUT"] = str(output)
    env["MSPROF_HOOK_RAW_OP"] = TARGET_OP
    env.pop("MSPROF_HOOK_RAW_OP_HASH", None)
    env.pop("MSPROF_HOOK_VERBOSE", None)
    result = subprocess.run(
        [str(probe)], cwd=output, env=env, text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"probe exited {result.returncode}; stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if f"FORMAT_CONTRACT_PROBE_OK count={EXPECTED_FORMAT_COUNT}" not in result.stdout:
        raise AssertionError(f"probe did not report every format value: {result.stdout!r}")
    raw_path = output / "raw_reports.jsonl"
    if not raw_path.is_file():
        raise AssertionError("raw report capture was not created")
    records: list[dict[str, object]] = []
    for line_no, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AssertionError(f"invalid raw JSON at line {line_no}: {error}") from error
        if not isinstance(record, dict):
            raise AssertionError(f"raw JSON line {line_no} is not an object")
        records.append(record)
    return result.stdout, records


def check_results(output: Path, stdout: str,
                  records: list[dict[str, object]]) -> None:
    observed_line = next(
        (line for line in stdout.splitlines() if line.startswith("FORMAT_VALUES=")), None
    )
    if observed_line is None:
        raise AssertionError("probe did not expose the C_Format values it reported")
    observed_formats = [int(value) for value in observed_line.partition("=")[2].split(",")]
    packed_hwcn = 0x12340000 | 16
    packed_ndhwc = 0xABCD0000 | 27
    packed_max = 0xFACE0000 | 0xFF
    expected_formats = [*range(56), 255, 56, packed_hwcn, packed_ndhwc, packed_max]
    if observed_formats != expected_formats:
        raise AssertionError(
            f"installed C_Format values changed: expected {expected_formats!r}, "
            f"got {observed_formats!r}"
        )

    actual_files = {path.name for path in output.iterdir() if path.is_file()}
    if actual_files != {"ops.csv", "launch.csv", "hash_dic.csv", "raw_reports.jsonl"}:
        raise AssertionError(f"unexpected output files: {sorted(actual_files)!r}")
    ops = read_csv(output / "ops.csv", EXPECTED_OPS_HEADER)
    launches = read_csv(output / "launch.csv", EXPECTED_LAUNCH_HEADER)
    hashes = read_csv(output / "hash_dic.csv", EXPECTED_HASH_HEADER)
    if len(ops) != 1:
        raise AssertionError(f"expected one ops.csv row, got {len(ops)}")

    expected_labels = [EXPECTED_FORMAT_NAMES.get(value, "UNKNOWN") for value in range(56)]
    expected_labels.extend([
        "UNKNOWN",  # C_FORMAT_MAX is a range sentinel.
        "UNKNOWN",  # First value past C_FORMAT_END.
        "HWCN",     # Packed subformat and C0 bits do not change the primary format.
        "NDHWC",
        "UNKNOWN",  # Packed C_FORMAT_MAX remains an unknown primary value.
    ])
    row = ops[0]
    expected_fields = {
        "thread_id": "53",
        "timestamp": "123456789",
        "op_name": TARGET_OP,
        "op_type": "FormatContractType",
        "task_type": "2",
        "task_type_name": "AI_VECTOR_CORE",
        "block_dim": "9",
        "input_num": str(len(expected_formats)),
        "input_formats": ";".join(expected_labels),
        "input_dtypes": ";".join(["FLOAT"] * len(expected_formats)),
        "input_shapes": ";".join(["1"] * len(expected_formats)),
        "output_num": "0",
        "output_formats": "N/A",
        "output_dtypes": "N/A",
        "output_shapes": "N/A",
    }
    for field, expected in expected_fields.items():
        if row.get(field) != expected:
            raise AssertionError(
                f"ops.csv {field} differs from the CANN format contract; "
                f"got {row.get(field)!r}"
            )

    if launches != [{
        "thread_id": "53", "op_name": TARGET_OP, "begin_ns": "2000",
        "end_ns": "2050", "duration_ns": "50",
    }]:
        raise AssertionError(f"launch.csv schema or row changed: {launches!r}")
    if {item["string"] for item in hashes} != {TARGET_OP, "FormatContractType"}:
        raise AssertionError(f"hash_dic.csv schema or names changed: {hashes!r}")

    additions = [record for record in records if record.get("event") == "additional_tensor_info"]
    if len(additions) != (len(expected_formats) + 4) // 5:
        raise AssertionError(f"unexpected number of raw tensor reports: {len(additions)}")
    raw_formats: list[int] = []
    for record in additions:
        tensor_num = record.get("tensor_num")
        slots = record.get("raw_tensor_slots")
        if not isinstance(tensor_num, int) or not isinstance(slots, list):
            raise AssertionError(f"raw tensor record omitted numeric format slots: {record!r}")
        if tensor_num > len(slots):
            raise AssertionError(f"raw tensor record count exceeds its slots: {record!r}")
        for slot in slots[:tensor_num]:
            if not isinstance(slot, dict) or not isinstance(slot.get("format"), int):
                raise AssertionError(f"raw format was not retained as a number: {slot!r}")
            raw_formats.append(slot["format"])
    if raw_formats != expected_formats:
        raise AssertionError(
            f"raw JSON formats were changed: expected {expected_formats!r}, got {raw_formats!r}"
        )


def run(cxx: str, sdk_include: Path) -> None:
    if not (sdk_include / "graph" / "c_types.h").is_file():
        raise AssertionError(f"graph/c_types.h not found under SDK include dir: {sdk_include}")
    with tempfile.TemporaryDirectory(prefix="msprof-format-contract-") as temporary:
        work = Path(temporary)
        hook_library, fake_library, probe = build_artifacts(
            work, cxx, sdk_include, probe_source=PROBE_SOURCE
        )
        output = work / "profile-output"
        output.mkdir()
        stdout, records = run_probe(probe, hook_library, fake_library, output)
        check_results(output, stdout, records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--sdk-include-dir", "--acl-include-dir", dest="sdk_include_dir",
                        type=Path, help="CANN SDK include directory containing graph/c_types.h")
    args = parser.parse_args()
    try:
        run(args.cxx, resolve_sdk_include(args.sdk_include_dir))
    except (AssertionError, OSError, subprocess.SubprocessError) as error:
        print(f"test_format_contract: FAIL: {error}", file=sys.stderr)
        return 1
    print("test_format_contract: PASS: ops.csv formats match CANN C_Format")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
