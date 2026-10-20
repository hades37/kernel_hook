#!/usr/bin/env python3
"""Compile the registered Triton producer's CompactInfo excerpts against SDK ABI.

This is a CPU-only SDK contract probe. It loads the source excerpt fixtures in
this repository, passes each through the registered-backend factory overlay,
then compiles and calls the captured C++ report path with the actual SDK header.
No installed Triton driver, CANN runtime library, or NPU is needed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LOCAL_EXCERPT = ROOT / "test" / "triton_launcher_local_compact_excerpt.cxx"
PHYSICAL_EXCERPT = ROOT / "test" / "triton_launcher_physical_compact_excerpt.cxx"

HEADER_SOURCE = r'''#include <profiling/aprof_pub.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
'''

CPP_SUPPORT = r'''
namespace {
constexpr uint64_t kExpectedOpName = UINT64_C(0x1122334455667788);
constexpr uint32_t kExpectedThreadId = UINT32_C(0x3456);
constexpr uint64_t kExpectedTimestamp = UINT64_C(0x778899AABBCCDDEE);
constexpr uint32_t kExpectedBlockDim = UINT32_C(73);
constexpr size_t kExpectedCallCount = 2;

struct CapturedCompactInfo {
    bool has_data;
    uint32_t non_persistent_flag;
    uint32_t report_length;
    uintptr_t data_address;
    uint16_t magic_number;
    uint16_t level;
    uint32_t type;
    uint32_t thread_id;
    uint32_t data_len;
    uint64_t timestamp;
    uint64_t op_name;
    uint64_t op_type;
    uint32_t task_type;
    uint32_t block_dim;
    uint32_t op_flag;
};

CapturedCompactInfo g_captured[kExpectedCallCount]{};
size_t g_compact_call_count = 0;
size_t g_additional_call_count = 0;

int Fail(const char *message)
{
    std::fprintf(stderr, "compact_header_contract_probe: %s\n", message);
    return 1;
}
}  // namespace

extern "C" int32_t MsprofReportCompactInfo(uint32_t nonPersistantFlag,
                                            const VOID_PTR data,
                                            uint32_t length)
{
    if (g_compact_call_count >= kExpectedCallCount) {
        return -1;
    }
    CapturedCompactInfo &captured = g_captured[g_compact_call_count++];
    captured.has_data = data != nullptr;
    captured.non_persistent_flag = nonPersistantFlag;
    captured.report_length = length;
    captured.data_address = reinterpret_cast<uintptr_t>(data);
    if (data == nullptr) {
        return 102;
    }

    // Read through the real SDK declaration, including its packed union and
    // MsprofNodeBasicInfo field layout; do not mirror the ABI in this probe.
    const auto *record = static_cast<const MsprofCompactInfo *>(data);
    captured.magic_number = record->magicNumber;
    captured.level = record->level;
    captured.type = record->type;
    captured.thread_id = record->threadId;
    captured.data_len = record->dataLen;
    captured.timestamp = record->timeStamp;
    captured.op_name = record->data.nodeBasicInfo.opName;
    captured.op_type = record->data.nodeBasicInfo.opType;
    captured.task_type = record->data.nodeBasicInfo.taskType;
    captured.block_dim = record->data.nodeBasicInfo.blockDim;
    captured.op_flag = record->data.nodeBasicInfo.opFlag;
    return 102;
}

extern "C" int32_t MsprofReportAdditionalInfo(uint32_t,
                                               const VOID_PTR,
                                               uint32_t)
{
    ++g_additional_call_count;
    return 103;
}
'''

RAW_COMPACT_DECLARATION = re.compile(
    r"(?m)^(?P<indent>[ \t]*)MsprofCompactInfo nodeBasicInfo"
    r"(?P<initializer>\{\}|[ \t]*=[ \t]*\{\})?;[ \t]*$"
)


def load_overlay(path: Path):
    module_name = "_compact_header_contract_overlay"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load overlay module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def registered_backend(module_alias: str, wrapper_source: str):
    native_calls: list[dict[str, Any]] = []
    factory_result = object()
    namespace: dict[str, Any] = {
        "__name__": module_alias,
        "_wrapper_source": wrapper_source,
        "_native_calls": native_calls,
        "_factory_result": factory_result,
    }
    exec(
        """
def generate_npu_wrapper_src():
    return _wrapper_source

def make_launcher():
    return _wrapper_source

def make_npu_launcher_stub(header_src, wrapper_src, debug=False):
    _native_calls.append({
        "header_src": header_src,
        "wrapper_src": wrapper_src,
        "debug": debug,
    })
    return _factory_result

class NPUDriver:
    def __init__(self, *args, **kwargs):
        raise AssertionError("contract probe must use the registered class namespace")

class NPULauncher:
    def _make_launcher_stub_path(self):
        if self.template_entrypoint == "generate_npu_wrapper_src":
            wrapper_src = generate_npu_wrapper_src()
        elif self.template_entrypoint == "make_launcher":
            wrapper_src = make_launcher()
        else:
            raise AssertionError(self.template_entrypoint)
        return make_npu_launcher_stub(self.header_src, wrapper_src, self.debug)
""",
        namespace,
        namespace,
    )
    registry = {"ascend": types.SimpleNamespace(driver=namespace["NPUDriver"])}
    return namespace, registry, native_calls, factory_result


def capture_patched_source(overlay, module_alias: str, entrypoint: str,
                           excerpt_path: Path) -> str:
    original_source = excerpt_path.read_text(encoding="utf-8")
    if original_source.count("MsprofCompactInfo nodeBasicInfo;") != 2:
        raise AssertionError(
            f"{excerpt_path.name} must contain both captured CompactInfo report sections"
        )
    if original_source.count("MsprofAdditionalInfo tensorInfo;") != 2:
        raise AssertionError(
            f"{excerpt_path.name} must contain the two TensorInfo source blocks "
            "required by the current shared overlay contract"
        )

    header_source = HEADER_SOURCE
    namespace, registry, native_calls, factory_result = registered_backend(
        module_alias, original_source
    )
    original_factory = namespace[overlay.FACTORY_NAME]
    launcher_class = namespace["NPULauncher"]
    launcher = object.__new__(launcher_class)
    launcher.template_entrypoint = entrypoint
    launcher.header_src = header_source
    launcher.debug = entrypoint == "make_launcher"

    guard = overlay.install_registered_backend_patch(registry)
    try:
        result = launcher._make_launcher_stub_path()
    finally:
        guard.restore()

    if result is not factory_result:
        raise AssertionError("overlay changed the native factory return value")
    if namespace[overlay.FACTORY_NAME] is not original_factory:
        raise AssertionError("registered native factory binding was not restored")
    if len(native_calls) != 1:
        raise AssertionError(f"native factory call count changed: {len(native_calls)}")
    call = native_calls[0]
    if call["header_src"] != header_source:
        raise AssertionError("overlay changed header_src instead of only wrapper_src")
    if call["debug"] is not launcher.debug:
        raise AssertionError("overlay changed the factory debug argument")
    effective_source = call["wrapper_src"]
    if not isinstance(effective_source, str):
        raise AssertionError("native factory did not receive C++ wrapper source")
    report_call = (
        "MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), "
        "sizeof(MsprofCompactInfo));"
    )
    if effective_source.count(report_call) != 2:
        raise AssertionError(
            "the two original CompactInfo report calls and arguments were not preserved"
        )
    if effective_source.count("MsprofAdditionalInfo tensorInfo{};") != 2:
        raise AssertionError("the shared TensorInfo source patch did not pass through the factory")
    if effective_source.count("tensorInfo.dataLen = sizeof(MsprofTensorInfo);") != 2:
        raise AssertionError("the shared TensorInfo dataLen assignments were not preserved")
    return effective_source


def add_deterministic_poison(source: str) -> str:
    """Poison CompactInfo fields after construction, before source field writes.

    The writes make an otherwise uninitialized report deterministic without
    reading indeterminate bytes. Any producer assignments emitted after the
    declaration run later and must restore the ABI values before reporting.
    """
    declaration_count = len(RAW_COMPACT_DECLARATION.findall(source))
    if declaration_count != 2:
        raise AssertionError(
            "expected exactly two CompactInfo declarations in generated source; "
            f"found {declaration_count}"
        )

    def poison(match: re.Match[str]) -> str:
        indent = match.group("indent")
        initializer = match.group("initializer") or ""
        return (
            f"{indent}MsprofCompactInfo nodeBasicInfo{initializer};\n"
            f"{indent}nodeBasicInfo.dataLen = 0xA5A5A5A5u;\n"
            f"{indent}nodeBasicInfo.data.nodeBasicInfo.opFlag = 0xA5A5A5A5u;"
        )

    return RAW_COMPACT_DECLARATION.sub(poison, source)


def make_cpp_source(effective_wrapper_source: str) -> str:
    wrapper_source = add_deterministic_poison(effective_wrapper_source)
    return (
        HEADER_SOURCE
        + CPP_SUPPORT
        + r'''
int main()
{
    unsigned int __MsprofFlagL1 = 1;
    unsigned int threadId = kExpectedThreadId;
    unsigned long int endTime = kExpectedTimestamp;
    unsigned long int opNameHashID = kExpectedOpName;
    unsigned int nodeBasicBlockDim = kExpectedBlockDim;
    std::vector<int> tensorShapes;

'''
        + wrapper_source
        + r'''
    if (g_compact_call_count != kExpectedCallCount) {
        return Fail("expected exactly two original CompactInfo reports");
    }
    if (g_additional_call_count != kExpectedCallCount) {
        return Fail("expected exactly two original AdditionalInfo reports");
    }
    for (size_t index = 0; index < kExpectedCallCount; ++index) {
        const CapturedCompactInfo &record = g_captured[index];
        if (!record.has_data || record.data_address == 0 ||
            record.data_address % alignof(MsprofCompactInfo) != 0) {
            return Fail("CompactInfo report pointer was not preserved/aligned");
        }
        if (record.non_persistent_flag != 0 ||
            record.report_length != sizeof(MsprofCompactInfo)) {
            return Fail("MsprofReportCompactInfo call parameters changed");
        }
        if (record.magic_number != 0x5a5a ||
            record.level != MSPROF_REPORT_NODE_LEVEL ||
            record.type != MSPROF_REPORT_NODE_BASIC_INFO_TYPE ||
            record.thread_id != kExpectedThreadId ||
            record.data_len != sizeof(MsprofNodeBasicInfo) ||
            record.timestamp != kExpectedTimestamp) {
            return Fail("CompactInfo header identity/dataLen changed");
        }
        if (record.op_name != kExpectedOpName ||
            record.op_type != kExpectedOpName ||
            record.task_type != MSPROF_GE_TASK_TYPE_AIV ||
            record.block_dim != kExpectedBlockDim || record.op_flag != 0) {
            return Fail("MsprofNodeBasicInfo fields were not preserved/initialized");
        }
    }
    std::printf("COMPACT_HEADER_CONTRACT_PROBE_OK calls=%zu node_basic_size=%zu "
                "compact_size=%zu\n",
                g_compact_call_count, sizeof(MsprofNodeBasicInfo),
                sizeof(MsprofCompactInfo));
    return 0;
}
'''
    )


def compile_and_run(source: str, cxx: str, sdk_include_dir: Path,
                    label: str) -> None:
    with tempfile.TemporaryDirectory(prefix="compact-header-contract-") as temp_dir:
        work = Path(temp_dir)
        source_path = work / "compact_header_contract_probe.cpp"
        executable = work / "compact_header_contract_probe"
        source_path.write_text(source, encoding="utf-8")

        command = [
            cxx,
            "-std=c++11",
            "-Wall",
            "-Wextra",
            f"-I{sdk_include_dir}",
            str(source_path),
            "-o",
            str(executable),
        ]
        compiled = subprocess.run(
            command, cwd=work, text=True, capture_output=True, check=False
        )
        if compiled.returncode != 0:
            raise AssertionError(
                f"{label}: SDK contract probe compilation failed ({compiled.returncode})\n"
                f"command: {' '.join(command)}\nstdout:\n{compiled.stdout}\n"
                f"stderr:\n{compiled.stderr}"
            )

        result = subprocess.run(
            [str(executable)], cwd=work, text=True, capture_output=True, check=False
        )
        if result.returncode != 0:
            raise AssertionError(
                f"{label}: CompactInfo SDK contract failed ({result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if (
            "COMPACT_HEADER_CONTRACT_PROBE_OK calls=2 node_basic_size=" not in result.stdout
            or "compact_size=" not in result.stdout
        ):
            raise AssertionError(f"{label}: probe success marker missing: {result.stdout!r}")


def run(cxx: str, sdk_include_dir: Path, overlay_script: Path) -> None:
    sdk_include_dir = sdk_include_dir.resolve(strict=True)
    if not (sdk_include_dir / "profiling" / "aprof_pub.h").is_file():
        raise FileNotFoundError(
            "--sdk-include-dir must contain profiling/aprof_pub.h from the target SDK"
        )
    overlay = load_overlay(overlay_script.resolve(strict=True))

    cases = (
        (
            "local registered namespace / generate_npu_wrapper_src",
            "ascend",
            "generate_npu_wrapper_src",
            LOCAL_EXCERPT,
        ),
        (
            "physical registered namespace / make_launcher",
            "triton.backends.ascend.driver",
            "make_launcher",
            PHYSICAL_EXCERPT,
        ),
    )
    for label, module_alias, entrypoint, excerpt in cases:
        if not excerpt.is_file():
            raise FileNotFoundError(f"CompactInfo source fixture is missing: {excerpt}")
        patched_source = capture_patched_source(
            overlay, module_alias, entrypoint, excerpt
        )
        compile_and_run(
            make_cpp_source(patched_source), cxx, sdk_include_dir, label
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--sdk-include-dir", type=Path, required=True)
    parser.add_argument(
        "--overlay-script",
        type=Path,
        default=SCRIPTS / "triton_tensorinfo_overlay.py",
        help="overlay implementation to exercise (defaults to the workspace version)",
    )
    args = parser.parse_args()
    try:
        run(args.cxx, args.sdk_include_dir, args.overlay_script)
    except (AssertionError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"test_compact_header_contract: FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "test_compact_header_contract: PASS: registered local/physical factories "
        "preserve CompactInfo SDK fields and report calls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
