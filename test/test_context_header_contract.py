#!/usr/bin/env python3
"""Exercise the registered Triton factory's ContextInfo output against SDK ABI.

This CPU-only probe combines the full ContextInfo report branches saved from
the local and physical drivers with their existing launcher report excerpts.
It forwards each generated wrapper through the registered-backend factory
overlay, instruments raw local declarations in test-only storage poisoned
before placement construction, and compiles/captures the real SDK structs.
No installed driver, CANN runtime, or NPU is loaded.

Run with the SDK include directory that contains profiling/aprof_pub.h, e.g.:
  python3 test/test_context_header_contract.py --cxx /usr/bin/g++ \\
    --sdk-include-dir /root/miniconda3/envs/cann910/Ascend/cann-9.1.0/include/experiment/msprof

To demonstrate the known RED against a frozen overlay, pass its script with
--overlay-script. The current overlay is used by default.
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
LOCAL_LAUNCHER = ROOT / "test" / "triton_launcher_local_compact_excerpt.cxx"
PHYSICAL_LAUNCHER = ROOT / "test" / "triton_launcher_physical_compact_excerpt.cxx"
LOCAL_CONTEXT = ROOT / "test" / "triton_context_local_excerpt.cxx"
PHYSICAL_CONTEXT = ROOT / "test" / "triton_context_physical_excerpt.cxx"

RAW_TASK_TYPE_ASSIGNMENT = (
    "nodeBasicInfo.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_AIV;"
)
CONTEXT_SPLIT = "/* NEXT_CONTEXT_REPORT_SITE */"
CONTEXT_DECLARATIONS = {
    "info": re.compile(
        r"(?m)^(?P<indent>[ \t]*)MsprofAdditionalInfo[ \t]+info"
        r"(?P<initializer>[ \t]*\{[ \t]*\})?;[ \t]*\r?$"
    ),
    "ctxId": re.compile(
        r"(?m)^(?P<indent>[ \t]*)MsprofContextIdInfo[ \t]+ctxId"
        r"(?P<initializer>[ \t]*\{[ \t]*\})?;[ \t]*\r?$"
    ),
}

HEADER_SOURCE = r'''#include <profiling/aprof_pub.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>
'''

CPP_SUPPORT = r'''
namespace {
constexpr uint64_t kExpectedOpName = UINT64_C(0x1122334455667788);
constexpr uint32_t kExpectedThreadId = UINT32_C(0x3456);
constexpr uint64_t kExpectedTimestamp = UINT64_C(0x778899AABBCCDDEE);
constexpr uint32_t kExpectedBlockDim = UINT32_C(73);
constexpr uint32_t kExpectedTaskType = __EXPECTED_TASK_TYPE__;
constexpr bool kExpectedMix = __EXPECTED_MIX__;
constexpr bool kProfileEnabled = __PROFILE_ENABLED__;
constexpr size_t kCompactReports = 2;
constexpr size_t kAdditionalReportsMax = 4;

static_assert(sizeof(MsprofContextIdInfo) <= MSPROF_ADDTIONAL_INFO_DATA_LENGTH,
              "SDK ContextIdInfo does not fit AdditionalInfo payload");

struct CapturedCompact {
    bool has_data{};
    uint32_t report_flag{};
    uint32_t report_size{};
    uintptr_t address{};
    uint16_t magic{};
    uint16_t level{};
    uint32_t type{};
    uint32_t thread_id{};
    uint32_t data_len{};
    uint64_t timestamp{};
    uint64_t op_name{};
    uint64_t op_type{};
    uint32_t task_type{};
    uint32_t block_dim{};
    uint32_t op_flag{};
};

struct CapturedAdditional {
    bool has_data{};
    uint32_t report_flag{};
    uint32_t report_size{};
    uintptr_t address{};
    uint16_t magic{};
    uint16_t level{};
    uint32_t type{};
    uint32_t thread_id{};
    uint32_t data_len{};
    uint64_t timestamp{};
    uint64_t payload_op_name{};
    uint32_t payload_ctx_count{};
    uint32_t payload_ctx_ids[MSPROF_CTX_ID_MAX_NUM]{};
};

CapturedCompact g_compact[kCompactReports]{};
CapturedAdditional g_additional[kAdditionalReportsMax]{};
size_t g_compact_count = 0;
size_t g_additional_count = 0;

int Fail(const char *message)
{
    std::fprintf(stderr, "context_header_contract_probe: %s\n", message);
    return 1;
}
}  // namespace

extern "C" int32_t MsprofReportCompactInfo(uint32_t reportFlag,
                                            const VOID_PTR data,
                                            uint32_t length)
{
    if (g_compact_count >= kCompactReports) {
        return -1;
    }
    CapturedCompact &captured = g_compact[g_compact_count++];
    captured.has_data = data != nullptr;
    captured.report_flag = reportFlag;
    captured.report_size = length;
    captured.address = reinterpret_cast<uintptr_t>(data);
    if (data == nullptr) {
        return 102;
    }
    const auto *record = static_cast<const MsprofCompactInfo *>(data);
    captured.magic = record->magicNumber;
    captured.level = record->level;
    captured.type = record->type;
    captured.thread_id = record->threadId;
    std::memcpy(&captured.data_len, &record->dataLen, sizeof(captured.data_len));
    captured.timestamp = record->timeStamp;
    captured.op_name = record->data.nodeBasicInfo.opName;
    captured.op_type = record->data.nodeBasicInfo.opType;
    captured.task_type = record->data.nodeBasicInfo.taskType;
    captured.block_dim = record->data.nodeBasicInfo.blockDim;
    captured.op_flag = record->data.nodeBasicInfo.opFlag;
    return 102;
}

extern "C" int32_t MsprofReportAdditionalInfo(uint32_t reportFlag,
                                               const VOID_PTR data,
                                               uint32_t length)
{
    if (g_additional_count >= kAdditionalReportsMax) {
        return -1;
    }
    CapturedAdditional &captured = g_additional[g_additional_count++];
    captured.has_data = data != nullptr;
    captured.report_flag = reportFlag;
    captured.report_size = length;
    captured.address = reinterpret_cast<uintptr_t>(data);
    if (data == nullptr) {
        return 103;
    }
    const auto *record = static_cast<const MsprofAdditionalInfo *>(data);
    captured.magic = record->magicNumber;
    captured.level = record->level;
    captured.type = record->type;
    captured.thread_id = record->threadId;
    std::memcpy(&captured.data_len, &record->dataLen, sizeof(captured.data_len));
    captured.timestamp = record->timeStamp;
    if (record->type == MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE) {
        MsprofContextIdInfo context{};
        std::memcpy(&context, record->data, sizeof(context));
        captured.payload_op_name = context.opName;
        captured.payload_ctx_count = context.ctxIdNum;
        for (size_t i = 0; i < MSPROF_CTX_ID_MAX_NUM; ++i) {
            captured.payload_ctx_ids[i] = context.ctxIds[i];
        }
    } else if (record->type == MSPROF_REPORT_NODE_TENSOR_INFO_TYPE) {
        std::memcpy(&captured.payload_op_name, record->data,
                    sizeof(captured.payload_op_name));
    }
    return 103;
}
'''


def load_overlay(path: Path):
    module_name = "_context_header_contract_overlay"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load overlay module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def registered_backend(module_alias: str, wrapper_source: str, failure=None):
    native_calls: list[dict[str, Any]] = []
    factory_result = object()
    namespace: dict[str, Any] = {
        "__name__": module_alias,
        "_wrapper_source": wrapper_source,
        "_native_calls": native_calls,
        "_factory_result": factory_result,
        "_factory_error": failure,
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
    if _factory_error is not None:
        raise _factory_error
    return _factory_result

class NPUDriver:
    def __init__(self, *args, **kwargs):
        raise AssertionError("probe must use the registered driver class namespace")

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


def load_context_sites(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"ContextInfo source fixture is missing: {path}")
    source = path.read_text(encoding="utf-8")
    sites = source.split(CONTEXT_SPLIT)
    if len(sites) != 2:
        raise AssertionError(f"{path.name} must contain two ContextInfo report sites")
    return [site.strip() for site in sites]


def compose_wrapper_source(
    launcher_path: Path,
    context_path: Path,
    task_type: str,
    is_mix: bool,
) -> str:
    source = launcher_path.read_text(encoding="utf-8")
    if source.count(RAW_TASK_TYPE_ASSIGNMENT) != 2:
        raise AssertionError(f"{launcher_path.name} must retain both CompactInfo blocks")
    source = source.replace(
        RAW_TASK_TYPE_ASSIGNMENT,
        RAW_TASK_TYPE_ASSIGNMENT.replace("MSPROF_GE_TASK_TYPE_AIV", task_type),
    )
    context_sites = load_context_sites(context_path)
    condition = "true" if is_mix else "false"
    context_sites = [site.replace("__IS_MIX_TASK_TYPE__", condition) for site in context_sites]
    tensor_markers = source.split("      // Report tensor info")
    if len(tensor_markers) != 3:
        raise AssertionError(f"{launcher_path.name} must contain two tensor report sites")
    return (
        tensor_markers[0]
        + "\n"
        + context_sites[0]
        + "\n\n      // Report tensor info"
        + tensor_markers[1]
        + "\n"
        + context_sites[1]
        + "\n\n      // Report tensor info"
        + tensor_markers[2]
    )


def invoke_registered_factory(overlay, module_alias: str, entrypoint: str,
                              source: str, debug: bool):
    namespace, registry, native_calls, factory_result = registered_backend(
        module_alias, source
    )
    original_factory = namespace[overlay.FACTORY_NAME]
    launcher = object.__new__(namespace["NPULauncher"])
    launcher.template_entrypoint = entrypoint
    launcher.header_src = "// context header contract fixture"
    launcher.debug = debug

    guard = overlay.install_registered_backend_patch(registry)
    try:
        result = launcher._make_launcher_stub_path()
    finally:
        guard.restore()

    if result is not factory_result:
        raise AssertionError("overlay changed the native factory return value")
    if namespace[overlay.FACTORY_NAME] is not original_factory:
        raise AssertionError("registered factory binding was not restored")
    if len(native_calls) != 1:
        raise AssertionError(f"native factory call count changed: {len(native_calls)}")
    call = native_calls[0]
    if call["header_src"] != launcher.header_src or call["debug"] is not debug:
        raise AssertionError("overlay changed native factory arguments")
    expected_source = overlay.patch_wrapper_source(source)
    if call["wrapper_src"] != expected_source:
        raise AssertionError("registered native factory did not receive patched C++ source")
    return call["wrapper_src"]


def verify_native_factory_error(overlay, module_alias: str, entrypoint: str,
                                source: str, debug: bool) -> None:
    failure = RuntimeError("intentional native factory failure")
    namespace, registry, native_calls, _ = registered_backend(
        module_alias, source, failure=failure
    )
    original_factory = namespace[overlay.FACTORY_NAME]
    launcher = object.__new__(namespace["NPULauncher"])
    launcher.template_entrypoint = entrypoint
    launcher.header_src = "// context header error fixture"
    launcher.debug = debug

    guard = overlay.install_registered_backend_patch(registry)
    caught = None
    try:
        launcher._make_launcher_stub_path()
    except BaseException as error:
        caught = error
    finally:
        guard.restore()

    if caught is not failure:
        raise AssertionError("native factory exception identity was not preserved")
    if namespace[overlay.FACTORY_NAME] is not original_factory:
        raise AssertionError("factory binding was not restored after native error")
    if len(native_calls) != 1:
        raise AssertionError("native factory error path call count changed")


def instrument_context_storage(source: str) -> tuple[str, dict[str, list[bool]]]:
    """Re-run each source declaration's actual init form on poisoned storage.

    The test instrumentation is applied only to the wrapper returned through
    the registered factory. Placement construction with no initializer keeps
    poison bytes; placement construction with `{}` performs C++ value-init and
    zeroes them before the original producer assignments run.
    """
    states: dict[str, list[bool]] = {name: [] for name in CONTEXT_DECLARATIONS}

    def replace_declarations(kind: str, cxx_type: str) -> None:
        pattern = CONTEXT_DECLARATIONS[kind]

        def replace(match: re.Match[str]) -> str:
            index = len(states[kind])
            states[kind].append(match.group("initializer") is not None)
            indent = match.group("indent")
            storage = f"__context_probe_{kind}_{index}"
            initializer = match.group("initializer") or ""
            if initializer:
                initializer = "{}"
            return (
                f"{indent}alignas({cxx_type}) unsigned char {storage}[sizeof({cxx_type})];\n"
                f"{indent}std::memset({storage}, 0xA5, sizeof({cxx_type}));\n"
                f"{indent}{cxx_type} &{kind} = *::new "
                f"(static_cast<void *>({storage})) {cxx_type}{initializer};"
            )

        nonlocal source
        source, _ = pattern.subn(replace, source)

    replace_declarations("info", "MsprofAdditionalInfo")
    replace_declarations("ctxId", "MsprofContextIdInfo")
    if len(states["info"]) != 2 or len(states["ctxId"]) != 2:
        raise AssertionError(
            "expected two ContextInfo declarations of each SDK type; "
            f"info={len(states['info'])}, ctxId={len(states['ctxId'])}"
        )
    return source, states


def make_cpp_source(wrapper_source: str, task_type: str, is_mix: bool,
                    profile_enabled: bool) -> str:
    support = CPP_SUPPORT.replace("__EXPECTED_TASK_TYPE__", task_type)
    support = support.replace("__EXPECTED_MIX__", "true" if is_mix else "false")
    support = support.replace(
        "__PROFILE_ENABLED__", "true" if profile_enabled else "false"
    )
    return (
        HEADER_SOURCE
        + support
        + r'''
int main()
{
    auto execute_generated_wrapper = [](unsigned int profile_flag) {
        unsigned int __MsprofFlagL1 = profile_flag;
        unsigned int threadId = kExpectedThreadId;
        unsigned long int endTime = kExpectedTimestamp;
        unsigned long int opNameHashID = kExpectedOpName;
        unsigned int nodeBasicBlockDim = kExpectedBlockDim;
        std::vector<int> tensorShapes;
'''
        + wrapper_source
        + r'''
    };

    execute_generated_wrapper(kProfileEnabled ? 1U : 0U);
    const size_t compact_after_requested_run = g_compact_count;
    const size_t additional_after_requested_run = g_additional_count;
    execute_generated_wrapper(0U);
    if (g_compact_count != compact_after_requested_run ||
        g_additional_count != additional_after_requested_run) {
        return Fail("__MsprofFlagL1=0 changed the original report guard behavior");
    }

    const size_t expected_compact = kProfileEnabled ? kCompactReports : 0;
    const size_t expected_context = kProfileEnabled && kExpectedMix ? 2 : 0;
    const size_t expected_tensor = kProfileEnabled ? 2 : 0;
    if (g_compact_count != expected_compact ||
        g_additional_count != expected_context + expected_tensor) {
        return Fail("original CompactInfo/ContextInfo/TensorInfo report counts changed");
    }

    for (size_t index = 0; index < g_compact_count; ++index) {
        const CapturedCompact &record = g_compact[index];
        if (!record.has_data || record.address == 0 ||
            record.address % alignof(MsprofCompactInfo) != 0 ||
            record.report_flag != 0 || record.report_size != sizeof(MsprofCompactInfo) ||
            record.magic != 0x5a5a || record.level != MSPROF_REPORT_NODE_LEVEL ||
            record.type != MSPROF_REPORT_NODE_BASIC_INFO_TYPE ||
            record.thread_id != kExpectedThreadId ||
            record.data_len != sizeof(MsprofNodeBasicInfo) ||
            record.timestamp != kExpectedTimestamp ||
            record.op_name != kExpectedOpName || record.op_type != kExpectedOpName ||
            record.task_type != kExpectedTaskType ||
            record.block_dim != kExpectedBlockDim || record.op_flag != 0) {
            return Fail("neighboring CompactInfo header/report contract changed");
        }
    }

    size_t context_count = 0;
    size_t tensor_count = 0;
    for (size_t index = 0; index < g_additional_count; ++index) {
        const CapturedAdditional &record = g_additional[index];
        if (!record.has_data || record.address == 0 ||
            record.address % alignof(MsprofAdditionalInfo) != 0 ||
            record.report_flag != 0 ||
            record.report_size != sizeof(MsprofAdditionalInfo) ||
            record.magic != MSPROF_REPORT_DATA_MAGIC_NUM ||
            record.level != MSPROF_REPORT_NODE_LEVEL ||
            record.thread_id != kExpectedThreadId ||
            record.timestamp != kExpectedTimestamp) {
            return Fail("AdditionalInfo header/report parameters changed");
        }
        if (record.type == MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE) {
            ++context_count;
            if (record.data_len != sizeof(MsprofContextIdInfo) ||
                record.payload_op_name != kExpectedOpName ||
                record.payload_ctx_count != 1 || record.payload_ctx_ids[0] != 0) {
                return Fail("ContextInfo dataLen/opName/ctxIdNum/ctxIds[0] contract failed");
            }
            for (size_t ctx_index = 1; ctx_index < MSPROF_CTX_ID_MAX_NUM; ++ctx_index) {
                if (record.payload_ctx_ids[ctx_index] != 0) {
                    return Fail("value-initialized ContextIdInfo tail is not zero");
                }
            }
        } else if (record.type == MSPROF_REPORT_NODE_TENSOR_INFO_TYPE) {
            ++tensor_count;
            if (record.data_len != sizeof(MsprofTensorInfo) ||
                record.payload_op_name != kExpectedOpName) {
                return Fail("neighboring TensorInfo report contract changed");
            }
        } else {
            return Fail("unexpected AdditionalInfo report type");
        }
    }
    if (context_count != expected_context || tensor_count != expected_tensor) {
        return Fail("MIX-only ContextInfo guard or TensorInfo report count changed");
    }
    std::printf("CONTEXT_HEADER_CONTRACT_PROBE_OK context_reports=%zu "
                "context_size=%zu additional_size=%zu context_slots=%u\n",
                context_count, sizeof(MsprofContextIdInfo),
                sizeof(MsprofAdditionalInfo), MSPROF_CTX_ID_MAX_NUM);
    return 0;
}
'''
    )


def compile_and_run(source: str, cxx: str, sdk_include_dir: Path,
                    label: str) -> str:
    with tempfile.TemporaryDirectory(prefix="context-header-contract-") as temp_dir:
        work = Path(temp_dir)
        source_path = work / "context_header_contract_probe.cpp"
        executable = work / "context_header_contract_probe"
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
                f"{label}: SDK probe compilation failed ({compiled.returncode})\n"
                f"command: {' '.join(command)}\nstdout:\n{compiled.stdout}\n"
                f"stderr:\n{compiled.stderr}"
            )
        result = subprocess.run(
            [str(executable)], cwd=work, text=True, capture_output=True, check=False
        )
        if result.returncode != 0:
            raise AssertionError(
                f"{label}: ContextInfo SDK contract failed ({result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if "CONTEXT_HEADER_CONTRACT_PROBE_OK context_reports=" not in result.stdout:
            raise AssertionError(f"{label}: success marker missing: {result.stdout!r}")
        return result.stdout.strip()


def run(cxx: str, sdk_include_dir: Path, overlay_script: Path) -> None:
    sdk_include_dir = sdk_include_dir.resolve(strict=True)
    if not (sdk_include_dir / "profiling" / "aprof_pub.h").is_file():
        raise FileNotFoundError(
            "--sdk-include-dir must contain profiling/aprof_pub.h from the target SDK"
        )
    overlay = load_overlay(overlay_script.resolve(strict=True))
    cases = (
        (
            "local registered namespace",
            "ascend",
            "generate_npu_wrapper_src",
            LOCAL_LAUNCHER,
            LOCAL_CONTEXT,
        ),
        (
            "physical registered namespace",
            "triton.backends.ascend.driver",
            "make_launcher",
            PHYSICAL_LAUNCHER,
            PHYSICAL_CONTEXT,
        ),
    )

    for label, module_alias, entrypoint, launcher_path, context_path in cases:
        for task_type, is_mix in (
            ("MSPROF_GE_TASK_TYPE_MIX_AIC", True),
            ("MSPROF_GE_TASK_TYPE_MIX_AIV", True),
            ("MSPROF_GE_TASK_TYPE_AIV", False),
            ("MSPROF_GE_TASK_TYPE_AI_CORE", False),
        ):
            source = compose_wrapper_source(
                launcher_path, context_path, task_type, is_mix
            )
            patched_source = invoke_registered_factory(
                overlay, module_alias, entrypoint, source,
                debug=entrypoint == "make_launcher",
            )
            verify_native_factory_error(
                overlay, module_alias, entrypoint, source,
                debug=entrypoint == "make_launcher",
            )
            instrumented, initializer_states = instrument_context_storage(patched_source)
            label_with_task = f"{label} / {task_type}"
            output = compile_and_run(
                make_cpp_source(instrumented, task_type, is_mix, True),
                cxx,
                sdk_include_dir,
                label_with_task,
            )
            if not all(initializer_states["info"]) or not all(initializer_states["ctxId"]):
                raise AssertionError(
                    f"{label_with_task}: ContextInfo and ContextIdInfo must be value-initialized; "
                    f"source declarations={initializer_states}"
                )
            print(f"{label_with_task}: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--sdk-include-dir", type=Path, required=True)
    parser.add_argument(
        "--overlay-script",
        type=Path,
        default=SCRIPTS / "triton_tensorinfo_overlay.py",
        help="overlay implementation to exercise (defaults to workspace version)",
    )
    args = parser.parse_args()
    try:
        run(args.cxx, args.sdk_include_dir, args.overlay_script)
    except (
        AssertionError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(f"test_context_header_contract: FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "test_context_header_contract: PASS: registered local/physical MIX context "
        "reports satisfy the SDK header and payload contract"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
