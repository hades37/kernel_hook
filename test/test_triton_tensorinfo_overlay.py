#!/usr/bin/env python3
"""CPU-only contract tests for the process-local Triton producer overlay.

These tests use fake registered backends and the native launcher-factory call
shape. They do not import Triton, compile a launcher, or require CANN/NPU.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import triton_tensorinfo_overlay as overlay


HEADER_SRC = "#include <runtime/runtime/rt.h>\n"
RAW_DECLARATION = "MsprofAdditionalInfo tensorInfo;"
INITIALIZED_DECLARATION = (
    "MsprofAdditionalInfo tensorInfo{};\n"
    "      tensorInfo.dataLen = sizeof(MsprofTensorInfo);"
)


def raw_wrapper_source(template: str = "fixture") -> str:
    """Stand in for either registered driver's generated wrapper source."""
    return f"""// {template} launcher template
void c_triton_launch_kernel() {{
      MsprofAdditionalInfo info;
      info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      int tensorCapacity = MSPROF_GE_TENSOR_DATA_NUM;
      int shapeCapacity = MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
}}
void python_launch_entrypoint() {{
      MsprofAdditionalInfo info;
      info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      int tensorCapacity = MSPROF_GE_TENSOR_DATA_NUM;
      int shapeCapacity = MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
}}
"""


def expected_initialized_source(source: str) -> str:
    return source.replace(RAW_DECLARATION, INITIALIZED_DECLARATION)


class RecordingFactory:
    """Keep both the callable's signature and its original argument shape."""

    __signature__ = inspect.Signature(
        [
            inspect.Parameter("header_src", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("wrapper_src", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                "debug", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=False
            ),
        ]
    )

    def __init__(self, calls, result=None, failure=None):
        self.calls = calls
        self.result = result
        self.failure = failure

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.failure is not None:
            raise self.failure
        return self.result


def make_registered_backend(
    module_name: str, wrapper_sources: dict[str, str] | None = None
):
    """Build a registry driver whose methods resolve globals in one namespace."""
    factory_calls: list[dict[str, object]] = []

    def source_builder(template: str) -> str:
        if wrapper_sources is not None and template in wrapper_sources:
            return wrapper_sources[template]
        return raw_wrapper_source(template)

    namespace: dict[str, object] = {
        "__name__": module_name,
        "_source_builder": source_builder,
        "_factory_calls": factory_calls,
        "hashlib": hashlib,
    }
    exec(
        """
def generate_npu_wrapper_src():
    return _source_builder("local-generate_npu_wrapper_src")

def make_launcher():
    return _source_builder("physical-make_launcher")

def make_npu_launcher_stub(header_src, wrapper_src, debug=False):
    cache_key = hashlib.sha256((header_src + "\\0" + wrapper_src).encode("utf-8")).hexdigest()
    _factory_calls.append({
        "header_src": header_src,
        "wrapper_src": wrapper_src,
        "debug": debug,
        "cache_key": cache_key,
    })
    return {"cache_key": cache_key}

class NPULauncher:
    def __init__(self, *args, **kwargs):
        raise AssertionError("the registered driver must not be instantiated")

    def _make_launcher_stub_path(self):
        if self.template_entrypoint == "generate_npu_wrapper_src":
            wrapper_src = generate_npu_wrapper_src()
        elif self.template_entrypoint == "make_launcher":
            wrapper_src = make_launcher()
        else:
            raise AssertionError(self.template_entrypoint)
        return make_npu_launcher_stub("registered-header", wrapper_src, self.debug)

class NPUDriver:
    def __init__(self, *args, **kwargs):
        raise AssertionError("the registered driver must not be instantiated")
""",
        namespace,
        namespace,
    )
    driver_class = namespace["NPUDriver"]
    launcher_class = namespace["NPULauncher"]
    assert isinstance(driver_class, type)
    assert isinstance(launcher_class, type)
    assert driver_class.__init__.__globals__ is namespace
    assert launcher_class._make_launcher_stub_path.__globals__ is namespace
    registry = {"ascend": types.SimpleNamespace(driver=driver_class)}
    return namespace, registry, factory_calls


def saved_generated_wrapper_sources() -> dict[str, str]:
    fixture_dir = Path(__file__).parent
    return {
        "local-generate_npu_wrapper_src": (
            fixture_dir / "triton_launcher_local_compact_excerpt.cxx"
        ).read_text(encoding="utf-8"),
        "physical-make_launcher": (
            fixture_dir / "triton_launcher_physical_compact_excerpt.cxx"
        ).read_text(encoding="utf-8"),
    }


COMPACT_REPORT = (
    "MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), "
    "sizeof(MsprofCompactInfo));"
)
COMPACT_REPORT_BLOCK = re.compile(
    r"MsprofCompactInfo nodeBasicInfo(?:\{\})?;"
    r"(?P<body>.*?MsprofReportCompactInfo\(0,\s*static_cast<void\s*\*>"
    r"\(&nodeBasicInfo\),\s*sizeof\(MsprofCompactInfo\)\);)",
    re.DOTALL,
)


def compact_report_blocks(source: str) -> list[str]:
    return [match.group(0) for match in COMPACT_REPORT_BLOCK.finditer(source)]


def source_with_explicit_hf32_compact_flags(source: str) -> str:
    assignment = (
        "      nodeBasicInfo.dataLen = sizeof(MsprofNodeBasicInfo);\n"
        "      nodeBasicInfo.data.nodeBasicInfo.opFlag = 1;\n"
    )
    if source.count(COMPACT_REPORT) != 2:
        raise AssertionError("expected two captured CompactInfo report sites")
    return source.replace(COMPACT_REPORT, assignment + COMPACT_REPORT)


def assert_default_compact_metadata(test_case: unittest.TestCase, source: str) -> None:
    blocks = compact_report_blocks(source)
    test_case.assertEqual(len(blocks), 2)
    for block in blocks:
        test_case.assertEqual(
            len(
                re.findall(
                    r"nodeBasicInfo\.dataLen\s*=\s*"
                    r"sizeof\(MsprofNodeBasicInfo\)\s*;",
                    block,
                )
            ),
            1,
        )
        test_case.assertEqual(
            len(
                re.findall(
                    r"nodeBasicInfo\.data\.nodeBasicInfo\.opFlag\s*=\s*0U?\s*;",
                    block,
                )
            ),
            1,
        )


def remove_default_compact_metadata(source: str) -> str:
    return re.sub(
        r"(?m)^[ \t]*(?P<name>[A-Za-z_]\w*)\.dataLen\s*=\s*"
        r"sizeof\(MsprofNodeBasicInfo\);\r?\n"
        r"^[ \t]*(?P=name)\.data\.nodeBasicInfo\.opFlag\s*=\s*0;\r?\n",
        "",
        source,
    )


class PatchWrapperSourceTests(unittest.TestCase):
    def test_initializes_both_tensor_info_records_and_preserves_other_source(self) -> None:
        for template in ("local-generate_npu_wrapper_src", "physical-make_launcher"):
            with self.subTest(template=template):
                source = raw_wrapper_source(template)
                self.assertEqual(source.count(RAW_DECLARATION), 2)

                patched = overlay.patch_wrapper_source(source)

                self.assertEqual(patched, expected_initialized_source(source))
                self.assertEqual(patched.count("MsprofAdditionalInfo tensorInfo{};"), 2)
                self.assertEqual(
                    patched.count("tensorInfo.dataLen = sizeof(MsprofTensorInfo);"), 2
                )
                self.assertEqual(patched.count("MsprofAdditionalInfo info;"), 2)
                self.assertEqual(patched.count("MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE"), 2)
                self.assertEqual(patched.count("MSPROF_REPORT_NODE_TENSOR_INFO_TYPE"), 2)
                self.assertEqual(patched.count("MSPROF_GE_TENSOR_DATA_NUM"), 2)
                self.assertEqual(patched.count("MSPROF_GE_TENSOR_DATA_SHAPE_LEN"), 2)

    def test_two_correctly_initialized_records_are_a_noop(self) -> None:
        source = expected_initialized_source(raw_wrapper_source("already-patched"))

        self.assertEqual(overlay.patch_wrapper_source(source), source)

    def test_partial_or_unknown_patch_shapes_fail_closed(self) -> None:
        source = raw_wrapper_source("partial")
        partial = source.replace(RAW_DECLARATION, INITIALIZED_DECLARATION, 1)
        unknown = source + "\n      " + RAW_DECLARATION + "\n"
        malformed = source.replace(
            RAW_DECLARATION,
            RAW_DECLARATION + "\n      tensorInfo.dataLen = 1;",
            1,
        )

        for value in (partial, unknown, malformed, "void unrelated_launcher() {}\n"):
            with self.subTest(source=value[:48]):
                with self.assertRaises(ValueError):
                    overlay.patch_wrapper_source(value)


class FactoryPatchTests(unittest.TestCase):
    def test_factory_arguments_cache_identity_and_return_value_are_preserved(self) -> None:
        source = raw_wrapper_source("factory-forwarding")
        expected_source = expected_initialized_source(source)
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        result = object()
        native_factory = RecordingFactory(calls, result=result)
        namespace = {"make_npu_launcher_stub": native_factory}
        baseline_key = hashlib.sha256((HEADER_SRC + "\0" + source).encode("utf-8")).hexdigest()

        with overlay.install_factory_patch(namespace):
            positional_result = namespace["make_npu_launcher_stub"](
                HEADER_SRC, source, True
            )

        self.assertIs(positional_result, result)
        self.assertEqual(calls[0], ((HEADER_SRC, expected_source, True), {}))
        self.assertIs(namespace["make_npu_launcher_stub"], native_factory)
        patched_key = hashlib.sha256(
            (HEADER_SRC + "\0" + calls[0][0][1]).encode("utf-8")
        ).hexdigest()
        self.assertNotEqual(patched_key, baseline_key)

        calls.clear()
        with overlay.install_factory_patch(namespace):
            keyword_result = namespace["make_npu_launcher_stub"](
                header_src=HEADER_SRC, wrapper_src=source, debug=False
            )

        self.assertIs(keyword_result, result)
        self.assertEqual(
            calls[0],
            ((), {"header_src": HEADER_SRC, "wrapper_src": expected_source, "debug": False}),
        )
        self.assertIs(namespace["make_npu_launcher_stub"], native_factory)

    def test_factory_exception_is_rethrown_unchanged_and_patch_is_restored(self) -> None:
        failure = RuntimeError("native launcher factory failed")
        audit_records: list[dict[str, object]] = []
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        native_factory = RecordingFactory(calls, failure=failure)
        namespace = {"make_npu_launcher_stub": native_factory}
        with self.assertRaises(RuntimeError) as raised:
            with overlay.install_factory_patch(
                namespace, audit_callback=audit_records.append
            ):
                namespace["make_npu_launcher_stub"](
                    HEADER_SRC, raw_wrapper_source("factory-failure"), True
                )

        self.assertIs(raised.exception, failure)
        self.assertIs(namespace["make_npu_launcher_stub"], native_factory)
        self.assertEqual(len(audit_records), 1)

    def test_restore_preserves_replacement_and_is_idempotent(self) -> None:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        native_factory = RecordingFactory(calls, result="native-result")

        def replacement_factory(*args, **kwargs):
            return "replacement"

        namespace = {"make_npu_launcher_stub": native_factory}
        guard = overlay.install_factory_patch(namespace)

        namespace["make_npu_launcher_stub"] = replacement_factory
        with self.assertRaisesRegex(RuntimeError, "preserving that binding"):
            guard.restore()
        self.assertIs(namespace["make_npu_launcher_stub"], replacement_factory)

        namespace["make_npu_launcher_stub"] = native_factory
        next_guard = overlay.install_factory_patch(namespace)
        namespace["make_npu_launcher_stub"] = native_factory
        next_guard.restore()
        next_guard.restore()
        self.assertIs(namespace["make_npu_launcher_stub"], native_factory)

    def test_audit_callback_failure_is_not_silently_ignored(self) -> None:
        audit_failure = OSError("audit sink failed")
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        native_factory = RecordingFactory(calls, result="native-result")
        def fail_audit(record):
            raise audit_failure

        namespace = {"make_npu_launcher_stub": native_factory}
        with overlay.install_factory_patch(namespace, audit_callback=fail_audit):
            with self.assertRaises(OSError) as raised:
                namespace["make_npu_launcher_stub"](
                    HEADER_SRC, raw_wrapper_source("audit-failure"), False
                )

        self.assertIs(raised.exception, audit_failure)
        self.assertIs(namespace["make_npu_launcher_stub"], native_factory)

    def test_audit_call_numbers_remain_unique_under_concurrent_factory_calls(self) -> None:
        audit_records: list[dict[str, object]] = []
        audit_lock = threading.Lock()

        def collect(record):
            with audit_lock:
                audit_records.append(record)

        def native_factory(header_src, wrapper_src, debug=False):
            return hashlib.sha256(
                (header_src + "\0" + wrapper_src).encode("utf-8")
            ).hexdigest()

        namespace = {"make_npu_launcher_stub": native_factory}
        call_count = 24
        with overlay.install_factory_patch(namespace, audit_callback=collect):
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(
                    lambda index: namespace["make_npu_launcher_stub"](
                        HEADER_SRC, raw_wrapper_source(f"concurrent-{index}"), index % 2 == 0
                    ),
                    range(call_count),
                ))

        self.assertEqual(len(results), call_count)
        call_numbers = [record["call"] for record in audit_records]
        self.assertEqual(sorted(call_numbers), list(range(1, call_count + 1)))


class CapturedCompactInfoFactoryTests(unittest.TestCase):
    def test_saved_local_and_physical_templates_get_defaults_through_registry_factory(self) -> None:
        sources = saved_generated_wrapper_sources()
        for source in sources.values():
            self.assertEqual(source.count(COMPACT_REPORT), 2)
            self.assertNotIn("nodeBasicInfo.dataLen =", source)
            self.assertNotIn("nodeBasicInfo.data.nodeBasicInfo.opFlag =", source)

        for module_name in ("ascend", "triton.backends.ascend.driver"):
            namespace, registry, factory_calls = make_registered_backend(
                module_name, wrapper_sources=sources
            )
            original_factory = namespace["make_npu_launcher_stub"]
            audit_records: list[dict[str, object]] = []

            with overlay.install_registered_backend_patch(
                registry, audit_callback=audit_records.append
            ):
                launcher_class = namespace["NPULauncher"]
                for entrypoint in (
                    "generate_npu_wrapper_src",
                    "make_launcher",
                ):
                    with self.subTest(module_name=module_name, entrypoint=entrypoint):
                        template_source = sources[
                            "local-generate_npu_wrapper_src"
                            if entrypoint == "generate_npu_wrapper_src"
                            else "physical-make_launcher"
                        ]
                        launcher = object.__new__(launcher_class)
                        launcher.template_entrypoint = entrypoint
                        launcher.debug = False
                        launcher._make_launcher_stub_path()

                        forwarded_source = factory_calls[-1]["wrapper_src"]
                        assert isinstance(forwarded_source, str)
                        assert_default_compact_metadata(self, forwarded_source)
                        self.assertEqual(
                            remove_default_compact_metadata(forwarded_source),
                            expected_initialized_source(template_source),
                        )
                        self.assertEqual(
                            overlay.patch_wrapper_source(forwarded_source),
                            forwarded_source,
                        )
                        reparsed_source, tensor_count, compact_count = (
                            overlay._patch_wrapper_source(forwarded_source)
                        )
                        self.assertEqual(reparsed_source, forwarded_source)
                        self.assertEqual(tensor_count, 0)
                        self.assertEqual(compact_count, 0)
                        self.assertEqual(
                            forwarded_source.count("MsprofCompactInfo nodeBasicInfo;"), 2
                        )
                        self.assertNotIn("MsprofCompactInfo nodeBasicInfo{};", forwarded_source)
                        self.assertEqual(forwarded_source.count(COMPACT_REPORT), 2)
                        self.assertEqual(
                            forwarded_source.count("MsprofReportAdditionalInfo("), 2
                        )
                        self.assertLess(
                            forwarded_source.index(COMPACT_REPORT),
                            forwarded_source.index("MsprofReportAdditionalInfo("),
                        )
                        self.assertEqual(audit_records[-1]["replacement_count"], 2)
                        self.assertEqual(
                            audit_records[-1]["compact_replacement_count"], 2
                        )

            self.assertIs(namespace["make_npu_launcher_stub"], original_factory)
            self.assertEqual(len(factory_calls), 2)

    def test_existing_hf32_compact_flags_are_preserved_or_rejected_before_factory(self) -> None:
        source = source_with_explicit_hf32_compact_flags(
            saved_generated_wrapper_sources()["local-generate_npu_wrapper_src"]
        )
        namespace, registry, factory_calls = make_registered_backend(
            "triton.backends.ascend.driver",
            wrapper_sources={"local-generate_npu_wrapper_src": source},
        )
        original_factory = namespace["make_npu_launcher_stub"]

        with overlay.install_registered_backend_patch(registry):
            launcher_class = namespace["NPULauncher"]
            launcher = object.__new__(launcher_class)
            launcher.template_entrypoint = "generate_npu_wrapper_src"
            launcher.debug = False
            try:
                launcher._make_launcher_stub_path()
            except (RuntimeError, ValueError):
                self.assertEqual(factory_calls, [])
            else:
                self.assertEqual(len(factory_calls), 1)
                forwarded_source = factory_calls[0]["wrapper_src"]
                assert isinstance(forwarded_source, str)
                blocks = compact_report_blocks(forwarded_source)
                self.assertEqual(len(blocks), 2)
                for block in blocks:
                    flags = re.findall(
                        r"nodeBasicInfo\.data\.nodeBasicInfo\.opFlag\s*=\s*([^;]+);",
                        block,
                    )
                    self.assertEqual(flags, ["1"])
                    self.assertRegex(
                        block,
                        r"nodeBasicInfo\.dataLen\s*=\s*"
                        r"sizeof\(MsprofNodeBasicInfo\)\s*;",
                    )

        self.assertIs(namespace["make_npu_launcher_stub"], original_factory)

    def test_one_correct_compact_block_is_preserved_while_the_other_is_repaired(self) -> None:
        source = saved_generated_wrapper_sources()["local-generate_npu_wrapper_src"]
        fixed_block = (
            "      nodeBasicInfo.dataLen = sizeof(MsprofNodeBasicInfo);\n"
            "      nodeBasicInfo.data.nodeBasicInfo.opFlag = MSPROF_GE_OP_FLAG_HF32;\n"
        )
        partial_source = source.replace(COMPACT_REPORT, fixed_block + COMPACT_REPORT, 1)
        namespace, registry, factory_calls = make_registered_backend(
            "triton.backends.ascend.driver",
            wrapper_sources={"local-generate_npu_wrapper_src": partial_source},
        )
        audit_records: list[dict[str, object]] = []

        with overlay.install_registered_backend_patch(
            registry, audit_callback=audit_records.append
        ):
            launcher_class = namespace["NPULauncher"]
            launcher = object.__new__(launcher_class)
            launcher.template_entrypoint = "generate_npu_wrapper_src"
            launcher.debug = False
            launcher._make_launcher_stub_path()

        forwarded_source = factory_calls[0]["wrapper_src"]
        assert isinstance(forwarded_source, str)
        blocks = compact_report_blocks(forwarded_source)
        self.assertEqual(len(blocks), 2)
        self.assertRegex(blocks[0], r"opFlag\s*=\s*MSPROF_GE_OP_FLAG_HF32\s*;")
        self.assertRegex(blocks[1], r"opFlag\s*=\s*0\s*;")
        self.assertEqual(audit_records[0]["replacement_count"], 2)
        self.assertEqual(audit_records[0]["compact_replacement_count"], 1)

    def test_conflicting_unknown_or_non_aiv_compact_shapes_fail_closed(self) -> None:
        source = saved_generated_wrapper_sources()["local-generate_npu_wrapper_src"]
        compact_blocks = list(COMPACT_REPORT_BLOCK.finditer(source))
        self.assertEqual(len(compact_blocks), 2)
        bad_sources = {
            "conflicting-data-len": source.replace(
                COMPACT_REPORT,
                "      nodeBasicInfo.dataLen = 1;\n" + COMPACT_REPORT,
                1,
            ),
            "duplicate-data-len": source.replace(
                COMPACT_REPORT,
                "      nodeBasicInfo.dataLen = sizeof(MsprofNodeBasicInfo);\n"
                "      nodeBasicInfo.dataLen = sizeof(MsprofNodeBasicInfo);\n"
                + COMPACT_REPORT,
                1,
            ),
            "duplicate-op-flag": source.replace(
                COMPACT_REPORT,
                "      nodeBasicInfo.data.nodeBasicInfo.opFlag = 0;\n"
                "      nodeBasicInfo.data.nodeBasicInfo.opFlag = 1;\n"
                + COMPACT_REPORT,
                1,
            ),
            "unknown-op-flag-expression": source.replace(
                COMPACT_REPORT,
                "      nodeBasicInfo.data.nodeBasicInfo.opFlag = get_op_flag();\n"
                + COMPACT_REPORT,
                1,
            ),
            "non-aiv-without-op-flag": source.replace(
                "nodeBasicInfo.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_AIV;",
                "nodeBasicInfo.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_AICORE;",
                1,
            ),
            "exec-mode-without-op-flag": source.replace(
                COMPACT_REPORT,
                "      nodeBasicInfo.data.nodeBasicInfo.execMode = 0;\n" + COMPACT_REPORT,
                1,
            ),
            "conditioned-data-len": source.replace(
                COMPACT_REPORT,
                "      if (should_set_length) nodeBasicInfo.dataLen = "
                "sizeof(MsprofNodeBasicInfo);\n"
                + COMPACT_REPORT,
                1,
            ),
            "conditioned-op-flag": source.replace(
                COMPACT_REPORT,
                "      if (use_hf32) nodeBasicInfo.data.nodeBasicInfo.opFlag = 1;\n"
                + COMPACT_REPORT,
                1,
            ),
            "whole-payload-overwrite": source.replace(
                COMPACT_REPORT,
                "      nodeBasicInfo.data = payload;\n" + COMPACT_REPORT,
                1,
            ),
            "payload-alias": source.replace(
                COMPACT_REPORT,
                "      auto &payload = nodeBasicInfo.data.nodeBasicInfo;\n"
                + COMPACT_REPORT,
                1,
            ),
            "exec-mode-alias": source.replace(
                COMPACT_REPORT,
                "      auto execMode = get_exec_mode();\n" + COMPACT_REPORT,
                1,
            ),
            "unsupported-report-cast": source.replace(
                "static_cast<void *>(&nodeBasicInfo)",
                "reinterpret_cast<void *>(&nodeBasicInfo)",
                1,
            ),
            "duplicate-report": source.replace(
                COMPACT_REPORT,
                COMPACT_REPORT + "\n      " + COMPACT_REPORT,
                1,
            ),
            "one-compact-block": (
                source[: compact_blocks[1].start()]
                + source[compact_blocks[1].end() :]
            ),
            "three-compact-blocks": (
                source + "\n" + compact_blocks[0].group(0) + "\n"
            ),
            "unknown-extra-declaration": source + "\nMsprofCompactInfo unreported;\n",
        }
        for name, bad_source in bad_sources.items():
            with self.subTest(shape=name):
                with self.assertRaises(ValueError):
                    overlay.patch_wrapper_source(bad_source)

    def test_tensor_and_compact_replacement_totals_remain_independent(self) -> None:
        source = saved_generated_wrapper_sources()["local-generate_npu_wrapper_src"]
        tensor_only_fixed_source = expected_initialized_source(source)
        factory_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def native_factory(header_src, wrapper_src, debug=False):
            factory_calls.append(((header_src, wrapper_src, debug), {}))
            return "native-result"

        with tempfile.TemporaryDirectory(prefix="compact audit ") as temporary:
            writer = overlay.AuditWriter(Path(temporary))
            namespace = {"make_npu_launcher_stub": native_factory}
            with overlay.install_factory_patch(namespace, audit_callback=writer):
                namespace["make_npu_launcher_stub"](HEADER_SRC, source, False)
                namespace["make_npu_launcher_stub"](
                    HEADER_SRC, tensor_only_fixed_source, False
                )

            records = [
                json.loads(line)
                for line in writer.calls_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["replacement_count"] for record in records], [2, 0])
            self.assertEqual(
                [record["compact_replacement_count"] for record in records], [2, 2]
            )
            snapshot = writer.snapshot()
            self.assertEqual(snapshot["replacement_total"], 2)
            self.assertEqual(snapshot["compact_replacement_total"], 4)
            self.assertEqual(len(factory_calls), 2)


class RegisteredBackendTests(unittest.TestCase):
    def test_registered_namespace_wins_over_both_module_aliases_and_templates(self) -> None:
        for module_name in ("ascend", "triton.backends.ascend.driver"):
            namespace, registry, factory_calls = make_registered_backend(module_name)
            original_factory = namespace["make_npu_launcher_stub"]
            decoy = types.ModuleType(module_name)

            def decoy_factory(header_src, wrapper_src, debug=False):
                return "decoy"

            decoy.make_npu_launcher_stub = decoy_factory
            audit_records: list[dict[str, object]] = []

            with self.subTest(module_name=module_name):
                with mock.patch.dict(sys.modules, {module_name: decoy}):
                    with overlay.install_registered_backend_patch(
                        registry, audit_callback=audit_records.append
                    ):
                        self.assertIsNot(
                            namespace["make_npu_launcher_stub"], original_factory
                        )
                        self.assertIs(decoy.make_npu_launcher_stub, decoy_factory)

                        launcher_class = namespace["NPULauncher"]
                        for entrypoint, debug in (
                            ("generate_npu_wrapper_src", False),
                            ("make_launcher", True),
                        ):
                            launcher = object.__new__(launcher_class)
                            launcher.template_entrypoint = entrypoint
                            launcher.debug = debug
                            result = launcher._make_launcher_stub_path()
                            self.assertEqual(
                                result["cache_key"], factory_calls[-1]["cache_key"]
                            )
                            self.assertEqual(
                                factory_calls[-1]["wrapper_src"],
                                expected_initialized_source(
                                    raw_wrapper_source(
                                        "local-generate_npu_wrapper_src"
                                        if entrypoint == "generate_npu_wrapper_src"
                                        else "physical-make_launcher"
                                    )
                                ),
                            )
                            self.assertEqual(factory_calls[-1]["debug"], debug)
                            self.assertNotEqual(result, "decoy")

                    self.assertIs(namespace["make_npu_launcher_stub"], original_factory)
                    self.assertIs(decoy.make_npu_launcher_stub, decoy_factory)

            self.assertEqual(len(factory_calls), 2)
            self.assertEqual(len(audit_records), 2)

    def test_real_cli_run_path_preserves_relative_argv_and_restores_process_state(self) -> None:
        namespace, registry, factory_calls = make_registered_backend(
            "triton.backends.ascend.driver"
        )
        native_factory = namespace["make_npu_launcher_stub"]

        triton_module = types.ModuleType("triton")
        triton_module.__path__ = []
        backends_module = types.ModuleType("triton.backends")
        backends_module.__path__ = []
        backends_module.backends = registry
        triton_module.backends = backends_module
        ascend_module = types.ModuleType("triton.backends.ascend")
        ascend_module.__path__ = []
        decoy_driver_module = types.ModuleType("triton.backends.ascend.driver")

        def decoy_factory(header_src, wrapper_src, debug=False):
            return "decoy"

        decoy_driver_module.make_npu_launcher_stub = decoy_factory
        fake_modules = {
            "triton": triton_module,
            "triton.backends": backends_module,
            "triton.backends.ascend": ascend_module,
            "triton.backends.ascend.driver": decoy_driver_module,
        }

        with tempfile.TemporaryDirectory(prefix="tensorinfo cli ") as temporary:
            temporary_path = Path(temporary)
            script_dir = temporary_path / "trainer dir"
            script_dir.mkdir()
            script_path = script_dir / "train script.py"
            relative_script = str(script_path.relative_to(temporary_path))
            script_path.write_text(
                "import json, sys\n"
                "from pathlib import Path\n"
                "from triton.backends import backends\n"
                "namespace = backends.get('ascend').driver.__init__.__globals__\n"
                "if sys.argv[4] == 'no-call':\n"
                "    result = None\n"
                "else:\n"
                "    launcher_class = namespace['NPULauncher']\n"
                "    launcher = object.__new__(launcher_class)\n"
                "    launcher.template_entrypoint = sys.argv[3]\n"
                "    launcher.debug = True\n"
                "    result = launcher._make_launcher_stub_path()\n"
                "Path(sys.argv[2]).write_text(json.dumps({\n"
                "    'argv': sys.argv, 'path0': sys.path[0], 'result': result,\n"
                "}), encoding='utf-8')\n"
                "if sys.argv[4] == 'exit-zero':\n"
                "    raise SystemExit(0)\n"
                "if sys.argv[4] == 'exit-nonzero':\n"
                "    raise SystemExit(7)\n"
                "if sys.argv[4] == 'raise':\n"
                "    raise RuntimeError('trainer script failure')\n",
                encoding="utf-8",
            )

            original_cwd = Path.cwd()
            original_argv_object = sys.argv
            original_argv = list(sys.argv)
            original_path_object = sys.path
            original_path = list(sys.path)
            try:
                os.chdir(temporary_path)
                with mock.patch.dict(sys.modules, fake_modules):
                    scenarios = (
                        ("normal", "generate_npu_wrapper_src", "ok", 0, True),
                        ("exit-zero", "make_launcher", "ok", 0, True),
                        ("exit-nonzero", "generate_npu_wrapper_src", "system_exit", 7, True),
                        ("raise", "make_launcher", "error", 1, True),
                        ("no-call", "generate_npu_wrapper_src", "no_launcher_calls", 0, False),
                    )
                    for index, (mode, template, status, exit_code, calls_factory) in enumerate(scenarios):
                        with self.subTest(mode=mode, template=template):
                            calls_before = len(factory_calls)
                            capture_path = temporary_path / f"capture {index}.json"
                            audit_dir = temporary_path / f"audit {index}"
                            script_arguments = [
                                relative_script,
                                "argument with spaces",
                                str(capture_path),
                                template,
                                mode,
                            ]
                            caught_error = None
                            try:
                                cli_exit = overlay.main(
                                    ["--audit-dir", str(audit_dir), "--", *script_arguments]
                                )
                            except BaseException as error:
                                caught_error = error
                                cli_exit = None

                            if mode in ("normal", "no-call"):
                                self.assertIsNone(caught_error)
                                self.assertEqual(cli_exit, 0)
                            elif mode.startswith("exit-"):
                                self.assertIsInstance(caught_error, SystemExit)
                                self.assertEqual(caught_error.code, exit_code)
                            else:
                                self.assertIsInstance(caught_error, RuntimeError)
                                self.assertEqual(str(caught_error), "trainer script failure")

                            captured = json.loads(capture_path.read_text(encoding="utf-8"))
                            self.assertEqual(
                                captured["argv"],
                                [
                                    relative_script,
                                    "argument with spaces",
                                    str(capture_path),
                                    template,
                                    mode,
                                ],
                            )
                            self.assertEqual(captured["path0"], str(script_dir))
                            if calls_factory:
                                self.assertEqual(len(factory_calls), calls_before + 1)
                                self.assertEqual(
                                    captured["result"]["cache_key"],
                                    factory_calls[-1]["cache_key"],
                                )
                                expected_template = (
                                    "local-generate_npu_wrapper_src"
                                    if template == "generate_npu_wrapper_src"
                                    else "physical-make_launcher"
                                )
                                self.assertEqual(
                                    factory_calls[-1]["wrapper_src"],
                                    expected_initialized_source(
                                        raw_wrapper_source(expected_template)
                                    ),
                                )
                            else:
                                self.assertIsNone(captured["result"])
                                self.assertEqual(len(factory_calls), calls_before)
                            self.assertIs(namespace["make_npu_launcher_stub"], native_factory)
                            self.assertIs(
                                decoy_driver_module.make_npu_launcher_stub, decoy_factory
                            )

                            manifest = json.loads(
                                (audit_dir / "run_manifest.json").read_text(encoding="utf-8")
                            )
                            self.assertEqual(manifest["run_status"], status)
                            self.assertEqual(manifest["exit_code"], exit_code)
                            self.assertEqual(manifest["launcher_calls"], int(calls_factory))
                            self.assertEqual(manifest["repair_activated"], calls_factory)
                            self.assertTrue(manifest["factory_restored"])
                            self.assertEqual(manifest["script_path_argument"], relative_script)
                            self.assertEqual(manifest["script_argv"], script_arguments)

                            self.assertIs(sys.argv, original_argv_object)
                            self.assertEqual(sys.argv, original_argv)
                            self.assertIs(sys.path, original_path_object)
                            self.assertEqual(sys.path, original_path)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
