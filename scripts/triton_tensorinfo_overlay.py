#!/usr/bin/env python3
"""Repair TensorInfo producer initialization for one Python process."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import runpy
import sys
import threading
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any


FACTORY_NAME = "make_npu_launcher_stub"
DEFAULT_ZERO_OP_FLAG_TASK_TYPES = frozenset(
    {
        "MSPROF_GE_TASK_TYPE_AIV",
        "MSPROF_GE_TASK_TYPE_AI_CORE",
        "MSPROF_GE_TASK_TYPE_MIX_AIC",
        "MSPROF_GE_TASK_TYPE_MIX_AIV",
    }
)
RAW_DECLARATION = re.compile(
    r"(?m)^(?P<indent>[ \t]*)MsprofAdditionalInfo tensorInfo;[ \t]*(?P<cr>\r?)$"
)
INITIALIZED_DECLARATION = re.compile(
    r"(?m)^[ \t]*MsprofAdditionalInfo tensorInfo\{\};[ \t]*\r?$"
)
INITIALIZED_BLOCK = re.compile(
    r"(?m)^(?P<indent>[ \t]*)MsprofAdditionalInfo tensorInfo\{\};[ \t]*\r?\n"
    r"(?P=indent)tensorInfo\.dataLen[ \t]*=[ \t]*sizeof\(MsprofTensorInfo\);[ \t]*\r?$"
)
TENSOR_DATA_LEN_ASSIGNMENT = re.compile(r"(?m)^[ \t]*tensorInfo\.dataLen[ \t]*=")
COMPACT_DECLARATION = re.compile(
    r"(?m)^(?P<indent>[ \t]*)MsprofCompactInfo[ \t]+"
    r"(?P<name>[A-Za-z_]\w*)(?P<initializer>\{\})?;[ \t]*(?=\r?$)"
)
COMPACT_REPORT_CALL = re.compile(r"\bMsprofReportCompactInfo\s*\(")
COMPACT_TYPE_REFERENCE = re.compile(r"\bMsprofCompactInfo\b")


def _compact_report_pattern(name: str) -> re.Pattern[str]:
    escaped_name = re.escape(name)
    return re.compile(
        rf"\bMsprofReportCompactInfo\s*\(\s*[^,\r\n]+,\s*"
        rf"static_cast\s*<\s*void\s*\*>\s*\(\s*&\s*{escaped_name}\s*\)\s*,\s*"
        rf"sizeof\s*\(\s*MsprofCompactInfo\s*\)\s*\)\s*;"
    )


def _compact_member_assignment(
    name: str, member: str, rhs_name: str = "rhs"
) -> re.Pattern[str]:
    escaped_name = re.escape(name)
    if member == "dataLen":
        field = rf"{escaped_name}\.dataLen"
    elif member == "opFlag":
        field = rf"{escaped_name}\.data\.nodeBasicInfo\.opFlag"
    else:
        raise ValueError(f"unsupported CompactInfo member: {member}")
    return re.compile(rf"\b{field}\s*=\s*(?P<{rhs_name}>[^;\r\n]+?)\s*;")


def _recognized_simple_value(value: str) -> bool:
    value = value.strip()
    integer_literal = re.fullmatch(
        r"(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|0[0-7]*|[1-9][0-9]*)"
        r"(?:[uU](?:ll|LL|[lL])?|(?:ll|LL|[lL])[uU]?)?",
        value,
    )
    simple_symbol = re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", value)
    return integer_literal is not None or simple_symbol is not None


COMPACT_SIMPLE_ASSIGNMENT = re.compile(
    r"(?P<name>[A-Za-z_]\w*)\.(?P<field>level|magicNumber|type|threadId|timeStamp|"
    r"dataLen|data\.nodeBasicInfo\.(?:opName|opType|taskType|blockDim|opFlag))"
    r"\s*=\s*(?P<rhs>[^;\r\n]+?)\s*;\s*(?://.*)?"
)


def _validate_compact_block(block: str, name: str) -> None:
    declaration_pattern = re.compile(
        rf"MsprofCompactInfo\s+{re.escape(name)}(?:\{{\}})?;"
    )
    report_pattern = _compact_report_pattern(name)
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if declaration_pattern.fullmatch(stripped) or report_pattern.fullmatch(stripped):
            continue
        assignment = COMPACT_SIMPLE_ASSIGNMENT.fullmatch(stripped)
        if (
            assignment is not None
            and assignment.group("name") == name
            and assignment.group("field") == "dataLen"
        ):
            rhs = assignment.group("rhs").strip()
            if re.fullmatch(r"sizeof\s*\(\s*MsprofNodeBasicInfo\s*\)", rhs):
                continue
        if (
            assignment is None
            or assignment.group("name") != name
            or not _recognized_simple_value(assignment.group("rhs"))
        ):
            raise ValueError(
                f"CompactInfo {name!r} contains an unsupported statement or control-flow shape"
            )


def _patch_compact_info_source(source: str) -> tuple[str, int]:
    declarations = list(COMPACT_DECLARATION.finditer(source))
    report_calls = list(COMPACT_REPORT_CALL.finditer(source))
    type_references = list(COMPACT_TYPE_REFERENCE.finditer(source))

    if not declarations:
        if report_calls or type_references:
            raise ValueError(
                "launcher wrapper contains an unsupported CompactInfo declaration/report shape"
            )
        return source, 0

    if len(declarations) != 2 or any(
        declaration.group("name") != "nodeBasicInfo" for declaration in declarations
    ):
        raise ValueError("launcher wrapper must contain exactly two nodeBasicInfo blocks")

    if len(report_calls) != 2:
        raise ValueError(
            "each CompactInfo declaration must have exactly one supported report call"
        )

    additions: list[tuple[int, str]] = []
    covered_type_references: list[tuple[int, int]] = []
    replacement_count = 0

    for index, declaration in enumerate(declarations):
        name = declaration.group("name")
        next_declaration_start = (
            declarations[index + 1].start()
            if index + 1 < len(declarations)
            else len(source)
        )
        calls_in_block = [
            call
            for call in report_calls
            if declaration.end() <= call.start() < next_declaration_start
        ]
        if len(calls_in_block) != 1:
            raise ValueError(
                f"CompactInfo declaration {name!r} has an unknown report boundary"
            )

        report = calls_in_block[0]
        report_match = _compact_report_pattern(name).match(source, report.start())
        if report_match is None or report_match.end() > next_declaration_start:
            raise ValueError(
                f"CompactInfo declaration {name!r} has an unsupported report call"
            )

        block_start = declaration.start()
        block_end = report_match.end()
        block = source[block_start:block_end]
        _validate_compact_block(block, name)
        covered_type_references.extend(
            (match.start(), match.end())
            for match in type_references
            if block_start <= match.start() < block_end
        )

        data_len_reference = re.compile(rf"\b{re.escape(name)}\.dataLen\b")
        data_len_references = list(data_len_reference.finditer(block))
        data_len_assignments = list(_compact_member_assignment(name, "dataLen").finditer(block))
        if len(data_len_references) != len(data_len_assignments) or len(data_len_assignments) > 1:
            raise ValueError(f"CompactInfo {name!r} has an unknown or duplicate dataLen assignment")

        needs_data_len = not data_len_assignments
        if data_len_assignments and not re.fullmatch(
            r"sizeof\s*\(\s*MsprofNodeBasicInfo\s*\)",
            data_len_assignments[0].group("rhs").strip(),
        ):
            raise ValueError(f"CompactInfo {name!r} has a conflicting dataLen assignment")

        op_flag_reference = re.compile(
            rf"\b{re.escape(name)}\.data\.nodeBasicInfo\.opFlag\b"
        )
        op_flag_references = list(op_flag_reference.finditer(block))
        op_flag_assignments = list(_compact_member_assignment(name, "opFlag").finditer(block))
        if len(op_flag_references) != len(op_flag_assignments) or len(op_flag_assignments) > 1:
            raise ValueError(f"CompactInfo {name!r} has an unknown or duplicate opFlag assignment")

        needs_op_flag = not op_flag_assignments
        if op_flag_assignments and not _recognized_simple_value(
            op_flag_assignments[0].group("rhs")
        ):
            raise ValueError(f"CompactInfo {name!r} has an unrecognized opFlag assignment")

        if needs_op_flag:
            mode_or_exec_mode = re.compile(
                rf"\b{re.escape(name)}\.data\.nodeBasicInfo\.(?:mode|execMode)\b"
            )
            task_type_reference = re.compile(
                rf"\b{re.escape(name)}\.data\.nodeBasicInfo\.taskType\b"
            )
            task_type_assignment = re.compile(
                rf"\b{re.escape(name)}\.data\.nodeBasicInfo\.taskType\s*=\s*"
                r"(?P<rhs>[^;\r\n]+?)\s*;"
            )
            task_type_assignments = list(task_type_assignment.finditer(block))
            task_type_value = (
                task_type_assignments[0].group("rhs").strip()
                if len(task_type_assignments) == 1
                else "<missing-or-ambiguous>"
            )
            if (
                mode_or_exec_mode.search(block)
                or len(list(task_type_reference.finditer(block))) != 1
                or len(task_type_assignments) != 1
                or task_type_value not in DEFAULT_ZERO_OP_FLAG_TASK_TYPES
            ):
                raise ValueError(
                    f"CompactInfo {name!r} needs an explicit opFlag for unsupported "
                    f"taskType {task_type_value!r} or mode/execMode template"
                )

        assignments: list[str] = []
        if needs_data_len:
            assignments.append(f"{name}.dataLen = sizeof(MsprofNodeBasicInfo);")
        if needs_op_flag:
            assignments.append(f"{name}.data.nodeBasicInfo.opFlag = 0;")
        if assignments:
            line_ending = "\r\n" if source.startswith("\r\n", declaration.end()) else "\n"
            insertion = line_ending + line_ending.join(
                f"{declaration.group('indent')}{assignment}" for assignment in assignments
            )
            additions.append((declaration.end(), insertion))
            replacement_count += 1

    for reference in type_references:
        if not any(start <= reference.start() < end for start, end in covered_type_references):
            raise ValueError("launcher wrapper contains an unsupported CompactInfo type use")

    for position, insertion in reversed(additions):
        source = source[:position] + insertion + source[position:]
    return source, replacement_count


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _patch_wrapper_source(source: str) -> tuple[str, int, int]:
    if not isinstance(source, str):
        raise TypeError("generated launcher wrapper source must be a string")

    raw_count = len(RAW_DECLARATION.findall(source))
    initialized_count = len(INITIALIZED_DECLARATION.findall(source))
    initialized_block_count = len(INITIALIZED_BLOCK.findall(source))
    data_len_count = len(TENSOR_DATA_LEN_ASSIGNMENT.findall(source))

    if raw_count == 2 and initialized_count == 0 and data_len_count == 0:
        def initialize(match: re.Match[str]) -> str:
            indent = match.group("indent")
            line_ending = "\r\n" if match.group("cr") else "\n"
            return (
                f"{indent}MsprofAdditionalInfo tensorInfo{{}};{line_ending}"
                f"{indent}tensorInfo.dataLen = sizeof(MsprofTensorInfo);"
            )

        tensor_source, tensor_replacement_count = RAW_DECLARATION.subn(initialize, source)

    elif (
        raw_count == 0
        and initialized_count == 2
        and initialized_block_count == 2
        and data_len_count == 2
    ):
        tensor_source, tensor_replacement_count = source, 0
    else:
        raise ValueError(
            "launcher wrapper must contain exactly two uninitialized TensorInfo blocks "
            "or exactly two correctly initialized blocks; refusing partial/unknown source"
        )

    effective_source, compact_replacement_count = _patch_compact_info_source(tensor_source)
    return effective_source, tensor_replacement_count, compact_replacement_count


def patch_wrapper_source(source: str) -> str:
    """Initialize TensorInfo and supported CompactInfo metadata, preserving other C++."""
    return _patch_wrapper_source(source)[0]


def _parameter_index(signature: inspect.Signature, name: str) -> int:
    positional = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    try:
        return positional.index(name)
    except ValueError as error:
        raise RuntimeError(f"factory parameter {name!r} is not positional or keyword") from error


def _argument_value(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    parameter = signature.parameters.get(name)
    if parameter is None:
        raise RuntimeError(f"registered factory has no {name!r} parameter")
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
        raise TypeError(f"missing required factory argument: {name}")
    index = _parameter_index(signature, name)
    if index < len(args):
        return args[index]
    raise TypeError(f"missing required factory argument: {name}")


def _replace_argument(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    forwarded_kwargs = dict(kwargs)
    if name in forwarded_kwargs:
        forwarded_kwargs[name] = value
        return args, forwarded_kwargs

    parameter = signature.parameters.get(name)
    if parameter is None:
        raise RuntimeError(f"registered factory has no {name!r} parameter")
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
        forwarded_kwargs[name] = value
        return args, forwarded_kwargs

    index = _parameter_index(signature, name)
    if index < len(args):
        forwarded_args = list(args)
        forwarded_args[index] = value
        return tuple(forwarded_args), forwarded_kwargs

    forwarded_kwargs[name] = value
    return args, forwarded_kwargs


def _safe_error_fields(error: BaseException) -> dict[str, str]:
    try:
        message = str(error)
    except BaseException:
        message = "<exception message unavailable>"
    return {"error_type": type(error).__name__, "error_message": message}


def _add_secondary_error_note(primary: BaseException, source: str, error: BaseException) -> None:
    fields = _safe_error_fields(error)
    note = f"TensorInfo overlay {source} also failed: {fields['error_type']}: {fields['error_message']}"
    try:
        primary.add_note(note)
    except BaseException:
        try:
            setattr(primary, "_triton_tensorinfo_overlay_secondary_error", fields)
        except BaseException:
            # The active primary exception still propagates unchanged.
            pass


class PatchGuard:
    """Restore the exact factory binding installed by the overlay."""

    def __init__(self, namespace: MutableMapping[str, Any], original: Callable[..., Any], wrapped: Callable[..., Any]):
        self.namespace = namespace
        self.original = original
        self.wrapped = wrapped
        self._active = True

    def restore(self) -> None:
        if not self._active:
            return
        with _INSTALL_LOCK:
            current_factory = self.namespace.get(FACTORY_NAME)
            if current_factory is self.original:
                self._active = False
                return
            if current_factory is not self.wrapped:
                self._active = False
                raise RuntimeError(
                    "cannot restore TensorInfo factory overlay: the registered factory "
                    "binding was replaced by another component; preserving that binding"
                )
            self.namespace[FACTORY_NAME] = self.original
            self._active = False

    def __enter__(self) -> "PatchGuard":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.restore()
        return False


_INSTALL_LOCK = threading.RLock()


def install_factory_patch(
    namespace: MutableMapping[str, Any],
    *,
    audit_callback: Callable[[dict[str, Any]], Any] | None = None,
) -> PatchGuard:
    """Wrap one driver's native factory without replacing its compile/cache flow."""
    if not isinstance(namespace, MutableMapping):
        raise TypeError("registered driver globals must be a mutable mapping")
    original_factory = namespace.get(FACTORY_NAME)
    if not callable(original_factory):
        raise RuntimeError("registered driver has no callable make_npu_launcher_stub")
    if getattr(original_factory, "_triton_tensorinfo_overlay", False):
        raise RuntimeError("TensorInfo factory overlay is already installed")

    signature = inspect.signature(original_factory)
    if "wrapper_src" not in signature.parameters or "header_src" not in signature.parameters:
        raise RuntimeError("registered launcher factory has an unsupported signature")
    call_lock = threading.Lock()
    call_number = 0

    def emit(record: dict[str, Any], primary: BaseException | None = None) -> None:
        nonlocal call_number
        if audit_callback is None:
            return
        with call_lock:
            call_number += 1
            record["call"] = call_number
            try:
                audit_callback(record)
            except BaseException as audit_error:
                if primary is None:
                    raise
                _add_secondary_error_note(primary, "audit callback", audit_error)

    def record_source_failure(
        error: BaseException,
        header_src: Any = None,
        original_src: Any = None,
    ) -> None:
        record: dict[str, Any] = {
            "return_status": "error",
            "error_stage": "source_validation",
            "replacement_count": None,
            "compact_replacement_count": None,
            "original_cache_key": None,
            "effective_cache_key": None,
            "effective_cpp_sha256": None,
        }
        if isinstance(header_src, str):
            record["header_sha256"] = sha256_text(header_src)
        if isinstance(original_src, str):
            record["original_cpp_sha256"] = sha256_text(original_src)
        record.update(_safe_error_fields(error))
        emit(record, primary=error)

    def wrapped_factory(*args: Any, **kwargs: Any) -> Any:
        try:
            header_src = _argument_value(signature, args, kwargs, "header_src")
            original_src = _argument_value(signature, args, kwargs, "wrapper_src")
            if not isinstance(header_src, str) or not isinstance(original_src, str):
                raise TypeError("launcher source arguments must be strings")
            effective_src, replacement_count, compact_replacement_count = _patch_wrapper_source(
                original_src
            )
            forwarded_args, forwarded_kwargs = _replace_argument(
                signature, args, kwargs, "wrapper_src", effective_src
            )
            original_key = hashlib.sha256(
                (header_src + "\0" + original_src).encode("utf-8")
            ).hexdigest()
            effective_key = hashlib.sha256(
                (header_src + "\0" + effective_src).encode("utf-8")
            ).hexdigest()
            record: dict[str, Any] = {
                "header_sha256": sha256_text(header_src),
                "original_cpp_sha256": sha256_text(original_src),
                "effective_cpp_sha256": sha256_text(effective_src),
                "replacement_count": replacement_count,
                "compact_replacement_count": compact_replacement_count,
                "original_cache_key": original_key,
                "effective_cache_key": effective_key,
                "return_status": "running",
            }
        except BaseException as source_error:
            record_source_failure(
                source_error,
                locals().get("header_src"),
                locals().get("original_src"),
            )
            raise

        try:
            result = original_factory(*forwarded_args, **forwarded_kwargs)
        except BaseException as factory_error:
            record["return_status"] = "error"
            record["error_stage"] = "factory"
            record.update(_safe_error_fields(factory_error))
            emit(record, primary=factory_error)
            raise

        record["return_status"] = "ok"
        if isinstance(result, (str, os.PathLike)):
            try:
                result_path = Path(result).resolve()
                record["returned_so"] = str(result_path)
                if result_path.is_file():
                    digest = hashlib.sha256()
                    with result_path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    record["returned_so_sha256"] = digest.hexdigest()
                else:
                    record["returned_so_sha256"] = None
            except OSError as error:
                record["returned_so"] = os.fspath(result)
                record["returned_so_sha256"] = None
                record["returned_so_hash_error"] = _safe_error_fields(error)
        else:
            record["returned_so"] = None
            record["returned_so_type"] = type(result).__name__

        emit(record)
        return result

    wrapped_factory.__wrapped__ = original_factory  # type: ignore[attr-defined]
    wrapped_factory._triton_tensorinfo_overlay = True  # type: ignore[attr-defined]
    wrapped_factory.__name__ = getattr(original_factory, "__name__", FACTORY_NAME)
    with _INSTALL_LOCK:
        if namespace.get(FACTORY_NAME) is not original_factory:
            raise RuntimeError("registered launcher factory changed during installation")
        namespace[FACTORY_NAME] = wrapped_factory
    return PatchGuard(namespace, original_factory, wrapped_factory)


def install_registered_backend_patch(
    backend_registry: Any = None,
    *,
    audit_callback: Callable[[dict[str, Any]], Any] | None = None,
) -> PatchGuard:
    """Install into the registered Ascend driver namespace, without a second import."""
    if backend_registry is None:
        from triton.backends import backends

        backend_registry = backends
    backend = backend_registry.get("ascend")
    driver_class = getattr(backend, "driver", None) if backend is not None else None
    if driver_class is None:
        raise RuntimeError("registered Triton Ascend backend is unavailable")

    driver_init = getattr(driver_class, "__init__", None)
    namespace = getattr(driver_init, "__globals__", None)
    if not isinstance(namespace, MutableMapping):
        raise RuntimeError("registered Ascend driver has no Python globals namespace")
    if namespace.get("__name__") != getattr(driver_class, "__module__", None):
        raise RuntimeError("registered driver class module does not match its globals namespace")

    launcher_class = namespace.get("NPULauncher")
    launcher_path = getattr(launcher_class, "_make_launcher_stub_path", None)
    if launcher_path is None or getattr(launcher_path, "__globals__", None) is not namespace:
        raise RuntimeError("registered NPULauncher does not use the registered driver namespace")
    if getattr(launcher_class, "__module__", None) != namespace.get("__name__"):
        raise RuntimeError("NPULauncher class module does not match its globals namespace")

    return install_factory_patch(namespace, audit_callback=audit_callback)


class AuditWriter:
    def __init__(self, audit_dir: Path):
        self.audit_dir = audit_dir
        self.calls_path = audit_dir / "launcher_calls.jsonl"
        self._lock = threading.Lock()
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.replacement_total = 0
        self.compact_replacement_total = 0

    def __call__(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, default=str)
        with self._lock:
            with self.calls_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
            self.call_count += 1
            if record.get("return_status") == "ok":
                self.success_count += 1
            elif record.get("return_status") == "error":
                self.error_count += 1
            self.replacement_total += int(record.get("replacement_count", 0) or 0)
            self.compact_replacement_total += int(
                record.get("compact_replacement_count", 0) or 0
            )

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "launcher_calls": self.call_count,
                "launcher_successes": self.success_count,
                "launcher_errors": self.error_count,
                "replacement_total": self.replacement_total,
                "compact_replacement_total": self.compact_replacement_total,
            }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fresh_audit_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("--audit-dir must be an absolute path")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("--audit-dir must be a fresh or empty directory")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _parse_args(argv: list[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, help="fresh absolute audit directory")
    if "--" not in argv:
        parser.error("expected -- followed by a Python script and its arguments")
    separator = argv.index("--")
    options = parser.parse_args(argv[:separator])
    script_argv = argv[separator + 1 :]
    if not script_argv or script_argv[0].startswith("-"):
        parser.error("expected a script path after --")
    try:
        audit_dir = _fresh_audit_dir(options.audit_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return audit_dir, script_argv


def _script_exit_code(error: BaseException) -> int:
    if isinstance(error, SystemExit):
        code = error.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    if isinstance(error, KeyboardInterrupt):
        return 130
    return 1


def _registered_driver_manifest(namespace: MutableMapping[str, Any], driver_class: type) -> dict[str, Any]:
    path_value = namespace.get("__file__")
    driver_path = Path(path_value).resolve() if isinstance(path_value, str) else None
    try:
        driver_hash = sha256_bytes(driver_path.read_bytes()) if driver_path and driver_path.is_file() else None
    except OSError:
        driver_hash = None
    return {
        "registered_driver_class": driver_class.__qualname__,
        "registered_driver_class_module": driver_class.__module__,
        "driver_namespace_name": namespace.get("__name__"),
        "driver_namespace_file": str(driver_path) if driver_path else None,
        "driver_sha256": driver_hash,
        "launcher_class": namespace["NPULauncher"].__qualname__,
    }


def main(argv: list[str] | None = None) -> int:
    audit_dir, script_argv = _parse_args(sys.argv[1:] if argv is None else argv)
    script_argument = script_argv[0]
    script_path = Path(script_argument)
    if not script_path.is_absolute():
        script_path = Path.cwd() / script_path
    script_path = script_path.resolve(strict=True)
    if not script_path.is_file():
        raise FileNotFoundError(f"Python script is missing: {script_path}")

    manifest_path = audit_dir / "run_manifest.json"
    audit_writer = AuditWriter(audit_dir)
    manifest: dict[str, Any] = {
        "run_status": "running",
        "exit_code": None,
        "script_path_argument": script_argument,
        "script_path_resolved": str(script_path),
        "script_argv": script_argv,
        "working_directory": os.getcwd(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "overlay_path": str(Path(__file__).resolve()),
    }
    _write_manifest(manifest_path, manifest)

    original_argv_object = sys.argv
    original_argv = list(sys.argv)
    original_path_object = sys.path
    original_path = list(sys.path)
    guard: PatchGuard | None = None
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None

    try:
        from triton.backends import backends

        backend = backends.get("ascend")
        driver_class = getattr(backend, "driver", None) if backend is not None else None
        if driver_class is None:
            raise RuntimeError("registered Triton Ascend backend is unavailable")
        namespace = driver_class.__init__.__globals__
        manifest.update(_registered_driver_manifest(namespace, driver_class))
        guard = install_registered_backend_patch(backends, audit_callback=audit_writer)

        sys.argv = list(script_argv)
        sys.path[:] = original_path
        if sys.path:
            sys.path[0] = str(script_path.parent)
        else:
            sys.path.append(str(script_path.parent))
        runpy.run_path(script_argument, run_name="__main__")
        manifest["run_status"] = "ok"
        manifest["exit_code"] = 0
    except BaseException as error:
        primary_error = error
        exit_code = _script_exit_code(error)
        manifest["run_status"] = (
            "ok" if isinstance(error, SystemExit) and exit_code == 0
            else "system_exit" if isinstance(error, SystemExit)
            else "error"
        )
        manifest["exit_code"] = exit_code
        if not (isinstance(error, SystemExit) and exit_code == 0):
            manifest.update(_safe_error_fields(error))
        raise
    finally:
        try:
            if guard is not None:
                guard.restore()
        except BaseException as error:
            cleanup_error = error

        try:
            original_argv_object[:] = original_argv
            sys.argv = original_argv_object
            original_path_object[:] = original_path
            sys.path = original_path_object
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
            else:
                _add_secondary_error_note(cleanup_error, "argument/path restoration", error)

        manifest.update(audit_writer.snapshot())
        if manifest.get("exit_code") == 0 and audit_writer.call_count == 0:
            manifest["run_status"] = "no_launcher_calls"
        manifest["repair_activated"] = bool(
            manifest.get("replacement_total", 0) > 0
            or manifest.get("compact_replacement_total", 0) > 0
        )
        manifest["factory_restored"] = (
            guard.namespace.get(FACTORY_NAME) is guard.original if guard is not None else None
        )
        try:
            _write_manifest(manifest_path, manifest)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
            else:
                _add_secondary_error_note(cleanup_error, "manifest write", error)

        if primary_error is not None:
            if cleanup_error is not None:
                _add_secondary_error_note(primary_error, "overlay cleanup", cleanup_error)
        elif cleanup_error is not None:
            raise cleanup_error

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
