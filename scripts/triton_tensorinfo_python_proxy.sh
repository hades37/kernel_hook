#!/usr/bin/env bash
set -Eeuo pipefail

real_python=${KIMIK3_PRODUCER_REAL_PYTHON:-}
if [[ -z "${real_python}" || "${real_python}" != /* || ! -x "${real_python}" ]]; then
    echo "triton_tensorinfo_python_proxy: KIMIK3_PRODUCER_REAL_PYTHON must be an absolute executable" >&2
    exit 64
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
overlay="${script_dir}/triton_tensorinfo_overlay.py"
trainer_relative=mindspeed_mm/fsdp/train/trainer.py
trainer_entry=no

if [[ ${1:-} == "${trainer_relative}" ]]; then
    trainer_entry=yes
elif [[ ${1:-} == /* && -n ${MINDSPEED_ROOT:-} ]]; then
    expected="${MINDSPEED_ROOT}/${trainer_relative}"
    actual_path=$(readlink -f -- "$1" 2>/dev/null || true)
    expected_path=$(readlink -f -- "${expected}" 2>/dev/null || true)
    if [[ -n "${actual_path}" && "${actual_path}" == "${expected_path}" ]]; then
        trainer_entry=yes
    fi
fi

if [[ "${trainer_entry}" != yes ]]; then
    exec "${real_python}" "$@"
fi

audit_dir=${KIMIK3_TENSORINFO_AUDIT_DIR:-${KIMIK3_PRODUCER_OUT:-}}
if [[ -z "${audit_dir}" && -n ${KIMIK3_FAKE_ARTIFACT_DIR:-} ]]; then
    audit_dir="${KIMIK3_FAKE_ARTIFACT_DIR}/producer-audit"
fi
if [[ -z "${audit_dir}" || "${audit_dir}" != /* ]]; then
    echo "triton_tensorinfo_python_proxy: set an absolute KIMIK3_TENSORINFO_AUDIT_DIR or KIMIK3_PRODUCER_OUT" >&2
    exit 64
fi

exec "${real_python}" "${overlay}" --audit-dir "${audit_dir}" -- "$@"
