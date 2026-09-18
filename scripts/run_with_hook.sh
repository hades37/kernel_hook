#!/bin/bash
# Run any application with the msprof hook preloaded.
#
# Usage:
#   ./run_with_hook.sh <output_dir> <app> [args...]
#
# Example:
#   source /home/developer/Ascend/cann/bin/setenv.bash
#   ./run_with_hook.sh /tmp/hook_out ./my_app arg1 arg2
#   ./run_with_hook.sh /tmp/hook_out msprof --output=/tmp/prof --task-time=on ./my_app

set -e

OUT_DIR="${1:?usage: run_with_hook.sh <output_dir> <app> [args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK_LIB="${SCRIPT_DIR}/../build/libmsprof_hook.so"

if [ ! -f "${HOOK_LIB}" ]; then
    echo "[run_with_hook] hook library not found, running build.sh ..."
    bash "${SCRIPT_DIR}/../build.sh"
fi

mkdir -p "${OUT_DIR}"

export LD_PRELOAD="${HOOK_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
export MSPROF_HOOK_OUT="${OUT_DIR}"
export MSPROF_HOOK_VERBOSE="${MSPROF_HOOK_VERBOSE:-1}"

echo "[run_with_hook] LD_PRELOAD=${LD_PRELOAD}"
echo "[run_with_hook] MSPROF_HOOK_OUT=${MSPROF_HOOK_OUT}"
echo "[run_with_hook] exec: $*"
exec "$@"
