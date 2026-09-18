#!/bin/bash
# ---------------------------------------------------------------------------
# Build script for kernel_hook (LD_PRELOAD hook library).
#
# Usage:
#   bash build.sh              # build (Release)
#   bash build.sh -j 16        # parallel jobs
#   bash build.sh --debug      # build with Debug flags
#   bash build.sh clean        # remove build dir
#   bash build.sh clean build  # clean then build
#
# Output:
#   build/libmsprof_hook.so
# ---------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="Release"
JOBS="$(nproc 2>/dev/null || echo 8)"
DO_CLEAN=0
DO_BUILD=1

usage() {
    sed -n '2,15p' "$0"
}

# ---- parse args ----
while [ $# -gt 0 ]; do
    case "$1" in
        clean)
            DO_CLEAN=1
            shift
            ;;
        build)
            DO_BUILD=1
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        -j)
            JOBS="$2"
            shift 2
            ;;
        -j*)
            JOBS="${1#-j}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[build] unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ "${DO_CLEAN}" -eq 1 ]; then
    echo "[build] cleaning ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

if [ "${DO_BUILD}" -eq 0 ]; then
    exit 0
fi

# ---- tool check ----
if ! command -v cmake >/dev/null 2>&1; then
    echo "[build] cmake not found in PATH" >&2
    exit 1
fi
if ! command -v g++ >/dev/null 2>&1; then
    echo "[build] g++ not found in PATH" >&2
    exit 1
fi

echo "[build] source dir : ${SCRIPT_DIR}"
echo "[build] build dir  : ${BUILD_DIR}"
echo "[build] build type : ${BUILD_TYPE}"
echo "[build] jobs       : ${JOBS}"

mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "${BUILD_DIR}" -j "${JOBS}"

HOOK_LIB="${BUILD_DIR}/libmsprof_hook.so"
if [ ! -f "${HOOK_LIB}" ]; then
    echo "[build] expected artifact not found: ${HOOK_LIB}" >&2
    exit 1
fi

echo "[build] done: ${HOOK_LIB}"
