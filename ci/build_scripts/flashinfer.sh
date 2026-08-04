#!/usr/bin/env bash
# Builds flashinfer wheels for the wheelhouse:
#   - flashinfer-python: compiled from the pinned flashinfer submodule checkout
#   - flashinfer-cubin:    prebuilt wheel from https://flashinfer.ai/whl
#   - flashinfer-jit-cache: prebuilt wheel from https://flashinfer.ai/whl/cu<XY>
#
# Mirrors sglang/docker/Dockerfile "PARALLEL STAGE 3: FlashInfer Cache" and verl's
# Dockerfiles: cubin is CUDA-version-agnostic; jit-cache is fetched from the
# CUDA-specific flashinfer.ai index.
#
# Of those three, only jit-cache is arch-specific, so on a non-x86_64 build
# this script emits jit-cache alone (see BUILD_ANY_ARCH_WHEELS below).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

export_extra_env

pip install -q build wheel ninja numpy

mkdir -p dist

# flashinfer-python and flashinfer-cubin are both py3-none-any: one wheel
# serves every architecture, and pip matches it on arm64 just as well as on
# x86_64. Building them on more than one arch would produce byte-identical
# wheels that then race the same asset name onto the same release, so the
# x86_64 row owns them and the arm64 row contributes only the arch-specific
# flashinfer-jit-cache below. (TARGET_ARCH is exported by _build.yml and is
# the build's target arch, which it has already asserted matches `uname -m`.)
BUILD_ANY_ARCH_WHEELS=false
if [ "${TARGET_ARCH:-x86_64}" = "x86_64" ]; then
  BUILD_ANY_ARCH_WHEELS=true
fi

if [ "${BUILD_ANY_ARCH_WHEELS}" = true ]; then
  echo "::group::flashinfer-python (JIT core wheel)"
  python -m build --wheel --outdir dist .
  echo "::endgroup::"
else
  echo "Skipping flashinfer-python on ${TARGET_ARCH}: py3-none-any wheel is built on x86_64 only"
fi

FLASHINFER_VERSION="$(tr -d '[:space:]' < version.txt)"
FLASHINFER_CU_INDEX="$(flashinfer_jit_cache_cu_index "${CUDA_VERSION}")"
FLASHINFER_CUBIN_INDEX="https://flashinfer.ai/whl"
FLASHINFER_JIT_CACHE_INDEX="https://flashinfer.ai/whl/${FLASHINFER_CU_INDEX}"

echo "flashinfer version=${FLASHINFER_VERSION} cuda=${CUDA_VERSION} jit-cache index=${FLASHINFER_JIT_CACHE_INDEX}"

if [ "${BUILD_ANY_ARCH_WHEELS}" = true ]; then
  echo "::group::flashinfer-cubin (prebuilt wheel)"
  download_prebuilt_wheel \
    "flashinfer-cubin" \
    "${FLASHINFER_VERSION}" \
    "${FLASHINFER_CUBIN_INDEX}" \
    dist
  echo "::endgroup::"
else
  echo "Skipping flashinfer-cubin on ${TARGET_ARCH}: py3-none-any wheel is rehosted from x86_64 only"
fi

echo "::group::flashinfer-jit-cache (prebuilt wheel)"
download_prebuilt_wheel \
  "flashinfer-jit-cache" \
  "${FLASHINFER_VERSION}" \
  "${FLASHINFER_JIT_CACHE_INDEX}" \
  dist
echo "::endgroup::"

echo "Built wheels:"
ls -al dist
