#!/usr/bin/env bash
# Builds the flash-mla (FlashMLA) wheel, mirroring the source build verl's
# docker/Dockerfile.uv.cu130 runs for the same pinned commit (CCCL headers on
# CPATH, no build isolation so setup.py sees the installed torch), with
# `pip wheel --no-deps -w dist` in place of the in-place `uv sync` build.
# Run with CWD = the FlashMLA submodule checkout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

export_extra_env

pip install -q ninja packaging wheel setuptools

# setup.py imports torch.utils.cpp_extension at module scope, so the build must
# run --no-build-isolation against the torch the workflow installed.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

ensure_cuda_cccl_include_path

# FlashMLA emits its own -gencode flags (sm_90a + sm_100f) and ignores
# TORCH_CUDA_ARCH_LIST, which is why versions.yaml leaves torch_cuda_arch_list
# null for this component; FLASH_MLA_DISABLE_SM90 / FLASH_MLA_DISABLE_SM100 are
# the switches to reach for if a future matrix needs to drop one of them.
#
# Its setup.py defaults NVCC_THREADS to 32. Effective concurrency here is
# MAX_JOBS x NVCC_THREADS, so that would oversubscribe a 4 vCPU GitHub runner
# by an order of magnitude and get nvcc OOM-killed; versions.yaml pins a sane
# value through extra_env.
export NVCC_THREADS="${NVCC_THREADS:-2}"

# csrc/cutlass is FlashMLA's own submodule. _build.yml already checked it out
# (`git submodule update --init --recursive --depth 1`); setup.py re-runs that
# init itself at import time, which is then a no-op instead of a full-depth
# clone of cutlass.
mkdir -p dist
MAX_JOBS="${MAX_JOBS}" pip wheel -v \
  --no-build-isolation \
  --no-deps \
  -w dist \
  .

strip_wheel_local_version dist

echo "Built wheels:"
ls -al dist
