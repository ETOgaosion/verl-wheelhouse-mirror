#!/usr/bin/env bash
# Builds the deep-ep (DeepEP) wheel with the toolchain verl's
# docker/Dockerfile.uv.cu130 sets up for its own source build of the same
# pinned commit - system NVSHMEM at that image's absolute path, CCCL headers on
# CPATH, a Hopper+Blackwell arch list - swapping the in-place `uv sync` build
# for `pip wheel --no-deps -w dist` so a distributable artifact is produced.
# Run with CWD = the DeepEP submodule checkout.
#
# Caveat carried over from that source build: setup.py's second extension
# (hybrid_ep_cpp) bakes `-DBASE_PATH="<build dir>"` in for its runtime JIT, so
# in a prebuilt wheel that path points at this runner's checkout. verl does not
# use the hybrid-EP backend, and its own build has the same property (uv builds
# in a throwaway temp env that is gone by run time), so nothing regresses here.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

export_extra_env

pip install -q ninja packaging wheel setuptools

# setup.py imports torch.utils.cpp_extension at module scope, so the build must
# run --no-build-isolation against the torch the workflow installed.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Host-compiler include path for CUDA 13's relocated libcu++/cub/thrust, the
# <infiniband/mlx5dv.h> that csrc/kernels/configs.cuh includes for IBGDA, and
# NVSHMEM (headers + libs) for the internode / low-latency kernels. Without
# NVSHMEM_DIR set, setup.py silently falls back to -DDISABLE_NVSHMEM and builds
# a wheel with those kernels compiled out.
ensure_cuda_cccl_include_path
ensure_cuda_stub_library_path
install_rdma_devel
install_nvshmem

# TORCH_CUDA_ARCH_LIST is already in the environment (_build.yml's "Build wheel"
# step exports it from versions.yaml); deep-ep's setup.py consumes the canonical
# dotted form as-is, so there is nothing to convert here.
#
# setup.py asserts this is 1 for any arch list other than exactly "9.0" - the
# aggressive `.L1::no_allocate` LD/ST tricks it guards are Hopper-only. That is
# already the default; setting it explicitly keeps the assert from firing if a
# future ref changes the default.
export DISABLE_AGGRESSIVE_PTX_INSTRS=1

# The hybrid_ep extension links -lnvtx3interop and -lcuda. Both come from the
# toolkit _build.yml installs: libnvtx3interop.so from its nvtx sub-package
# (in targets/<t>/lib, which $CUDA_HOME/lib64 symlinks to, and which torch puts
# on the link path), and the libcuda.so stub from driver-dev, reached via the
# LIBRARY_PATH set above.
mkdir -p dist
MAX_JOBS="${MAX_JOBS}" pip wheel -v \
  --no-build-isolation \
  --no-deps \
  -w dist \
  .

strip_wheel_local_version dist

echo "Built wheels:"
ls -al dist
