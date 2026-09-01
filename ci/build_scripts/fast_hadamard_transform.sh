#!/usr/bin/env bash
# Builds the fast-hadamard-transform wheel, mirroring the build step in
# fast-hadamard-transform's own .github/workflows/publish.yml (its setup.py is a
# close relative of flash-attention's, down to the cached-wheel bdist_wheel
# override, so this script is a close relative of flash_attention.sh).
# Run with CWD = the fast-hadamard-transform submodule checkout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

export_extra_env

pip install -q ninja packaging wheel setuptools

# setup.py imports torch at module scope, and `python setup.py bdist_wheel`
# below runs without build isolation, so it sees the torch the workflow
# installed.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Without FORCE_BUILD, setup.py's CachedWheelsCommand tries to download a
# prebuilt wheel from upstream's releases first, guessing a URL from the torch /
# CUDA / Python / ABI combination. Upstream publishes nothing for cu13 +
# torch 2.11, so that guess 404s and it falls back to a source build anyway -
# but building a *wheelhouse* wheel by rehosting somebody else's is not what we
# want even when the guess does hit, so short-circuit it. Same reason
# flash_attention.sh sets FLASH_ATTENTION_FORCE_BUILD.
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE
export FAST_HADAMARD_TRANSFORM_FORCE_CXX11_ABI="${CXX11_ABI}"

# No TORCH_CUDA_ARCH_LIST handling: setup.py derives its gencode flags from the
# toolkit version alone and never reads that variable, which is why
# versions.yaml leaves torch_cuda_arch_list null for this component.
#
# NVCC_THREADS is likewise left at setup.py's own default of 4; see the max_jobs
# comment in versions.yaml.
MAX_JOBS="${MAX_JOBS}" python setup.py bdist_wheel --dist-dir=dist

# No strip_wheel_local_version call: unlike deep-ep / flash-mla, this version is
# plain unless FAST_HADAMARD_TRANSFORM_LOCAL_VERSION is set, and it isn't.
# (Upstream's publish.yml appends "+cu..torch..cxx11abi.." by *renaming the file*
# after the build, which desyncs the filename from the wheel's own METADATA -
# deliberately not reproduced here.)
echo "Built wheels:"
ls -al dist
