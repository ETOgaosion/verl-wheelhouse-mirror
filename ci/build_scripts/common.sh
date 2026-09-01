#!/usr/bin/env bash
# Shared helpers sourced by every ci/build_scripts/<component>.sh script (and,
# for free_disk_space/install_cudnn, by .github/workflows/_build.yml itself).
#
# The calling workflow is expected to have already exported the relevant
# matrix fields as environment variables before invoking a build script, e.g.
# CUDA_VERSION, TORCH_CUDA_ARCH_LIST, MAX_JOBS, EXTRA_ENV (see _build.yml's
# "Build wheel" step). This file must be *sourced*, not executed.

set -euo pipefail

# ---------------------------------------------------------------------------
# maybe_sudo: run a privileged command through sudo only where that is both
# needed and possible. GitHub-hosted runners build as an unprivileged user
# with passwordless sudo; the self-hosted machine (apex, TransformerEngine)
# builds as root in a container that has no sudo binary at all.
# ---------------------------------------------------------------------------
maybe_sudo() {
  if [ "$(id -u)" -eq 0 ] || ! command -v sudo >/dev/null 2>&1; then
    "$@"
  else
    sudo "$@"
  fi
}

# ---------------------------------------------------------------------------
# free_disk_space: same cleanup flash-attention's own _build.yml runs before
# a CUDA build, to claw back space on standard GitHub-hosted runners. Only
# called for those (see _build.yml) - on the persistent self-hosted machine
# these paths belong to the host, not to a throwaway VM image.
# ---------------------------------------------------------------------------
free_disk_space() {
  echo "::group::Free up disk space"
  maybe_sudo rm -rf /usr/share/dotnet || true
  maybe_sudo rm -rf /opt/ghc || true
  maybe_sudo rm -rf /opt/hostedtoolcache/CodeQL || true
  maybe_sudo rm -rf /usr/local/lib/android || true
  echo "::endgroup::"
}

# ---------------------------------------------------------------------------
# install_cudnn: mirrors the cuDNN network-repo install both verl Dockerfiles
# run before building TransformerEngine. Expects CUDA_VERSION to be exported.
# ---------------------------------------------------------------------------
install_cudnn() {
  local cuda_major arch
  cuda_major="$(echo "${CUDA_VERSION}" | cut -d. -f1)"
  arch="$(uname -m)"
  if [ "${arch}" = "aarch64" ]; then
    arch="sbsa"
  fi

  echo "::group::Install cuDNN ${cuda_major}"
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${arch}/cuda-keyring_1.1-1_all.deb"
  maybe_sudo dpkg -i cuda-keyring_1.1-1_all.deb
  rm -f cuda-keyring_1.1-1_all.deb
  maybe_sudo apt-get update
  maybe_sudo apt-get install -y --allow-downgrades --allow-change-held-packages \
    "cudnn9-cuda-${cuda_major}"
  echo "::endgroup::"
}

# ---------------------------------------------------------------------------
# install_nccl: TransformerEngine's common/util/logging.h always includes
# nccl.h, and NCCL EP (enabled when NVTE_CUDA_ARCHS contains arch >= 90)
# links libnccl at build time. Mirrors TE's own wheel Dockerfile
# (libnccl2 + libnccl-dev) and vllm/docker/Dockerfile's CUDA-matched pin.
# Expects CUDA_VERSION to be exported and the NVIDIA CUDA apt repo to already
# be configured (Jimver/cuda-toolkit's network install step does this).
# ---------------------------------------------------------------------------
install_nccl() {
  local cuda_short nccl_ver
  cuda_short="$(echo "${CUDA_VERSION}" | cut -d. -f1,2)"

  echo "::group::Install NCCL (+cuda${cuda_short})"
  maybe_sudo apt-get update
  nccl_ver="$(
    apt-cache madison libnccl-dev 2>/dev/null \
      | grep "+cuda${cuda_short}" \
      | head -1 \
      | awk -F'|' '{gsub(/^ +| +$/, "", $2); print $2}'
  )"
  if [ -z "${nccl_ver}" ]; then
    echo "::error::No libnccl-dev package found for +cuda${cuda_short}" >&2
    exit 1
  fi
  maybe_sudo apt-get install -y --no-install-recommends --allow-change-held-packages \
    "libnccl-dev=${nccl_ver}" "libnccl2=${nccl_ver}"
  echo "Installed libnccl-dev=${nccl_ver} libnccl2=${nccl_ver}"
  echo "::endgroup::"
}

# ---------------------------------------------------------------------------
# ensure_cuda_cccl_include_path: put CUDA 13's CCCL headers on CPATH.
#
# CUDA 13 moved libcu++ / cub / thrust out of the toolkit's top-level include/
# and into include/cccl/. nvcc finds them there on its own, but the *host*
# compiler does not, so a `#include <cuda/std/...>` reached from a .cpp
# translation unit (deep-ep's csrc/deep_ep.cpp, FlashMLA's cutlass headers)
# fails to resolve without help. verl's own Dockerfiles do the same thing -
# `export CPATH=/usr/local/cuda/targets/<target>/include/cccl:$CPATH` in
# docker/Dockerfile.stable.vllm / .sglang, a /usr/local/cuda-cccl symlink on
# CPATH in docker/Dockerfile.uv.cu130. A no-op warning on toolkits that
# predate the move (their headers are already on the default search path).
# ---------------------------------------------------------------------------
ensure_cuda_cccl_include_path() {
  local cuda_home target include_dir
  cuda_home="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
  case "$(uname -m)" in
    aarch64) target="sbsa-linux" ;;
    *) target="x86_64-linux" ;;
  esac

  for include_dir in \
    "${cuda_home}/targets/${target}/include/cccl" \
    "${cuda_home}/include/cccl"; do
    if [ -d "${include_dir}" ]; then
      export CPATH="${include_dir}${CPATH:+:${CPATH}}"
      echo "CCCL headers: added ${include_dir} to CPATH"
      return 0
    fi
  done

  echo "::warning::No cccl include directory found under ${cuda_home}; leaving CPATH unchanged."
}

# ---------------------------------------------------------------------------
# install_nvshmem: install NVIDIA's NVSHMEM pip package at the same absolute
# path verl's runtime image puts it, and export NVSHMEM_DIR pointing there.
#
# deep-ep links NVSHMEM and bakes `-Wl,-rpath,${NVSHMEM_DIR}/lib` into
# deep_ep_cpp.so, so a *prebuilt* wheel only resolves libnvshmem at run time if
# that same directory exists on the target machine. verl's
# docker/Dockerfile.uv.cu130 deliberately installs nvidia-nvshmem-cu<major> as
# a system pip package under /usr/local/lib/python<X.Y>/dist-packages (not into
# a venv), so this targets that exact path instead of the runner's own
# site-packages. Keep NVSHMEM_VERSION (extra_env in versions.yaml) in sync with
# that Dockerfile's NVSHMEM_VERSION arg.
#
# Setting NVSHMEM_DIR also makes deep-ep's setup.py link the unversioned
# `-l:libnvshmem_host.so`, which the pip package does not ship - hence the
# symlink, mirroring the one that Dockerfile creates. Only the build needs it:
# what gets recorded in the .so is the library's SONAME
# (libnvshmem_host.so.<major>), which the pip package does ship.
# ---------------------------------------------------------------------------
install_nvshmem() {
  local version="${NVSHMEM_VERSION:?NVSHMEM_VERSION must be set (see extra_env in versions.yaml)}"
  local cuda_major site_dir host_lib
  local host_libs=()
  cuda_major="$(echo "${CUDA_VERSION}" | cut -d. -f1)"
  site_dir="/usr/local/lib/python${PYTHON_VERSION}/dist-packages"

  echo "::group::Install nvidia-nvshmem-cu${cuda_major}==${version} into ${site_dir}"
  maybe_sudo pip install --upgrade --target "${site_dir}" \
    "nvidia-nvshmem-cu${cuda_major}==${version}"

  export NVSHMEM_DIR="${site_dir}/nvidia/nvshmem"
  host_libs=("${NVSHMEM_DIR}"/lib/libnvshmem_host.so.*)
  if [ ! -e "${host_libs[0]}" ]; then
    echo "::error::No libnvshmem_host.so.* under ${NVSHMEM_DIR}/lib after installing the wheel" >&2
    return 1
  fi
  host_lib="$(basename "${host_libs[0]}")"
  maybe_sudo ln -sf "${host_lib}" "${NVSHMEM_DIR}/lib/libnvshmem_host.so"

  echo "NVSHMEM_DIR=${NVSHMEM_DIR} (build-time symlink -> ${host_lib})"
  echo "::endgroup::"
}

# ---------------------------------------------------------------------------
# strip_wheel_local_version: rewrite the wheels in a dist dir to drop the PEP
# 440 local version segment their build appended (deep-ep and flash-mla both
# tack "+<short git sha>" onto their own version).
#
# GitHub's release-asset upload rewrites "+" to "." in the stored filename (it
# arrives URL-encoded as a space), so deep_ep-1.2.1+3f601f7-...whl would land
# on the release as deep_ep-1.2.1.3f601f7-...whl - a filename whose version no
# longer matches the METADATA inside, which pip and uv both reject. Neither
# project has an opt-out env var (there is no equivalent of TransformerEngine's
# NVTE_NO_LOCAL_VERSION), so the segment is stripped from the built artifact
# here instead. Nothing is lost: the exact commit a wheel came from is what the
# component's release tag and title record.
#
# Requires the `wheel` package, which every caller installs anyway.
# ---------------------------------------------------------------------------
strip_wheel_local_version() {
  local dist_dir="${1:-dist}"
  local wheel unpack_dir pkg_dir clean_pkg_dir meta_dir version clean_version

  for wheel in "${dist_dir}"/*.whl; do
    [ -e "${wheel}" ] || continue
    case "$(basename "${wheel}")" in
      *+*) ;;
      *) continue ;;
    esac

    unpack_dir="$(mktemp -d)"
    python -m wheel unpack --dest "${unpack_dir}" "${wheel}"
    pkg_dir="$(find "${unpack_dir}" -mindepth 1 -maxdepth 1 -type d)"
    meta_dir="$(find "${pkg_dir}" -mindepth 1 -maxdepth 1 -type d -name '*.dist-info')"

    version="$(sed -n 's/^Version: //p' "${meta_dir}/METADATA" | head -1)"
    clean_version="${version%%+*}"
    sed "s/^Version: .*/Version: ${clean_version}/" "${meta_dir}/METADATA" > "${meta_dir}/METADATA.rewritten"
    mv "${meta_dir}/METADATA.rewritten" "${meta_dir}/METADATA"

    # The version is spelled out again in the unpacked root's name and in the
    # .dist-info (plus .data, for wheels that have one) directories inside it.
    # Substitute on each basename in turn - a path-wide substitution would
    # rewrite the parent directory's copy of the version instead. `wheel pack`
    # reads the name and version back out of .dist-info to build the new
    # filename, and regenerates RECORD itself.
    clean_pkg_dir="${unpack_dir}/$(basename "${pkg_dir}" | sed "s/${version}\$/${clean_version}/")"
    mv "${pkg_dir}" "${clean_pkg_dir}"
    for meta_dir in "${clean_pkg_dir}"/*.dist-info "${clean_pkg_dir}"/*.data; do
      [ -d "${meta_dir}" ] || continue
      mv "${meta_dir}" "${clean_pkg_dir}/$(basename "${meta_dir}" | sed "s/${version}\./${clean_version}./")"
    done

    echo "Repacking $(basename "${wheel}") without its local version segment (${version} -> ${clean_version})"
    rm -f "${wheel}"
    python -m wheel pack --dest-dir "${dist_dir}" "${clean_pkg_dir}"
    rm -rf "${unpack_dir}"
  done
}

# ---------------------------------------------------------------------------
# arch_list_strip_dots: convert the canonical "8.0;9.0;10.0;12.0" arch list
# into the undotted "80;90;100;120" form that flash-attention's
# FLASH_ATTN_CUDA_ARCHS and TransformerEngine's NVTE_CUDA_ARCHS expect.
# apex consumes TORCH_CUDA_ARCH_LIST in the canonical dotted form directly
# (no conversion needed), and flashinfer's list is given pre-formatted in
# versions.yaml (with PTX-family suffix letters like "9.0a"/"12.0f") since
# those can't be derived mechanically.
# ---------------------------------------------------------------------------
arch_list_strip_dots() {
  echo "$1" | tr -d '.'
}

# ---------------------------------------------------------------------------
# flashinfer_jit_cache_cu_index: map a CUDA toolkit version to the flashinfer.ai
# jit-cache wheel index suffix (e.g. 13.0.2 -> cu130). Cubin wheels are fetched
# from the CUDA-agnostic https://flashinfer.ai/whl index instead.
# Mirrors sglang/docker/Dockerfile and vllm/docker/Dockerfile.
# ---------------------------------------------------------------------------
flashinfer_jit_cache_cu_index() {
  echo "cu$(echo "$1" | cut -d. -f1,2 | tr -d '.')"
}

# ---------------------------------------------------------------------------
# download_prebuilt_wheel: pip download a prebuilt wheel (wheels only, no deps)
# with retry logic, writing the .whl into dest_dir. Used to vendor wheels this
# repo intentionally does not build from source - currently flashinfer's
# companion wheels (flashinfer-cubin / -jit-cache from flashinfer.ai; jit-cache
# is ~1.2 GB and that index can be flaky).
# --only-binary=:all: guarantees we rehost the upstream binary wheel and never
# silently fall back to building an sdist.
# ---------------------------------------------------------------------------
download_prebuilt_wheel() {
  local package="$1"
  local version="$2"
  local index_url="$3"
  local dest_dir="$4"
  local attempt max_attempts=5

  for attempt in $(seq 1 "${max_attempts}"); do
    if pip download "${package}==${version}" \
      --index-url "${index_url}" \
      --no-deps \
      --only-binary=:all: \
      -d "${dest_dir}"; then
      echo "Downloaded ${package}==${version} from ${index_url}"
      return 0
    fi
    if [ "${attempt}" -lt "${max_attempts}" ]; then
      echo "::warning::Attempt ${attempt}/${max_attempts} to download ${package} failed; retrying in 10s..."
      sleep 10
    fi
  done

  echo "::error::Failed to download ${package}==${version} from ${index_url} after ${max_attempts} attempts" >&2
  return 1
}

# ---------------------------------------------------------------------------
# export_extra_env: EXTRA_ENV is exported as a JSON object string by the
# workflow (e.g. '{"NVTE_BUILD_THREADS_PER_JOB": "4"}'); turn its entries
# into real exported environment variables.
# ---------------------------------------------------------------------------
export_extra_env() {
  if [ -z "${EXTRA_ENV:-}" ] || [ "${EXTRA_ENV}" = "{}" ]; then
    return 0
  fi
  local key value
  while IFS='=' read -r key value; do
    [ -n "${key}" ] || continue
    export "${key}=${value}"
    echo "Exported ${key}=${value} (from extra_env)"
  done < <(echo "${EXTRA_ENV}" | jq -r 'to_entries[] | "\(.key)=\(.value)"')
}
