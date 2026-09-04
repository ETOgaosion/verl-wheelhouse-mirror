#!/usr/bin/env python3
"""Rewrite a DeepEP checkout so one fat binary can carry sm_80 plus Hopper+.

Upstream DeepEP treats A100 and Hopper as mutually exclusive compile
configurations: DISABLE_SM90_FEATURES=1 forces TORCH_CUDA_ARCH_LIST=8.0,
drops NVSHMEM, and is a global preprocessor flag, so you cannot also keep
TMA / cluster launch / internode kernels for 9.0/10.0 in the same nvcc
invocation.

The wheelhouse wants one x86_64 wheel, so this script (applied by
ci/build_scripts/deep_ep.sh against the pinned DeepEP checkout, never
committed into the submodule) does four local edits:

1. Device-side SM90 features follow __CUDA_ARCH__ instead of a global
   -DDISABLE_SM90_FEATURES. Ampere compilation of intranode.cu then takes
   the existing #else Ampere paths; Hopper/Blackwell compilation is
   unchanged.
2. configs.cuh always includes the toolkit <cuda_fp8.h>. Upstream's
   Ampere fallback (#else of the SM90 guard) re-declares the FP8 types
   with int/uint8_t typedefs; during the sm_80 device pass torch/nvshmem
   headers pull in the real <cuda_fp8.h> transitively, causing
   "invalid redeclaration of __nv_fp8_interpretation_t / __nv_fp8x4_e4m3"
   errors. The sm_80 pass references no FP8 symbols (the only FP8 call
   sites are in the Hopper-only sources of edit 4), so the real header
   is harmless there; host and sm_90/sm_100 passes already used it.
3. Host launch macros in launch.cuh pick cluster/TMA vs classic <<<>>> at
   runtime from the current device's compute capability, so the same host
   object can launch either cubin. The runtime SET_SHARED_MEMORY_FOR_TMA
   always expands to code referencing the caller's `smem_size` (upstream's
   Ampere build #defines that macro to void()), so intranode.cu's two
   `smem_size` constants move out from under the SM90 guard.
4. Internode / low-latency / PCIe .cu files have no Ampere fallbacks.
   Their device pass for sm_80 is compiled out; launching those kernels on
   A100 fails with "no kernel image", which matches upstream's
   "A100 support (intranode only)" claim.

Idempotent: safe to re-run on a resumable-build checkout.
"""

from __future__ import annotations

from pathlib import Path

MARKER = "DEEP_EP_SM80_FATBIN"

CONFIGS_SNIPPET = f"""\
// {MARKER}: when nvcc compiles the sm_80 gencode, take the existing Ampere
// fallbacks (#else of #ifndef DISABLE_SM90_FEATURES for TMA/cluster/elect)
// without forcing that flag for the host pass or for sm_90/sm_100.
// Upstream setup.py cannot express this because -DDISABLE_SM90_FEATURES is
// process-global. The FP8 fallback typedefs are removed separately below:
// the sm_80 pass uses the real <cuda_fp8.h> like every other pass.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
#ifndef DISABLE_SM90_FEATURES
#define DISABLE_SM90_FEATURES
#endif
#endif

"""

CONFIGS_FP8_OLD = """\
#ifndef DISABLE_SM90_FEATURES
#include <cuda_fp8.h>
#else
// Ampere does not support FP8 features
#define __NV_E4M3 0
#define __NV_E5M2 1
typedef int __nv_fp8_interpretation_t;
typedef int __nv_fp8x4_e4m3;
typedef uint8_t __nv_fp8_storage_t;
#endif
"""

CONFIGS_FP8_NEW = f"""\
// {MARKER}: always include the toolkit FP8 header. The upstream Ampere
// fallback re-declares these types with int/uint8_t typedefs, which collide
// with the real <cuda_fp8.h> that torch/nvshmem headers pull in
// transitively during the sm_80 device pass. No sm_80 compilation unit
// references FP8 symbols (the only FP8 call sites live in the Hopper-only
// sources whose sm_80 cubin is compiled out), so the real header is
// harmless here and identical to the host / sm_90 / sm_100 passes.
#include <cuda_fp8.h>
"""

# intranode.cu declares `smem_size` only under #ifndef DISABLE_SM90_FEATURES
# (two identical sites: dispatch and combine launch wrappers). The patched
# SET_SHARED_MEMORY_FOR_TMA expands in every pass and references that
# variable (the cudaFuncSetAttribute call is a runtime Hopper+ check), so
# the constant must be declared unconditionally.
INTRANODE_SMEM_OLD = """\
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif
"""

INTRANODE_SMEM_NEW = f"""\
// {MARKER}: declared unconditionally - the runtime-dispatched
// SET_SHARED_MEMORY_FOR_TMA macro references smem_size in every compilation
// pass; upstream's Ampere build #defines that macro to void() instead.
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
"""

LAUNCH_MACROS = f"""\
#ifndef SETUP_LAUNCH_CONFIG
// {MARKER}: both launch paths in one host object. Cluster attributes and
// cudaLaunchKernelEx are Hopper+; Ampere uses the classic launch that
// upstream selects via DISABLE_SM90_FEATURES=1.
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \\
    cudaLaunchConfig_t cfg = {{(num_sms), (num_threads), 0, stream, nullptr, 0}}; \\
    cudaLaunchAttribute attr[2]; \\
    int __num_sms = (num_sms); \\
    int __num_threads = (num_threads); \\
    auto __stream = (stream); \\
    int __sm_device = 0, __sm_major = 0; \\
    cudaGetDevice(&__sm_device); \\
    cudaDeviceGetAttribute(&__sm_major, cudaDevAttrComputeCapabilityMajor, __sm_device); \\
    const bool __use_sm90_launch = (__sm_major >= 9); \\
    if (__use_sm90_launch) {{ \\
        attr[0].id = cudaLaunchAttributeCooperative; \\
        attr[0].val.cooperative = 1; \\
        attr[1].id = cudaLaunchAttributeClusterDimension; \\
        attr[1].val.clusterDim.x = ((num_sms) % 2 == 0 ? 2 : 1); \\
        attr[1].val.clusterDim.y = 1; \\
        attr[1].val.clusterDim.z = 1; \\
        cfg.attrs = attr; \\
        cfg.numAttrs = 2; \\
    }}
#endif

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...) \\
do {{ \\
    if (__use_sm90_launch) {{ \\
        CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__)); \\
    }} else {{ \\
        kernel<<<__num_sms, __num_threads, 0, __stream>>>(__VA_ARGS__); \\
        cudaError_t e = cudaGetLastError(); \\
        if (e != cudaSuccess) {{ \\
            EPException cuda_exception("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \\
            fprintf(stderr, "%s\\n", cuda_exception.what()); \\
            throw cuda_exception; \\
        }} \\
    }} \\
}} while (0)
#endif

#ifndef SET_SHARED_MEMORY_FOR_TMA
#define SET_SHARED_MEMORY_FOR_TMA(kernel) \\
do {{ \\
    if (__use_sm90_launch) {{ \\
        EP_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess); \\
        cfg.dynamicSmemBytes = smem_size; \\
    }} \\
}} while (0)
#endif
"""

IS_SM90_COMPILED = f"""\
bool is_sm90_compiled() {{
#ifndef DISABLE_SM90_FEATURES
    // {MARKER}: the fatbin has both Ampere and Hopper cubins; report the
    // current device so callers skip FP8 / internode on A100.
    int device = 0, major = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return true;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        return true;
    return major >= 9;
#else
    return false;
#endif
}}
"""

HOPPER_ONLY_SOURCES = (
    "csrc/kernels/internode.cu",
    "csrc/kernels/internode_ll.cu",
    "csrc/kernels/pcie.cu",
)

ORIGINAL_LAUNCH_START = "#ifndef SETUP_LAUNCH_CONFIG"
ORIGINAL_LAUNCH_END = "#define SWITCH_RANKS"

ORIGINAL_IS_SM90 = """\
bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}
"""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")


def patch_configs(root: Path) -> None:
    path = root / "csrc/kernels/configs.cuh"
    text = _read(path)
    if MARKER in text:
        return
    pragma = "#pragma once\n"
    if not text.startswith(pragma):
        raise SystemExit(f"{path}: expected to start with #pragma once")
    if CONFIGS_FP8_OLD not in text:
        raise SystemExit(f"{path}: could not find the FP8 fallback block to replace")
    text = text.replace(CONFIGS_FP8_OLD, CONFIGS_FP8_NEW, 1)
    _write(path, pragma + "\n" + CONFIGS_SNIPPET + text[len(pragma) :])


def patch_intranode_smem(root: Path) -> None:
    path = root / "csrc/kernels/intranode.cu"
    text = _read(path)
    if MARKER in text:
        return
    count = text.count(INTRANODE_SMEM_OLD)
    if count != 2:
        raise SystemExit(f"{path}: expected 2 guarded smem_size declarations, found {count}")
    _write(path, text.replace(INTRANODE_SMEM_OLD, INTRANODE_SMEM_NEW))


def patch_launch(root: Path) -> None:
    path = root / "csrc/kernels/launch.cuh"
    text = _read(path)
    if MARKER in text:
        return
    start = text.find(ORIGINAL_LAUNCH_START)
    end = text.find(ORIGINAL_LAUNCH_END)
    if start < 0 or end < 0 or end <= start:
        raise SystemExit(f"{path}: could not find launch-macro block to replace")
    _write(path, text[:start] + LAUNCH_MACROS + "\n" + text[end:])


def patch_is_sm90_compiled(root: Path) -> None:
    path = root / "csrc/deep_ep.cpp"
    text = _read(path)
    if MARKER in text:
        return
    if ORIGINAL_IS_SM90 not in text:
        raise SystemExit(f"{path}: could not find is_sm90_compiled() to replace")
    _write(path, text.replace(ORIGINAL_IS_SM90, IS_SM90_COMPILED, 1))


def wrap_hopper_only_source(path: Path) -> None:
    text = _read(path)
    if MARKER in text:
        return
    header = (
        f"#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)\n"
        f"// {MARKER}: no Ampere implementation; omit the sm_80 cubin.\n"
        f"// Keep a dummy kernel so -rdc=true device-link still sees this TU.\n"
        f"__global__ void deep_ep_sm80_placeholder_{path.stem}() {{}}\n"
        f"#else\n"
    )
    footer = "\n#endif  // !(__CUDA_ARCH__ < 900)\n"
    _write(path, header + text + footer)


def main() -> None:
    root = Path.cwd()
    if not (root / "setup.py").is_file() or not (root / "csrc/kernels").is_dir():
        raise SystemExit(f"{root} does not look like a DeepEP checkout (run from the submodule CWD)")
    patch_configs(root)
    patch_intranode_smem(root)
    patch_launch(root)
    patch_is_sm90_compiled(root)
    for rel in HOPPER_ONLY_SOURCES:
        wrap_hopper_only_source(root / rel)
    print("DeepEP sm_80 fatbin rewrites applied")


if __name__ == "__main__":
    main()
