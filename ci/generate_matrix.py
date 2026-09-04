#!/usr/bin/env python3
"""Expand versions.yaml (in the project's base directory) into a GitHub
Actions build matrix.

versions.yaml is the single editable/extendable source of truth for CPU
arch / CUDA / Python / Torch combinations and per-component build config -
it is plain data (no Python), so it can be read and edited without touching
any code.
This script is the primary place that turns it into something GitHub
Actions can consume (ci/release_meta.py, which computes per-component
release tag/title/notes, imports this module's load_versions/release_tag
helpers rather than re-reading versions.yaml itself).

Every entry in the emitted JSON list is a flat dict that maps 1:1 onto the
inputs of .github/workflows/_build.yml, so a workflow can do:

    strategy:
      matrix:
        include: ${{ fromJSON(needs.compute-matrix.outputs.matrix) }}

Usage:
    python ci/generate_matrix.py --component apex
    python ci/generate_matrix.py --component all
    python ci/generate_matrix.py --component flash-attention --arch aarch64
    python ci/generate_matrix.py --list-components
    python ci/generate_matrix.py --component apex --github-output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from cuda_archs import base_arch, base_arch_set, parse_arch_list, wheel_cuda_archs

# versions.yaml lives in the project's base directory (one level up from ci/).
VERSIONS_FILE = Path(__file__).resolve().parent.parent / "versions.yaml"

# Characters not in this set get collapsed to "-" when building a release tag
# out of a git ref, since refs (e.g. branch names) aren't guaranteed to be
# valid/clean git tag components on their own.
_TAG_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Arch assumed for a build_matrix entry that doesn't spell one out, and the
# one left implicit in release titles (see release_title).
DEFAULT_ARCH = "x86_64"

# GitHub identifies every self-hosted runner by this label, whatever else it
# carries, so its presence in a `runs_on` is what distinguishes a self-hosted
# row from a GitHub-hosted one.
SELF_HOSTED_LABEL = "self-hosted"

# Wheel platform-tag suffixes mapped to the build_matrix `arch` they satisfy,
# e.g. "manylinux_2_28_aarch64" -> aarch64.
_PLATFORM_TAG_ARCHES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}

# Snapshot of per-wheel build config stored in the release body so skip
# detection can compare torch_cuda_arch_list / extra_env / builder / etc.
# without putting those fields in the title or wheel filename.
# Schema history:
#   1: initial build inputs (cuda/python/torch/arch list/extra_env/...).
#   2: adds `cuda_archs` (the arch list parsed into a canonical SM-token set,
#      e.g. ["8.0", "9.0a", "10.0"]) and `build_inputs` (sha256 of the builder
#      script, common.sh and every declared patch), so editing the deep-ep
#      sm80 patch or a build script invalidates the skip even when no
#      versions.yaml field changed. Skip detection additionally verifies the
#      wheels attached to the release really fat-bin those arches.
BUILD_CONFIG_SCHEMA = 2
BUILD_CONFIG_MARKER = "wheelhouse-build-config"
# Machine-readable build snapshot uploaded as a release asset alongside the
# wheels. The release body still carries the same JSON in a hidden HTML
# comment as a fallback, but the file is the source of truth: it survives
# release-note edits, holds far more detail than a title can, and skip
# detection downloads it with the same `gh release` path as the wheels.
BUILD_MANIFEST_FILENAME = "wheelhouse-build-manifest.json"

# Wheels larger than this are not downloaded for fatbin inspection (the
# flashinfer rehosts a ~1.2 GB data wheel); such components set
# verify_wheel_archs: false in versions.yaml. Hitting this cap is a
# fail-closed "could not verify" -> rebuild, never a silent skip.
MAX_WHEEL_INSPECT_BYTES = 2_000_000_000


class WheelInspectionError(Exception):
    """A release wheel could not be downloaded or read for arch verification."""


def load_versions() -> Dict[str, Any]:
    with open(VERSIONS_FILE, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def combo_arch(combo: Dict[str, Any]) -> str:
    return str(combo.get("arch", DEFAULT_ARCH))


def matrix_arches(versions: Dict[str, Any]) -> List[str]:
    """Every arch the build_matrix covers, in first-seen order."""
    return list(dict.fromkeys(combo_arch(combo) for combo in versions["build_matrix"]))


def component_arches(versions: Dict[str, Any], component: str) -> List[str]:
    """The arches one component opts into, in build_matrix order.

    A component without an `arches` field builds every arch in the matrix;
    one with it builds only the listed subset (see versions.yaml for why a
    component would opt out of an arch).
    """
    cfg = get_component(versions, component)
    available = matrix_arches(versions)

    unknown = sorted(set(cfg.get("arch_overrides") or {}) - set(available))
    if unknown:
        raise SystemExit(
            f"Component {component!r} has arch_overrides for arch(es) not in build_matrix: "
            f"{', '.join(unknown)}. Known arches: {', '.join(available)}"
        )

    configured = cfg.get("arches")
    if configured is None:
        return available

    unknown = sorted({str(arch) for arch in configured} - set(available))
    if unknown:
        raise SystemExit(
            f"Component {component!r} lists arch(es) not in build_matrix: {', '.join(unknown)}. "
            f"Known arches: {', '.join(available)}"
        )
    return [arch for arch in available if arch in {str(a) for a in configured}]


def component_combos(versions: Dict[str, Any], component: str) -> List[Dict[str, Any]]:
    """The build_matrix rows one component is actually built for."""
    arches = set(component_arches(versions, component))
    return [combo for combo in versions["build_matrix"] if combo_arch(combo) in arches]


def check_runner_policy(component: str, arch: str, runs_on: Any) -> None:
    """Reject a self-hosted runner on anything but the default arch.

    The only self-hosted machine this repo builds on is x86_64, and non-x86_64
    builds are deliberately kept on GitHub's hosted runners (`ubuntu-*-arm`
    for aarch64) so that adding an arch never depends on someone standing up
    matching hardware first. A self-hosted label on such a row would otherwise
    queue forever waiting for a runner that does not exist, or - if one is
    ever registered under a mismatched arch - burn the job's setup time before
    _build.yml's `uname -m` assertion catches it.
    """
    if arch == DEFAULT_ARCH:
        return
    labels = runs_on if isinstance(runs_on, list) else [runs_on]
    if any(str(label).strip().lower() == SELF_HOSTED_LABEL for label in labels):
        raise SystemExit(
            f"Component {component!r} points its {arch} build at a self-hosted runner "
            f"({runs_on!r}). Non-{DEFAULT_ARCH} builds must use a GitHub-hosted runner "
            f"(e.g. 'ubuntu-24.04-arm' for aarch64); see arch_overrides in versions.yaml."
        )


def component_config(versions: Dict[str, Any], component: str, arch: str) -> Dict[str, Any]:
    """One component's config with its `arch_overrides` for `arch` applied."""
    cfg = dict(get_component(versions, component))
    cfg.update((cfg.pop("arch_overrides", None) or {}).get(arch) or {})
    check_runner_policy(component, arch, cfg["runs_on"])
    return cfg


def sanitize_ref(ref: str) -> str:
    """Make a git ref safe to splice into a release tag."""
    sanitized = _TAG_UNSAFE_RE.sub("-", ref).strip("-.")
    return sanitized or "unknown"


def release_tag(component: str, ref: str) -> str:
    """Persistent per-component release tag, e.g. "apex-master".

    Each component gets its own GitHub Release, keyed by its pinned ref
    rather than by a shared "latest"/repo-level tag: rebuilding the same ref
    re-uploads wheels onto the same release, and bumping the ref in
    versions.yaml starts a new release, leaving the old one as history.
    Shared with ci/release_meta.py so the release that gets created/updated
    and the tag _build.yml uploads wheels to always match.
    """
    return f"{component}-{sanitize_ref(ref)}"


def release_title(ref: str, component: str, combos: List[Dict[str, Any]]) -> str:
    """Describe the dependency combinations covered by a component release.

    One "cu.. py.. torch.." segment per combination the component is built
    for, prefixed by the arch for everything except DEFAULT_ARCH - x86_64 is
    the baseline every component builds, so leaving it implicit keeps the
    titles of x86_64-only releases stable (and their builds skippable) as
    other arches are added to the matrix.
    """
    segments = "; ".join(
        f"{'' if combo_arch(combo) == DEFAULT_ARCH else combo_arch(combo) + ' '}"
        f"cu{combo['cuda']} py{combo['python']} torch{combo['torch']}"
        for combo in combos
    )
    return f"{component} {ref} - {segments}"


def normalize_package_name(name: str) -> str:
    """Apply PEP 503 package-name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_arches(filename: str, arches: List[str]) -> Set[str]:
    """Which of `arches` a wheel file's platform tag can be installed on.

    A wheel filename ends in "-<python>-<abi>-<platform>.whl", where platform
    may itself be a "."-joined set of compressed tags (e.g.
    "manylinux_2_27_x86_64.manylinux_2_28_x86_64"). Pure-Python wheels are
    tagged "any" and count for every arch.
    """
    platform_tag = filename[: -len(".whl")].rsplit("-", 1)[-1]
    matched = set()
    for tag in platform_tag.split("."):
        if tag == "any":
            return set(arches)
        for suffix, arch in _PLATFORM_TAG_ARCHES.items():
            if tag.endswith(suffix):
                matched.add(arch)
                break
    return matched & set(arches)


def _merged_component_fields(
    versions: Dict[str, Any], component: str, arch: str
) -> Dict[str, Any]:
    """Component config with arch_overrides applied, without requiring runs_on."""
    cfg = dict(get_component(versions, component))
    cfg.update((cfg.pop("arch_overrides", None) or {}).get(arch) or {})
    return cfg


def build_input_fingerprint(
    versions: Dict[str, Any], component: str, arch: str
) -> List[Dict[str, str]]:
    """Sha256 of every repo file whose content changes what the wheel contains.

    versions.yaml captures build *inputs that are data* (ref, env, arch list);
    the builder script, common.sh, the reusable workflow that provisions the
    runner (CUDA toolkit / cuDNN / Python / cache) and any patches the builder
    applies are *inputs that are code*, so editing e.g.
    ci/patches/enable_deep_ep_sm80.py or the CUDA install step in _build.yml
    would otherwise leave the stored manifest identical and skip a rebuild the
    edit was meant to produce. Paths are repo-root-relative; patches are
    declared per component in versions.yaml (`patches:`).
    """
    cfg = _merged_component_fields(versions, component, arch)
    rel_paths: List[str] = [".github/workflows/_build.yml"]
    builder = cfg.get("builder")
    if builder:
        rel_paths.append(f"ci/build_scripts/{builder}.sh")
    rel_paths.append("ci/build_scripts/common.sh")
    rel_paths.extend(str(path) for path in (cfg.get("patches") or []))

    fingerprint = []
    for rel in rel_paths:
        path = VERSIONS_FILE.parent / rel
        if not path.is_file():
            raise SystemExit(
                f"Component {component!r} fingerprints build input {rel!r}, but that file "
                f"does not exist. Fix the component's builder/patches in versions.yaml."
            )
        fingerprint.append({"path": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return sorted(fingerprint, key=lambda entry: entry["path"])


def normalize_build_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical per-wheel build config used for skip comparison."""
    extra = config.get("extra_env") or {}
    if not isinstance(extra, dict):
        extra = {}
    packages = config.get("wheel_packages") or []
    if not isinstance(packages, list):
        packages = []
    build_inputs = config.get("build_inputs") or []
    if not isinstance(build_inputs, list):
        build_inputs = []
    runs_on = config.get("runs_on")
    if isinstance(runs_on, list):
        runs_on_norm: Any = sorted(str(label) for label in runs_on)
    else:
        runs_on_norm = str(runs_on or "")
    return {
        "arch": str(config.get("arch") or DEFAULT_ARCH),
        "builder": str(config.get("builder") or ""),
        "build_inputs": [
            {"path": str(entry.get("path") or ""), "sha256": str(entry.get("sha256") or "")}
            for entry in sorted(build_inputs, key=lambda entry: str(entry.get("path") or ""))
        ],
        # The exact build command _build.yml runs (the rest of the environment
        # is the explicit fields below + extra_env). Recorded verbatim so a
        # change in how the builder is invoked is a visible manifest diff.
        "command": str(config.get("command") or ""),
        "cuda": str(config.get("cuda") or ""),
        "cuda_archs": parse_arch_list(str(config.get("torch_cuda_arch_list") or "")),
        "cxx11_abi": str(config.get("cxx11_abi") or ""),
        "extra_env": {str(key): str(value) for key, value in extra.items()},
        "max_jobs": str(config.get("max_jobs") or ""),
        "python": str(config.get("python") or ""),
        "requires_cudnn": bool(config.get("requires_cudnn")),
        "runs_on": runs_on_norm,
        "torch": str(config.get("torch") or ""),
        "torch_audio": str(config.get("torch_audio") or ""),
        "torch_cuda_arch_list": str(config.get("torch_cuda_arch_list") or ""),
        "torch_vision": str(config.get("torch_vision") or ""),
        "wheel_packages": sorted(normalize_package_name(str(name)) for name in packages),
    }


def canonical_build_config(config: Dict[str, Any]) -> str:
    return json.dumps(normalize_build_config(config), sort_keys=True, separators=(",", ":"))


def combo_build_config(
    versions: Dict[str, Any], component: str, combo: Dict[str, Any]
) -> Dict[str, Any]:
    """The build inputs that affect one matrix row's wheel contents."""
    arch = combo_arch(combo)
    cfg = _merged_component_fields(versions, component, arch)
    builder = str(cfg.get("builder") or "")
    return normalize_build_config(
        {
            "arch": arch,
            "build_inputs": build_input_fingerprint(versions, component, arch),
            "builder": builder,
            "command": f"bash ci/build_scripts/{builder}.sh" if builder else "",
            "cuda": combo.get("cuda") or "",
            "cxx11_abi": combo.get("cxx11_abi") or "",
            "extra_env": cfg.get("extra_env") or {},
            "max_jobs": cfg.get("max_jobs", 1),
            "python": combo.get("python") or "",
            "requires_cudnn": bool(cfg.get("requires_cudnn")),
            "runs_on": cfg.get("runs_on") or "",
            "torch": combo.get("torch") or "",
            "torch_audio": combo.get("torch_audio") or "",
            "torch_cuda_arch_list": cfg.get("torch_cuda_arch_list") or "",
            "torch_vision": combo.get("torch_vision") or "",
            "wheel_packages": cfg.get("wheel_packages") or [],
        }
    )


def component_build_manifest(versions: Dict[str, Any], component: str) -> Dict[str, Any]:
    """JSON snapshot of every wheel-producing combo for one component."""
    return {
        "schema": BUILD_CONFIG_SCHEMA,
        "component": component,
        "builds": [
            combo_build_config(versions, component, combo)
            for combo in component_combos(versions, component)
        ],
    }


def format_release_notes(versions: Dict[str, Any], component: str) -> str:
    """Human-readable notes plus a hidden JSON snapshot of each wheel's build config."""
    cfg = get_component(versions, component)
    ref = str(cfg["ref"])
    blob = json.dumps(component_build_manifest(versions, component), indent=2, sort_keys=True)
    return (
        f"Prebuilt CUDA wheel(s) for {component}, pinned to `{ref}`. "
        "See versions.yaml at this ref for the exact dependency versions.\n\n"
        f"<!-- {BUILD_CONFIG_MARKER}\n{blob}\n-->"
    )


def parse_stored_build_manifest(body: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract the wheelhouse build-config snapshot from a release body, if any."""
    if not body:
        return None
    marker = f"<!-- {BUILD_CONFIG_MARKER}"
    start = body.find(marker)
    if start < 0:
        return None
    payload = body[start + len(marker) :].lstrip()
    try:
        obj, _ = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("schema") != BUILD_CONFIG_SCHEMA:
        return None
    if not isinstance(obj.get("builds"), list):
        return None
    return obj


def released_wheel_packages(release: Dict[str, Any], arches: List[str]) -> Set[Tuple[str, str]]:
    return {
        (normalize_package_name(str(asset["name"]).split("-", 1)[0]), arch)
        for asset in release.get("assets", [])
        if isinstance(asset, dict)
        and isinstance(asset.get("name"), str)
        and asset["name"].endswith(".whl")
        for arch in wheel_arches(asset["name"], arches)
    }


def release_has_manifest_asset(release: Dict[str, Any]) -> bool:
    return any(
        isinstance(asset, dict) and asset.get("name") == BUILD_MANIFEST_FILENAME
        for asset in release.get("assets", [])
    )


def fetch_release_manifest(
    repo: Optional[str], tag: str, release: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """The stored build manifest for skip comparison.

    Primary source is the wheelhouse-build-manifest.json asset uploaded with
    the wheels (the release body is human-editable; an asset is not); the
    hidden JSON comment in the release body remains a fallback for releases
    cut before the asset existed. Anything unreadable/old-schema degrades to
    None, which is the fail-closed "no comparable snapshot -> rebuild" case.
    """
    if release_has_manifest_asset(release) and repo:
        try:
            with tempfile.TemporaryDirectory(prefix="wheelhouse-manifest-") as tmp:
                path = download_release_asset(repo, tag, BUILD_MANIFEST_FILENAME, Path(tmp))
                obj = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(obj, dict)
                and obj.get("schema") == BUILD_CONFIG_SCHEMA
                and isinstance(obj.get("builds"), list)
            ):
                return obj
        except (WheelInspectionError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(
                f"Could not read {BUILD_MANIFEST_FILENAME} from release {tag!r}: {exc}; "
                f"falling back to the release-body manifest.",
                file=sys.stderr,
            )
    body = release.get("body")
    return parse_stored_build_manifest(body if isinstance(body, str) else "")


def _wheel_platform_is_any(filename: str) -> bool:
    """Whether a wheel filename's platform tag is the pure-Python "any" tag."""
    platform_tag = filename[: -len(".whl")].rsplit("-", 1)[-1]
    return platform_tag == "any"


def download_release_asset(repo: str, tag: str, asset_name: str, dest_dir: Path) -> Path:
    """Download one release asset (a wheel) via gh; raise on any failure."""
    try:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                tag,
                "--repo",
                repo,
                "--dir",
                str(dest_dir),
                "--pattern",
                asset_name,
                "--clobber",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise WheelInspectionError(
            f"`gh release download {tag}/{asset_name}` failed: {detail or 'unknown error'}"
        ) from exc
    except (subprocess.SubprocessError, OSError) as exc:
        raise WheelInspectionError(
            f"could not download {tag}/{asset_name} with gh: {exc}"
        ) from exc
    path = dest_dir / asset_name
    if not path.is_file():
        raise WheelInspectionError(f"download of {tag}/{asset_name} produced no local file")
    return path


def verify_release_wheel_archs(
    repo: Optional[str],
    tag: str,
    release: Dict[str, Any],
    required_packages: Set[str],
    cpu_arch: str,
    required_arch_tokens: List[str],
    all_arches: List[str],
) -> Tuple[bool, str]:
    """Ground-truth skip check: the release's wheels must fat-bin every required SM.

    Downloads the platform-specific wheels attached to the release and reads
    the SM arches actually embedded in their .nv_fatbin sections (cuobjdump if
    present, a direct ELF section scan otherwise), so e.g. a deep-ep wheel
    whose sm_80 cubin never materialized - because the sm80 patch did not
    apply, nvcc dropped a gencode, or a stale wheel was uploaded - is rebuilt
    even though versions.yaml still says "8.0;9.0;10.0". Every condition that
    prevents verification fails *closed* (keep the build); a false skip could
    leave a broken wheel published, while a false rebuild only costs CI time.
    """
    required_bases = base_arch_set(set(required_arch_tokens))
    if not required_bases:
        return True, "no CUDA arch coverage required for this build (pure-Python or self-described gencodes)"
    if not repo:
        return False, "cannot verify wheel CUDA arch coverage without a repo; keeping build"

    candidates: List[Tuple[str, int]] = []
    for asset in release.get("assets", []):
        if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
            continue
        name = asset["name"]
        if not name.endswith(".whl") or _wheel_platform_is_any(name):
            continue  # pure-Python wheel: no fatbin to inspect
        package = normalize_package_name(name.split("-", 1)[0])
        if package not in required_packages:
            continue
        if cpu_arch not in wheel_arches(name, all_arches):
            continue
        candidates.append((name, int(asset.get("size") or 0)))

    if not candidates:
        return False, (
            f"no platform-specific wheel for {sorted(required_packages)} on {cpu_arch} "
            f"available for CUDA arch verification; keeping build"
        )

    found_tokens: Set[str] = set()
    scanned: List[str] = []
    with tempfile.TemporaryDirectory(prefix="wheelhouse-verify-") as tmp:
        tmp_dir = Path(tmp)
        for name, size in candidates:
            if size > MAX_WHEEL_INSPECT_BYTES:
                return False, (
                    f"wheel {name} ({size} bytes) exceeds the {MAX_WHEEL_INSPECT_BYTES}-byte "
                    f"inspection cap; set verify_wheel_archs: false in versions.yaml to opt "
                    f"out; keeping build"
                )
            try:
                wheel_path = download_release_asset(repo, tag, name, tmp_dir)
                found_tokens |= wheel_cuda_archs(wheel_path)
            except WheelInspectionError as exc:
                return False, f"wheel arch verification unavailable for {name}: {exc}; keeping build"
            except Exception as exc:  # noqa: BLE001 - read/scan boundary: fail closed
                return False, f"wheel {name} could not be inspected ({exc}); keeping build"
            scanned.append(name)

    found_bases = base_arch_set(found_tokens)
    missing = required_bases - found_bases
    if missing:
        found_text = ", ".join(sorted(found_bases)) or "no CUDA arch markers"
        return False, (
            f"published wheel(s) {', '.join(scanned)} fat-bin {found_text}, missing required "
            f"SM arch(es) {', '.join(sorted(missing))}; rebuilding"
        )
    return True, (
        f"wheel fatbin covers required SM arch(es) {', '.join(sorted(required_bases))} "
        f"(scanned {', '.join(scanned)}; found {', '.join(sorted(found_bases))})"
    )


def release_covers_combo(
    versions: Dict[str, Any],
    component: str,
    combo: Dict[str, Any],
    release: Dict[str, Any],
    repo: Optional[str] = None,
    verify_wheels: bool = True,
) -> Tuple[bool, str]:
    """Skip one matrix row only when title, stored build config, wheels, and
    the wheels' actual fatbin SM-arch coverage all match.

    `verify_wheels=False` skips the (multi-GB) release-wheel download and
    fatbin inspection - used by the PR dry-run report, where the decision is
    shown to a human and the real re-verification still runs at push time.
    """
    cfg = get_component(versions, component)
    ref = str(cfg["ref"])
    expected_title = release_title(ref, component, component_combos(versions, component))
    actual_title = release.get("name")
    if actual_title != expected_title:
        return False, f"title mismatch: expected {expected_title!r}, found {actual_title!r}"

    configured_packages = cfg.get("wheel_packages")
    if not isinstance(configured_packages, list) or not configured_packages:
        return False, "wheel_packages is not configured"

    manifest = fetch_release_manifest(repo, release_tag(component, ref), release)
    if manifest is None:
        return False, "no stored build config (release manifest asset/body missing or from an older schema)"

    current = combo_build_config(versions, component, combo)
    stored = {
        canonical_build_config(build)
        for build in manifest["builds"]
        if isinstance(build, dict)
    }
    if canonical_build_config(current) not in stored:
        arch = combo_arch(combo)
        return False, f"build config mismatch for {arch}"

    arch = combo_arch(combo)
    arches = component_arches(versions, component)
    required = {
        (normalize_package_name(str(name)), arch) for name in configured_packages
    }
    released = released_wheel_packages(release, arches)
    missing = sorted(required - released)
    if missing:
        formatted = ", ".join(f"{package} ({pkg_arch})" for package, pkg_arch in missing)
        return False, f"missing wheel package(s): {formatted}"

    effective_cfg = _merged_component_fields(versions, component, arch)
    if not verify_wheels:
        return True, (
            "exact title, matching build config, and expected wheel packages are present "
            "(wheel fatbin re-verification deferred: dry-run mode)"
        )
    if effective_cfg.get("verify_wheel_archs", True):
        required_packages = {normalize_package_name(str(name)) for name in configured_packages}
        covered, reason = verify_release_wheel_archs(
            repo,
            release_tag(component, ref),
            release,
            required_packages,
            arch,
            current["cuda_archs"],
            arches,
        )
        if not covered:
            return False, reason
        return True, (
            "exact title, matching build config, expected wheel packages present, and "
            f"{reason}"
        )

    return True, (
        "exact title, matching build config, and expected wheel packages are present "
        "(wheel CUDA arch verification opted out)"
    )


def release_covers_component(
    versions: Dict[str, Any],
    component: str,
    release: Dict[str, Any],
    repo: Optional[str] = None,
    verify_wheels: bool = True,
) -> Tuple[bool, str]:
    """Check that every matrix row for a component is covered by the release."""
    last_reason = "exact title, matching build config, and all expected wheel packages are present"
    for combo in component_combos(versions, component):
        covered, reason = release_covers_combo(
            versions, component, combo, release, repo, verify_wheels=verify_wheels
        )
        if not covered:
            return False, reason
        last_reason = reason
    return True, last_reason


def inspect_release(repo: str, tag: str) -> Optional[Dict[str, Any]]:
    """Read the release metadata needed for skip detection with the GitHub CLI."""
    try:
        result = subprocess.run(
            ["gh", "release", "view", tag, "--repo", repo, "--json", "name,assets,body"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print(f"Could not inspect release {tag!r}: {exc}; keeping its build.", file=sys.stderr)
        return None

    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh exited with status {result.returncode}"
        print(f"Could not inspect release {tag!r}: {detail}; keeping its build.", file=sys.stderr)
        return None

    try:
        release = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(
            f"Could not parse release {tag!r}: {exc}; keeping its build.",
            file=sys.stderr,
        )
        return None
    if not isinstance(release, dict):
        print(f"Release {tag!r} returned invalid metadata; keeping its build.", file=sys.stderr)
        return None
    return release


def components_needing_build(
    versions: Dict[str, Any], components: List[str], repo: str
) -> List[str]:
    """Drop components whose every matrix row is already covered by the release."""
    needed = []
    for component in components:
        cfg = get_component(versions, component)
        tag = release_tag(component, str(cfg["ref"]))
        release = inspect_release(repo, tag)
        if release is None:
            needed.append(component)
            continue

        covered, reason = release_covers_component(versions, component, release, repo)
        if covered:
            print(f"Skipping {component}: release {tag!r} {reason}.", file=sys.stderr)
        else:
            print(f"Keeping {component}: release {tag!r} has {reason}.", file=sys.stderr)
            needed.append(component)
    return needed


def combo_for_matrix_entry(
    versions: Dict[str, Any], entry: Dict[str, Any]
) -> Dict[str, Any]:
    """Find the versions.yaml build_matrix row that produced a matrix entry."""
    component = str(entry["component"])
    for combo in component_combos(versions, component):
        if (
            combo_arch(combo) == entry["arch"]
            and str(combo["cuda"]) == str(entry["cuda"])
            and str(combo["python"]) == str(entry["python"])
            and str(combo["torch"]) == str(entry["torch"])
        ):
            return combo
    raise SystemExit(
        f"No build_matrix combo matches {component} arch={entry['arch']} "
        f"cuda={entry['cuda']} python={entry['python']} torch={entry['torch']}"
    )


def evaluate_matrix_rows(
    versions: Dict[str, Any],
    entries: List[Dict[str, Any]],
    repo: str,
    verify_wheels: bool = True,
) -> List[Dict[str, Any]]:
    """Decide build vs skip for every matrix row, with the reason why.

    Returns one record per input row: {"entry", "decision": "build"|"skip",
    "reason"}. Each component's release is inspected at most once. Fail-closed:
    a release that is missing/unreadable, a missing/stale build manifest, a
    missing wheel, or a fatbin-coverage mismatch all decide "build".
    """
    decisions = []
    releases: Dict[str, Optional[Dict[str, Any]]] = {}
    for entry in entries:
        component = str(entry["component"])
        tag = str(entry["release_tag"])
        label = f"{component} ({entry['arch']})"
        if component not in releases:
            releases[component] = inspect_release(repo, tag)
        release = releases[component]
        if release is None:
            reason = "release not found or unreadable; it would be created"
            print(f"Keeping {label}: {reason}.", file=sys.stderr)
            decisions.append({"entry": entry, "decision": "build", "reason": reason})
            continue

        combo = combo_for_matrix_entry(versions, entry)
        covered, reason = release_covers_combo(
            versions, component, combo, release, repo, verify_wheels=verify_wheels
        )
        if covered:
            print(f"Skipping {label}: release {tag!r} {reason}.", file=sys.stderr)
            decisions.append({"entry": entry, "decision": "skip", "reason": reason})
        else:
            print(f"Keeping {label}: release {tag!r} has {reason}.", file=sys.stderr)
            decisions.append({"entry": entry, "decision": "build", "reason": reason})
    return decisions


def rows_needing_build(
    versions: Dict[str, Any],
    entries: List[Dict[str, Any]],
    repo: str,
    verify_wheels: bool = True,
) -> List[Dict[str, Any]]:
    """Keep only matrix rows whose wheel build config is not already on the release."""
    decisions = evaluate_matrix_rows(
        versions, entries, repo, verify_wheels=verify_wheels
    )
    return [d["entry"] for d in decisions if d["decision"] == "build"]


def format_dry_run_report(decisions: List[Dict[str, Any]], repo: str) -> str:
    """Markdown build/skip preview for PRs: no builds run, a human judges it."""
    builds = [d for d in decisions if d["decision"] == "build"]
    skips = [d for d in decisions if d["decision"] == "skip"]
    lines = [
        "## Wheelhouse build plan (dry run)",
        "",
        (
            f"Compared the `versions.yaml` in this change against the releases on "
            f"`{repo}`. Nothing is built or published from this check - it only "
            "shows which matrix rows the next push would rebuild and which it "
            "would skip, so a human can confirm the plan matches the intent of "
            "the change."
        ),
        "",
        f"**{len(builds)} row(s) would build, {len(skips)} row(s) would be skipped.**",
        "",
        "| Component | Ref | Arch | CUDA | Python | Torch | Decision | Reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in decisions:
        entry = d["entry"]
        decision = ":hammer: build" if d["decision"] == "build" else ":fast_forward: skip"
        reason = str(d["reason"]).replace("|", "\\|")
        lines.append(
            f"| `{entry['component']}` | `{entry['ref']}` | `{entry['arch']}` "
            f"| {entry['cuda']} | {entry['python']} | {entry['torch']} "
            f"| {decision} | {reason} |"
        )
    lines.extend(
        [
            "",
            (
                "<sub>Each row is decided from the release title, the "
                "`wheelhouse-build-manifest.json` release asset (dependency "
                "versions, CUDA arch list, extra env, builder command, "
                "`max_jobs`, `runs_on`, and sha256 fingerprints of "
                "`_build.yml`, the builder script, `common.sh` and declared "
                "patches), and wheel package presence. This dry run does not "
                "download wheels; the published wheels' fatbin SM-arch "
                "coverage is re-verified at push time and gated before "
                "upload. Anything that cannot be verified fails closed "
                "towards a rebuild.</sub>"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def component_names(versions: Dict[str, Any]) -> List[str]:
    return sorted(versions["components"])


def get_component(versions: Dict[str, Any], name: str) -> Dict[str, Any]:
    try:
        return versions["components"][name]
    except KeyError as exc:
        known = ", ".join(component_names(versions))
        raise SystemExit(f"Unknown component {name!r}. Known components: {known}") from exc


def build_matrix_entries(versions: Dict[str, Any], component: str) -> List[Dict[str, Any]]:
    """Cartesian-product one component's config against the build_matrix.

    Only the rows whose arch the component opts into, and with that arch's
    `arch_overrides` merged in (so e.g. an aarch64 row can carry its own
    runner and CUDA arch list).
    """
    ref = str(get_component(versions, component)["ref"])
    entries = []
    for combo in component_combos(versions, component):
        arch = combo_arch(combo)
        cfg = component_config(versions, component, arch)
        entries.append(
            {
                "component": component,
                "path": cfg["path"],
                "ref": ref,
                "release_tag": release_tag(component, ref),
                "builder": cfg["builder"],
                "arch": arch,
                # Reusable-workflow inputs are strings, while GitHub Actions
                # accepts either a label or a label array for `runs-on`.
                # Preserve both config forms by sending a JSON-encoded value;
                # _build.yml restores it with fromJSON().
                "runs_on": json.dumps(cfg["runs_on"]),
                "cuda": str(combo["cuda"]),
                "python": str(combo["python"]),
                "torch": str(combo["torch"]),
                "torch_vision": str(combo.get("torch_vision") or ""),
                "torch_audio": str(combo.get("torch_audio") or ""),
                "cxx11_abi": str(combo["cxx11_abi"]),
                "torch_cuda_arch_list": cfg.get("torch_cuda_arch_list") or "",
                # Whether _build.yml should gate the upload on the produced
                # wheels fat-binning every arch in torch_cuda_arch_list.
                # Rehosted/prebuilt packages (e.g. flashinfer) set false.
                "verify_wheel_archs": "false" if cfg.get("verify_wheel_archs") is False else "true",
                "requires_cudnn": "true" if cfg.get("requires_cudnn") else "false",
                "max_jobs": str(cfg.get("max_jobs", 1)),
                "build_timeout": str(cfg.get("build_timeout", "5h")),
                "job_timeout_minutes": int(cfg.get("job_timeout_minutes", 360)),
                # Already a JSON *string*; the calling workflow passes it
                # straight through to _build.yml's extra-env input (do not
                # re-encode it with toJSON() in the workflow, or it will be
                # double-escaped).
                "extra_env": json.dumps(cfg.get("extra_env") or {}, sort_keys=True),
            }
        )
    return entries


def build_full_matrix(versions: Dict[str, Any], components: List[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for name in components:
        entries.extend(build_matrix_entries(versions, name))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--component",
        default="all",
        help="Component name from versions.yaml, or 'all' (default) for every component.",
    )
    parser.add_argument(
        "--arch",
        default="all",
        help=(
            "Restrict the matrix to one build_matrix arch (e.g. aarch64), or 'all' "
            "(default) for every arch each component opts into. Useful to test one "
            "arch without also spending a runner on the others."
        ),
    )
    parser.add_argument(
        "--list-components",
        action="store_true",
        help="Print known component names (one per line) and exit.",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Also write matrix=<json> and has_builds=<true|false> to $GITHUB_OUTPUT.",
    )
    parser.add_argument(
        "--skip-existing-releases",
        action="store_true",
        help=(
            "Omit matrix rows whose target release already has a matching title, "
            "an identical per-wheel build config, and the expected wheels."
        ),
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub owner/repo used with --skip-existing-releases (defaults to $GITHUB_REPOSITORY).",
    )
    parser.add_argument(
        "--no-verify-wheels",
        action="store_true",
        help=(
            "Do not download release wheels to re-verify their fatbin SM-arch "
            "coverage; decide from the release title, build-manifest asset and "
            "wheel filenames only. Used by the PR dry-run report to stay cheap "
            "(wheels can be multi-GB); the real re-verification runs at push time."
        ),
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        help=(
            "Dry-run mode: write a markdown build/skip report for every matrix "
            "row (with the reason) to PATH and exit 0 without printing the JSON "
            "matrix. Intended for PR CI, where a human judges whether the plan "
            "matches the change. Requires --repo or $GITHUB_REPOSITORY."
        ),
    )
    args = parser.parse_args()

    versions = load_versions()

    if args.list_components:
        print("\n".join(component_names(versions)))
        return

    if args.component == "all":
        components = component_names(versions)
    else:
        get_component(versions, args.component)  # validates, raises a clear SystemExit if unknown
        components = [args.component]

    matrix = build_full_matrix(versions, components)

    if args.arch != "all":
        available = matrix_arches(versions)
        if args.arch not in available:
            parser.error(f"Unknown arch {args.arch!r}. Known arches: {', '.join(available)}, all")
        # An empty result is expected and fine here (e.g. --component apex
        # --arch aarch64, for a component that opts out of that arch): the
        # has_builds output below lets the calling workflow skip its build job
        # rather than fail on an empty matrix.
        matrix = [entry for entry in matrix if entry["arch"] == args.arch]

    if args.report:
        if not args.repo:
            parser.error("--report requires --repo or $GITHUB_REPOSITORY")
        decisions = evaluate_matrix_rows(
            versions,
            matrix,
            args.repo,
            verify_wheels=not args.no_verify_wheels,
        )
        Path(args.report).write_text(
            format_dry_run_report(decisions, args.repo), encoding="utf-8"
        )
        return

    if args.skip_existing_releases:
        if not args.repo:
            parser.error("--skip-existing-releases requires --repo or $GITHUB_REPOSITORY")
        matrix = rows_needing_build(
            versions, matrix, args.repo, verify_wheels=not args.no_verify_wheels
        )

    payload = json.dumps(matrix)
    print(payload)

    if args.github_output:
        github_output = os.environ.get("GITHUB_OUTPUT")
        if not github_output:
            raise SystemExit("--github-output requires $GITHUB_OUTPUT to be set")
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"matrix={payload}\n")
            fh.write(f"has_builds={'true' if matrix else 'false'}\n")


if __name__ == "__main__":
    main()
