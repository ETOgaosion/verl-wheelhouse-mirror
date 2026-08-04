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
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

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


def release_covers_component(
    versions: Dict[str, Any], component: str, release: Dict[str, Any]
) -> Tuple[bool, str]:
    """Check that a release has the exact dependency title and all expected wheels."""
    cfg = get_component(versions, component)
    ref = str(cfg["ref"])
    expected_title = release_title(ref, component, component_combos(versions, component))
    actual_title = release.get("name")
    if actual_title != expected_title:
        return False, f"title mismatch: expected {expected_title!r}, found {actual_title!r}"

    configured_packages = cfg.get("wheel_packages")
    if not isinstance(configured_packages, list) or not configured_packages:
        return False, "wheel_packages is not configured"

    # Every expected wheel is tracked per arch, not just per package name:
    # a component built for both x86_64 and aarch64 is only covered once both
    # platform-tagged wheels are on the release, otherwise the first arch to
    # upload would mark the other one's build as redundant forever.
    arches = component_arches(versions, component)
    required = {
        (normalize_package_name(str(name)), arch)
        for name in configured_packages
        for arch in arches
    }
    released = {
        (normalize_package_name(str(asset["name"]).split("-", 1)[0]), arch)
        for asset in release.get("assets", [])
        if isinstance(asset, dict)
        and isinstance(asset.get("name"), str)
        and asset["name"].endswith(".whl")
        for arch in wheel_arches(asset["name"], arches)
    }
    missing = sorted(required - released)
    if missing:
        formatted = ", ".join(f"{package} ({arch})" for package, arch in missing)
        return False, f"missing wheel package(s): {formatted}"

    return True, "exact dependency title and all expected wheel packages are present"


def inspect_release(repo: str, tag: str) -> Optional[Dict[str, Any]]:
    """Read the release metadata needed for skip detection with the GitHub CLI."""
    try:
        result = subprocess.run(
            ["gh", "release", "view", tag, "--repo", repo, "--json", "name,assets"],
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
    """Drop components already represented by complete, dependency-matching releases."""
    needed = []
    for component in components:
        cfg = get_component(versions, component)
        tag = release_tag(component, str(cfg["ref"]))
        release = inspect_release(repo, tag)
        if release is None:
            needed.append(component)
            continue

        covered, reason = release_covers_component(versions, component, release)
        if covered:
            print(f"Skipping {component}: release {tag!r} {reason}.", file=sys.stderr)
        else:
            print(f"Keeping {component}: release {tag!r} has {reason}.", file=sys.stderr)
            needed.append(component)
    return needed


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
                "torch_vision": str(combo["torch_vision"]),
                "torch_audio": str(combo["torch_audio"]),
                "cxx11_abi": str(combo["cxx11_abi"]),
                "torch_cuda_arch_list": cfg.get("torch_cuda_arch_list") or "",
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
        help="Omit components whose target release already has matching dependencies and wheels.",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub owner/repo used with --skip-existing-releases (defaults to $GITHUB_REPOSITORY).",
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

    if args.skip_existing_releases:
        if not args.repo:
            parser.error("--skip-existing-releases requires --repo or $GITHUB_REPOSITORY")
        components = components_needing_build(versions, components, args.repo)

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
