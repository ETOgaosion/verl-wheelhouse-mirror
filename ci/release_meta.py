#!/usr/bin/env python3
"""Compute GitHub Release metadata (tag/title/notes) for one or every
component in versions.yaml.

Each (component, Python version, torch version) triple gets its own
persistent GitHub Release, keyed by the component's currently-pinned ref:

    tag:   "<component>-<ref>-pyX.Y-torch<torch>"
           e.g. "apex-master-py3.12-torch2.13.0"
           (both versions are in the tag because the wheel is ABI-bound to
           each and its filename records neither: sharing a tag across torch
           versions would make the newer build clobber the older one's
           assets, silently changing what every pinned download URL serves)
    title: "<component> <ref> - [<arch> ]cu<cuda> py<python> torch<torch>[; ...]"
           (one segment per versions.yaml build_matrix row of that
           component *for that Python and torch only*, with the x86_64 arch
           left implicit) e.g.
           "apex master - cu13.0.2 py3.11 torch2.13.0; aarch64 cu13.0.2 py3.11 torch2.13.0"
    notes: human-readable pin plus a hidden JSON snapshot of each wheel's
           build config (CUDA/Python/Torch, torch_cuda_arch_list, extra_env,
           builder, ...). Skip detection requires that snapshot to match
           versions.yaml exactly; title and wheel filename stay free of GPU arch.

Rebuilding the same (ref, Python, torch) re-uploads (--clobber) wheels onto
the same release; bumping a component's ref - or the matrix's torch - starts
a brand new set of releases (new tags), leaving the previous ones attached to
the old versions as a historical record. This module reuses
generate_matrix.py's helpers so the tag computed here always matches the
"release_tag" field _build.yml is given for that matrix row.

With no --python, the running interpreter's major.minor selects the
release: _build.yml runs this after setup-python has put the matrix row's
Python on PATH, so the correct metadata is computed without another workflow
input. --python X.Y selects one interpreter explicitly; --python all emits
one entry per (component, interpreter the component opts into). --torch
likewise pins the torch version; when omitted it is inferred from the
build_matrix rows of the selected interpreter, which normally pin exactly
one.

Usage:
    python ci/release_meta.py --component apex
    python ci/release_meta.py --component apex --python 3.11
    python ci/release_meta.py --component apex --torch 2.13.0
    python ci/release_meta.py --component all --python all
    python ci/release_meta.py --component apex --github-output
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from typing import Any, Dict, List, Optional

from generate_matrix import (
    BUILD_MANIFEST_FILENAME,
    component_build_manifest,
    component_combos_for_release,
    component_cuda_for,
    component_names,
    component_python_versions,
    component_release_keys,
    component_torch_versions,
    format_release_notes,
    get_component,
    load_versions,
    matrix_python_versions,
    release_tag,
    release_title,
)


def runtime_python() -> str:
    """Major.minor of the running interpreter (setup-python's Python in CI)."""
    return ".".join(platform.python_version().split(".")[:2])


def resolve_torch(
    versions: Dict[str, Any], component: str, python: str, torch: Optional[str]
) -> str:
    """The torch version a release is for, inferred when not given explicitly."""
    available = component_torch_versions(versions, component, python)
    if torch is not None:
        if str(torch) not in available:
            raise SystemExit(
                f"Component {component!r} does not build Python {python} against torch "
                f"{torch}. It builds: {', '.join(available) or '(nothing)'}."
            )
        return str(torch)
    if len(available) != 1:
        raise SystemExit(
            f"Component {component!r} builds Python {python} against "
            f"{len(available)} torch versions ({', '.join(available) or 'none'}); "
            "pass --torch to pick one."
        )
    return available[0]


def resolve_cuda(
    versions: Dict[str, Any],
    component: str,
    python: str,
    torch: str,
    cuda: Optional[str],
) -> str:
    """The CUDA version a release is for, inferred when not passed explicitly.

    Each active component pins one CUDA per (Python, torch), so the
    _build.yml path (which passes --torch but no --cuda under setup-python)
    resolves it unambiguously; a historical/multi-CUDA matrix must name it.
    """
    available = [
        c
        for py, t, c in component_release_keys(versions, component)
        if py == python and t == torch
    ]
    if cuda is not None:
        if str(cuda) not in available:
            raise SystemExit(
                f"Component {component!r} does not build Python {python} / torch {torch} "
                f"against CUDA {cuda}. It builds: {', '.join(available) or '(nothing)'}."
            )
        return str(cuda)
    return component_cuda_for(versions, component, python, torch)


def component_release_meta(
    versions: Dict[str, Any],
    component: str,
    python: str,
    torch: Optional[str] = None,
    cuda: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = get_component(versions, component)
    ref = str(cfg["ref"])
    if not component_torch_versions(versions, component, python):
        available = ", ".join(component_python_versions(versions, component))
        raise SystemExit(
            f"Component {component!r} has no build_matrix rows for Python {python}. "
            f"It builds for: {available or '(nothing)'}."
        )
    torch = resolve_torch(versions, component, python, torch)
    cuda = resolve_cuda(versions, component, python, torch, cuda)
    combos = component_combos_for_release(versions, component, python, torch, cuda)
    return {
        "component": component,
        "python": str(python),
        "torch": torch,
        "cuda": cuda,
        "ref": ref,
        "tag": release_tag(component, ref, python, torch, cuda),
        "title": release_title(ref, component, combos),
        "notes": format_release_notes(versions, component, python, torch, cuda),
        # Machine-readable build snapshot uploaded as a release asset
        # (BUILD_MANIFEST_FILENAME); the notes above embed the same JSON as a
        # fallback for older skip-detection code.
        "manifest": component_build_manifest(versions, component, python, torch, cuda),
        "manifest_asset": BUILD_MANIFEST_FILENAME,
    }


def release_meta_entries(
    versions: Dict[str, Any],
    components: List[str],
    python: str,
    torch: Optional[str] = None,
    cuda: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """One metadata record per release the given components publish.

    With python == "all" every component contributes one record per
    (interpreter, torch, CUDA) tuple it opts into; otherwise components that
    don't opt into that interpreter are skipped (an explicitly named such
    component is rejected by main() before reaching here).
    """
    if python == "all":
        return [
            component_release_meta(versions, name, py, release_torch, release_cuda)
            for name in components
            for py, release_torch, release_cuda in component_release_keys(versions, name)
            if (torch is None or release_torch == torch)
            and (cuda is None or release_cuda == cuda)
        ]
    return [
        component_release_meta(versions, name, python, torch, cuda)
        for name in components
        if python in component_python_versions(versions, name)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--component",
        default="all",
        help="Component name from versions.yaml, or 'all' (default) for every component.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "Python version the release is for (e.g. 3.11), or 'all' for every "
            "interpreter each component opts into. Defaults to the running "
            "interpreter's major.minor (CI: the matrix row's setup-python)."
        ),
    )
    parser.add_argument(
        "--torch",
        default=None,
        help=(
            "Torch version the release is for (e.g. 2.13.0). Defaults to the one "
            "the build_matrix pins for the selected interpreter, which is "
            "unambiguous unless the matrix builds that interpreter against "
            "several torch versions."
        ),
    )
    parser.add_argument(
        "--cuda",
        default=None,
        help=(
            "Full CUDA version the release is for (e.g. 13.2.2). Defaults to the "
            "single CUDA the build_matrix pins for the selected Python/torch "
            "(unambiguous while each component opts into one CUDA); required if "
            "a matrix ever builds one (Python, torch) against several."
        ),
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Also write matrix=<json> to $GITHUB_OUTPUT, for use in a workflow step.",
    )
    args = parser.parse_args()

    versions = load_versions()

    if args.component == "all":
        components = component_names(versions)
        explicit = False
    else:
        get_component(versions, args.component)  # validates, raises a clear SystemExit if unknown
        components = [args.component]
        explicit = True

    python = args.python if args.python is not None else runtime_python()
    if python != "all":
        known = matrix_python_versions(versions)
        if python not in known:
            raise SystemExit(
                f"Python {python!r} is not in the build_matrix. Known versions: "
                f"{', '.join(known)}. Pass --python X.Y or 'all'."
            )
        if explicit and python not in component_python_versions(versions, args.component):
            available = ", ".join(component_python_versions(versions, args.component))
            raise SystemExit(
                f"Component {args.component!r} does not build for Python {python}. "
                f"It builds for: {available or '(nothing)'}."
            )

    entries = release_meta_entries(
        versions, components, python, args.torch, args.cuda
    )
    payload = json.dumps(entries)
    print(payload)

    if args.github_output:
        github_output = os.environ.get("GITHUB_OUTPUT")
        if not github_output:
            raise SystemExit("--github-output requires $GITHUB_OUTPUT to be set")
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"matrix={payload}\n")


if __name__ == "__main__":
    main()
