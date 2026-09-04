#!/usr/bin/env python3
"""Compute GitHub Release metadata (tag/title/notes) for one or every
component in versions.yaml.

Each component gets its own persistent GitHub Release, keyed by the
component's currently-pinned ref rather than by a shared "latest"/repo-level
tag:

    tag:   "<component>-<ref>"                     e.g. "transformer-engine-v2.16.1"
    title: "<component> <ref> - [<arch> ]cu<cuda> py<python> torch<torch>[; ...]"
           (one segment per versions.yaml build_matrix entry the component is
           built for, with the x86_64 arch left implicit) e.g.
           "transformer-engine v2.16.1 - cu13.0.2 py3.12 torch2.11.0"
    notes: human-readable pin plus a hidden JSON snapshot of each wheel's
           build config (CUDA/Python/Torch, torch_cuda_arch_list, extra_env,
           builder, ...). Skip detection requires that snapshot to match
           versions.yaml exactly; title and wheel filename stay free of GPU arch.

Rebuilding the same ref re-uploads (--clobber) wheels onto the same
release; bumping a component's ref in versions.yaml starts a brand new
release (new tag), leaving the previous one attached to the old ref as a
historical record. Release notes include a hidden JSON snapshot of each
wheel's build config (CUDA arch list, extra_env, builder, ...) so skip
detection can rebuild when those change without putting them in the title.
This module reuses generate_matrix.py's load_versions/release_tag helpers
so the tag computed here always matches the "release_tag" field _build.yml
is given for that component's rows.

Usage:
    python ci/release_meta.py --component apex
    python ci/release_meta.py --component all
    python ci/release_meta.py --component apex --github-output
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from generate_matrix import (
    BUILD_MANIFEST_FILENAME,
    component_build_manifest,
    component_combos,
    component_names,
    format_release_notes,
    get_component,
    load_versions,
    release_tag,
    release_title,
)


def component_release_meta(versions: Dict[str, Any], component: str) -> Dict[str, Any]:
    cfg = get_component(versions, component)
    ref = str(cfg["ref"])
    return {
        "component": component,
        "ref": ref,
        "tag": release_tag(component, ref),
        "title": release_title(ref, component, component_combos(versions, component)),
        "notes": format_release_notes(versions, component),
        # Machine-readable build snapshot uploaded as a release asset
        # (BUILD_MANIFEST_FILENAME); the notes above embed the same JSON as a
        # fallback for older skip-detection code.
        "manifest": component_build_manifest(versions, component),
        "manifest_asset": BUILD_MANIFEST_FILENAME,
    }


def all_release_meta(versions: Dict[str, Any], components: List[str]) -> List[Dict[str, Any]]:
    return [component_release_meta(versions, name) for name in components]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--component",
        default="all",
        help="Component name from versions.yaml, or 'all' (default) for every component.",
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
    else:
        get_component(versions, args.component)  # validates, raises a clear SystemExit if unknown
        components = [args.component]

    entries = all_release_meta(versions, components)
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
