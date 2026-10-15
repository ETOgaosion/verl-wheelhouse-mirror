#!/usr/bin/env python3
"""Build static PEP 503 "simple" package indexes from this repo's GitHub
Release assets, for deployment to GitHub Pages.

Every .whl attached to any GitHub Release in the repo is listed under its
(PEP 503-normalized) distribution name, linking straight back to the release
asset's own download URL - GitHub Pages only ever serves the tiny index
HTML, never the wheel bytes themselves.

One index is published per CUDA/torch world the releases cover:

    /cu130/torch2.13/simple/   every wheel built against cu13.0.2 + torch 2.13
    /cu130/torch2.11/simple/   ... and the ones still on torch 2.11
    /simple/                   alias for the build_matrix's current default

The split exists because a wheel filename records neither CUDA nor torch,
so two torch builds of the same component produce byte-different wheels with
identical names and versions. A single index could not hold both, and a
consumer resolving from it would have no way to ask for the build matching
its own torch; a per-world index URL makes that choice explicit and
unambiguous - the same shape download.pytorch.org and flashinfer.ai use.

Which world a release belongs to is read from its title, which
ci/release_meta.py generates with a "cu<cuda> py<python> torch<torch>"
segment per build.

Exception: a small set of genuinely CUDA/torch-independent distributions -
pure-Python `py3-none-any` wheels with no compiled or data payload - are
declared `world_agnostic: true` in versions.yaml (currently megatron-bridge).
Every release of such a distribution is mirrored into EVERY world, so a
consumer pinned to an older version keeps resolving it after the default
world moves (e.g. megatron-bridge 0.5.2 stays installable from the cu132
index even though it was first published under cu130). Different versions
have different filenames, so they coexist on one page; the newest is the
default and an exact `==<old>` pin still finds its wheel. This must not be
used for wheels that merely *name* py3-none-any but carry CUDA payloads
(flashinfer-cubin/jit-cache are CUDA-specific data despite the tag).

Usage:
    python ci/build_index.py --repo <owner>/<repo> --out-dir _site

Requires the `gh` CLI to be authenticated (e.g. via $GITHUB_TOKEN) and on
PATH, which is already true on GitHub-hosted runners.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, TypedDict

from generate_matrix import load_versions


class WheelAsset(TypedDict):
    name: str
    url: str
    worlds: List[str]


# "cu13.0.2 py3.12 torch2.13.0" segments of a release title (see
# ci/release_meta.py's release_title). One per build the release covers;
# under the current tag scheme they all agree on CUDA and torch and differ
# only by arch, but parsing every segment keeps older multi-world titles
# (and hand-made ones) from landing in the wrong index.
_TITLE_SEGMENT_RE = re.compile(r"cu(?P<cuda>[\w.]+)\s+py(?P<python>[\w.]+)\s+torch(?P<torch>[\w.+]+)")


def normalize(name: str) -> str:
    """PEP 503 project-name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_distribution(filename: str) -> str:
    """PEP 503-normalized distribution name from a wheel filename."""
    return normalize(filename.split("-")[0])


def is_universal_wheel(filename: str) -> bool:
    """True for a py3-none-any (interpreter/ABI/platform-independent) wheel."""
    return filename[: -len(".whl")].rsplit("-", 1)[-1] == "any"


def world_agnostic_distributions(versions: Dict[str, object]) -> set:
    """Normalized distributions mirrored into every index world.

    Sourced from `world_agnostic: true` components in versions.yaml and their
    `wheel_packages`. Only their py3-none-any wheels are actually mirrored
    (see assign_worlds); a compiled/data wheel is never promoted even if the
    flag is set, so flashinfer-style data wheels stay world-bound.
    """
    agnostic: set = set()
    for cfg in versions.get("components", {}).values():  # type: ignore[union-attr]
        if cfg.get("world_agnostic") is True:  # type: ignore[union-attr]
            agnostic.update(normalize(p) for p in cfg.get("wheel_packages", []))  # type: ignore[union-attr]
    return agnostic


def assign_worlds(asset: WheelAsset, all_worlds: List[str], agnostic: set) -> List[str]:
    """Index worlds one asset belongs to.

    World-agnostic py3-none-any wheels go to every world; every other wheel
    stays on the world(s) its release title declares.
    """
    if is_universal_wheel(asset["name"]) and wheel_distribution(asset["name"]) in agnostic:
        return list(all_worlds)
    return asset["worlds"]


def cuda_tag(cuda: str) -> str:
    """CUDA version as an index path segment: "13.0.2" -> "cu130"."""
    major_minor = "".join(str(cuda).split(".")[:2])
    return f"cu{major_minor}"


def torch_tag(torch: str) -> str:
    """Torch version as an index path segment: "2.13.0" -> "torch2.13".

    Patch releases keep libtorch's ABI, so wheels are interchangeable within
    a major.minor - the same granularity download.pytorch.org and
    flashinfer.ai use for their index paths.
    """
    major_minor = ".".join(str(torch).split(".")[:2])
    return f"torch{major_minor}"


def world_path(cuda: str, torch: str) -> str:
    """Index subtree one (CUDA, torch) pair is published under."""
    return f"{cuda_tag(cuda)}/{torch_tag(torch)}"


def release_worlds(title: str) -> List[str]:
    """The CUDA/torch worlds a release's title says its wheels were built for."""
    return list(
        dict.fromkeys(
            world_path(match["cuda"], match["torch"])
            for match in _TITLE_SEGMENT_RE.finditer(title or "")
        )
    )


def _gh_json(*args: str) -> object:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def list_wheel_assets(repo: str) -> List[WheelAsset]:
    """Enumerate every .whl asset across every release in the repo.

    A release whose title carries no parseable "cu.. py.. torch.." segment
    cannot be placed in a world, so its wheels are reported and left out of
    every index rather than guessed into one: an unlisted wheel is a visible
    404, a misfiled one installs and then fails at import with an ABI error.
    """
    releases = _gh_json(
        "release", "list", "--repo", repo, "--json", "tagName,name", "--limit", "1000"
    )
    assets: List[WheelAsset] = []
    for release in releases:  # type: ignore[union-attr]
        tag = release["tagName"]
        worlds = release_worlds(release.get("name") or "")
        view = _gh_json("release", "view", tag, "--repo", repo, "--json", "assets")
        wheels = [
            asset
            for asset in view.get("assets", [])  # type: ignore[union-attr]
            if asset["name"].endswith(".whl")
        ]
        if wheels and not worlds:
            print(
                f"Release {tag!r} has {len(wheels)} wheel(s) but no CUDA/torch segment "
                f"in its title ({release.get('name')!r}); leaving them out of the index.",
                file=sys.stderr,
            )
            continue
        for asset in wheels:
            assets.append({"name": asset["name"], "url": asset["url"], "worlds": worlds})
    return assets


def group_by_world(
    assets: List[WheelAsset],
    all_worlds: List[str],
    agnostic: set,
) -> Dict[str, List[WheelAsset]]:
    by_world: Dict[str, List[WheelAsset]] = defaultdict(list)
    for asset in assets:
        for world in assign_worlds(asset, all_worlds, agnostic):
            by_world[world].append(asset)
    return by_world


def group_by_package(assets: List[WheelAsset]) -> Dict[str, List[WheelAsset]]:
    by_package: Dict[str, List[WheelAsset]] = defaultdict(list)
    for asset in assets:
        # Per the wheel filename spec, the distribution name is everything
        # before the first "-" (version/build/tag segments follow).
        distribution = asset["name"].split("-")[0]
        by_package[normalize(distribution)].append(asset)
    return by_package


def render_page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "  <head>\n"
        '    <meta charset="utf-8">\n'
        f"    <title>{html.escape(title)}</title>\n"
        "  </head>\n"
        "  <body>\n"
        f"{body}\n"
        "  </body>\n"
        "</html>\n"
    )


def write_simple_tree(assets: List[WheelAsset], simple_dir: Path, label: str) -> Dict[str, List[WheelAsset]]:
    """Write one PEP 503 "simple" tree (root listing + one page per package)."""
    by_package = group_by_package(assets)
    simple_dir.mkdir(parents=True, exist_ok=True)

    for package, package_assets in by_package.items():
        pkg_dir = simple_dir / package
        pkg_dir.mkdir(parents=True, exist_ok=True)
        links = "\n".join(
            f'    <a href="{html.escape(a["url"])}">{html.escape(a["name"])}</a><br/>'
            for a in sorted(package_assets, key=lambda a: a["name"])
        )
        (pkg_dir / "index.html").write_text(render_page(package, links))

    root_links = "\n".join(
        f'    <a href="{html.escape(pkg)}/">{html.escape(pkg)}</a><br/>'
        for pkg in sorted(by_package)
    )
    (simple_dir / "index.html").write_text(render_page(label, root_links))
    return by_package


def all_declared_worlds(assets: List[WheelAsset]) -> List[str]:
    """Every world any release title declares, in first-seen order."""
    return list(dict.fromkeys(world for asset in assets for world in asset["worlds"]))


def build_index(
    assets: List[WheelAsset],
    out_dir: Path,
    default_world: str,
    agnostic: set,
) -> Dict[str, List[WheelAsset]]:
    """Publish one index per CUDA/torch world, plus the default-world alias."""
    declared = all_declared_worlds(assets)
    by_world = group_by_world(assets, declared, agnostic)

    for world, world_assets in by_world.items():
        write_simple_tree(
            world_assets,
            out_dir / Path(world) / "simple",
            f"verl-wheelhouse {world} simple index",
        )

    # /simple/ keeps serving the world versions.yaml currently builds, so URLs
    # published before the split (and anything that just wants "what the
    # wheelhouse builds today") keep working. Pin the world path instead to
    # stay on one torch across a future bump.
    write_simple_tree(
        by_world.get(default_world, []),
        out_dir / "simple",
        f"verl-wheelhouse simple index ({default_world})",
    )

    worlds_list = "\n".join(
        f'      <li><code>{html.escape(world)}/simple/</code> - '
        f"{len(group_by_package(by_world[world]))} package(s)"
        f"{' (also served at <code>/simple/</code>)' if world == default_world else ''}</li>"
        for world in sorted(by_world)
    )
    landing_body = (
        "    <h1>verl-wheelhouse</h1>\n"
        "    <p>Prebuilt CUDA wheels for apex, TransformerEngine, flash-attention, "
        "flashinfer, Megatron-Bridge, DeepEP and FlashMLA, built by "
        '<a href="https://github.com/verl-project/verl">verl</a>\'s wheelhouse CI.</p>\n'
        "    <p>Wheels are ABI-bound to the CUDA and torch they were built "
        "against, and their filenames record neither, so each combination has "
        "its own index. Pick the one matching your environment:</p>\n"
        "    <ul>\n"
        f"{worlds_list}\n"
        "    </ul>\n"
        "    <p>Install with:</p>\n"
        "    <pre>pip install --extra-index-url &lt;this-pages-url&gt;/"
        f"{html.escape(default_world)}/simple/ &lt;package&gt;</pre>\n"
        f'    <p><a href="{html.escape(default_world)}/simple/">Browse the current '
        "index</a></p>\n"
    )
    (out_dir / "index.html").write_text(render_page("verl-wheelhouse", landing_body))

    return by_world


def default_world(versions: Dict[str, object]) -> str:
    """The world /simple/ aliases: the build_matrix's first row."""
    matrix = versions["build_matrix"]  # type: ignore[index]
    first = matrix[0]  # type: ignore[index]
    return world_path(str(first["cuda"]), str(first["torch"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="owner/repo, e.g. verl-project/verl-wheelhouse")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--default-world",
        default=None,
        help=(
            "Index subtree to also serve at /simple/, e.g. cu130/torch2.13. "
            "Defaults to the world versions.yaml's first build_matrix row pins."
        ),
    )
    args = parser.parse_args()

    versions = load_versions()
    world = args.default_world or default_world(versions)
    agnostic = world_agnostic_distributions(versions)
    if agnostic:
        print(f"World-agnostic distributions (mirrored into every world): "
              f"{', '.join(sorted(agnostic))}", file=sys.stderr)

    assets = list_wheel_assets(args.repo)
    if not assets:
        print("No .whl release assets found yet; publishing an empty index.", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_world = build_index(assets, args.out_dir, world, agnostic)
    if world not in by_world:
        print(
            f"No release wheels for the default world {world!r}; /simple/ is empty "
            "(worlds found: " + (", ".join(sorted(by_world)) or "none") + ").",
            file=sys.stderr,
        )
    summary = ", ".join(
        f"{name} ({len(group_by_package(world_assets))} package(s))"
        for name, world_assets in sorted(by_world.items())
    )
    print(f"Indexed {len(assets)} wheel(s) across {len(by_world)} world(s): {summary or 'none'}.")


if __name__ == "__main__":
    main()
