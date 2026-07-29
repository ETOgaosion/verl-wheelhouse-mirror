#!/usr/bin/env python3
"""Build a static PEP 503 "simple" package index from this repo's GitHub
Release assets, for deployment to GitHub Pages.

Every .whl attached to any GitHub Release in the repo is listed under its
(PEP 503-normalized) distribution name, linking straight back to the release
asset's own download URL - GitHub Pages only ever serves the tiny index
HTML, never the wheel bytes themselves.

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


class WheelAsset(TypedDict):
    name: str
    url: str


def normalize(name: str) -> str:
    """PEP 503 project-name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _gh_json(*args: str) -> object:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def list_wheel_assets(repo: str) -> List[WheelAsset]:
    """Enumerate every .whl asset across every release in the repo."""
    releases = _gh_json(
        "release", "list", "--repo", repo, "--json", "tagName", "--limit", "1000"
    )
    assets: List[WheelAsset] = []
    for release in releases:  # type: ignore[union-attr]
        tag = release["tagName"]
        view = _gh_json("release", "view", tag, "--repo", repo, "--json", "assets")
        for asset in view.get("assets", []):  # type: ignore[union-attr]
            if asset["name"].endswith(".whl"):
                assets.append({"name": asset["name"], "url": asset["url"]})
    return assets


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


def build_index(assets: List[WheelAsset], out_dir: Path) -> Dict[str, List[WheelAsset]]:
    by_package = group_by_package(assets)

    simple_dir = out_dir / "simple"
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
    (simple_dir / "index.html").write_text(render_page("verl-wheelhouse simple index", root_links))

    landing_body = (
        "    <h1>verl-wheelhouse</h1>\n"
        "    <p>Prebuilt CUDA wheels for apex, TransformerEngine, flash-attention, "
        "flashinfer, and Megatron-Bridge, built by "
        '<a href="https://github.com/verl-project/verl">verl</a>\'s wheelhouse CI.</p>\n'
        "    <p>Install with:</p>\n"
        "    <pre>pip install --extra-index-url &lt;this-pages-url&gt;/simple/ &lt;package&gt;</pre>\n"
        '    <p><a href="simple/">Browse the package index</a></p>\n'
    )
    (out_dir / "index.html").write_text(render_page("verl-wheelhouse", landing_body))

    return by_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="owner/repo, e.g. verl-project/verl-wheelhouse")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    assets = list_wheel_assets(args.repo)
    if not assets:
        print("No .whl release assets found yet; publishing an empty index.", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_package = build_index(assets, args.out_dir)
    print(f"Indexed {len(assets)} wheel(s) across {len(by_package)} package(s).")


if __name__ == "__main__":
    main()
