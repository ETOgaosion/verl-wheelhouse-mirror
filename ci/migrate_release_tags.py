#!/usr/bin/env python3
"""One-time migration: retag existing releases to the per-(torch, CUDA) scheme.

Releases used to be keyed by (component, Python) on a bare
"<component>-<ref>" tag, then by (component, Python, torch) as
"<...>-pyX.Y-torch<ver>"; they are now keyed by (component, Python, torch,
CUDA) as "<...>-pyX.Y-torch<ver>-cu<maj><min>" - see release_tag in
ci/generate_matrix.py. Adding CUDA to the tag is what stops a cu130 -> cu132
bump (same torch) from overwriting the cu130 wheels, the way keying on torch
stopped torch bumps earlier. This script renames releases already on GitHub
to match, so their assets keep their identity instead of being rebuilt:

    apex-master-py3.12-torch2.13.0
        -> apex-master-py3.12-torch2.13.0-cu130   (when its title says cu13.0.2)
    apex-master-py3.12-torch2.13.0               (cu13.2.2 wheels)
        -> apex-master-py3.12-torch2.13.0-cu132

Nothing is recompiled and no asset is re-uploaded; `gh release edit --tag`
only moves the release (and creates the new git tag). The old tag's download
URLs stop resolving, which is the intended outcome: a 404 is a loud signal to
re-pin, where a CUDA bump silently overwriting the wheels behind a URL lock
files reference was a silent ABI mismatch.

The new tag is derived from the release *title*, which release_meta.py
generates as "<component> <ref> - [<arch> ]cu<cuda> py<python> torch<torch>",
so component/ref and the CUDA world never have to be guessed out of a
dash-separated tag. Two consequences worth knowing before running it:

  * Run it only when no build is in flight. A release's title is refreshed
    at the start of a build and its wheels are uploaded at the end, so a
    running build means the title already describes a CUDA the attached
    wheels are not from yet - and retagging mid-run also moves the release
    out from under the job's own upload step.
  * `gh release edit --tag` creates the new git tag but leaves the old one
    behind, pointing at the same commit with no release on it. Prune them
    with `git push --delete origin <old-tag>` once the new tags look right.

Usage:
    python ci/migrate_release_tags.py --repo <owner>/<repo>            # dry run
    python ci/migrate_release_tags.py --repo <owner>/<repo> --apply
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Dict, List, Optional

from generate_matrix import release_tag

# "<component> <ref> - <segments>", e.g.
# "flash-attention v2.8.3 - cu13.0.2 py3.11 torch2.11.0; aarch64 ...".
_TITLE_RE = re.compile(r"^(?P<component>\S+)\s+(?P<ref>\S+)\s+-\s+(?P<segments>.+)$")
# Every build segment carries the same CUDA/Python/torch within one release;
# the first one is enough to key it.
_SEGMENT_RE = re.compile(
    r"cu(?P<cuda>[\w.]+)\s+py(?P<python>[\w.]+)\s+torch(?P<torch>[\w.+]+)"
)


def gh_json(*args: str) -> object:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def planned_tag(title: str) -> Optional[str]:
    """The tag `title`'s release should live under, or None if unparseable."""
    title_match = _TITLE_RE.match(title or "")
    if not title_match:
        return None
    segments = {
        (match["cuda"], match["python"], match["torch"])
        for match in _SEGMENT_RE.finditer(title_match["segments"])
    }
    if len(segments) != 1:
        # A release spanning several CUDA/Python/torch tuples cannot be
        # expressed as one tag; leave it for a human.
        return None
    cuda, python, torch = segments.pop()
    return release_tag(title_match["component"], title_match["ref"], python, torch, cuda)


def plan_migration(repo: str) -> List[Dict[str, str]]:
    releases = gh_json(
        "release", "list", "--repo", repo, "--json", "tagName,name", "--limit", "1000"
    )
    plan: List[Dict[str, str]] = []
    for release in releases:  # type: ignore[union-attr]
        old = release["tagName"]
        title = release.get("name") or ""
        new = planned_tag(title)
        if new is None:
            plan.append({"old": old, "new": "", "status": "unparseable title", "title": title})
        elif new == old:
            plan.append({"old": old, "new": new, "status": "already migrated", "title": title})
        else:
            plan.append({"old": old, "new": new, "status": "rename", "title": title})
    return plan


def check_plan(plan: List[Dict[str, str]]) -> List[str]:
    """Reasons the migration must not run, if any."""
    problems = []
    targets: Dict[str, str] = {}
    for entry in plan:
        if entry["status"] == "unparseable title":
            problems.append(f"{entry['old']}: cannot derive a tag from title {entry['title']!r}")
            continue
        clash = targets.get(entry["new"])
        if clash:
            problems.append(
                f"{entry['old']} and {clash} both want tag {entry['new']!r}"
            )
        targets[entry["new"]] = entry["old"]
    return problems


def apply_migration(repo: str, plan: List[Dict[str, str]]) -> None:
    for entry in plan:
        if entry["status"] != "rename":
            continue
        subprocess.run(
            ["gh", "release", "edit", entry["old"], "--repo", repo, "--tag", entry["new"]],
            check=True,
        )
        print(f"Retagged {entry['old']} -> {entry['new']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", required=True, help="owner/repo, e.g. verl-project/verl-wheelhouse")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually retag. Without it the plan is printed and nothing changes.",
    )
    args = parser.parse_args()

    plan = plan_migration(args.repo)
    for entry in sorted(plan, key=lambda item: item["old"]):
        arrow = f" -> {entry['new']}" if entry["status"] == "rename" else ""
        print(f"[{entry['status']}] {entry['old']}{arrow}")

    renames = [entry for entry in plan if entry["status"] == "rename"]
    problems = check_plan(plan)
    if problems:
        print("\nRefusing to migrate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(1)

    if not args.apply:
        print(f"\nDry run: {len(renames)} release(s) would be retagged. Re-run with --apply.")
        return
    apply_migration(args.repo, plan)
    print(f"\nRetagged {len(renames)} release(s).")


if __name__ == "__main__":
    main()
