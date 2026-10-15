#!/usr/bin/env python3
"""One-time recovery: republish wheels from workflow artifacts to a release.

Before releases were keyed by torch version, bumping torch in versions.yaml
re-uploaded wheels onto the *same* release, overwriting the previous torch's
assets (the wheel filename records no torch, so the names collide). The
overwritten wheels are not lost while their build's workflow artifacts are
still within the repo's 90-day retention: every _build.yml run uploads each
wheel as "wheel-<component>-<arch>-cu<cuda>-py<python>-torch<torch>".

This script reads those artifacts and publishes them to the release the new
tag scheme gives them, so a torch world that was clobbered becomes
installable again without recompiling anything.

Release titles are computed from versions.yaml *as of the commit that
produced the artifacts* (--versions-ref), so they describe the build that
actually happened rather than today's matrix. No build manifest is attached:
a restored release is not a reproducible skip-detection target, and writing
one from today's build scripts would claim a provenance these wheels do not
have. Push builds never consult it either - the matrix has moved on to
another torch, which is a different release.

Usage:
    python ci/restore_release_wheels.py --repo <owner>/<repo> \\
        --run 34934509976 --run 34941009059 --versions-ref 815a454     # dry run
    ... --apply
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from generate_matrix import (
    component_combos_for_release,
    get_component,
    release_tag,
    release_title,
)

# Artifact naming from _build.yml's "Upload wheel artifact" step. Component
# names contain dashes, so every other field is anchored from the right.
_ARTIFACT_RE = re.compile(
    r"^wheel-(?P<component>.+)-(?P<arch>x86_64|aarch64)"
    r"-cu(?P<cuda>[\w.]+)-py(?P<python>[\w.]+)-torch(?P<torch>[\w.+]+)$"
)


def gh_json(*args: str) -> Any:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def run_with_retries(cmd: List[str], attempts: int = 4) -> None:
    """Run a gh command, retrying it on failure.

    Artifacts here are multi-GB, and a single read timeout against GitHub's
    blob backend used to abort the whole restore partway through. Every
    command this wraps is idempotent (a download into a fresh directory, an
    upload with --clobber), so retrying is always safe.
    """
    for attempt in range(1, attempts + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit status {result.returncode}"
        if attempt == attempts:
            raise RuntimeError(f"{' '.join(cmd[:3])} failed after {attempts} attempts: {reason}")
        wait = 5 * attempt
        print(f"  attempt {attempt}/{attempts} failed ({reason}); retrying in {wait}s", flush=True)
        time.sleep(wait)


def versions_at(ref: str) -> Dict[str, Any]:
    """versions.yaml as of a git ref, i.e. what the artifacts were built from."""
    result = subprocess.run(
        ["git", "show", f"{ref}:versions.yaml"], check=True, capture_output=True, text=True
    )
    return yaml.safe_load(result.stdout)


def run_artifacts(repo: str, run_id: str) -> List[Dict[str, Any]]:
    payload = gh_json("api", f"repos/{repo}/actions/runs/{run_id}/artifacts", "--paginate")
    return [
        artifact
        for artifact in payload.get("artifacts", [])
        if not artifact.get("expired")
    ]


def plan_restores(
    repo: str, run_ids: List[str], versions: Dict[str, Any], torch: str
) -> Dict[str, Dict[str, Any]]:
    """Group the matching artifacts by the release tag they belong to."""
    by_tag: Dict[str, Dict[str, Any]] = {}
    for run_id in run_ids:
        for artifact in run_artifacts(repo, run_id):
            match = _ARTIFACT_RE.match(artifact["name"])
            if not match or match["torch"] != torch:
                continue
            component = match["component"]
            try:
                cfg = get_component(versions, component)
            except SystemExit:
                print(
                    f"Skipping artifact {artifact['name']}: {component!r} is not in the "
                    f"versions.yaml being used for titles.",
                    file=sys.stderr,
                )
                continue
            ref = str(cfg["ref"])
            python, torch_version, cuda = (
                match["python"],
                match["torch"],
                match["cuda"],
            )
            tag = release_tag(component, ref, python, torch_version, cuda)
            combos = component_combos_for_release(
                versions, component, python, torch_version, cuda
            )
            entry = by_tag.setdefault(
                tag,
                {
                    "component": component,
                    "ref": ref,
                    "python": python,
                    "torch": torch_version,
                    "cuda": cuda,
                    "title": release_title(ref, component, combos),
                    "artifacts": [],
                },
            )
            entry["artifacts"].append({"name": artifact["name"], "run_id": run_id})
    return by_tag


def release_notes(entry: Dict[str, Any], run_ids: List[str]) -> str:
    runs = ", ".join(sorted({artifact["run_id"] for artifact in entry["artifacts"]}))
    return (
        f"Prebuilt CUDA wheel(s) for {entry['component']} on Python {entry['python']} / "
        f"torch {entry['torch']} / CUDA {entry['cuda']}, pinned to `{entry['ref']}`.\n\n"
        "Restored from the build's workflow artifacts (runs "
        f"{runs}) after a torch bump overwrote them on the release that predated "
        "per-torch release tags. The wheels are the original build's bytes; no "
        "build manifest is attached, so this release is not a skip-detection "
        "target - rebuilding this torch would produce a fresh one."
    )


def released_wheel_count(repo: str, tag: str) -> Optional[int]:
    """How many .whl assets the release already has, or None if it is absent."""
    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo, "--json", "assets"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    assets = json.loads(result.stdout).get("assets", [])
    return sum(1 for asset in assets if str(asset.get("name", "")).endswith(".whl"))


def restore(
    repo: str, tag: str, entry: Dict[str, Any], run_ids: List[str], apply: bool, force: bool
) -> bool:
    """Publish one release from its artifacts. True if anything was uploaded."""
    names = sorted(artifact["name"] for artifact in entry["artifacts"])
    print(f"{tag}\n  title: {entry['title']}\n  artifacts: {', '.join(names)}", flush=True)
    if not apply:
        return False

    # Each artifact holds one wheel, so a release already carrying that many
    # is done. Re-running after a failure (these downloads are multi-GB and
    # the network is not always kind) then costs nothing for what succeeded.
    if not force:
        existing = released_wheel_count(repo, tag)
        if existing is not None and existing >= len(entry["artifacts"]):
            print(f"  already has {existing} wheel(s); skipping (use --force to redo)")
            return False

    with tempfile.TemporaryDirectory(prefix="wheelhouse-restore-") as tmp:
        dest = Path(tmp)
        for artifact in entry["artifacts"]:
            run_with_retries(
                [
                    "gh", "run", "download", artifact["run_id"],
                    "--repo", repo,
                    "--name", artifact["name"],
                    "--dir", str(dest / artifact["name"]),
                ]
            )
        wheels = sorted(dest.rglob("*.whl"))
        if not wheels:
            raise RuntimeError(f"{tag}: artifacts contained no .whl files")

        notes_file = dest / "notes.md"
        notes_file.write_text(release_notes(entry, run_ids), encoding="utf-8")
        verb = "edit" if released_wheel_count(repo, tag) is not None else "create"
        run_with_retries(
            ["gh", "release", verb, tag, "--repo", repo,
             "--title", entry["title"], "--notes-file", str(notes_file)]
        )
        run_with_retries(
            ["gh", "release", "upload", tag, *[str(path) for path in wheels],
             "--repo", repo, "--clobber"]
        )
        print(f"  uploaded {len(wheels)} wheel(s)", flush=True)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", required=True, help="owner/repo, e.g. verl-project/verl-wheelhouse")
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        metavar="RUN_ID",
        help="Workflow run whose wheel artifacts to restore from (repeatable).",
    )
    parser.add_argument(
        "--torch",
        required=True,
        help="Only restore artifacts built against this torch version, e.g. 2.11.0.",
    )
    parser.add_argument(
        "--versions-ref",
        required=True,
        help=(
            "Git ref whose versions.yaml describes those builds (the commit the "
            "runs were triggered from), used for component refs and release titles."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually download and publish. Without it the plan is printed only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-upload releases that already carry their wheels.",
    )
    args = parser.parse_args()

    versions = versions_at(args.versions_ref)
    plan = plan_restores(args.repo, args.runs, versions, args.torch)
    if not plan:
        raise SystemExit(
            f"No unexpired artifacts for torch {args.torch} in run(s) {', '.join(args.runs)}."
        )

    # One release failing (a read timeout on a multi-GB artifact, most
    # likely) must not strand the ones after it: keep going and report at the
    # end, so a re-run only has the failures left to do.
    restored, failures = 0, []
    for tag in sorted(plan):
        try:
            if restore(args.repo, tag, plan[tag], args.runs, args.apply, args.force):
                restored += 1
        except RuntimeError as exc:
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            failures.append(tag)

    if not args.apply:
        print(f"\nDry run: {len(plan)} release(s) would be restored. Re-run with --apply.")
        return
    print(f"\nRestored {restored} release(s) of {len(plan)} planned.")
    if failures:
        print(f"Failed: {', '.join(failures)}. Re-run to retry just those.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
