#!/usr/bin/env python3

import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_matrix
from test_cuda_archs import fatbin_elf, make_wheel


def make_versions(component):
    return {"build_matrix": [{"cuda": "13.0.2", "python": "3.12", "torch": "2.11.0"}],
            "components": {"demo": component}}


def make_release(versions, assets):
    """A release dict whose title/notes are computed the way release_meta.py
    computes them, so the hidden body manifest matches versions.yaml exactly."""
    cfg = versions["components"]["demo"]
    ref = str(cfg["ref"])
    return {
        "name": generate_matrix.release_title(ref, "demo", generate_matrix.component_combos(versions, "demo")),
        "body": generate_matrix.format_release_notes(versions, "demo"),
        "assets": [{"name": name, "size": size} for name, size in assets],
    }


DEMO_WHEELS = [
    ("demo-1.2.3-cp312-cp312-manylinux_2_28_x86_64.whl", 1000),
    ("demo_helper-1.2.3-py3-none-any.whl", 500),
]

# A deterministic stand-in for the builder/common/patch sha256 fingerprint;
# these tests don't care about real repo files, only about skip logic.
STUB_FINGERPRINT = [{"path": "ci/build_scripts/demo.sh", "sha256": "deadbeef"}]


class ExistingReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._fp_patch = patch(
            "generate_matrix.build_input_fingerprint", return_value=STUB_FINGERPRINT
        )
        self._cuobjdump_patch = patch("cuda_archs.find_cuobjdump", return_value=None)
        self._fp_patch.start()
        self._cuobjdump_patch.start()
        self.addCleanup(self._fp_patch.stop)
        self.addCleanup(self._cuobjdump_patch.stop)

        self.versions = make_versions(
            {"ref": "v1.2.3", "wheel_packages": ["demo", "demo-helper"]}
        )
        self.release = make_release(self.versions, DEMO_WHEELS)

    def test_exact_release_covers_component(self) -> None:
        covered, _ = generate_matrix.release_covers_component(
            self.versions, "demo", self.release
        )
        self.assertTrue(covered)

    def test_dependency_title_must_match_exactly(self) -> None:
        self.release["name"] = "demo v1.2.3 - cu12.8.1 py3.12 torch2.11.0"
        covered, reason = generate_matrix.release_covers_component(
            self.versions, "demo", self.release
        )
        self.assertFalse(covered)
        self.assertIn("title mismatch", reason)

    def test_every_expected_wheel_package_is_required(self) -> None:
        self.release["assets"].pop()
        covered, reason = generate_matrix.release_covers_component(
            self.versions, "demo", self.release
        )
        self.assertFalse(covered)
        self.assertIn("demo-helper", reason)

    def test_missing_manifest_forces_rebuild(self) -> None:
        self.release["body"] = "human notes only, no snapshot"
        covered, reason = generate_matrix.release_covers_component(
            self.versions, "demo", self.release
        )
        self.assertFalse(covered)
        self.assertIn("no stored build config", reason)

    @patch("generate_matrix.inspect_release")
    def test_matching_component_is_removed_from_builds(self, inspect_release) -> None:
        inspect_release.return_value = self.release
        needed = generate_matrix.components_needing_build(
            self.versions, ["demo"], "owner/repo"
        )
        self.assertEqual([], needed)
        inspect_release.assert_called_once_with("owner/repo", "demo-v1.2.3")

    @patch("generate_matrix.inspect_release")
    def test_detection_failure_keeps_build(self, inspect_release) -> None:
        inspect_release.return_value = None
        needed = generate_matrix.components_needing_build(
            self.versions, ["demo"], "owner/repo"
        )
        self.assertEqual(["demo"], needed)


class BuildConfigManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._fp_patch = patch(
            "generate_matrix.build_input_fingerprint", return_value=STUB_FINGERPRINT
        )
        self._cuobjdump_patch = patch("cuda_archs.find_cuobjdump", return_value=None)
        self._fp_patch.start()
        self._cuobjdump_patch.start()
        self.addCleanup(self._fp_patch.stop)
        self.addCleanup(self._cuobjdump_patch.stop)

        self.versions = make_versions(
            {
                "ref": "v1.2.3",
                "wheel_packages": ["demo"],
                "builder": "demo",
                "torch_cuda_arch_list": "8.0;9.0;10.0",
                "max_jobs": 4,
                "runs_on": ["self-hosted", "Linux", "X64"],
            }
        )
        self.combo = generate_matrix.component_combos(self.versions, "demo")[0]

    def test_manifest_records_command_and_build_factors(self) -> None:
        config = generate_matrix.combo_build_config(self.versions, "demo", self.combo)
        self.assertEqual(config["command"], "bash ci/build_scripts/demo.sh")
        self.assertEqual(config["max_jobs"], "4")
        self.assertEqual(config["runs_on"], ["Linux", "X64", "self-hosted"])
        self.assertEqual(config["cuda_archs"], ["10.0", "8.0", "9.0"])
        self.assertEqual(config["build_inputs"], STUB_FINGERPRINT)

    def test_smaller_arch_set_in_manifest_forces_rebuild(self) -> None:
        # A release cut when versions.yaml only promised 8.0/9.0 must not be
        # trusted for a current config that also promises 10.0, even before
        # the wheels' fatbin is downloaded.
        release = make_release(self.versions, [DEMO_WHEELS[0]])
        manifest = generate_matrix.parse_stored_build_manifest(release["body"])
        manifest["builds"][0]["torch_cuda_arch_list"] = "8.0;9.0"
        manifest["builds"][0]["cuda_archs"] = ["8.0", "9.0"]
        blob = json.dumps(manifest, indent=2, sort_keys=True)
        release["body"] = re.sub(
            r"<!-- wheelhouse-build-config.*?-->",
            f"<!-- {generate_matrix.BUILD_CONFIG_MARKER}\n{blob}\n-->",
            release["body"],
            flags=re.DOTALL,
        )
        covered, reason = generate_matrix.release_covers_combo(
            self.versions, "demo", self.combo, release, repo=None
        )
        self.assertFalse(covered)
        self.assertIn("build config mismatch", reason)

    def test_changed_build_input_hash_forces_rebuild(self) -> None:
        # A patch/builder/_build.yml edit changes a sha256 in build_inputs;
        # no versions.yaml pin moved, but the rebuild the edit was meant to
        # ship must not be skipped.
        release = make_release(self.versions, [DEMO_WHEELS[0]])
        manifest = generate_matrix.parse_stored_build_manifest(release["body"])
        manifest["builds"][0]["build_inputs"][0]["sha256"] = "00000000-changed"
        blob = json.dumps(manifest, indent=2, sort_keys=True)
        release["body"] = re.sub(
            r"<!-- wheelhouse-build-config.*?-->",
            f"<!-- {generate_matrix.BUILD_CONFIG_MARKER}\n{blob}\n-->",
            release["body"],
            flags=re.DOTALL,
        )
        covered, reason = generate_matrix.release_covers_combo(
            self.versions, "demo", self.combo, release, repo=None
        )
        self.assertFalse(covered)
        self.assertIn("build config mismatch", reason)

    def test_no_repo_fails_closed_when_arches_declared(self) -> None:
        release = make_release(self.versions, [DEMO_WHEELS[0]])
        covered, reason = generate_matrix.release_covers_combo(
            self.versions, "demo", self.combo, release, repo=None
        )
        self.assertFalse(covered)
        self.assertIn("cannot verify", reason)

    def test_verify_opt_out_skips_fatbin_check(self) -> None:
        self.versions["components"]["demo"]["verify_wheel_archs"] = False
        release = make_release(self.versions, [DEMO_WHEELS[0]])
        covered, reason = generate_matrix.release_covers_combo(
            self.versions, "demo", self.combo, release, repo=None
        )
        self.assertTrue(covered, reason)
        self.assertIn("opted out", reason)

    def _release_with_wheel(self, arches_in_wheel):
        wheel_name = "demo-1.2.3-cp312-cp312-manylinux_2_28_x86_64.whl"
        release = make_release(self.versions, [(wheel_name, 1000)])

        def fake_download(_repo, _tag, asset_name, dest_dir):
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            path = dest_dir / asset_name
            tokens = [f"sm_{a.replace('.', '')}".encode() for a in arches_in_wheel]
            make_wheel(path, [fatbin_elf(tokens)])
            return path

        return release, wheel_name, fake_download

    def test_fatbin_covering_all_arches_is_skipped(self) -> None:
        release, _wheel, fake_download = self._release_with_wheel(["8.0", "9.0", "10.0"])
        with patch("generate_matrix.download_release_asset", side_effect=fake_download):
            covered, reason = generate_matrix.release_covers_combo(
                self.versions, "demo", self.combo, release, repo="owner/repo"
            )
        self.assertTrue(covered, reason)
        self.assertIn("wheel fatbin covers", reason)

    def test_fatbin_missing_sm80_forces_rebuild(self) -> None:
        # The deep-ep failure mode: patch didn't apply, so the sm_80 cubin
        # never materialized even though versions.yaml declares 8.0.
        release, _wheel, fake_download = self._release_with_wheel(["9.0", "10.0"])
        with patch("generate_matrix.download_release_asset", side_effect=fake_download):
            covered, reason = generate_matrix.release_covers_combo(
                self.versions, "demo", self.combo, release, repo="owner/repo"
            )
        self.assertFalse(covered)
        self.assertIn("8.0", reason)
        self.assertIn("rebuilding", reason)

    def test_download_failure_fails_closed(self) -> None:
        release, wheel_name, _ = self._release_with_wheel(["8.0", "9.0", "10.0"])

        def boom(_repo, _tag, asset_name, _dest_dir):
            raise generate_matrix.WheelInspectionError("gh not authenticated")

        with patch("generate_matrix.download_release_asset", side_effect=boom):
            covered, reason = generate_matrix.release_covers_combo(
                self.versions, "demo", self.combo, release, repo="owner/repo"
            )
        self.assertFalse(covered)
        self.assertIn("keeping build", reason)
        self.assertIn(wheel_name, reason)

    def test_manifest_asset_file_is_the_primary_source(self) -> None:
        # Release carries the wheelhouse-build-manifest.json asset and no body
        # snapshot at all: the asset must be downloaded and used.
        release, wheel_name, fake_download = self._release_with_wheel(["8.0", "9.0", "10.0"])
        manifest_blob = json.dumps(
            generate_matrix.component_build_manifest(self.versions, "demo"),
            indent=2,
            sort_keys=True,
        )
        release["body"] = None
        release["assets"].append(
            {"name": generate_matrix.BUILD_MANIFEST_FILENAME, "size": len(manifest_blob)}
        )

        def fake_download(_repo, _tag, asset_name, dest_dir):
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            path = dest_dir / asset_name
            if asset_name == generate_matrix.BUILD_MANIFEST_FILENAME:
                path.write_text(manifest_blob, encoding="utf-8")
            else:
                make_wheel(path, [fatbin_elf([b"sm_80", b"sm_90", b"sm_100"])])
            return path

        with patch("generate_matrix.download_release_asset", side_effect=fake_download):
            covered, reason = generate_matrix.release_covers_combo(
                self.versions, "demo", self.combo, release, repo="owner/repo"
            )
        self.assertTrue(covered, reason)

    def test_manifest_asset_with_old_schema_falls_back_and_rebuilds(self) -> None:
        release, wheel_name, fake_download = self._release_with_wheel(["8.0", "9.0", "10.0"])
        release["body"] = "no body snapshot"
        release["assets"].append(
            {"name": generate_matrix.BUILD_MANIFEST_FILENAME, "size": 100}
        )

        def fake_download(_repo, _tag, asset_name, dest_dir):
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            path = dest_dir / asset_name
            if asset_name == generate_matrix.BUILD_MANIFEST_FILENAME:
                path.write_text(json.dumps({"schema": 1, "builds": []}), encoding="utf-8")
            else:
                make_wheel(path, [fatbin_elf([b"sm_80", b"sm_90", b"sm_100"])])
            return path

        with patch("generate_matrix.download_release_asset", side_effect=fake_download):
            covered, reason = generate_matrix.release_covers_combo(
                self.versions, "demo", self.combo, release, repo="owner/repo"
            )
        self.assertFalse(covered)
        self.assertIn("no stored build config", reason)


if __name__ == "__main__":
    unittest.main()
