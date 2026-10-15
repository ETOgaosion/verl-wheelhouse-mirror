#!/usr/bin/env python3

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_index
import generate_matrix
import release_meta
from test_cuda_archs import fatbin_elf, make_wheel


# Torch every fixture matrix below pins. Its value is arbitrary (these tests
# never touch a real wheel), but releases are keyed by it, so helpers and tag
# assertions have to agree on one.
DEMO_TORCH = "2.11.0"


def make_versions(component):
    return {
        "build_matrix": [
            {
                "cuda": "13.0.2",
                "python": "3.12",
                "torch": DEMO_TORCH,
                "torch_vision": "0.26.0",
                "torch_audio": "2.11.0",
                "cxx11_abi": "TRUE",
            }
        ],
        "components": {"demo": component},
    }


DEMO_CUDA = "13.0.2"


def make_release(versions, assets, python="3.12", torch=DEMO_TORCH, cuda=DEMO_CUDA):
    """A release dict whose title/notes are computed the way release_meta.py
    computes them for one (component, Python, torch, CUDA) release, so the
    hidden body manifest matches versions.yaml exactly."""
    cfg = versions["components"]["demo"]
    ref = str(cfg["ref"])
    return {
        "name": generate_matrix.release_title(
            ref,
            "demo",
            generate_matrix.component_combos_for_release(
                versions, "demo", python, torch, cuda
            ),
        ),
        "body": generate_matrix.format_release_notes(
            versions, "demo", python, torch, cuda
        ),
        "assets": [{"name": name, "size": size} for name, size in assets],
    }


DEMO_WHEELS = [
    ("demo-1.2.3-cp312-cp312-manylinux_2_28_x86_64.whl", 1000),
    ("demo_helper-1.2.3-py3-none-any.whl", 500),
]

# A deterministic stand-in for the toolchain/builder/lib/patch sha256
# fingerprint; these tests don't care about real repo files, only skip logic.
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
            self.versions, "demo", "3.12", DEMO_TORCH, DEMO_CUDA, self.release
        )
        self.assertTrue(covered)

    def test_dependency_title_must_match_exactly(self) -> None:
        self.release["name"] = "demo v1.2.3 - cu12.8.1 py3.12 torch2.11.0"
        covered, reason = generate_matrix.release_covers_component(
            self.versions, "demo", "3.12", DEMO_TORCH, DEMO_CUDA, self.release
        )
        self.assertFalse(covered)
        self.assertIn("title mismatch", reason)

    def test_every_expected_wheel_package_is_required(self) -> None:
        self.release["assets"].pop()
        covered, reason = generate_matrix.release_covers_component(
            self.versions, "demo", "3.12", DEMO_TORCH, DEMO_CUDA, self.release
        )
        self.assertFalse(covered)
        self.assertIn("demo-helper", reason)

    def test_missing_manifest_forces_rebuild(self) -> None:
        self.release["body"] = "human notes only, no snapshot"
        covered, reason = generate_matrix.release_covers_component(
            self.versions, "demo", "3.12", DEMO_TORCH, DEMO_CUDA, self.release
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
        inspect_release.assert_called_once_with(
            "owner/repo", f"demo-v1.2.3-py3.12-torch{DEMO_TORCH}-cu130"
        )

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
        # A patch/builder/lib/toolchain edit changes a sha256 in build_inputs;
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
            generate_matrix.component_build_manifest(
                self.versions, "demo", "3.12", DEMO_TORCH, DEMO_CUDA
            ),
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


class DryRunReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._fp_patch = patch(
            "generate_matrix.build_input_fingerprint", return_value=STUB_FINGERPRINT
        )
        self._cuobjdump_patch = patch("cuda_archs.find_cuobjdump", return_value=None)
        self._fp_patch.start()
        self._cuobjdump_patch.start()
        self.addCleanup(self._fp_patch.stop)
        self.addCleanup(self._cuobjdump_patch.stop)

        self.component_cfg = {
            "ref": "v1.2.3",
            "path": "demo",
            "builder": "demo",
            "wheel_packages": ["demo"],
            "torch_cuda_arch_list": "8.0;9.0;10.0",
            "max_jobs": 4,
            "runs_on": ["self-hosted", "Linux", "X64"],
        }
        self.versions = make_versions(dict(self.component_cfg))
        # A second component whose release does not exist yet.
        self.versions["components"]["ghost"] = dict(self.component_cfg, ref="v9.9.9")
        self.entries = generate_matrix.build_full_matrix(
            self.versions, ["demo", "ghost"]
        )

    def test_dry_run_covers_without_downloading_wheels(self) -> None:
        release = make_release(self.versions, [DEMO_WHEELS[0]])

        def boom(*_args, **_kwargs):
            raise AssertionError("dry-run must not download release wheels")

        combo = generate_matrix.component_combos(self.versions, "demo")[0]
        with patch("generate_matrix.download_release_asset", side_effect=boom):
            covered, reason = generate_matrix.release_covers_combo(
                self.versions,
                "demo",
                combo,
                release,
                repo="owner/repo",
                verify_wheels=False,
            )
        self.assertTrue(covered, reason)
        self.assertIn("dry-run mode", reason)

    @patch("generate_matrix.inspect_release")
    def test_report_lists_build_and_skip_decisions_with_reasons(
        self, inspect_release
    ) -> None:
        covered_release = make_release(self.versions, [DEMO_WHEELS[0]])

        def fake_inspect(_repo, tag):
            return None if tag.startswith("ghost-") else covered_release

        inspect_release.side_effect = fake_inspect
        decisions = generate_matrix.evaluate_matrix_rows(
            self.versions, self.entries, "owner/repo", verify_wheels=False
        )
        by_component = {d["entry"]["component"]: d for d in decisions}
        self.assertEqual(by_component["demo"]["decision"], "skip")
        self.assertEqual(by_component["ghost"]["decision"], "build")
        self.assertIn("would be created", by_component["ghost"]["reason"])

        report = generate_matrix.format_dry_run_report(decisions, "owner/repo")
        self.assertIn("1 row(s) would build, 1 row(s) would be skipped", report)
        self.assertIn("`demo`", report)
        self.assertIn("`ghost`", report)
        self.assertIn(":hammer: build", report)
        self.assertIn(":fast_forward: skip", report)
        self.assertIn("dry run", report)

    @patch("generate_matrix.inspect_release", return_value=None)
    def test_missing_release_is_a_build_decision(self, inspect_release) -> None:
        decisions = generate_matrix.evaluate_matrix_rows(
            self.versions,
            self.entries,
            "owner/repo",
            verify_wheels=False,
        )
        self.assertTrue(all(d["decision"] == "build" for d in decisions))
        self.assertEqual(2, len(decisions))
        inspect_release.assert_called()


class BuildInputFingerprintTests(unittest.TestCase):
    """build_input_fingerprint over the real repo files: the shared toolchain
    action fingerprints every target; a leaf lib fingerprints only the builders
    that source it (resolved transitively); pure CI orchestration never does."""

    def _paths(self, components):
        versions = {"components": components}
        result = {}
        for name in components:
            result[name] = {
                entry["path"]
                for entry in generate_matrix.build_input_fingerprint(
                    versions, name, "x86_64"
                )
            }
        return result

    def test_toolchain_action_fingerprints_every_target_and_workflow_does_not(self) -> None:
        paths = self._paths(
            {
                "apex": {"builder": "apex"},
                "transformer_engine": {
                    "builder": "transformer_engine",
                    "requires_cudnn": True,
                },
                "deep_ep": {
                    "builder": "deep_ep",
                    "patches": ["ci/patches/enable_deep_ep_sm80.py"],
                },
            }
        )
        action = ".github/actions/build-toolchain/action.yml"
        for component_paths in paths.values():
            self.assertIn(action, component_paths)
            # The reusable workflow is pure orchestration; editing it must skip.
            self.assertNotIn(".github/workflows/_build.yml", component_paths)
            # The old monolith must be gone from every fingerprint.
            self.assertNotIn("ci/build_scripts/common.sh", component_paths)

    def test_builder_only_sources_its_own_leaf_lib_closure(self) -> None:
        paths = self._paths(
            {
                "apex": {"builder": "apex"},
                "transformer_engine": {"builder": "transformer_engine"},
                "flash_attention": {"builder": "flash_attention"},
                "flash_mla": {"builder": "flash_mla"},
                "deep_ep": {"builder": "deep_ep"},
                "flashinfer": {"builder": "flashinfer"},
            }
        )
        # env.sh is sourced (directly or via another lib) by every builder.
        for component_paths in paths.values():
            self.assertIn("ci/build_scripts/lib/env.sh", component_paths)
        # nccl.sh is reached only by transformer_engine (it sources env.sh
        # itself, exercising the lib -> lib transitive edge).
        self.assertIn("ci/build_scripts/lib/nccl.sh", paths["transformer_engine"])
        for component in ("apex", "flash_attention", "flash_mla", "deep_ep", "flashinfer"):
            self.assertNotIn("ci/build_scripts/lib/nccl.sh", paths[component])
        # arch.sh only feeds the two builders that flatten the arch list.
        self.assertIn("ci/build_scripts/lib/arch.sh", paths["transformer_engine"])
        self.assertIn("ci/build_scripts/lib/arch.sh", paths["flash_attention"])
        self.assertNotIn("ci/build_scripts/lib/arch.sh", paths["apex"])
        # wheel_pack.sh only feeds the two builders that strip the local tag.
        self.assertIn("ci/build_scripts/lib/wheel_pack.sh", paths["deep_ep"])
        self.assertIn("ci/build_scripts/lib/wheel_pack.sh", paths["flash_mla"])
        self.assertNotIn("ci/build_scripts/lib/wheel_pack.sh", paths["transformer_engine"])
        # The flashinfer-only download helper stays on the flashinfer target.
        self.assertIn("ci/build_scripts/lib/flashinfer.sh", paths["flashinfer"])
        self.assertNotIn("ci/build_scripts/lib/flashinfer.sh", paths["deep_ep"])

    def test_each_builder_itself_and_its_patch_are_fingerprinted(self) -> None:
        paths = self._paths(
            {
                "deep_ep": {
                    "builder": "deep_ep",
                    "patches": ["ci/patches/enable_deep_ep_sm80.py"],
                }
            }
        )["deep_ep"]
        self.assertIn("ci/build_scripts/deep_ep.sh", paths)
        self.assertIn("ci/patches/enable_deep_ep_sm80.py", paths)

    def test_cudnn_provisioning_only_for_requires_cudnn(self) -> None:
        paths = self._paths(
            {
                "transformer_engine": {
                    "builder": "transformer_engine",
                    "requires_cudnn": True,
                },
                "apex": {"builder": "apex"},
            }
        )
        cudnn = "ci/build_scripts/provision/install_cudnn.sh"
        self.assertIn(cudnn, paths["transformer_engine"])
        self.assertNotIn(cudnn, paths["apex"])

    def test_missing_build_input_fails_closed(self) -> None:
        versions = {"components": {"ghost": {"builder": "no_such_builder"}}}
        with self.assertRaises(SystemExit):
            generate_matrix.build_input_fingerprint(versions, "ghost", "x86_64")


class RebuildScopeTests(unittest.TestCase):
    """End-to-end scope of the rebuild decision over an isolated temp repo:

    for an edit to each kind of build input, assert exactly which targets'
    fingerprints change (=> rebuild) and which stay byte-identical (=> skip).
    Influential edits (toolchain, a sourced lib, a builder, a patch, cuDNN)
    must change only the targets that consume them; no-influence orchestration
    (the reusable workflow) must change none.
    """

    COMPONENTS = ("alpha", "beta", "gamma")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        # build_input_fingerprint resolves every path from VERSIONS_FILE's
        # parent, so repointing it redirects the whole scan at the temp repo.
        self._versions_patch = patch.object(
            generate_matrix, "VERSIONS_FILE", self.root / "versions.yaml"
        )
        self._versions_patch.start()
        self.addCleanup(self._versions_patch.stop)

        self._write(
            ".github/actions/build-toolchain/action.yml",
            "name: toolchain\nsteps: []\n",
        )
        self._write(".github/workflows/_build.yml", "# reusable orchestration\n")
        self._write("ci/build_scripts/lib/env.sh", "export_extra_env() { :; }\n")
        self._write("ci/build_scripts/lib/arch.sh", "arch_list_strip_dots() { :; }\n")
        self._write(
            "ci/build_scripts/lib/nccl.sh",
            '__lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'source "${__lib_dir}/env.sh"\n'
            "unset __lib_dir\n"
            "install_nccl() { :; }\n",
        )
        self._write(
            "ci/build_scripts/provision/install_cudnn.sh",
            '__p="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'source "${__p}/../lib/env.sh"\n'
            "install_cudnn() { :; }\n",
        )
        # alpha sources only env; beta sources nccl (which transitively pulls
        # env); gamma sources env, needs cuDNN and declares a patch.
        self._write(
            "ci/build_scripts/alpha.sh",
            'SCRIPT_DIR="x"\nsource "${SCRIPT_DIR}/lib/env.sh"\nalpha_build\n',
        )
        self._write(
            "ci/build_scripts/beta.sh",
            'SCRIPT_DIR="x"\nsource "${SCRIPT_DIR}/lib/nccl.sh"\nbeta_build\n',
        )
        self._write(
            "ci/build_scripts/gamma.sh",
            'SCRIPT_DIR="x"\nsource "${SCRIPT_DIR}/lib/env.sh"\ngamma_build\n',
        )
        self._write("ci/patches/gamma.py", "# gamma build-time patch\n")

        self.versions = {
            "components": {
                "alpha": {"builder": "alpha"},
                "beta": {"builder": "beta"},
                "gamma": {
                    "builder": "gamma",
                    "requires_cudnn": True,
                    "patches": ["ci/patches/gamma.py"],
                },
            }
        }

    def _write(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _append(self, rel: str, line: str) -> None:
        with (self.root / rel).open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _snapshot(self):
        return {
            name: {
                entry["path"]: entry["sha256"]
                for entry in generate_matrix.build_input_fingerprint(
                    self.versions, name, "x86_64"
                )
            }
            for name in self.COMPONENTS
        }

    def _changed(self, before, after):
        """{component: set of fingerprint paths whose sha/set changed}."""
        changed = {}
        for name in self.COMPONENTS:
            paths = set(before[name]) | set(after[name])
            changed[name] = {
                path for path in paths if before[name].get(path) != after[name].get(path)
            }
        return changed

    def test_baseline_closures(self) -> None:
        snap = self._snapshot()
        action = ".github/actions/build-toolchain/action.yml"
        for name in self.COMPONENTS:
            self.assertIn(action, snap[name])
            self.assertNotIn(".github/workflows/_build.yml", snap[name])
            self.assertIn("ci/build_scripts/lib/env.sh", snap[name])
        # nccl.sh reaches beta alone (via the lib -> lib transitive edge);
        # arch.sh is sourced by nobody and therefore nowhere; cuDNN only gamma.
        self.assertNotIn("ci/build_scripts/lib/nccl.sh", snap["alpha"])
        self.assertIn("ci/build_scripts/lib/nccl.sh", snap["beta"])
        self.assertNotIn("ci/build_scripts/lib/nccl.sh", snap["gamma"])
        for name in self.COMPONENTS:
            self.assertNotIn("ci/build_scripts/lib/arch.sh", snap[name])
        self.assertIn(
            "ci/build_scripts/provision/install_cudnn.sh", snap["gamma"]
        )
        self.assertNotIn(
            "ci/build_scripts/provision/install_cudnn.sh", snap["beta"]
        )
        self.assertIn("ci/patches/gamma.py", snap["gamma"])

    def test_fingerprint_is_stable_without_an_edit(self) -> None:
        before = self._snapshot()
        after = self._snapshot()
        self.assertEqual(self._changed(before, after), {n: set() for n in self.COMPONENTS})

    def test_workflow_orchestration_edit_has_no_influence(self) -> None:
        # The exact TE failure class: touching upload timeouts/cache/checkout in
        # the reusable workflow must not rebuild a single wheel.
        before = self._snapshot()
        self._append(".github/workflows/_build.yml", "# tweak upload timeout\n")
        changed = self._changed(before, self._snapshot())
        self.assertEqual(changed, {n: set() for n in self.COMPONENTS})

    def test_toolchain_edit_rebuilds_every_target(self) -> None:
        before = self._snapshot()
        self._append(".github/actions/build-toolchain/action.yml", "# bump cuda\n")
        changed = self._changed(before, self._snapshot())
        action = ".github/actions/build-toolchain/action.yml"
        for name in self.COMPONENTS:
            self.assertEqual(changed[name], {action}, name)

    def test_shared_lib_edit_rebuilds_only_targets_that_source_it(self) -> None:
        # env.sh is on every target's closure -> all rebuild.
        before = self._snapshot()
        self._append("ci/build_scripts/lib/env.sh", "env_changed=1\n")
        changed = self._changed(before, self._snapshot())
        env = "ci/build_scripts/lib/env.sh"
        for name in self.COMPONENTS:
            self.assertEqual(changed[name], {env}, name)

        # nccl.sh is on beta's closure alone -> only beta rebuilds.
        before = self._snapshot()
        self._append("ci/build_scripts/lib/nccl.sh", "nccl_changed=1\n")
        changed = self._changed(before, self._snapshot())
        nccl = "ci/build_scripts/lib/nccl.sh"
        self.assertEqual(changed["alpha"], set())
        self.assertEqual(changed["beta"], {nccl})
        self.assertEqual(changed["gamma"], set())

        # arch.sh is sourced by nobody -> editing it rebuilds nothing, even
        # though it lives under the fingerprinted ci/build_scripts tree.
        before = self._snapshot()
        self._append("ci/build_scripts/lib/arch.sh", "arch_changed=1\n")
        changed = self._changed(before, self._snapshot())
        self.assertEqual(changed, {n: set() for n in self.COMPONENTS})

    def test_builder_edit_rebuilds_only_that_target(self) -> None:
        before = self._snapshot()
        self._append("ci/build_scripts/alpha.sh", "alpha_tweak=1\n")
        changed = self._changed(before, self._snapshot())
        builder = "ci/build_scripts/alpha.sh"
        self.assertEqual(changed["alpha"], {builder})
        self.assertEqual(changed["beta"], set())
        self.assertEqual(changed["gamma"], set())

    def test_patch_edit_rebuilds_only_declaring_target(self) -> None:
        before = self._snapshot()
        self._append("ci/patches/gamma.py", "# patch tweak\n")
        changed = self._changed(before, self._snapshot())
        self.assertEqual(changed["alpha"], set())
        self.assertEqual(changed["beta"], set())
        self.assertEqual(changed["gamma"], {"ci/patches/gamma.py"})

    def test_cudnn_provisioning_edit_rebuilds_only_requires_cudnn_target(self) -> None:
        before = self._snapshot()
        self._append(
            "ci/build_scripts/provision/install_cudnn.sh", "cudnn_tweak=1\n"
        )
        changed = self._changed(before, self._snapshot())
        cudnn = "ci/build_scripts/provision/install_cudnn.sh"
        self.assertEqual(changed["alpha"], set())
        self.assertEqual(changed["beta"], set())
        self.assertEqual(changed["gamma"], {cudnn})

    def test_adding_a_source_line_expands_that_targets_closure_only(self) -> None:
        # The source closure is self-registering: a builder that newly pulls in
        # a helper immediately fingerprints it, with no per-component list to
        # update, and other targets are unaffected.
        before = self._snapshot()
        self._append("ci/build_scripts/alpha.sh", 'source "${SCRIPT_DIR}/lib/arch.sh"\n')
        changed = self._changed(before, self._snapshot())
        arch = "ci/build_scripts/lib/arch.sh"
        self.assertIn(arch, changed["alpha"])
        self.assertIn("ci/build_scripts/alpha.sh", changed["alpha"])
        self.assertEqual(changed["beta"], set())
        self.assertEqual(changed["gamma"], set())


def make_python_matrix_versions(component_cfg, arches=("x86_64", "aarch64"),
                                pythons=("3.11", "3.12")):
    """A 2 arch x 2 python build_matrix, ordered exactly like versions.yaml
    (x86_64 rows before aarch64, and 3.11 before 3.12 within each arch)."""
    return {
        "build_matrix": [
            {
                "arch": arch,
                "cuda": "13.0.2",
                "python": python,
                "torch": DEMO_TORCH,
                "torch_vision": "0.26.0",
                "torch_audio": "2.11.0",
                "cxx11_abi": "TRUE",
            }
            for arch in arches
            for python in pythons
        ],
        "components": {"demo": component_cfg},
    }


class PythonVersionFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "ref": "v1.2.3",
            "path": "demo",
            "builder": "demo",
            "wheel_packages": ["demo"],
            "torch_cuda_arch_list": "8.0;9.0;10.0",
            "max_jobs": 4,
            # A GitHub-hosted (not self-hosted) label so the 2-arch fixture
            # passes check_runner_policy for its aarch64 rows.
            "runs_on": ["ubuntu-24.04"],
        }
        self.versions = make_python_matrix_versions(dict(self.cfg))

    def test_matrix_python_versions_follow_first_seen_order(self) -> None:
        self.assertEqual(
            ["3.11", "3.12"],
            generate_matrix.matrix_python_versions(self.versions),
        )

    def test_component_without_python_versions_builds_every_python(self) -> None:
        self.assertEqual(
            ["3.11", "3.12"],
            generate_matrix.component_python_versions(self.versions, "demo"),
        )
        combos = generate_matrix.component_combos(self.versions, "demo")
        self.assertEqual(
            [
                ("x86_64", "3.11"),
                ("x86_64", "3.12"),
                ("aarch64", "3.11"),
                ("aarch64", "3.12"),
            ],
            [
                (generate_matrix.combo_arch(c), generate_matrix.combo_python(c))
                for c in combos
            ],
        )

    def test_python_versions_subset_like_py3_none_any_component(self) -> None:
        # megatron-bridge/flashinfer pin python_versions: ["3.12"] so the same
        # py3-none-any asset is not uploaded once per interpreter.
        self.versions["components"]["demo"]["python_versions"] = ["3.12"]
        self.assertEqual(
            ["3.12"],
            generate_matrix.component_python_versions(self.versions, "demo"),
        )
        combos = generate_matrix.component_combos(self.versions, "demo")
        self.assertEqual(
            [("x86_64", "3.12"), ("aarch64", "3.12")],
            [
                (generate_matrix.combo_arch(c), generate_matrix.combo_python(c))
                for c in combos
            ],
        )

    def test_python_filter_combines_with_arch_filter(self) -> None:
        cfg = dict(self.cfg, arches=["aarch64"], python_versions=["3.11"])
        self.versions["components"]["demo"] = cfg
        combos = generate_matrix.component_combos(self.versions, "demo")
        self.assertEqual(
            [("aarch64", "3.11")],
            [
                (generate_matrix.combo_arch(c), generate_matrix.combo_python(c))
                for c in combos
            ],
        )

    def test_unknown_python_version_is_rejected(self) -> None:
        self.versions["components"]["demo"]["python_versions"] = ["2.7"]
        with self.assertRaises(SystemExit) as ctx:
            generate_matrix.component_python_versions(self.versions, "demo")
        self.assertIn("not in build_matrix", str(ctx.exception))
        self.assertIn("2.7", str(ctx.exception))

    def _run_cli(self, argv):
        out = io.StringIO()
        with patch.object(sys, "argv", ["generate_matrix.py"] + argv), patch(
            "generate_matrix.load_versions", return_value=self.versions
        ), contextlib.redirect_stdout(out):
            generate_matrix.main()
        return json.loads(out.getvalue())

    def test_cli_python_filter_keeps_only_requested_interpreter(self) -> None:
        rows = self._run_cli(["--component", "demo", "--python", "3.11"])
        self.assertEqual(2, len(rows))
        self.assertTrue(all(row["python"] == "3.11" for row in rows))
        self.assertEqual({"x86_64", "aarch64"}, {row["arch"] for row in rows})

    def test_cli_python_all_default_builds_every_interpreter(self) -> None:
        rows = self._run_cli(["--component", "demo"])
        self.assertEqual(4, len(rows))
        # build_matrix order is preserved verbatim (arch outer, python inner).
        self.assertEqual(
            [
                ("x86_64", "3.11"),
                ("x86_64", "3.12"),
                ("aarch64", "3.11"),
                ("aarch64", "3.12"),
            ],
            [(r["arch"], r["python"]) for r in rows],
        )

    def test_cli_python_filter_can_leave_an_empty_matrix(self) -> None:
        # A 3.12-only component dispatched with --python 3.11 yields no rows;
        # has_builds lets the workflow skip rather than fail on an empty matrix.
        self.versions["components"]["demo"]["python_versions"] = ["3.12"]
        self.assertEqual([], self._run_cli(["--component", "demo", "--python", "3.11"]))

    def test_cli_unknown_python_exits_2(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._run_cli(["--component", "demo", "--python", "2.7"])
        self.assertEqual(2, ctx.exception.code)


def make_cuda_matrix_versions(component_cfg, arches=("x86_64", "aarch64"),
                              cudas=("13.2.2", "13.0.2")):
    """A 2 arch x 2 CUDA build_matrix, the current-cuda row first (so the
    /simple default world is 13.2), each pinned to one Python/torch pair."""
    return {
        "build_matrix": [
            {
                "arch": arch,
                "cuda": cuda,
                "python": "3.12",
                "torch": DEMO_TORCH,
                "torch_vision": "0.28.0",
                "torch_audio": "2.11.0",
                "cxx11_abi": "TRUE",
            }
            for arch in arches
            for cuda in cudas
        ],
        "components": {"demo": component_cfg},
    }


class CudaVersionFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "ref": "v1.2.3",
            "path": "demo",
            "builder": "demo",
            "wheel_packages": ["demo"],
            "cuda_versions": ["13.2.2"],
            "torch_cuda_arch_list": "9.0;10.0",
            "max_jobs": 4,
            "runs_on": ["ubuntu-24.04"],
        }
        self.versions = make_cuda_matrix_versions(dict(self.cfg))

    def test_matrix_cuda_versions_follow_first_seen_order(self) -> None:
        self.assertEqual(
            ["13.2.2", "13.0.2"],
            generate_matrix.matrix_cuda_versions(self.versions),
        )

    def test_component_without_cuda_versions_sees_every_cuda(self) -> None:
        del self.versions["components"]["demo"]["cuda_versions"]
        self.assertEqual(
            ["13.2.2", "13.0.2"],
            generate_matrix.component_cuda_versions(self.versions, "demo"),
        )
        combos = generate_matrix.component_combos(self.versions, "demo")
        self.assertEqual(
            {("x86_64", "13.2.2"), ("x86_64", "13.0.2"),
             ("aarch64", "13.2.2"), ("aarch64", "13.0.2")},
            {
                (generate_matrix.combo_arch(c), generate_matrix.combo_cuda(c))
                for c in combos
            },
        )

    def test_cuda_versions_subset_pins_one_world(self) -> None:
        self.assertEqual(
            ["13.2.2"],
            generate_matrix.component_cuda_versions(self.versions, "demo"),
        )
        combos = generate_matrix.component_combos(self.versions, "demo")
        self.assertEqual(
            [("x86_64", "13.2.2"), ("aarch64", "13.2.2")],
            [
                (generate_matrix.combo_arch(c), generate_matrix.combo_cuda(c))
                for c in combos
            ],
        )

    def test_component_held_on_older_cuda(self) -> None:
        # flashinfer today: opt out of the current toolkit and stay on 13.0.
        self.versions["components"]["demo"]["cuda_versions"] = ["13.0.2"]
        combos = generate_matrix.component_combos(self.versions, "demo")
        self.assertTrue(
            all(generate_matrix.combo_cuda(c) == "13.0.2" for c in combos)
        )
        title = generate_matrix.release_title(
            "v1.2.3",
            "demo",
            generate_matrix.component_combos_for_release(
                self.versions, "demo", "3.12", DEMO_TORCH, DEMO_CUDA
            ),
        )
        self.assertEqual(["cu130/torch2.11"], build_index.release_worlds(title))

    def test_cuda_filter_combines_with_arch_filter(self) -> None:
        cfg = dict(self.cfg, arches=["aarch64"], cuda_versions=["13.0.2"])
        self.versions["components"]["demo"] = cfg
        combos = generate_matrix.component_combos(self.versions, "demo")
        self.assertEqual(
            [("aarch64", "13.0.2")],
            [
                (generate_matrix.combo_arch(c), generate_matrix.combo_cuda(c))
                for c in combos
            ],
        )

    def test_unknown_cuda_version_is_rejected(self) -> None:
        self.versions["components"]["demo"]["cuda_versions"] = ["13.9.9"]
        with self.assertRaises(SystemExit) as ctx:
            generate_matrix.component_cuda_versions(self.versions, "demo")
        self.assertIn("not in build_matrix", str(ctx.exception))
        self.assertIn("13.9.9", str(ctx.exception))

    def test_two_cudas_for_one_python_torch_get_distinct_releases(self) -> None:
        # With CUDA keyed into the release tag, a component may build one
        # (Python, torch) against both 13.0 and 13.2: each lands on its own
        # tag, so the second build can never clobber the first's wheels.
        self.versions["components"]["demo"]["cuda_versions"] = [
            "13.2.2",
            "13.0.2",
        ]
        keys = [
            (cuda, generate_matrix.release_tag("demo", "v1.2.3", py, torch, cuda))
            for py, torch, cuda in generate_matrix.component_release_keys(
                self.versions, "demo"
            )
            if py == "3.12" and torch == DEMO_TORCH
        ]
        self.assertEqual(
            [
                ("13.2.2", f"demo-v1.2.3-py3.12-torch{DEMO_TORCH}-cu132"),
                ("13.0.2", f"demo-v1.2.3-py3.12-torch{DEMO_TORCH}-cu130"),
            ],
            keys,
        )
        # Each cuda-keyed release groups only that CUDA's arch rows.
        self.assertEqual(
            2,
            len(
                generate_matrix.component_combos_for_release(
                    self.versions, "demo", "3.12", DEMO_TORCH, "13.2.2"
                )
            ),
        )
        self.assertEqual(
            2,
            len(
                generate_matrix.component_combos_for_release(
                    self.versions, "demo", "3.12", DEMO_TORCH, "13.0.2"
                )
            ),
        )
        # release_meta must be told which cuda when more than one exists.
        with patch("generate_matrix.build_input_fingerprint", return_value=STUB_FINGERPRINT):
            with self.assertRaises(SystemExit):
                release_meta.component_release_meta(
                    self.versions, "demo", "3.12", DEMO_TORCH
                )
            meta = release_meta.component_release_meta(
                self.versions, "demo", "3.12", DEMO_TORCH, "13.2.2"
            )
        self.assertTrue(meta["tag"].endswith("-cu132"))


class PerPythonReleaseTests(unittest.TestCase):
    """One GitHub Release per (component, Python version, torch version):
    each interpreter's rows get their own tag, title and manifest, and a
    torch bump lands on a new release instead of overwriting the old one."""

    PY312_WHEELS = [
        ("demo-1.2.3-cp312-cp312-manylinux_2_28_x86_64.whl", 1000),
        ("demo-1.2.3-cp312-cp312-manylinux_2_28_aarch64.whl", 1000),
    ]
    PY311_WHEELS = [
        ("demo-1.2.3-cp311-cp311-manylinux_2_28_x86_64.whl", 1000),
        ("demo-1.2.3-cp311-cp311-manylinux_2_28_aarch64.whl", 1000),
    ]

    def setUp(self) -> None:
        self._fp_patch = patch(
            "generate_matrix.build_input_fingerprint", return_value=STUB_FINGERPRINT
        )
        self._cuobjdump_patch = patch("cuda_archs.find_cuobjdump", return_value=None)
        self._fp_patch.start()
        self._cuobjdump_patch.start()
        self.addCleanup(self._fp_patch.stop)
        self.addCleanup(self._cuobjdump_patch.stop)

        self.cfg = {
            "ref": "v1.2.3",
            "path": "demo",
            "builder": "demo",
            "wheel_packages": ["demo"],
            "torch_cuda_arch_list": "8.0;9.0;10.0",
            "max_jobs": 4,
            "runs_on": ["ubuntu-24.04"],
        }
        self.versions = make_python_matrix_versions(dict(self.cfg))
        self.entries = generate_matrix.build_full_matrix(self.versions, ["demo"])

    def test_tag_spells_out_python_and_torch(self) -> None:
        tags = {(r["arch"], r["python"]): r["release_tag"] for r in self.entries}
        self.assertEqual(f"demo-v1.2.3-py3.12-torch{DEMO_TORCH}-cu130", tags[("x86_64", "3.12")])
        self.assertEqual(f"demo-v1.2.3-py3.12-torch{DEMO_TORCH}-cu130", tags[("aarch64", "3.12")])
        self.assertEqual(f"demo-v1.2.3-py3.11-torch{DEMO_TORCH}-cu130", tags[("x86_64", "3.11")])
        self.assertEqual(f"demo-v1.2.3-py3.11-torch{DEMO_TORCH}-cu130", tags[("aarch64", "3.11")])
        # Both arches of one interpreter share a release; the wheel filename's
        # platform tag is what tells them apart inside it.
        self.assertEqual(2, len(set(tags.values())))

    def test_two_torch_versions_never_share_a_release(self) -> None:
        """A matrix building one interpreter against two torch versions must
        produce two releases - otherwise the second build's wheels overwrite
        the first's, since neither version is in the wheel filename."""
        versions = make_python_matrix_versions(dict(self.cfg), pythons=("3.12",))
        newer = dict(versions["build_matrix"][0], torch="2.13.0")
        versions["build_matrix"].append(newer)

        self.assertEqual(
            [("3.12", DEMO_TORCH, "13.0.2"), ("3.12", "2.13.0", "13.0.2")],
            generate_matrix.component_release_keys(versions, "demo"),
        )
        tags = {r["release_tag"] for r in generate_matrix.build_full_matrix(versions, ["demo"])}
        self.assertEqual(
            {
                f"demo-v1.2.3-py3.12-torch{DEMO_TORCH}-cu130",
                "demo-v1.2.3-py3.12-torch2.13.0-cu130",
            },
            tags,
        )

    def test_release_title_lists_only_that_python_segments(self) -> None:
        title312 = generate_matrix.release_title(
            "v1.2.3",
            "demo",
            generate_matrix.component_combos_for_release(
                self.versions, "demo", "3.12", DEMO_TORCH, DEMO_CUDA
            ),
        )
        self.assertEqual(
            "demo v1.2.3 - cu13.0.2 py3.12 torch2.11.0; "
            "aarch64 cu13.0.2 py3.12 torch2.11.0",
            title312,
        )
        title311 = generate_matrix.release_title(
            "v1.2.3",
            "demo",
            generate_matrix.component_combos_for_release(
                self.versions, "demo", "3.11", DEMO_TORCH, DEMO_CUDA
            ),
        )
        self.assertEqual(
            "demo v1.2.3 - cu13.0.2 py3.11 torch2.11.0; "
            "aarch64 cu13.0.2 py3.11 torch2.11.0",
            title311,
        )

    def test_manifest_is_scoped_to_one_python(self) -> None:
        for python in ("3.11", "3.12"):
            manifest = generate_matrix.component_build_manifest(
                self.versions, "demo", python, DEMO_TORCH, DEMO_CUDA
            )
            self.assertEqual(python, manifest["python"])
            self.assertEqual(2, len(manifest["builds"]))
            self.assertEqual({python}, {build["python"] for build in manifest["builds"]})

    @patch("generate_matrix.inspect_release")
    def test_existing_py312_release_skips_while_new_py311_release_builds(
        self, inspect_release
    ) -> None:
        release312 = make_release(self.versions, self.PY312_WHEELS, python="3.12")
        tag312 = f"demo-v1.2.3-py3.12-torch{DEMO_TORCH}-cu130"
        seen_tags = []

        def fake_inspect(_repo, tag):
            seen_tags.append(tag)
            return release312 if tag == tag312 else None

        inspect_release.side_effect = fake_inspect
        decisions = generate_matrix.evaluate_matrix_rows(
            self.versions, self.entries, "owner/repo", verify_wheels=False
        )
        by = {(d["entry"]["arch"], d["entry"]["python"]): d for d in decisions}
        for arch in ("x86_64", "aarch64"):
            self.assertEqual("skip", by[(arch, "3.12")]["decision"])
            self.assertIn("exact title", by[(arch, "3.12")]["reason"])
            self.assertEqual("build", by[(arch, "3.11")]["decision"])
            self.assertIn("would be created", by[(arch, "3.11")]["reason"])
        # Each per-python release is inspected once, shared by both arch rows.
        self.assertEqual(2, inspect_release.call_count)
        self.assertEqual(
            [f"demo-v1.2.3-py3.11-torch{DEMO_TORCH}-cu130", tag312],
            sorted(seen_tags),
        )

    def test_one_python_release_never_covers_the_other_python_row(self) -> None:
        release311 = make_release(self.versions, self.PY311_WHEELS, python="3.11")
        combo312 = generate_matrix.component_combos_for_release(
            self.versions, "demo", "3.12", DEMO_TORCH, DEMO_CUDA
        )[0]
        covered, reason = generate_matrix.release_covers_combo(
            self.versions, "demo", combo312, release311, repo=None, verify_wheels=False
        )
        self.assertFalse(covered)
        self.assertIn("title mismatch", reason)

    def test_release_meta_explicit_python(self) -> None:
        meta = release_meta.component_release_meta(self.versions, "demo", "3.11")
        self.assertEqual(f"demo-v1.2.3-py3.11-torch{DEMO_TORCH}-cu130", meta["tag"])
        self.assertEqual("3.11", meta["python"])
        self.assertEqual(DEMO_TORCH, meta["torch"])
        self.assertEqual(DEMO_CUDA, meta["cuda"])
        self.assertIn("py3.11", meta["title"])
        self.assertEqual(2, len(meta["manifest"]["builds"]))

    def test_release_meta_infers_torch_from_the_matrix(self) -> None:
        """_build.yml passes --torch, but a caller that omits it still gets the
        right release as long as the interpreter pins exactly one torch."""
        self.assertEqual(
            DEMO_TORCH,
            release_meta.resolve_torch(self.versions, "demo", "3.12", None),
        )
        self.versions["build_matrix"].append(
            dict(self.versions["build_matrix"][0], python="3.12", torch="2.13.0")
        )
        with self.assertRaises(SystemExit):
            release_meta.resolve_torch(self.versions, "demo", "3.12", None)
        self.assertEqual(
            "2.13.0",
            release_meta.resolve_torch(self.versions, "demo", "3.12", "2.13.0"),
        )

    def test_release_meta_all_emits_one_record_per_opted_in_python(self) -> None:
        entries = release_meta.release_meta_entries(self.versions, ["demo"], "all")
        self.assertEqual(
            [
                f"demo-v1.2.3-py3.11-torch{DEMO_TORCH}-cu130",
                f"demo-v1.2.3-py3.12-torch{DEMO_TORCH}-cu130",
            ],
            [entry["tag"] for entry in entries],
        )

    def test_release_meta_skips_components_not_opting_into_python(self) -> None:
        # A 3.12-only py3-none-any component (megatron-bridge/flashinfer style)
        # must not produce an empty 3.11 release when metadata is requested
        # for every component.
        self.versions["components"]["bridge"] = dict(
            self.cfg, ref="v9.9.9", python_versions=["3.12"]
        )
        entries311 = release_meta.release_meta_entries(
            self.versions, ["demo", "bridge"], "3.11"
        )
        self.assertEqual(
            [f"demo-v1.2.3-py3.11-torch{DEMO_TORCH}-cu130"],
            [entry["tag"] for entry in entries311],
        )
        entries_all = release_meta.release_meta_entries(
            self.versions, ["demo", "bridge"], "all"
        )
        self.assertEqual(
            [
                f"demo-v1.2.3-py3.11-torch{DEMO_TORCH}-cu130",
                f"demo-v1.2.3-py3.12-torch{DEMO_TORCH}-cu130",
                f"bridge-v9.9.9-py3.12-torch{DEMO_TORCH}-cu130",
            ],
            [entry["tag"] for entry in entries_all],
        )

    def test_release_meta_rejects_python_component_does_not_build(self) -> None:
        self.versions["components"]["demo"]["python_versions"] = ["3.12"]
        with self.assertRaises(SystemExit):
            release_meta.component_release_meta(self.versions, "demo", "3.11")


if __name__ == "__main__":
    unittest.main()
