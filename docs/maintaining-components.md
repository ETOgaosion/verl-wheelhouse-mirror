# Maintaining verl-wheelhouse components

This guide covers the two most common maintenance tasks for this repo:
bumping an existing component's pinned version, and adding a brand-new
component to the wheelhouse. Both are driven almost entirely by
`versions.yaml` (repo root) plus the small set of files listed below - no
other logic needs to change for routine work.

## Files involved

| File | Role |
|---|---|
| `versions.yaml` | The version map: CUDA/Python/Torch build matrix + per-component config |
| `ci/generate_matrix.py` | Expands `versions.yaml` into the GitHub Actions matrix (no edits needed for routine work) |
| `ci/release_meta.py` | Computes each component's release tag/title/notes (no edits needed for routine work) |
| `ci/build_scripts/common.sh` | Shared bash helpers (no edits needed unless adding new shared logic) |
| `ci/build_scripts/<builder>.sh` | The actual wheel-build command for one component |
| `.github/workflows/build-<component>.yml` | Per-component trigger workflow |
| `.github/workflows/_build.yml` | Reusable build workflow (no edits needed) |
| `.github/workflows/_ensure_release.yml` | Reusable create/update-release workflow (no edits needed) |
| `.github/workflows/build-all.yml`, `release.yml` | Already build every component via `--component all` (no edits needed) |

## Upgrading an existing component's version

1. Open `versions.yaml` and find the component under `components:`.
2. Update its `ref:` to the new git tag/branch/commit you want to pin.
3. Check the upstream project's release notes for anything else that changed:
   - New/removed supported GPU architectures → update `torch_cuda_arch_list`.
   - New build-time environment variables or flags → update
     `ci/build_scripts/<builder>.sh` and/or `extra_env` in `versions.yaml`.
   - A cuDNN version bump or new system package requirement → update
     `requires_cudnn` and/or `ci/build_scripts/common.sh`'s `install_cudnn`.
4. Regenerate and sanity-check the matrix locally:

   ```bash
   pip install pyyaml
   python3 ci/generate_matrix.py --component <name> | python3 -m json.tool
   ```

5. Commit the `versions.yaml` change. Pushing to `main` automatically
   triggers that component's `build-<component>.yml` (its `paths:` filter
   matches `versions.yaml`). If the target release already has the exact
   configured CUDA/Python/Torch title and all expected `wheel_packages`, the
   push skips the redundant build. Otherwise, on success, that push creates
   (or reuses) the component's own persistent release - tag
   `<component>-<new-ref>`, title `<component> <new-ref> - cu.. py..
   torch..` (see `ci/release_meta.py`) - uploads the new wheel there, and
   republishes the package index, so it's `pip install`-able right away.
   Bumping `ref` therefore starts a brand-new release; the previous ref's
   release is left untouched as history. You can also trigger the workflow
   manually from the Actions tab (`workflow_dispatch`) to test before
   merging (manual runs build but skip publishing).
6. Once you're happy, push a tag matching `v*` (`git tag vX.Y.Z && git push
   --tags`) to run `release.yml`: a full sweep that rebuilds every
   component and re-ensures/uploads to each one's own release, same as
   step 5 but across the whole matrix at once. The pushed tag is only a
   trigger - it does not itself become a release.

`apex` is the one exception - both verl Dockerfiles this repo mirrors build
it unpinned from `main`, so there's no fixed version to bump; every build of
`apex` simply picks up whatever `main` currently contains.

## Adding a brand-new component

Worked checklist, using a hypothetical `xformers` component as the example:

1. **Add the submodule:**

   ```bash
   git submodule add https://github.com/facebookresearch/xformers.git xformers
   ```

2. **Add a `components:` entry to `versions.yaml`**, following the schema
   documented in that file's own comments:

   ```yaml
   xformers:
     path: xformers
     ref: v0.0.29
     builder: xformers
     wheel_packages: [xformers]
     torch_cuda_arch_list: "8.0;9.0;10.0;12.0"
     requires_cudnn: false
     max_jobs: 2
     runs_on: ubuntu-24.04
     extra_env: {}
   ```

3. **Write `ci/build_scripts/xformers.sh`.** Use an existing script as a
   template (`ci/build_scripts/apex.sh` is a good default shape); every
   script follows the same pattern:

   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   source "${SCRIPT_DIR}/common.sh"

   export_extra_env
   # ... install prerequisite pip packages, export the project's own
   # documented build-time env vars (mirror its own CI/Dockerfile exactly) ...
   python setup.py bdist_wheel --dist-dir=dist   # or: pip wheel -w dist --no-deps .

   echo "Built wheels:"
   ls -al dist
   ```

   Then `chmod +x ci/build_scripts/xformers.sh`. The script always runs with
   its CWD already set to the component's checkout (see `_build.yml`'s
   `working-directory`), and must leave the final wheel(s) in `dist/`
   relative to that CWD - the reusable workflow uploads `<path>/dist/*.whl`.

4. **Copy a per-component workflow.** Duplicate
   `.github/workflows/build-flashinfer.yml` to
   `.github/workflows/build-xformers.yml` and adjust:
   - `name:` → `Build xformers`
   - the `paths:` entries → `versions.yaml`, `ci/build_scripts/common.sh`,
     `ci/build_scripts/xformers.sh`, `.github/workflows/_build.yml`,
     `.github/workflows/build-xformers.yml`
   - the `--component flashinfer` argument in the `compute-matrix` job →
     `--component xformers`
   - the `component: flashinfer` input under the `ensure-release` job → `component: xformers`

   Everything else - `workflow_dispatch`, the reusable `_ensure_release.yml`
   call's structure, and the trailing `publish-index` job that runs after a
   successful push build - is component-agnostic and can be copied as-is.
   `ci/release_meta.py` will automatically compute `xformers`'s own release
   tag/title once its `components:` entry exists in `versions.yaml`.

   If your new component's `runs_on` is **self-hosted**, copy
   `.github/workflows/build-apex.yml` instead: it additionally forwards a
   `secrets: { BYTED_PROXY: ... }` block to `_build.yml`, which routes
   otherwise-slow GitHub uploads through the optional `BYTED_PROXY` repo
   secret. **Drop that block for a normal `ubuntu-*` component** -
   `_build.yml`'s `runner.environment == 'self-hosted'` proxy gate is a no-op
   on GitHub-hosted runners anyway.

5. **Update `README.md`'s component table and repo-layout listing** to
   mention the new component/workflow.

6. **Validate before pushing:**

   ```bash
   pip install pyyaml
   python3 ci/generate_matrix.py --list-components   # should list the new name
   python3 ci/generate_matrix.py --component xformers | python3 -m json.tool
   bash -n ci/build_scripts/xformers.sh
   ```

7. `build-all.yml` and `release.yml` both use `--component all`, so the new
   component is automatically included in the weekly sanity sweep and in
   every future release - no changes needed there.

## Arch-list conventions

Different build systems want `torch_cuda_arch_list` in different string
shapes; the field in `versions.yaml` is treated as canonical and converted
where needed:

| Consumer | Format | Handled by |
|---|---|---|
| apex | dotted + semicolons, e.g. `8.0;9.0;12.0` | used as-is |
| flash-attention, TransformerEngine | undotted, e.g. `80;90;120` | `ci/build_scripts/common.sh`'s `arch_list_strip_dots` |
| flashinfer | space-separated with PTX-family suffixes, e.g. `8.0 9.0a 12.0f` | given verbatim in `versions.yaml` (suffixes can't be derived mechanically) |
| Megatron-Bridge (pure-Python) | n/a | set `torch_cuda_arch_list: null` for components that build no CUDA code, or that hardcode their own gencode flags |

## See also

- [`README.md`](../README.md) for the full pipeline architecture and how to
  trigger builds / install wheels.
- `versions.yaml`'s inline comments for the authoritative field-by-field
  schema of `build_matrix` and `components`.
