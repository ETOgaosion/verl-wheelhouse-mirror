---
name: manage-wheelhouse-components
description: >-
  Upgrade a pinned component version or add a brand-new CUDA wheel component
  to the verl-wheelhouse build pipeline. Covers editing versions.yaml,
  writing ci/build_scripts/<builder>.sh, and creating/copying a
  .github/workflows/build-<component>.yml trigger workflow. Use when asked
  to bump/update/pin a submodule version, add a new wheel/component to the
  wheelhouse, or edit versions.yaml in this repo.
---

# Managing verl-wheelhouse components

Everything is driven by `versions.yaml` (repo root, plain YAML data). For
the full step-by-step guide with a worked example, see
[docs/maintaining-components.md](../../../docs/maintaining-components.md).
This file has the quick-reference version.

## Upgrading an existing component's version

1. Edit the component's `ref:` under `components:` in `versions.yaml`.
2. If upstream changed supported GPU architectures, update
   `torch_cuda_arch_list` too (see arch-list conventions below).
3. Validate: `pip install pyyaml && python3 ci/generate_matrix.py --component <name>`.
4. Commit. Pushing to `main` auto-triggers `build-<component>.yml` (its
   `paths:` filter matches `versions.yaml`), and a successful build from a
   push also creates that component's new persistent release - tag
   `<component>-<new-ref>`, title `<component> <new-ref> - cu.. py..
   torch..` (see `ci/release_meta.py`) - uploads the wheel there, and
   republishes the index - no tag push needed to make it `pip
   install`-able. The previous ref's release is left untouched as history.

`apex` tracks `main` unpinned (matches verl's own Dockerfiles) - there is no
version to bump for it.

## Adding a brand-new component

Checklist:

- [ ] `git submodule add <url> <path>`
- [ ] Add a `components:` entry to `versions.yaml` (`path`, `ref`,
      `builder`, `wheel_packages`, `torch_cuda_arch_list`, `requires_cudnn`,
      `max_jobs`, `runs_on`, `extra_env`) - follow the schema comments already
      in that file. `wheel_packages` must list every distribution the build
      uploads; push builds use it to detect a complete matching release.
- [ ] Create `ci/build_scripts/<builder>.sh`. Copy the shape of an existing
      script (`ci/build_scripts/apex.sh` is a good default): shebang,
      `set -euo pipefail`, source `common.sh`, call `export_extra_env`,
      install prerequisite pip packages, run the project's own documented
      wheel-build command (mirror its own CI/Dockerfile exactly), leave the
      wheel(s) in `dist/` relative to CWD. Then `chmod +x` it.
- [ ] Copy an existing `.github/workflows/build-<component>.yml` to
      `build-<new-component>.yml`; update its `name:`, `paths:` filter
      entries, the `--component <name>` argument, and the `component:` input
      passed to `_ensure_release.yml` in the `ensure-release` job. Leave the
      reusable workflow calls' structure and the `publish-index` job
      untouched - they're otherwise component-agnostic. Start from
      `build-flashinfer.yml` for a GitHub-hosted component, or
      `build-apex.yml` for a self-hosted one (it also forwards the
      `BYTED_PROXY` secret).
- [ ] Update `README.md`'s component table and repo-layout listing.
- [ ] Validate:
      `pip install pyyaml && python3 ci/generate_matrix.py --component <name>`
      and `bash -n ci/build_scripts/<builder>.sh`.
- [ ] No changes needed in `build-all.yml` or `release.yml` - both already
      use `--component all` and pick up new components automatically.

## Arch-list conventions

`torch_cuda_arch_list` in `versions.yaml` is canonical dotted+semicolon form
(e.g. `8.0;9.0;12.0`); each build script converts it as needed:

- apex: used as-is.
- flash-attention, TransformerEngine: undotted via `common.sh`'s
  `arch_list_strip_dots` (e.g. `80;90;120`).
- flashinfer: given verbatim with PTX-family suffixes (e.g.
  `8.0 9.0a 12.0f`) since those can't be derived mechanically.
- Megatron-Bridge: set to `null` - the same goes for any component that
  builds no CUDA code or hardcodes its own gencode flags.

## Release naming

Each component publishes to its **own** persistent GitHub Release (no
single combined release for the whole repo): tag `<component>-<ref>`,
title `<component> <ref> - cu<cuda> py<python> torch<torch>[; ...]` (one
segment per `versions.yaml` `build_matrix` entry). This is computed by
`ci/release_meta.py` and created/refreshed by the reusable
`.github/workflows/_ensure_release.yml` workflow - don't hand-roll
`gh release create`/`edit` calls elsewhere. Bumping a component's `ref`
starts a brand-new release under a new tag; it never renames or reuses the
previous ref's release.

## Key invariant

Every build script must leave the final `.whl` file(s) in `dist/` relative
to its own CWD (the component's checkout path) - `_build.yml` uploads
`<path>/dist/*.whl` as both a workflow artifact and (on release) a GitHub
Release asset.
