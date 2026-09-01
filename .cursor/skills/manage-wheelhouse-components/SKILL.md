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
      `max_jobs`, `runs_on`, `arches`, `extra_env`) - follow the schema
      comments already in that file. `wheel_packages` must list every
      distribution the build uploads; push builds use it to detect a complete
      matching release. Start with `arches: [x86_64]` and add arm64 later
      (see below).
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

## CPU arches (x86_64 / aarch64)

`build_matrix` entries carry an `arch` field (`uname -m` spelling). A
component builds every arch in the matrix unless it narrows that with
`arches: [...]`, and can replace any field per arch via `arch_overrides`:

```yaml
arch_overrides:
  aarch64:
    runs_on: ubuntu-24.04-arm # GitHub-hosted arm64; self-hosted is rejected
    torch_cuda_arch_list: "9.0;10.0" # arm64 CUDA hosts are GH200/GB200-class only
```

To enable arm64 for a component: drop its `arches: [x86_64]`, add the
`arch_overrides.aarch64` block above, and - if it emits any `py3-none-any`
wheel - make the build script skip that wheel when `$TARGET_ARCH` is not
`x86_64`, or both arches will upload the same asset name. `_build.yml`
asserts the runner's `uname -m` matches the row's arch. No other change is
needed; CUDA, cuDNN, NCCL and torch installation are already arch-aware.

Non-x86_64 rows must stay on GitHub-hosted runners: the only self-hosted
machine here is x86_64, so a self-hosted arm64 row would queue forever.
`ci/generate_matrix.py` fails the matrix rather than emitting one. Expect
much lower `max_jobs` on the arm row (4 vCPU) and several resumed attempts
for the heavy builds.

`flash-attention`, `apex`, `transformer-engine`, `flashinfer`, `deep-ep`,
`flash-mla` and `fast-hadamard-transform` build both arches;
`megatron-bridge` is x86_64-only because its `py3-none-any` wheel already
covers arm64.

Each arch is a separate job. To build just one, dispatch
`build-<component>.yml` with its `arch` input (or pass
`--arch <arch>` to `ci/generate_matrix.py` locally); pushes always build
every arch. When adding a new arch to `build_matrix`, extend the static
`options:` list of every workflow's `arch` dispatch input to match.

## Arch-list conventions

`torch_cuda_arch_list` in `versions.yaml` is canonical dotted+semicolon form
(e.g. `8.0;9.0;12.0`); each build script converts it as needed:

- apex: used as-is.
- flash-attention, TransformerEngine: undotted via `common.sh`'s
  `arch_list_strip_dots` (e.g. `80;90;120`).
- flashinfer: given verbatim with PTX-family suffixes (e.g.
  `8.0 9.0a 12.0f`) since those can't be derived mechanically.
- Megatron-Bridge, flash-mla, fast-hadamard-transform: set to `null` - the
  same goes for any component that builds no CUDA code, or that derives /
  hardcodes its own gencode flags instead of reading the variable.

## Release naming

Each component publishes to its **own** persistent GitHub Release (no
single combined release for the whole repo): tag `<component>-<ref>`,
title `<component> <ref> - [<arch> ]cu<cuda> py<python> torch<torch>[; ...]`
(one segment per `versions.yaml` `build_matrix` entry the component is built
for, with the `x86_64` arch left implicit). This is computed by
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
