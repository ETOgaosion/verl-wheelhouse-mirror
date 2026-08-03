<!--
Thanks for contributing to verl-wheelhouse. Keep the section that matches your
change and delete the rest, along with any checklist item that doesn't apply.
See docs/maintaining-components.md for the full maintenance guide.
-->

## Type of change

- [ ] New image / new build combination (a `build_matrix` entry, or a new component)
- [ ] Component version bump (`ref` change in `versions.yaml`)
- [ ] Feature / enhancement to the pipeline
- [ ] Bug fix
- [ ] Docs only

## Summary

<!-- What changes and why, in a couple of sentences. -->

Closes #

## Dependency changes

<!-- Delete if none. Old -> new for every version this PR moves. -->

| What | Before | After |
|---|---|---|
| | | |

## Validation

<!-- Delete lines that don't apply; paste output or link the run. -->

- [ ] `python3 ci/generate_matrix.py --component <name> | python3 -m json.tool` produces the expected matrix
- [ ] `bash -n ci/build_scripts/<builder>.sh` passes
- [ ] Green `workflow_dispatch` run: <!-- link -->
- [ ] Built wheel installs and imports on the target GPU
- [ ] Release tag/title and `wheel_packages` assets are as expected
- [ ] Index refreshed and the wheel is `pip install`-able

## New image / new component checklist

<!-- Delete this whole section unless you're adding an image, a build_matrix
     entry, or a component. -->

- [ ] Submodule added (`git submodule add <url> <path>`)
- [ ] `components:` entry in `versions.yaml` complete (`path`, `ref`, `builder`,
      `wheel_packages`, `torch_cuda_arch_list`, `requires_cudnn`, `max_jobs`,
      `runs_on`, `extra_env`)
- [ ] `ci/build_scripts/<builder>.sh` added, executable, and leaves wheels in `dist/`
- [ ] `.github/workflows/build-<component>.yml` added (copied from
      `build-flashinfer.yml` for GitHub-hosted, `build-apex.yml` for self-hosted)
- [ ] `README.md` component table and repo layout updated
- [ ] No edits needed to `build-all.yml` / `release.yml` (both use `--component all`)

## Risk and rollback

<!-- Build time or runner impact, whether existing releases/wheels are affected,
     and how to revert if this breaks the nightly sweep. -->
