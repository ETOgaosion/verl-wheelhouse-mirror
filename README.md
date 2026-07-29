# verl-wheelhouse

A GitHub Actions-based "wheelhouse" that builds prebuilt CUDA wheels for the
heavy native-extension dependencies of [verl](https://github.com/verl-project/verl):

| Component | Submodule path | What gets built |
|---|---|---|
| [apex](https://github.com/NVIDIA/apex) | `apex` | full apex (`--cpp_ext --cuda_ext`) |
| [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | `TransformerEngine` | PyTorch TE extension |
| [flash-attention](https://github.com/Dao-AILab/flash-attention) | `flash-attention` | `flash_attn` |
| [flashinfer](https://github.com/flashinfer-ai/flashinfer) | `flashinfer` | `flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache` |
| [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | `Megatron-Bridge` | `megatron-bridge` (pure-Python `py3-none-any` wheel) |

The rollout engines (`vllm`, `sglang`) are deliberately *not* built here -
upstream's own published wheels are used instead.

Build commands follow the exact flags used in verl's own
[`docker/Dockerfile.stable.sglang`](https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.sglang)
and
[`docker/Dockerfile.stable.vllm`](https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.vllm),
and the GitHub Actions structure mirrors flash-attention's own
[`_build.yml`](https://github.com/Dao-AILab/flash-attention/blob/main/.github/workflows/_build.yml)
reusable workflow, including its resource-saving tricks for standard
GitHub-hosted runners (disk cleanup, swap space, capped parallelism,
resumable build caches).

Wheels are published to [GitHub Releases](../../releases) - one persistent
release per component, named after that component and its pinned dependency
versions (e.g. `transformer-engine v2.16.1 - cu13.0.2 py3.12 torch2.11.0`, tag
`transformer-engine-v2.16.1`) - and to a static [GitHub Pages](https://pages.github.com/)
PEP 503 "simple" package index, so they're directly `pip install`-able.

## Repo layout

```
versions.yaml             # THE editable/extendable version map (see below), kept in the project's base dir
ci/
  generate_matrix.py      # expands versions.yaml into a GH Actions matrix
  release_meta.py          # computes each component's release tag/title/notes
  build_index.py          # builds the static PEP 503 index from release assets
  build_scripts/
    common.sh              # shared bash helpers, sourced by every script below
    apex.sh
    transformer_engine.sh
    flash_attention.sh
    flashinfer.sh
    megatron_bridge.sh
.github/workflows/
  _build.yml              # reusable single-combination build workflow
  _ensure_release.yml     # reusable: create/update a component's release
  build-apex.yml
  build-transformer-engine.yml
  build-flash-attention.yml
  build-flashinfer.yml
  build-megatron-bridge.yml
  build-all.yml           # builds every component x every matrix combo
  release.yml             # on `v*` tag push: full-matrix build + upload
  publish-index.yml       # (re)publishes the GitHub Pages PEP 503 index
docs/
  maintaining-components.md  # step-by-step: upgrade a version / add a component
.cursor/skills/
  manage-wheelhouse-components/SKILL.md  # agent skill for the same tasks
```

## The version map (`versions.yaml`)

Everything version-related lives in one editable, plain-data YAML file at
the project's base directory - no Python, no workflow YAML - needs to
change for routine version bumps:

- **`build_matrix`**: the CUDA / Python / Torch combinations to build. Seeded
  with CUDA `13.0.2`, Python `3.12`, Torch `2.11.0` (matching both verl
  Dockerfiles' `ARG` defaults). Add another entry to build more combinations
  - every component is built once per entry here.
- **`components`**: per-submodule config - the git `ref` to build (branch,
  tag, or commit; overrides whatever commit the submodule pointer in this
  repo is on), which `ci/build_scripts/<builder>.sh` to run, the CUDA arch
  list, whether cuDNN is required, `max_jobs`, the runner label, and any
  extra environment variables.
- **`verl_reference_versions`**: versions verl's Dockerfiles pin for things
  this repo does *not* build (`transformers`, `trl`, `nsight_systems`,
  `megatron`, `verl` itself) - tracked here purely for compatibility
  bookkeeping.

To add a new CUDA/Python/Torch combination, append an entry to
`build_matrix`. To bump a component's version, edit its `ref`. To add a
brand-new component, add an entry to `components` and a matching
`ci/build_scripts/<builder>.sh`.

See [docs/maintaining-components.md](docs/maintaining-components.md) for
the full step-by-step guide (with a worked example) to both of these tasks.
A matching Cursor Agent Skill
([.cursor/skills/manage-wheelhouse-components](.cursor/skills/manage-wheelhouse-components/SKILL.md))
lets an agent apply the same checklist automatically.

`ci/generate_matrix.py` is the only code that reads `versions.yaml`; it
turns it into the flat JSON matrix GitHub Actions consumes. Inspect the
resulting matrix locally at any time (requires `pip install pyyaml`):

```bash
python3 ci/generate_matrix.py --list-components
python3 ci/generate_matrix.py --component apex | python3 -m json.tool
python3 ci/generate_matrix.py --component all
```

## Releases: one persistent release per component

Every component gets its **own** GitHub Release - there is no single
combined release for "the repo" as a whole. `ci/release_meta.py` computes,
for a component and its currently-pinned `ref` in `versions.yaml`:

- **tag**: `<component>-<ref>`, e.g. `transformer-engine-v2.16.1`
- **title**: `<component> <ref> - cu<cuda> py<python> torch<torch>[; ...]`
  (one `cu.. py.. torch..` segment per `build_matrix` entry), e.g.
  `transformer-engine v2.16.1 - cu13.0.2 py3.12 torch2.11.0`

The reusable `.github/workflows/_ensure_release.yml` workflow creates that
release if it doesn't exist yet, or refreshes its title (in case
`build_matrix` changed) if it does. Rebuilding the same `ref` re-uploads
(`--clobber`) wheels onto that same release; bumping a component's `ref` in
`versions.yaml` starts a **brand-new** release under a new tag, leaving the
previous release (and its wheels) attached to the old `ref` untouched as a
historical record.

## Triggering builds

- Each component has its own workflow (`build-<component>.yml`) that can be
  run on demand from the Actions tab (`workflow_dispatch`), and also runs
  automatically on pushes to `main` (including PR merges) that touch
  `versions.yaml`, `ci/generate_matrix.py`, `ci/build_scripts/common.sh`, or
  that component's build script.
- On a push, the workflow first checks the component's target release. If its
  title exactly matches the configured CUDA/Python/Torch matrix and it contains
  every distribution listed in that component's `wheel_packages`, the build is
  skipped. Otherwise, a successful build ensures/updates the release, uploads
  the wheel(s), and re-runs `publish-index.yml`. Manual `workflow_dispatch`
  runs always build but skip publishing, so they can force a fresh test build.
- `build-all.yml` builds every component and also runs on a weekly schedule
  as a sanity sweep. It does **not** publish - it's for validating the whole
  matrix still builds cleanly.
- Pushing a tag matching `v*` runs `release.yml`, which is purely a
  trigger - the tag itself is not a release. It runs the full component x
  matrix sweep and, for every component, ensures/updates that same
  per-component release described above and uploads every wheel, then
  finishes by republishing the package index. Use this to force a fresh,
  citable rebuild of everything at once; ordinary `main` pushes already keep
  each component's release up to date incrementally.
- `publish-index.yml` can also be run standalone (or fires automatically
  whenever a release is published/edited/deleted) to refresh the index
  without rebuilding any wheels.

> **Note on the very first push to a brand-new repo/branch:** GitHub sets
> the push event's "before" commit to all-zeros when a branch has no prior
> history, and `paths`-filtered `push` triggers silently don't fire for that
> specific push (a longstanding GitHub Actions quirk, not specific to this
> repo). Every *subsequent* push behaves normally. If nothing ran after your
> very first push, just trigger the relevant `build-<component>.yml` once
> via `workflow_dispatch` (or push any follow-up change) to prime things.

## Installing built wheels

Once GitHub Pages is enabled for this repo (Settings → Pages → Source:
GitHub Actions) and at least one wheel has been published (from a `main`
push or a `release.yml` sweep, either way lands on that component's own
release):

```bash
pip install --extra-index-url https://<owner>.github.io/<repo>/simple/ flash-attn
pip install --extra-index-url https://<owner>.github.io/<repo>/simple/ transformer-engine
```

Or install a specific wheel directly from that component's
[release page](../../releases) - look for the tag `<component>-<ref>`, e.g.
`transformer-engine-v2.16.1`.

## Caveats

- **Two components build on a self-hosted machine.** `apex` and
  `TransformerEngine` are the heavy CUDA builds here, so their `runs_on`
  points at a self-hosted runner (`[self-hosted, Linux, X64]`) and their
  `max_jobs` is raised accordingly (32 and 16 - TransformerEngine multiplies
  that by `NVTE_BUILD_THREADS_PER_JOB=4`). `flash-attention`, `flashinfer`
  and `Megatron-Bridge` stay on free GitHub-hosted runners with capped
  parallelism. `runs-on` and `max_jobs` are both plain per-component fields
  in `versions.yaml`, so moving a component between the two (or dialing
  parallelism up towards verl's own `128`/`256`) is a one-line edit with no
  workflow changes.
- **Build time on free runners.** Builds that stay on GitHub-hosted runners
  are slow even with capped `MAX_JOBS`/arch lists. The reusable workflow caps
  each build attempt at 5 hours (under GitHub's 6h job limit) and saves a
  resumable build-cache tarball on timeout, so re-running the workflow
  continues from where it left off - the same tradeoff flash-attention's own
  CI makes.
- **Self-hosted runners can route GitHub transfers through a proxy.** The
  self-hosted machine's direct egress to GitHub is slow and prone to hanging,
  which stalls the wheel/artifact and release uploads (and the resumable
  build cache). Set an optional `BYTED_PROXY` repository secret to an HTTP(S)
  egress proxy URL and those transfers are routed through it. It only takes
  effect on self-hosted runners (`runner.environment == 'self-hosted'`);
  GitHub-hosted builds always use a direct connection, so leaving the secret
  unset is a no-op. Only the self-hosted components' workflows forward it to
  the reusable `_build.yml`: `build-apex.yml`,
  `build-transformer-engine.yml`, and the `build-all.yml`/`release.yml`
  sweeps.
- **`apex` tracks `master`** (both verl Dockerfiles build it unpinned), so it
  effectively behaves like a rolling release under the fixed tag
  `apex-master`; every other component tracks a specific tag and gets a fresh
  release per version bump.
- **Old per-component releases aren't deleted automatically.** Bumping a
  `ref` starts a new release rather than replacing the old one, so the PEP
  503 index may list more than one version of a package (e.g. both
  `transformer-engine-v2.15` and `transformer-engine-v2.16.1` wheels) until
  you manually delete the stale release from the
  [releases page](../../releases) if that matters for your use case. The same
  applies to the retired `vllm-*`/`sglang-*` releases: dropping those
  components stops future builds but leaves their existing wheels published
  and indexed until the releases are deleted by hand.
