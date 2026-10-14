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
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | `DeepEP` | `deep-ep` (MoE all-to-all, linked against NVSHMEM) |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | `FlashMLA` | `flash-mla` (MLA decode/prefill, `sm90a` + `sm100f`) |
| [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform) | `fast-hadamard-transform` | `fast-hadamard-transform` (Hadamard kernels for DeepSeek sparse attention) |

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
release per component, Python version and torch version, named after that
component and its pinned dependency versions (e.g.
`transformer-engine v2.16.1 - cu13.2.2 py3.12 torch2.13.0`, tag
`transformer-engine-v2.16.1-py3.12-torch2.13.0` - the tag stays CUDA-free,
which is why one component cannot ship two CUDA worlds without a tag-scheme
change) - and to static
[GitHub Pages](https://pages.github.com/) PEP 503 "simple" package indexes,
one per CUDA/torch combination, so they're directly `pip install`-able.

## Repo layout

```
versions.yaml             # THE editable/extendable version map (see below), kept in the project's base dir
ci/
  generate_matrix.py      # expands versions.yaml into a GH Actions matrix
  release_meta.py          # release tag/title/notes per component x Python x torch
  build_index.py          # builds the per-CUDA/torch PEP 503 indexes from release assets
  migrate_release_tags.py # one-time: retag releases to the per-torch scheme
  restore_release_wheels.py # one-time: republish clobbered wheels from build artifacts
  build_scripts/
    apex.sh
    transformer_engine.sh
    flash_attention.sh
    flashinfer.sh
    megatron_bridge.sh
    deep_ep.sh
    flash_mla.sh
    fast_hadamard_transform.sh
    lib/                   # shared leaf helpers, sourced only by the builders that use them
      env.sh arch.sh nccl.sh cuda_paths.sh rdma_nvshmem.sh wheel_pack.sh flashinfer.sh
    provision/
      install_cudnn.sh     # cuDNN for requires_cudnn components (transformer-engine)
  runner/
    free_disk_space.sh     # GitHub-hosted disk cleanup; CI-only, never build-fingerprinted
  patches/
    enable_deep_ep_sm80.py  # fat-bin Ampere+Hopper for deep-ep
.github/workflows/
  _build.yml              # reusable single-combination build workflow
  _ensure_release.yml     # unused helper: pre-create/refresh a release (releases are made inline by _build.yml)
  build-apex.yml
  build-transformer-engine.yml
  build-flash-attention.yml
  build-flashinfer.yml
  build-megatron-bridge.yml
  build-deep-ep.yml
  build-flash-mla.yml
  build-fast-hadamard-transform.yml
  build-all.yml           # builds every component x every matrix combo
  release.yml             # on `v*` tag push: full-matrix build + upload
  publish-index.yml       # (re)publishes the GitHub Pages PEP 503 index
.github/actions/
  build-toolchain/action.yml  # Python/CUDA/cuDNN/PyTorch provisioning (shared build input)
docs/
  maintaining-components.md  # step-by-step: upgrade a version / add a component
.cursor/skills/
  manage-wheelhouse-components/SKILL.md  # agent skill for the same tasks
```

## The version map (`versions.yaml`)

Everything version-related lives in one editable, plain-data YAML file at
the project's base directory - no Python, no workflow YAML - needs to
change for routine version bumps:

- **`build_matrix`**: the CPU arch / CUDA / Python / Torch combinations to
  build. The current toolkit is CUDA `13.2.2`, Python `3.11` and `3.12`,
  Torch `2.13.0` (3.12 matches both verl Dockerfiles' `ARG` defaults, though
  torch is kept ahead of their `2.11.0` pin; 3.11 is added because torch
  publishes cp311 cu132 wheels for both arches and downstream envs still run
  it), on `x86_64` and `aarch64`. The matrix also still carries CUDA
  `13.0.2` rows, retained for `flashinfer` alone: it rehosts prebuilt
  jit-cache/cubin data wheels that upstream does not yet publish for cu132.
  Add another entry to build more combinations - every component is built
  once per entry here, minus the ones its `arches` / `python_versions` /
  `cuda_versions` filters exclude. Each component is pinned to exactly one
  CUDA per (Python, torch) via `cuda_versions`; releases are not keyed by
  CUDA (only by component/Python/torch), so spanning two CUDA worlds for one
  pair is rejected and the tag scheme would have to become CUDA-keyed first.

- **`components`**: per-submodule config - the git `ref` to build (branch,
  tag, or commit; overrides whatever commit the submodule pointer in this
  repo is on), which `ci/build_scripts/<builder>.sh` to run, the CUDA arch
  list, whether cuDNN is required, `max_jobs`, the runner label, which CPU
  arches to build for (`arches`), which Python versions (`python_versions`)
  and which CUDA versions (`cuda_versions`) to build for, with optional
  per-arch field overrides (`arch_overrides`), and any extra environment
  variables.

- **`verl_reference_versions`**: versions verl's Dockerfiles pin for things
  this repo does *not* build (`transformers`, `trl`, `nsight_systems`,
  `megatron`, `verl` itself) - tracked here purely for compatibility
  bookkeeping.

To add a new arch/CUDA/Python/Torch combination, append an entry to
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

## Releases: one persistent release per component, Python and torch

Every component gets **its own GitHub Release for each (Python version,
torch version) pair it builds for** - there is no single combined release
for "the repo" as a whole. `ci/release_meta.py` computes, for such a triple
and the component's currently-pinned `ref` in `versions.yaml`:

- **tag**: `<component>-<ref>-pyX.Y-torch<torch>`, e.g.
  `transformer-engine-v2.16.1-py3.11-torch2.13.0`
- **title**: `<component> <ref> - [<arch> ]cu<cuda> py<python> torch<torch>[; ...]`
  with one segment per `build_matrix` row **of that Python and torch** and
  the `x86_64` arch left implicit, e.g. `apex master - cu13.2.2 py3.11
  torch2.13.0; aarch64 cu13.2.2 py3.11 torch2.13.0`. The title carries CUDA
  (the index keys on it), the tag does not, so the rows grouped here must all
  share one CUDA - enforced by `cuda_versions` in `versions.yaml`.

Both versions are in the tag because a wheel is ABI-bound to each and its
filename records neither: `apex-0.1-cp312-cp312-linux_x86_64.whl` is the
name a torch 2.11 build and a torch 2.13 build both produce. Sharing one
tag across torch versions therefore meant the newer build silently
overwrote the older one's assets - changing what every already-published
download URL (including the ones pinned in a `uv.lock`) serves, with no
version or hash to notice it by. Arch stays *inside* a release, because
there the wheel filename's platform tag does distinguish the files.

Adding a Python or torch version now creates new releases instead of
editing existing ones: the new wheels go to the new tag, while the older
releases keep their exact tag/title/manifest and stay skippable (no churn
rebuild) and installable.

The build workflow creates the release inline while uploading: `_build.yml`
runs `ci/release_meta.py` under the matrix row's own setup-python
interpreter, so the interpreter needs no workflow input (no `--python` is
passed); torch has no such ambient source and is passed as `--torch`.
Rebuilding the same triple re-uploads (`--clobber`) wheels onto that same
release; bumping a component's `ref` - or the matrix's torch - starts
**brand-new** releases under new tags, leaving the previous releases (and
their wheels) untouched as a historical record.

Two one-time maintenance scripts exist for the migration to this scheme:
`ci/migrate_release_tags.py` retags releases created under the older
(component, Python) scheme, and `ci/restore_release_wheels.py` republishes
wheels that a torch bump overwrote back onto their own release, reading
them from the build's still-retained workflow artifacts. Both default to a
dry run and take `--apply` to act.

## Triggering builds

- Each component has its own workflow (`build-<component>.yml`) that can be
  run on demand from the Actions tab (`workflow_dispatch`), and also runs
  automatically on pushes to `main` (including PR merges) that touch
  `versions.yaml`, `ci/generate_matrix.py`, anything under
  `ci/build_scripts/` or `ci/patches/`, `.github/actions/`, or that
  component's workflow. The reusable `_build.yml` is intentionally not a push
  trigger: it is orchestration only, and editing it never rebuilds a wheel.
- Every arch/CUDA/Python/Torch combination is an independent job on its own
  runner, with its own build cache and wheel artifact, and `fail-fast: false`
  keeps one failure from cancelling the others. A manual run additionally
  takes an **arch** choice (`all` / `x86_64` / `aarch64`) and a **python**
  choice (`all` / `3.11` / `3.12`), so you can test one architecture or
  interpreter without spending hours of runner time on the others; a
  component that doesn't opt into the chosen arch/python version simply has
  nothing to build and its build job is skipped. Pushes always build every
  combination the component opts into.
- On a push, the workflow first checks the component's target release. If its
  title exactly matches the configured arch/CUDA/Python/Torch matrix and it
  contains every distribution listed in that component's `wheel_packages` for
  every arch it builds (matched on each wheel's own platform tag, so an
  x86_64 wheel never stands in for a missing aarch64 one), the build is
  skipped. Otherwise, a successful build ensures/updates the release, uploads
  the wheel(s), and re-runs `publish-index.yml`. Manual `workflow_dispatch`
  runs always build but skip publishing, so they can force a fresh test build.
- `build-all.yml` builds every component and also runs on a weekly schedule
  as a sanity sweep (the schedule always covers every arch; a manual run
  takes the same arch choice as the per-component workflows). It does **not**
  publish - it's for validating the whole matrix still builds cleanly.
- Pushing a tag matching `v*` runs `release.yml`, which is purely a
  trigger - the tag itself is not a release. It runs the full component x
  matrix sweep and, for every (component, Python) pair, ensures/updates the
  matching release described above and uploads every wheel, then
  finishes by republishing the package index. Use this to force a fresh,
  citable rebuild of everything at once; ordinary `main` pushes already keep
  each component's releases up to date incrementally.
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

Indexes are published to GitHub Pages **per CUDA/torch world**, and
refreshed whenever a wheel lands on one of a component's releases (from a
`main` push or a `release.yml` sweep):

```bash
# Pin the world matching your environment (recommended):
pip install --extra-index-url https://verl-project.github.io/verl-wheelhouse/cu132/torch2.13/simple/ flash-attn

# /simple/ is an alias for whatever versions.yaml currently builds (today
# cu132/torch2.13), so it moves when the matrix does:
pip install --extra-index-url https://verl-project.github.io/verl-wheelhouse/simple/ transformer-engine
```

`cu130/torch2.13` still exists but, after the 13.2 move, carries only
`flashinfer` - its rehosted jit-cache/cubin data wheels are not yet published
for cu132; every other component is under `cu132/torch2.13`.

The split is not cosmetic: two wheels of one component built against
different torch or CUDA versions have the same filename *and* the same
version, so one index cannot hold both and nothing in a plain
`pip install apex` could express which you need. Picking the URL is what
makes the choice explicit - the same shape `download.pytorch.org/whl/cu132`
and `flashinfer.ai/whl/cu130/torch2.13` use.
[`../../releases`](../../releases) lists every world's releases; their
landing page at the Pages root enumerates the ones currently published.

Or install a specific wheel directly from that component's
[release page](../../releases) - look for the tag
`<component>-<ref>-pyX.Y-torch<torch>` (e.g.
`transformer-engine-v2.16.1-py3.11-torch2.13.0`).

On a fork, substitute your own `https://<owner>.github.io/<repo>/`
and enable Pages first (Settings → Pages → Source: GitHub Actions);
everything in `ci/` and `.github/workflows/` derives the repo it is running
in from `${{ github.repository }}`, so no other change is needed.

## Caveats

- **Every package is available for arm64, but not every component has an
  arm64 job.** The `build_matrix` covers `x86_64` and `aarch64`, and
  `flash-attention`, `apex`, `TransformerEngine`, `flashinfer`, `deep-ep`,
  `flash-mla` and `fast-hadamard-transform` all build both.
  `Megatron-Bridge` deliberately stays `arches: [x86_64]`: its wheel is
  `py3-none-any`, so the single x86_64 build already installs on arm64, and a
  second job would only race a byte-identical asset name onto the same
  release. `flashinfer` is a hybrid for the same reason - two of its three
  wheels are `py3-none-any`, so `flashinfer.sh` emits those on x86_64 only and
  the arm64 job contributes just the arch-specific `flashinfer-jit-cache`.
- **The six CUDA-extension components build for both Python 3.11 and 3.12;
  `Megatron-Bridge` and `flashinfer` pin `python_versions: ["3.12"]`.** A
  `cp311`/`cp312` wheel is interpreter-specific, so flash-attention, apex,
  TransformerEngine, deep-ep, flash-mla and fast-hadamard-transform ship one
  wheel per interpreter per arch. The pure-Python components opt out of 3.11:
  their `py3-none-any` wheel already installs on every interpreter, so a
  second build would only re-upload an identical asset, and Megatron-Bridge
  additionally cannot build on 3.11 at all (upstream declares
  `requires-python >=3.12`). Adding a Python version is a two-step change:
  append its entries to `build_matrix` *and* add that version to the static
  `options:` list of every workflow's `python` dispatch input (GitHub Actions
  choice lists cannot be generated dynamically).
- **arm64 always builds on GitHub-hosted runners.** The self-hosted machine is
  x86_64-only, so every `arch_overrides.aarch64` points at `ubuntu-24.04-arm`
  (GitHub's free 4 vCPU / 16 GB arm64 runner) rather than waiting on arm
  hardware that doesn't exist. `ci/generate_matrix.py` fails the matrix if a
  non-x86_64 row names a self-hosted runner, and `_build.yml` separately
  asserts the runner's `uname -m` matches the row's arch. The practical cost
  is parallelism: `apex` drops from `max_jobs: 32` to `2` and
  `TransformerEngine` from an effective 64 compile threads to 4, so both are
  expected to exhaust their 5h budget and finish across a few re-runs off the
  resumable build cache. Everything else the toolchain needs already works
  there: `Jimver/cuda-toolkit` rewrites its apt repo path to NVIDIA's `sbsa`
  one, `provision/install_cudnn.sh`'s `install_cudnn` maps `aarch64` → `sbsa`, and
  download.pytorch.org publishes `manylinux_2_28_aarch64` CUDA wheels. Both
  arches' wheels live on the same per-(component, Python) release and the
  PEP 503 index lists them side by side - pip picks by platform tag.
- **Two components build on a self-hosted machine (x86_64 only).** `apex` and
  `TransformerEngine` are the heavy CUDA builds here, so their `runs_on`
  points at a self-hosted runner (`[self-hosted, Linux, X64]`) and their
  `max_jobs` is raised accordingly (32 and 16 - TransformerEngine multiplies
  that by `NVTE_BUILD_THREADS_PER_JOB=4`). Their aarch64 rows fall back to
  GitHub's arm64 runner, as above. `flash-attention`, `flashinfer` and
  `Megatron-Bridge` stay on free GitHub-hosted runners on both arches with
  capped parallelism. `runs-on` and `max_jobs` are both plain per-component
  fields in `versions.yaml`, so moving a component between the two (or dialing
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
  unset is a no-op for them. Only the self-hosted components' workflows
  forward it to the reusable `_build.yml`: `build-apex.yml`,
  `build-transformer-engine.yml`, and the `build-all.yml`/`release.yml`
  sweeps. Because secrets are per-repository and do not follow a fork or
  transfer, a self-hosted build in a new repo warns up front when
  `BYTED_PROXY` is missing, and every transfer that would use it carries a
  `timeout-minutes` cap so a stalled upload fails within the hour instead of
  sitting on the runner until the job times out.
- **The DeepSeek kernels carry two constraints the other components don't.**
  `deep-ep`'s x86_64 wheel also ships an Ampere (`8.0`, A100) cubin for the
  intranode kernels; internode / low-latency / NVSHMEM stay Hopper-and-newer,
  matching upstream. Ampere is compiled in via a build-time rewrite
  (`ci/patches/enable_deep_ep_sm80.py`) because DeepEP's own
  `DISABLE_SM90_FEATURES` flag cannot fat-bin both. The aarch64 wheel stays
  `9.0;10.0`. Separately, `deep-ep` links NVSHMEM and bakes
  `-Wl,-rpath,$NVSHMEM_DIR/lib` into its
  extension, so its wheel only resolves libnvshmem where that directory exists:
  `lib/rdma_nvshmem.sh`'s `install_nvshmem` therefore installs
  `nvidia-nvshmem-cu<major>==<extra_env NVSHMEM_VERSION>` into
  `/usr/local/lib/python<X.Y>/dist-packages`, the same absolute path verl's
  `docker/Dockerfile.uv.cu130` installs it to. Bumping the NVSHMEM version on
  either side without the other leaves the wheel pointing at a directory the
  image no longer has. Both projects also append `+<short git sha>` to their
  own version with no opt-out; since GitHub rewrites `+` to `.` in release
  asset filenames (which would desync the filename from the wheel's own
  METADATA), `lib/wheel_pack.sh`'s `strip_wheel_local_version` removes the segment
  after the build. Their wheels are therefore plain `deep-ep 1.2.1` /
  `flash-mla 1.0.0`, and - as with `apex` - the exact commit lives in the
  release tag and title rather than in the version.
- **`fast-hadamard-transform` ignores `torch_cuda_arch_list`.** Its `setup.py`
  derives gencode flags from the *toolkit* version alone, emitting nine of them
  on CUDA 13 (`sm_75` through `sm_121`) with no env var to narrow the list -
  which is both why it is slow enough to belong here (verl's image used to
  compile all nine on every build) and why its wheel covers more parts than any
  other component's. `versions.yaml` therefore sets `torch_cuda_arch_list: null`
  for it, as it does for `flash-mla`.
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
