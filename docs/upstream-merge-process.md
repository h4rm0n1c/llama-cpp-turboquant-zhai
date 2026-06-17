# Upstream Integration Process

The integration branch is built as a patch stack:

1. `upstream/master` from `ggml-org/llama.cpp`
2. The resolved TurboQuant baseline from `merge/upstream-sync`
3. New fork-only changes from `TheTom/feature/turboquant-kv-cache`
4. Local HTTP memory and KV usage reporting

`upstream/master` means GGML's llama.cpp mainline. CPU execution is not a
supported validation target for this fork.

## Why Use The Resolved Baseline

Replaying the complete TheTom branch onto current upstream duplicates many
commits that GGML main already contains under different hashes. It also creates
conflicts unrelated to TurboQuant.

The resolved `merge/upstream-sync` tree is the current TurboQuant baseline. A
squash merge of that tree onto current upstream preserves the fork behavior
without importing its historical topology. New TheTom commits are then reviewed
and applied only when their patches are not already represented.

## Prepare A Candidate

Refresh the remote-tracking refs:

```bash
git fetch upstream master
git fetch TheTom feature/turboquant-kv-cache
```

Create a detached candidate worktree:

```bash
scripts/prepare-upstream-integration.sh prepare
```

The default candidate path is `/tmp/llama-tq-integration`. Override it with
`TQ_INTEGRATION_WORKTREE`.

The baseline may conflict where GGML changed the same interfaces. Resolve each
conflict against current upstream behavior. In particular:

- Keep current upstream speculative decoding behavior and EAGLE3 support.
- Keep the current upstream `common/fit.h` API shape.
- Add the JSON declaration needed by `common_memory_breakdown_json()`.
- Preserve the router and child-server memory reporting behavior.

Stage the resolutions, then audit:

```bash
git -C /tmp/llama-tq-integration add <resolved-files>
scripts/prepare-upstream-integration.sh audit
```

## Review New TheTom Changes

List commits added after the resolved baseline:

```bash
git log --oneline --no-merges \
    merge/upstream-sync..TheTom/feature/turboquant-kv-cache
```

Classify each commit before applying it:

- Skip commits already reachable from or patch-equivalent to current upstream.
- Skip shared upstream lineage that TheTom imported under different hashes.
- Apply fork-specific TurboQuant, CUDA, router, and model compatibility fixes.
- Do not infer relevance from commit hashes or subjects alone; inspect the diff.

Apply reviewed commits through the helper:

```bash
scripts/prepare-upstream-integration.sh apply <sha> [<sha> ...]
```

The helper skips patches already present by reverse patch equivalence. It stops
on conflicts and leaves the candidate for manual resolution. It never commits,
pushes, tags, or changes the active branch.

## Local HTTP Memory Statistics

The local memory reporting is maintained as part of the resolved baseline:

- `common_memory_breakdown_json()` reports per-device total, free, model, KV,
  and compute bytes, plus host model, KV, and compute bytes.
- A child server attaches the data to its model information.
- The router exposes that information through `/v1/models`.

After every upstream integration, verify that these fields remain present and
that router mode does not initialize a CUDA context in the parent process.

## CUDA Build And Test

The Quantzhai build helper requires `.git` to be a directory, so materialize the
linked worktree as a full disposable clone:

```bash
scripts/prepare-upstream-integration.sh materialize
```

Build that clone with the Quantzhai Docker workflow:

```bash
cd ~/turboquant/quantzhai
QZ_BUILD_DIR=/tmp/qz-integration \
QZ_TQ_BRANCH=__detached__ \
scripts/qz-build-image
```

Validation is CUDA-only:

1. Build the CUDA server image.
2. Start the normal Quantzhai stack with the candidate image.
3. Run `scripts/qz-doctor`.
4. Load a TurboQuant model and perform a real generation request.
5. Query `/v1/models` and verify GPU model, KV, and compute byte fields.
6. Confirm the router process does not own GPU memory.

Do not promote the candidate based only on compilation. No commit, branch
rewrite, tag, push, or production switch is part of this procedure without
explicit human approval.
