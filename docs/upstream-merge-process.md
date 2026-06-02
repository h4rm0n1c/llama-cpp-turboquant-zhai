# Upstream Merge Process

## Agent Autonomy Scope

The agent is authorised to perform the following without confirmation:

- `git fetch upstream master` and `git fetch origin` (read-only operations)
- Cherry-pick upstream commits into `main` and resolve conflicts per the rules below
- Merge TheTom's `origin/feature/turboquant-kv-cache` into `main` and resolve conflicts
- Push `main` to `fork/main` after any of the above
- Notify the user of what was done and any notable conflict resolutions

The agent must NOT do without explicit user confirmation:

- Open PRs or issues on any upstream repository (ggml-org/llama.cpp, TheTom/llama-cpp-turboquant, etc.)
- Create new branches on the fork
- Change the build target
- Any action that affects production uptime without prior notice

## Branch Architecture

```
upstream/master (ggml-org/llama.cpp)
  └─ origin/feature/turboquant-kv-cache (TheTom's fork)
      └─ fork/main (our production branch)
          ├─ TheTom's base
          ├─ fix/srv-dining-philosophers-deadlock (bug chain)
          └─ feature/vram-http-metrics (VRAM/CUDA/load-timeout)
```

Our `main` contains two patch sets merged side by side. The bug chain (`fix/srv-dining-philosophers-deadlock`) covers the child process lifecycle deadlock, zombie slot, WIFSIGNALED encoding, abort callback, stop_mutex separation, loading timeout, and last_error/exit_signal in /v1/models. No auto-recover, no recovering flag, no reload_attempts — the proxy handles recovery.

The VRAM patches (`feature/vram-http-metrics`) cover CUDA OOM propagation, model-load-timeout CLI arg, child error reporting, and VRAM info in /v1/models.

## When Upstream (ggml-org/llama.cpp) has a fix we need

```bash
# 1. Find the upstream commit
git fetch upstream master
git log upstream/master --oneline --no-merges -20

# 2. Cherry-pick it directly onto our main
git checkout main
git cherry-pick <sha>
```

**If it conflicts** because TheTom doesn't have a prerequisite upstream commit:

```bash
# a) Find and cherry-pick the prerequisite first
git cherry-pick <prerequisite-sha>
# resolve conflicts, git add, git cherry-pick --continue

# b) Then cherry-pick the target commit
git cherry-pick <target-sha>
```

**Conflict resolution rule:**
- Accept our version (main) for files we patched: `tools/server/server-models.cpp`, `tools/server/server-models.h`, `tools/server/server.cpp`, `vendor/sheredom/subprocess.h`.
- Accept upstream's version for everything else.
- If the conflict is in our patched files, understand why upstream changed them and decide whether to keep our change or adopt upstream's.

**Don't:**
- Create branches for upstream fixes (cherry-pick directly onto main)
- Rebase our patch branches onto upstream (they're based on TheTom)

## When TheTom updates

TheTom maintains `feature/turboquant-kv-cache`. When he pushes new changes:

```bash
git fetch origin
git checkout main
git merge origin/feature/turboquant-kv-cache
```

After resolving any merge conflicts (accept TheTom's version unless it breaks our patches), push main.

## Cherry-pick from ggml directly (bypassing TheTom)

TheTom cherry-picks from ggml selectively, often weeks behind. If we need a specific fix or feature from ggml-org/llama.cpp before TheTom syncs it:

```bash
git fetch upstream master
git cherry-pick <sha-from-ggml>
```

When TheTom eventually merges that same commit, the cherry-pick's content is identical — git resolves it without conflict.

## Never

- Open a PR or issue against upstream without explicit user approval
- Submit work-in-progress branches
- Add workaround code when the root cause fix is simpler
- Rebase production branches (force push only in emergency)
- Detach threads with dangling references
- Guess at root causes — trace the actual execution path
