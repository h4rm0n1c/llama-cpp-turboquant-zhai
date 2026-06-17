#!/usr/bin/env bash

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
worktree=${TQ_INTEGRATION_WORKTREE:-/tmp/llama-tq-integration}
upstream_ref=${TQ_UPSTREAM_REF:-upstream/master}
baseline_ref=${TQ_BASELINE_REF:-merge/upstream-sync}
thetom_ref=${TQ_THETOM_REF:-TheTom/feature/turboquant-kv-cache}

usage() {
    cat <<EOF
Usage:
  $0 prepare
  $0 audit [<candidate-path>]
  $0 apply <commit> [<commit> ...]
  $0 materialize [<path>]

Environment:
  TQ_INTEGRATION_WORKTREE  Candidate path (default: $worktree)
  TQ_UPSTREAM_REF         GGML llama.cpp ref (default: $upstream_ref)
  TQ_BASELINE_REF         Resolved TurboQuant baseline (default: $baseline_ref)
  TQ_THETOM_REF           TheTom tracking ref (default: $thetom_ref)

This script never commits, pushes, tags, or changes the active branch.
EOF
}

require_worktree() {
    if [ ! -e "$worktree/.git" ]; then
        echo "Candidate worktree does not exist: $worktree" >&2
        exit 1
    fi
}

prepare() {
    if [ -e "$worktree" ]; then
        echo "Candidate path already exists: $worktree" >&2
        echo "Remove it explicitly with: git worktree remove $worktree" >&2
        exit 1
    fi

    git -C "$repo_root" rev-parse --verify "$upstream_ref^{commit}" >/dev/null
    git -C "$repo_root" rev-parse --verify "$baseline_ref^{commit}" >/dev/null
    git -C "$repo_root" rev-parse --verify "$thetom_ref^{commit}" >/dev/null

    git -C "$repo_root" worktree add --detach "$worktree" "$upstream_ref"
    if ! git -C "$worktree" merge --squash "$baseline_ref"; then
        cat <<EOF

The baseline has conflicts. Resolve them in $worktree, then:
  git -C $worktree add <resolved-files>
  $0 audit

Do not replay shared upstream commits from $thetom_ref.
EOF
        exit 2
    fi

    audit
}

audit() {
    require_worktree

    echo "upstream: $(git -C "$repo_root" rev-parse "$upstream_ref")"
    echo "baseline: $(git -C "$repo_root" rev-parse "$baseline_ref")"
    echo "TheTom:  $(git -C "$repo_root" rev-parse "$thetom_ref")"
    echo

    if git -C "$worktree" diff --name-only --diff-filter=U | grep -q .; then
        echo "Unresolved conflicts:"
        git -C "$worktree" diff --name-only --diff-filter=U
        return 2
    fi

    if git -C "$worktree" grep -n -E '^(<<<<<<<|=======|>>>>>>>)' -- \
        ':!vendor/**' ':!examples/**' ':!tests/**'; then
        echo "Conflict markers found." >&2
        return 2
    fi

    git -C "$worktree" diff --check
    git -C "$worktree" diff --cached --check

    echo "Candidate changes:"
    git -C "$worktree" diff --stat HEAD
    echo
    echo "TheTom commits not reachable from the baseline:"
    git -C "$repo_root" log --oneline --no-merges "$baseline_ref..$thetom_ref"
}

apply_commits() {
    require_worktree
    shift

    if [ "$#" -eq 0 ]; then
        usage
        exit 1
    fi

    if git -C "$worktree" diff --name-only --diff-filter=U | grep -q .; then
        echo "Resolve candidate conflicts before applying patches." >&2
        exit 2
    fi

    for commit in "$@"; do
        git -C "$repo_root" rev-parse --verify "$commit^{commit}" >/dev/null

        if git -C "$repo_root" show --format= --binary "$commit" |
            git -C "$worktree" apply --reverse --check >/dev/null 2>&1; then
            echo "$commit: already present by patch equivalence"
            continue
        fi

        echo "$commit: applying"
        git -C "$repo_root" show --format= --binary "$commit" |
            git -C "$worktree" apply --3way --index
    done

    audit
}

materialize() {
    require_worktree
    target=${2:-/tmp/qz-integration/llama-cpp-turboquant}

    if [ -e "$target" ]; then
        echo "Materialized path already exists: $target" >&2
        exit 1
    fi

    mkdir -p "$(dirname "$target")"
    git clone --no-local "$worktree" "$target"
    git -C "$worktree" diff --binary HEAD |
        git -C "$target" apply --index
    git -C "$target" diff --cached --check

    echo "Materialized candidate: $target"
}

case ${1:-} in
    prepare)
        prepare
        ;;
    audit)
        worktree=${2:-$worktree}
        audit
        ;;
    apply)
        apply_commits "$@"
        ;;
    materialize)
        materialize "$@"
        ;;
    *)
        usage
        exit 1
        ;;
esac
