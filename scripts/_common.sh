#!/usr/bin/env bash
# Shared configuration only; each entry point prints its own plan first.
set -u
set +x

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P) || fail 'cannot locate repository root'
CLUSTER_ENV=${CLUSTER_ENV:-"$REPO_ROOT/cluster.env"}
[[ -r "$CLUSTER_ENV" ]] || fail "missing cluster.env; copy cluster.env.example to $REPO_ROOT/cluster.env and edit it"
command -v python3 >/dev/null || fail 'python3 is required'
set -a
# shellcheck source=/dev/null
source "$CLUSTER_ENV" || fail 'could not read cluster.env'
set +a

run_recipe() {
  python3 -I "$REPO_ROOT/tools/cluster/recipe.py" "$1" --manifest "$REPO_ROOT/release/manifest.json"
}
