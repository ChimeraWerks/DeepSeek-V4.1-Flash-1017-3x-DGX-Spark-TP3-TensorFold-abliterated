#!/usr/bin/env bash
# Export committed release paths only; never copy Git history or model weights.
# Usage: tools/release/export_public.sh /absolute/fresh/destination
set -euo pipefail
umask 077
ROOT=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
DEST=${1:?provide a fresh absolute local destination}
[[ "$DEST" == /* && ! -e "$DEST" ]] || { echo 'FAIL: destination must be absolute and absent' >&2; exit 1; }
[[ -z "$(git -C "$ROOT" status --porcelain)" ]] || { echo 'FAIL: commit and review the clean source tree first' >&2; exit 1; }
command -v gitleaks >/dev/null
OWNER=$(git -C "$ROOT" config user.name)
EMAIL=$(git -C "$ROOT" config user.email)
[[ -n "$OWNER" && -n "$EMAIL" ]] || { echo 'FAIL: normal owner identity required' >&2; exit 1; }
DEST=$(realpath -m -- "$DEST")
[[ "$DEST" != "$ROOT" && "$DEST" != "$ROOT/"* ]] || { echo 'FAIL: destination must be outside source tree' >&2; exit 1; }
SOURCE=$(git -C "$ROOT" rev-parse HEAD)
# Reject symlinks/submodules before tar extraction, not after a possible escape.
git -C "$ROOT" ls-tree -r "$SOURCE" | awk '$1 != "100644" && $1 != "100755" {bad=1} END {exit bad}'

PATHS=(.gitignore .gitleaks.toml LICENSE NOTICE CREDITS.md THIRD_PARTY_NOTICES.md CHANGELOG.md
       CITATION.cff SECURITY.md README.md llms.txt cluster.env.example benchmarks docs engine release scripts serving tools)
mkdir -m 700 -- "$DEST"
# git archive exports only reviewed HEAD, not ignored files, build output or private refs.
git -C "$ROOT" archive --format=tar "$SOURCE" -- "${PATHS[@]}" | tar -xf - -C "$DEST"
python3 -I "$DEST/tools/release/audit_public.py" "$DEST"
gitleaks dir "$DEST" --config "$DEST/.gitleaks.toml" --redact=100 --no-banner \
  --gitleaks-ignore-path /dev/null --ignore-gitleaks-allow --max-archive-depth 3
git -C "$DEST" init --initial-branch=main
git -C "$DEST" config user.name "$OWNER"
git -C "$DEST" config user.email "$EMAIL"
# Normal hooks and real current timestamps. No signature, hook or identity bypass.
git -C "$DEST" add --all
git -C "$DEST" commit -m 'release: initial public release v1.1' \
  -m 'Initial public release v1.1. See CREDITS and NOTICE for upstream
components. No model weights are redistributed.'
test "$(git -C "$DEST" rev-list --count HEAD)" = 1
test "$(git -C "$DEST" rev-list --parents -n 1 HEAD | wc -w)" = 1
test -z "$(git -C "$DEST" remote)"
test -z "$(git -C "$DEST" status --porcelain)"
# Normal hooks may change staged content; verify the final clean snapshot too.
python3 -I "$DEST/tools/release/audit_public.py" "$DEST"
gitleaks git "$DEST" --config "$DEST/.gitleaks.toml" --log-opts=HEAD --redact=100 --no-banner \
  --gitleaks-ignore-path /dev/null --ignore-gitleaks-allow --max-archive-depth 3
printf 'Source snapshot: %s\nPublic local commit: ' "$SOURCE"
git -C "$DEST" rev-parse HEAD
printf 'Tracked paths: '
git -C "$DEST" ls-files | wc -l
echo 'Local export only: no remote configured, no publication performed.'
