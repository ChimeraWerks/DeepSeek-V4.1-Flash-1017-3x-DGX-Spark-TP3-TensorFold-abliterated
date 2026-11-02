#!/usr/bin/env bash
set -u
usage() {
  printf '%s\n' 'Usage: ./scripts/prep-sparks.sh status|plan|apply|explain [--only ID ...] [--yes]' \
    'status/plan read each Spark and change nothing; apply asks per Spark (sudo) and never reboots.' >&2
  exit 2
}
action=${1:-status}
case "$action" in status|plan|apply|explain) shift || true ;; -h|--help) usage ;; *) usage ;; esac
printf 'Plan: %s the DGX Spark host setup (firmware, 64K kernel, memory saver, memory policy, RoCE flags) on each Spark; never reboot.\n' "$action"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh" || exit 1

PREP="$REPO_ROOT/tools/spark/spark_prep.py"
if [[ "$action" == explain ]]; then
  exec python3 -I "$PREP" explain "$@"
fi
[[ -n "${SPARK_HOSTS:-}" && -n "${SPARK_USER:-}" && -n "${SSH_KEY:-}" ]] || fail 'set SPARK_HOSTS, SPARK_USER and SSH_KEY in cluster.env'
key=${SSH_KEY/#\~/$HOME}
remote='.cache/ds1017-prep/spark_prep.py'
status=0
for host in $SPARK_HOSTS; do
  printf '\n== %s\n' "$host"
  target="$SPARK_USER@$host"
  if [[ "$action" == apply ]]; then
    # apply needs sudo and a terminal for its prompt, so the script is copied first; status and plan write nothing.
    ssh -i "$key" -oBatchMode=yes -oStrictHostKeyChecking=yes -oConnectTimeout=10 "$target" \
      "mkdir -p ~/.cache/ds1017-prep && cat > ~/$remote" < "$PREP" || { printf 'FAIL: copy to %s\n' "$host" >&2; status=1; continue; }
    ssh -t -i "$key" -oStrictHostKeyChecking=yes -oConnectTimeout=10 "$target" sudo python3 -I "$remote" apply "$@" || status=1
  else
    ssh -i "$key" -oBatchMode=yes -oStrictHostKeyChecking=yes -oConnectTimeout=10 "$target" python3 -I - "$action" "$@" < "$PREP" || status=1
  fi
done
[[ "$action" == apply ]] && printf '\nReboot each Spark yourself when it is idle, then run ./scripts/prep-sparks.sh status again.\n'
exit "$status"
