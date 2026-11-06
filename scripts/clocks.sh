#!/usr/bin/env bash
set -u
usage() {
  printf '%s\n' 'Usage: ./scripts/clocks.sh status|set [MHz]|stock' \
    'status reads each Spark and changes nothing; set caps all three GPUs (default: GPU_CLOCK_CAP_MHZ); stock removes the cap.' \
    'set and stock ask for sudo on each Spark and need no restart.' >&2
  exit 2
}
action=${1:-status}
case "$action" in status|set|stock) ;; -h|--help) usage ;; *) usage ;; esac
printf 'Plan: %s the GPU clock cap on each Spark; never restart a rank.\n' "$action"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh" || exit 1

[[ -n "${SPARK_HOSTS:-}" && -n "${SPARK_USER:-}" && -n "${SSH_KEY:-}" ]] || fail 'set SPARK_HOSTS, SPARK_USER and SSH_KEY in cluster.env'
key=${SSH_KEY/#\~/$HOME}
smi=/usr/bin/nvidia-smi
case "$action" in
  set)
    cap=${2:-${GPU_CLOCK_CAP_MHZ:-2300}}
    [[ "$cap" =~ ^[1-9][0-9]*$ ]] || fail "cap must be a whole number of MHz, got: $cap"
    change=(-lgc "300,$cap")
    ;;
  stock) change=(-rgc) ;;
esac
status=0
for host in $SPARK_HOSTS; do
  printf '\n== %s\n' "$host"
  target="$SPARK_USER@$host"
  if [[ "$action" == status ]]; then
    ssh -i "$key" -oBatchMode=yes -oStrictHostKeyChecking=yes -oConnectTimeout=10 "$target" \
      "$smi" --query-gpu=clocks.current.graphics,clocks.max.graphics --format=csv,noheader || status=1
  else
    # sudo needs a terminal for its prompt; the lock lasts until reboot or the next change.
    ssh -t -i "$key" -oStrictHostKeyChecking=yes -oConnectTimeout=10 "$target" sudo "$smi" "${change[@]}" || status=1
  fi
done
[[ "$action" != status ]] && printf '\nstart.sh reapplies GPU_CLOCK_CAP_MHZ from cluster.env; set it there to keep this change.\n'
exit "$status"
