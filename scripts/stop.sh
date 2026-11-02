#!/usr/bin/env bash
set -u
printf '%s\n' 'Plan: stop the serving API on the head, then stop only the recipe rank units and memory guards; leave weights and spill files in place.'
# shellcheck source-path=SCRIPTDIR
# shellcheck source=_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh" || exit 1
run_recipe stop
