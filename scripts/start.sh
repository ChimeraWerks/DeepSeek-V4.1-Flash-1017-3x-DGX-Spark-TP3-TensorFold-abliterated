#!/usr/bin/env bash
set -u
printf '%s\n' 'Plan: install and validate the API config/private credentials first; validate all three ranks, replace recipe rank units, wait for warm numeric-effort engine health and arm guards; start the API, wait for readiness, then send three short greedy warmups directly to the head engine and report tokens/s.'
# shellcheck source-path=SCRIPTDIR
# shellcheck source=_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh" || exit 1
# recipe.py uses Python json to read all engine flags/sizes and calls deploy.py;
# no duplicated flags here. API setup uses chimera_sparks.lifecycle.start_api.
run_recipe start
