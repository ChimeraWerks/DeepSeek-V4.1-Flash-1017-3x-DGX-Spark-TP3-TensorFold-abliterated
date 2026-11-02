#!/usr/bin/env bash
set -u
printf '%s\n' 'Plan: download the pinned HF export on each Spark, build and SHA-256-check its rank file (skip a matching file), then stage built engine binaries/guards. No engine starts.'
# shellcheck source-path=SCRIPTDIR
# shellcheck source=_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh" || exit 1
run_recipe prepare
