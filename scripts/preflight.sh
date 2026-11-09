#!/usr/bin/env bash
set -u
printf '%s\n' 'Plan: read DGX OS, driver/GPU, RoCE links and GID slots, available memory, free disk and head-to-worker SSH on all three Sparks; change nothing.'
# shellcheck source-path=SCRIPTDIR
# shellcheck source=_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh" || exit 1
run_recipe preflight
