#!/usr/bin/env bash
set -u
printf '%s\n' 'Plan: read rank/guard user units, warm engine health including numeric_effort, API health and available memory on each Spark; change nothing.'
# shellcheck source-path=SCRIPTDIR
# shellcheck source=_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh" || exit 1
run_recipe status
