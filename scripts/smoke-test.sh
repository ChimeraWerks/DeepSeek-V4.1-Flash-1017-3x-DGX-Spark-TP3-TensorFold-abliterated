#!/usr/bin/env bash
set -u
printf '%s\n' 'Plan: send one short chat completion through the head API and print tokens/s. Read its credential only on the head; no service, configuration or weight changes.'
# shellcheck source-path=SCRIPTDIR
# shellcheck source=_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh" || exit 1
run_recipe smoke-test
