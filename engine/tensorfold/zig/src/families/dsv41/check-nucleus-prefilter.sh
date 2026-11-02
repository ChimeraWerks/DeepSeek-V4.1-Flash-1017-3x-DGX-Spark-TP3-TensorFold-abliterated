#!/usr/bin/env bash
# Offline only. Requires the attested baseline commit and Zig 0.17.0.
set -euo pipefail
: "${ZIG:?Set ZIG to the existing Zig 0.17.0 executable}"
: "${SAMPLER_RESULTS:?Set SAMPLER_RESULTS to a durable output directory}"
repo="$(git rev-parse --show-toplevel)"
mkdir -p "$SAMPLER_RESULTS"
base=ddaff844200c6a9ad7db6e19717433226bd7ef15
git show "$base:zig/src/families/dsv41/sampling.zig" > "$SAMPLER_RESULTS/reference.zig"
cd "$repo"
nice -n 10 "$ZIG" test -O ReleaseSafe -lc --dep reference   -Mroot=zig/src/families/dsv41/sampling_prefilter_test.zig   "-Mreference=$SAMPLER_RESULTS/reference.zig" > "$SAMPLER_RESULTS/unit-test.log" 2>&1
nice -n 10 "$ZIG" build-exe -O ReleaseSafe -lc --dep reference   -Mroot=zig/src/families/dsv41/sampling_prefilter_bench.zig   "-Mreference=$SAMPLER_RESULTS/reference.zig" "-femit-bin=$SAMPLER_RESULTS/sampler-bench"
nice -n 10 "$SAMPLER_RESULTS/sampler-bench" > "$SAMPLER_RESULTS/microbench.csv"
