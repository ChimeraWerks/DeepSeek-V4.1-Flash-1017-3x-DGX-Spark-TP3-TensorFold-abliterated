#!/usr/bin/env bash
# CPU-only aarch64 cross-build. See ../README.md for external prerequisites.
# W, ZIG, FATBINS, OUT and C may be absolute paths or relative to the caller.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
W=${W:-"$ROOT/engine/tensorfold"}
ZIG=${ZIG:-"$ROOT/engine/build/toolchains/zig/zig"}
FATBINS=${FATBINS:-"$ROOT/engine/build/fatbins-samplecand"}
OUT=${OUT:-"$ROOT/engine/build/out"}
C=${C:-"${XDG_CACHE_HOME:-$HOME/.cache}/tensorfold-zig/$(tr -d '\r\n' < "$W/.zig-version" 2>/dev/null || echo default)"}
JOBS=${JOBS:-8}
# KEYED_DRAFT: true embeds the isolated dsv41_keyed.fatbin; default false.
#              Runtime still requires TF_DS_KEYED_DRAFT=1 on every rank.
KEYED_DRAFT=${KEYED_DRAFT:-false}
fail() { printf 'FAIL: build: %s\n' "$*" >&2; exit 1; }
[[ "$KEYED_DRAFT" == true || "$KEYED_DRAFT" == false ]] || fail 'KEYED_DRAFT must be true or false'
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail 'JOBS must be a positive integer'
for tool in realpath sha256sum python3 nice ionice; do
  command -v "$tool" >/dev/null || fail "missing prerequisite: $tool"
done
W=$(realpath -m -- "$W")
FATBINS=$(realpath -m -- "$FATBINS")
OUT=$(realpath -m -- "$OUT")
C=$(realpath -m -- "$C")
ZIG=$(command -v "$ZIG") || fail 'set ZIG to the production-compatible compiler'
ZIG=$(realpath -- "$ZIG")
[[ -f "$W/build.zig" && -f "$W/.zig-version" ]] || fail "TensorFold tree is missing: $W"
[[ -s "$FATBINS/dsv41_sample_cand.fatbin" ]] || fail "sample-candidate fatbin is missing: $FATBINS"
required=$(tr -d '\r\n' < "$W/.zig-version")
actual=$("$ZIG" version)
[[ "$actual" == "$required" ]] || fail "Zig $required required; found $actual"
compiler_sha=$(sha256sum "$ZIG")
compiler_sha=${compiler_sha%% *}
if [[ -n "${EXPECTED_ZIG_SHA256:-}" ]]; then
  [[ "$compiler_sha" == "$EXPECTED_ZIG_SHA256" ]] || fail 'compiler SHA-256 mismatch'
fi
# Optional trusted manifest: resolve it before changing directories.
if [[ -n "${FATBINS_SHA256:-}" ]]; then
  manifest=$(realpath -- "$FATBINS_SHA256")
  (cd -- "$FATBINS" && sha256sum --strict -c "$manifest")
fi
mkdir -p -- "$OUT" "$C"
printf 'Plan: native + dsv41-tools; aarch64-linux-gnu; sample_cand=true; jobs=%s\n' "$JOBS"
printf 'Zig: %s (%s) SHA-256 %s\nOutput: %s\n' "$ZIG" "$actual" "$compiler_sha" "$OUT"
# A receipt is diagnostic provenance, not a promise of byte-identical rebuilds.
# Record all source files in zig/, the root build files and every supplied fatbin.
python3 - "$W" "$ZIG" "$FATBINS" "$OUT" "$C" "$JOBS" "$actual" <<'PY'
import hashlib, json, pathlib, sys
w, zig, fatbins, out, cache = map(pathlib.Path, sys.argv[1:6])
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
paths = sorted(p for p in (w / 'zig').rglob('*') if p.is_file())
paths += [p for p in (w / 'build.zig', w / 'build.zig.zon', w / '.zig-version') if p.is_file()]
source = {str(p.relative_to(w)): sha(p) for p in sorted(paths)}
receipt = {
    'schema': 1, 'status': 'started', 'source_root': str(w),
    'source_files_sha256': source,
    'source_manifest_sha256': hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
    'compiler': {'path': str(zig), 'version': sys.argv[7], 'executable_sha256': sha(zig)},
    'fatbins_root': str(fatbins),
    'fatbins_sha256': {p.name: sha(p) for p in sorted(fatbins.glob('*.fatbin'))},
    'cache_root': str(cache), 'output_root': str(out),
    'argv': ['build', 'native', 'dsv41-tools', '-j' + sys.argv[6],
             '-Dtarget=aarch64-linux-gnu', '-Dfatbins=' + str(fatbins),
             '-Dsample_cand=true', '--prefix', str(out), '--summary', 'all'],
    'runtime_assets_included': False,
}
(out / 'BUILD_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
PY
cd -- "$W"
# Preserve the measured build modes selected by the vendored build system:
# native safe; dsv41-tools fast. No CUDA compiler or GPU is invoked.
if ZIG_LOCAL_CACHE_DIR="$C/xlocal" ZIG_GLOBAL_CACHE_DIR="$C/global" nice -n 10 ionice -c3 \
  "$ZIG" build native dsv41-tools "-j$JOBS" -Dtarget=aarch64-linux-gnu \
  -Dfatbins="$FATBINS" -Dsample_cand=true -Dkeyed_draft="$KEYED_DRAFT" --prefix "$OUT" --summary all; then
  (cd -- "$OUT" && sha256sum native/bin/tensorfold-native bin/tf-dsv41-lanes bin/tf-dsv41-check > SHA256SUMS)
  python3 - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
p = out / 'BUILD_RECEIPT.json'
receipt = json.loads(p.read_text())
receipt['outputs_sha256'] = {line.split('  ', 1)[1]: line.split('  ', 1)[0]
                            for line in (out / 'SHA256SUMS').read_text().splitlines()}
receipt['status'] = 'succeeded'
p.write_text(json.dumps(receipt, indent=2) + '\n')
PY
  cat "$OUT/SHA256SUMS"
else
  fail 'Zig build failed; BUILD_RECEIPT.json remains status=started (not a successful build)'
fi
