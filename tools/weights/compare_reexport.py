"""Re-export one rank at a time and compare complete TFDSRK01 artifact SHA256s.

Uses the real rank_cache.py --write, including its immutable-source receipt
checks. Without --source-receipt, hash the needed local shards once to create a
fresh verification receipt in scratch. No network, GPU, or source writes.
Scratch needs room for ONE rank; .partial is a hardlink, not a second copy.
All scratch rank files, hardlinks and receipts are removed, including on error.

Example (one or several ranks, paired in argument order):
  python tools/weights/compare_reexport.py --plan EXPORT/plan.json \
    --keys rank0.keys rank1.keys rank2.keys \
    --rank-receipts rank0.receipt.json rank1.receipt.json rank2.receipt.json \
    --scratch-dir /path/with/room

Alternatively use --sha256 SHA [SHA ...]. Exits 1 if any rank mismatches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Under python -I the script's own folder is not on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unshard_ranks as unshard  # noqa: E402
from source_verify import local_receipt  # noqa: E402


def file_sha256(path):
    with Path(path).open('rb') as file:
        digest = hashlib.sha256()
        while data := file.read(8 << 20):
            digest.update(data)
        return digest.hexdigest()


def expected_digest(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', value):
        raise ValueError('Expected a complete SHA256 digest')
    return value.lower()


def local_source_receipt(root, source, shards):
    return local_receipt(root, source, shards)


def compare(*, plan_path, key_paths, scratch_dir, sha256s=None, rank_receipts=None,
            source_receipt=None):
    if not key_paths or (sha256s is None) == (rank_receipts is None):
        raise ValueError('Supply keys and exactly one of sha256s or rank_receipts')
    expected = []
    if rank_receipts is not None:
        if len(rank_receipts) != len(key_paths):
            raise ValueError('Pair one production receipt with each keys file')
        for path, keys_path in zip(rank_receipts, key_paths):
            receipt = unshard.read_json(path)
            keys_raw = Path(keys_path).read_bytes()
            if ('keys_sha256' in receipt and
                    receipt['keys_sha256'] != hashlib.sha256(keys_raw).hexdigest()):
                raise ValueError('Production receipt/keys SHA256 differs: ' + str(keys_path))
            if 'entries' in receipt and receipt['entries'] != len(keys_raw.decode().splitlines()):
                raise ValueError('Production receipt/keys count differs: ' + str(keys_path))
            expected.append(expected_digest(receipt['artifact_sha256']))
    else:
        if len(sha256s) != len(key_paths):
            raise ValueError('Pair one SHA256 with each keys file')
        expected = [expected_digest(value) for value in sha256s]
    plan_path = Path(plan_path).resolve()
    root = plan_path.parent
    plan = unshard.read_json(plan_path)
    if plan.get('schema') != unshard.PLAN_SCHEMA:
        raise ValueError('Unknown checkpoint plan schema')
    if any(t['source'] != 'mia' for t in plan['tensors'].values()):
        raise ValueError('Re-export expects a standalone all-mia plan')
    needed = set()
    for keys in key_paths:
        lines = Path(keys).read_text().splitlines()
        if len(set(lines)) != len(lines):
            raise ValueError('Duplicate native keys: ' + str(keys))
        for key in lines:
            needed.add(unshard.rank_cache.layout(key, plan['tensors'])['spec']['shard'])
    scratch_dir = Path(scratch_dir).resolve()
    if not scratch_dir.is_dir():
        raise FileNotFoundError(scratch_dir)
    script = unshard.TP3_TOOLS / 'rank_cache.py'
    results = []
    with tempfile.TemporaryDirectory(prefix='compare-reexport-', dir=scratch_dir) as temporary:
        work = Path(temporary)
        if source_receipt is None:
            receipt = local_source_receipt(root, plan['sources']['mia'], needed)
            source_receipt = work / 'source.receipt.json'
            unshard.write_json(source_receipt, receipt)
        else:
            source_receipt = Path(source_receipt).resolve()
        for keys, expected_sha in zip(key_paths, expected):
            # Separate directory per rank; reclaim it before starting the next.
            with tempfile.TemporaryDirectory(prefix='rank-', dir=work) as rank_work:
                artifact = Path(rank_work) / 'rank.bin'
                subprocess.run([sys.executable, str(script), '--plan', str(plan_path),
                                '--keys', str(Path(keys).resolve()), '--mia', str(root),
                                '--pollard', str(root), '--out', str(artifact),
                                '--source-receipt', 'mia=' + str(source_receipt), '--write'],
                               check=True)
                actual_sha = file_sha256(artifact)
                result = {'rank': Path(keys).stem, 'keys': str(Path(keys).resolve()),
                          'status': 'MATCH' if actual_sha == expected_sha else 'MISMATCH',
                          'expected_sha256': expected_sha, 'actual_sha256': actual_sha}
                results.append(result)
                print(f'{result["rank"]}: {result["status"]} {actual_sha}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--keys', nargs='+', required=True, type=Path)
    expected = parser.add_mutually_exclusive_group(required=True)
    expected.add_argument('--rank-receipts', nargs='+', type=Path)
    expected.add_argument('--sha256', nargs='+')
    parser.add_argument('--scratch-dir', required=True, type=Path)
    parser.add_argument('--source-receipt', type=Path,
                        help='Existing immutable-source receipt; otherwise hash local source shards')
    args = parser.parse_args()
    results = compare(plan_path=args.plan, key_paths=args.keys, scratch_dir=args.scratch_dir,
                      sha256s=args.sha256, rank_receipts=args.rank_receipts,
                      source_receipt=args.source_receipt)
    print(json.dumps(results, indent=2))
    return 0 if all(row['status'] == 'MATCH' for row in results) else 1


if __name__ == '__main__':
    sys.exit(main())
