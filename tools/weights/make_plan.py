#!/usr/bin/env python3
"""Build a rank-export plan from a standard, unpadded EXL3 folder's headers.

CPU-only and standard-library-only. Does not read or write ROOT/plan.json.
The shared unshard reader checks the index, descriptors and shard extents.
Local identities describe the header snapshot, not an upstream content digest;
prepare additionally hashes the source and checks the pinned rank artifact SHA.

Example:
  python3 -I tools/weights/make_plan.py --root EXL3 --out /scratch/plan.json
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

# Only trusted recipe code is added, never the downloaded model directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unshard_ranks as unshard


def build_plan(root, *, arm='EXL3', repo=None, revision=None):
    if (repo is None) != (revision is None):
        raise ValueError('Supply both repo and revision, or neither for a local header snapshot')
    with ExitStack() as stack:
        source = unshard.Template(root, None, stack)
        header_hash = hashlib.sha256(json.dumps(
            {'headers': source.headers}, separators=(',', ':')).encode()).hexdigest()
        tensors = {name: dict(spec, source='mia') for name, spec in source.tensors.items()}
        source.check_unchanged()
        return unshard.checkpoint_plan(
            tensors, arm=arm,
            source={'repo': repo if repo is not None else source.root.as_uri(),
                    'revision': revision if revision is not None else 'local-' + header_hash,
                    'headers_sha256': header_hash})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--arm', default='EXL3')
    parser.add_argument('--repo')
    parser.add_argument('--revision')
    args = parser.parse_args(argv)
    plan = build_plan(args.root, arm=args.arm, repo=args.repo, revision=args.revision)
    unshard.write_json(args.out, plan)
    print(json.dumps({'plan': str(args.out), 'tensors': len(plan['tensors']),
                      'payload_bytes': plan['stored_bytes_by_source']['mia']}))


if __name__ == '__main__':
    main()
