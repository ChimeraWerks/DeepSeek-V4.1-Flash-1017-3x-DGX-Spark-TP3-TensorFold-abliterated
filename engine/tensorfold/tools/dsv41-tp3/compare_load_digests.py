#!/usr/bin/env python3
"""Compare tf-dsv41-load --digests files from the same rank and weight pack."""
import argparse
import json
from pathlib import Path
import re

POINTER = re.compile(r'(?:layers|mtp)\.\d+\.ffn\.experts\.ptr[012]\Z')
HASH = re.compile(r'[0-9a-f]{64}\Z')


def read(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        name = row['name']
        if not isinstance(name, str) or name in rows:
            raise ValueError('Duplicate or invalid tensor name')
        if type(row['bytes']) is not int or row['bytes'] < 0:
            raise ValueError('Invalid tensor byte count: ' + name)
        if not isinstance(row['sha256'], str) or not HASH.fullmatch(row['sha256']):
            raise ValueError('Invalid tensor SHA-256: ' + name)
        rows[name] = row
    if not rows:
        raise ValueError('Empty digest file')
    return rows


def compare(before, after, strict_pointers=False):
    if before.keys() != after.keys():
        raise ValueError('Tensor names differ')
    checked = skipped = 0
    for name, a in before.items():
        b = after[name]
        if a['bytes'] != b['bytes']:
            raise ValueError('Tensor size differs: ' + name)
        if POINTER.fullmatch(name) and not strict_pointers:
            skipped += 1
            continue
        if a['sha256'] != b['sha256']:
            raise ValueError('Tensor bytes differ: ' + name)
        checked += 1
    if checked == 0:
        raise ValueError('No non-pointer tensors compared')
    return {'matched_tensors': checked, 'address_tables_excluded': skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('--strict-pointers', action='store_true',
                        help='Also compare absolute CUDA pointer values (normally differ across processes)')
    args = parser.parse_args()
    result = compare(read(args.before), read(args.after), args.strict_pointers)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
