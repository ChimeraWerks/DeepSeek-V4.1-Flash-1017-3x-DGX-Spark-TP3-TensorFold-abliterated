"""Downloaded-folder plans share the unshard reader and work without a private plan."""
import hashlib
import json
import subprocess
import sys

import pytest

import make_plan
import source_verify
from source_verify import check_receipt, local_receipt
from test_unshard_ranks import write_folder


def folder(tmp_path):
    root = tmp_path / 'standard'
    write_folder(root, {'weight': ('F16', [2], b'\x00\x3c\x00\x40'),
                        'empty': ('U8', [0], b''), 'scalar': ('I32', [], b'\0' * 4)})
    # A deliberately unusable private plan must never be consulted.
    (root / 'plan.json').write_text('not a public input')
    return root


def test_plan_from_headers_has_the_same_standalone_schema_and_offsets(tmp_path):
    root = folder(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    plan = make_plan.build_plan(root, arm='RELEASE', repo='example/export', revision='a' * 40)
    assert plan['schema'] == make_plan.unshard.PLAN_SCHEMA
    assert plan['stage'] == 'PRE_PADDING_SOURCE_SELECTION' and plan['serving_ready'] is False
    assert plan['arm'] == 'RELEASE' and plan['stored_bytes_by_source'] == {'mia': 8}
    assert plan['sources']['mia']['repo'] == 'example/export' and plan['sources']['mia']['revision'] == 'a' * 40
    spec = plan['tensors']['weight']
    assert spec == {'dtype': 'F16', 'shape': [2], 'data_offsets': [0, 4],
                    'shard': 'model-00001-of-00001.safetensors', 'bytes': 4, 'source': 'mia'}
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    assert make_plan.build_plan(root) == make_plan.build_plan(root)


def test_plan_cli_isolated_python_needs_no_model_code_or_plan(tmp_path):
    root = folder(tmp_path)
    # -I and trusted script paths must not execute modules planted in the model.
    (root / 'json.py').write_text("raise RuntimeError('model code imported')")
    out = tmp_path / 'fresh-plan.json'
    subprocess.run([sys.executable, '-I', str(make_plan.__file__), '--root', str(root),
                    '--out', str(out)], cwd=root, check=True, capture_output=True)
    plan = json.loads(out.read_bytes())
    assert plan['sources']['mia']['repo'] == root.as_uri()
    assert plan['sources']['mia']['revision'].startswith('local-')
    assert all(t['source'] == 'mia' for t in plan['tensors'].values())


def test_header_index_mismatch_uses_shared_validation(tmp_path):
    root = folder(tmp_path)
    path = root / 'model.safetensors.index.json'
    value = json.loads(path.read_bytes())
    value['weight_map']['missing'] = value['weight_map'].pop('weight')
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='header/index differ'):
        make_plan.build_plan(root)


def test_plan_cli_does_not_replace_an_existing_output(tmp_path):
    root = folder(tmp_path)
    out = tmp_path / 'plan.json'
    out.write_text('keep')
    with pytest.raises(FileExistsError):
        make_plan.main(['--root', str(root), '--out', str(out)])
    assert out.read_text() == 'keep'


def test_local_receipt_hashes_real_source_bytes_and_detects_mutation(tmp_path):
    root = folder(tmp_path)
    plan = make_plan.build_plan(root)
    shards = {t['shard'] for t in plan['tensors'].values()}
    receipt = local_receipt(root, plan['sources']['mia'], shards)
    assert receipt['verification'] == 'local-snapshot'
    for name in shards:
        assert receipt['files'][name]['digest'] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    check_receipt(receipt, root, plan['sources']['mia'], shards)
    shard = root / next(iter(shards))
    shard.write_bytes(shard.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Source changed since verification'):
        check_receipt(receipt, root, plan['sources']['mia'], shards)


def test_source_verify_keeps_the_pinned_manifest_mode(tmp_path, monkeypatch):
    root = folder(tmp_path)
    plan = make_plan.build_plan(root)
    shard = plan['tensors']['weight']['shard']
    raw = (root / shard).read_bytes()
    manifest = tmp_path / 'upstream.json'
    manifest.write_text(json.dumps({'id': 'example/export', 'sha': 'a' * 40,
                                   'siblings': [{'rfilename': shard, 'size': len(raw),
                                                 'lfs': {'sha256': hashlib.sha256(raw).hexdigest()}}]}))
    out = tmp_path / 'verified.json'
    monkeypatch.setattr(sys, 'argv', ['source_verify.py', '--manifest', str(manifest),
                                     '--root', str(root), '--out', str(out), '--weights-only'])
    source_verify.main()
    receipt = json.loads(out.read_bytes())
    assert receipt['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert 'verification' not in receipt
    assert receipt['files'][shard]['digest'] == hashlib.sha256(raw).hexdigest()
