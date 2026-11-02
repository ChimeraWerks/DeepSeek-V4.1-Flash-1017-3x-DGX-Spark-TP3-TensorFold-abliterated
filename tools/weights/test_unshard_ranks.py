"""Small CPU-only round trips through the real native rank-cache exporter."""
from __future__ import annotations

import copy
import json
import struct
import subprocess
import sys
from math import prod
from pathlib import Path

import compare_reexport
import pytest
import unshard_ranks as unshard


def pattern(size, seed):
    block = bytes((i + seed) % 256 for i in range(256))
    return (block * ((size + 255) // 256))[:size]


def write_folder(root, tensors):
    root.mkdir()
    header, payload, at = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        assert len(data) == prod(shape) * unshard.SIZES[dtype]
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [at, at + len(data)]}
        at += len(data)
        payload.append(data)
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    shard = 'model-00001-of-00001.safetensors'
    (root / shard).write_bytes(struct.pack('<Q', len(raw)) + raw + b''.join(payload))
    plan = {
        'schema': unshard.PLAN_SCHEMA, 'arm': 'SYNTHETIC',
        'stage': 'PRE_PADDING_SOURCE_SELECTION', 'serving_ready': False,
        'sources': {'mia': {'repo': 'local-synthetic', 'revision': 'v1', 'headers_sha256': '0' * 64}},
        'stored_bytes_by_source': {'mia': at},
        'tensors': {name: dict(t, shard=shard, source='mia',
                              bytes=t['data_offsets'][1] - t['data_offsets'][0])
                    for name, t in header.items()},
    }
    (root / 'plan.json').write_text(json.dumps(plan))
    (root / 'model.safetensors.index.json').write_text(json.dumps({
        'metadata': {'total_size': at}, 'weight_map': {name: shard for name in header}}))
    return plan


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    tensors = {}
    modules = [
        ('dense', [8, 24, 80], 128, 384, 64),
        ('rows', [24, 8, 64], 384, 128, 64),
        ('layers.0.attn.wo_b', [64, 8, 80], 1024, 128, 48),
        ('layers.0.attn.wo_a.slice.0', [8, 8, 80], 128, 128, 32),
        ('layers.0.attn.wq_b', [8, 256, 80], 128, 4096, 64),
    ]
    for seed, (module, shape, rows, cols, _) in enumerate(modules, 1):
        tensors[module + '.suh'] = ('F16', [rows], pattern(rows * 2, seed))
        tensors[module + '.svh'] = ('F16', [cols], pattern(cols * 2, seed + 2))
        tensors[module + '.mul1'] = ('I32', [], unshard.MUL1)
        tensors[module + '.trellis'] = ('I16', shape, pattern(prod(shape) * 2, seed + 3))
    tensors['replicated.weight'] = ('BF16', [2, 2], pattern(8, 72))
    halves = [0., -0., 1.5, 65504., 2 ** -24, -2., float('inf'), -float('inf')]
    tensors['cast.weight'] = ('F16', [8], struct.pack('<8e', *halves))
    tensors['half.weight'] = ('F16', [2], struct.pack('<2e', -0., 0.5))
    tensors['scalar'] = ('F32', [], struct.pack('<f', 2.))
    tensors['empty'] = ('U8', [0], b'')
    tensors['vision.image.weight'] = ('U8', [7], b'vision!')
    template = tmp_path / 'template'
    template_plan = write_folder(template, tensors)
    (template / 'config.json').write_text(json.dumps({
        'quantization_config': {'avg_bits': 5., 'quant_method': 'exl3'}, 'unchanged': 'preserved'}))
    quant = {'bits': 5., 'codebook': 'mul1', 'tensor_storage': {}}
    for module, _, _, _, _ in modules:
        stored = {}
        for name, (dtype, shape, data) in tensors.items():
            if name.rsplit('.', 1)[0] == module:
                stored[name] = {'shape': shape, 'n_bytes': len(data),
                                'dtype': 'torch.' + unshard.TORCH_DTYPES[dtype], 'extra': 'retained'}
        quant['tensor_storage'][module] = {
            'stored_tensors': stored, 'quant_format': 'exl3',
            'bits_per_weight': tensors[module + '.trellis'][1][-1] / 16,
            'mul1_multiplier': 0x83DCD12D, 'other': 'preserved',
        }
    (template / 'quantization_config.json').write_text(json.dumps(quant))
    for name, value in [('tokenizer.json', '{}'), ('tokenizer_config.json', '{}'),
                        ('chat_template.jinja', '{{ messages }}'), ('LICENSE', 'synthetic'),
                        ('generation_config.json', '{}')]:
        (template / name).write_text(value)
    (template / 'encoding' / 'nested').mkdir(parents=True)
    (template / 'encoding' / 'nested' / 'tokens.bin').write_bytes(b'encoding')
    donor = copy.deepcopy(tensors)
    for seed, (module, _, _, _, width) in enumerate(modules, 1):
        shape = list(donor[module + '.trellis'][1])
        shape[-1] = width
        donor[module + '.trellis'] = ('I16', shape, pattern(prod(shape) * 2, seed + 111))
    donor_root = tmp_path / 'donor'
    donor_plan = write_folder(donor_root, donor)
    receipt = compare_reexport.local_source_receipt(
        donor_root, donor_plan['sources']['mia'], {t['shard'] for t in donor_plan['tensors'].values()})
    receipt_path = tmp_path / 'donor.receipt.json'
    receipt_path.write_text(json.dumps(receipt))
    rank_paths, key_paths, receipts = [], [], []
    for rank in range(3):
        keys = ['replicated.weight|None', 'cast.weight|torch.float32',
                'half.weight|torch.float16', 'scalar|None', 'empty|None', 'dense.mul1|None']
        row_ranges = [(0, 768), (768, 1408), (1408, 2048)]
        col_ranges = [(0, 2688), (2688, 5376), (5376, 8192)]
        for module, col, row in [
            ('dense', (rank * 128, (rank + 1) * 128), None),
            ('rows', None, (rank * 128, (rank + 1) * 128)),
            ('layers.0.attn.wo_b', None, row_ranges[rank]),
            ('layers.0.attn.wq_b', col_ranges[rank], None),
        ]:
            for part in ('tr', 'suh', 'svh'):
                keys.append(f'{module}|{col}|{row}|{part}')
        if rank in (0, 2):
            for part in ('tr', 'suh', 'svh'):
                keys.append(f'layers.0.attn.wo_a.slice.{0 if rank == 0 else 8}|None|None|{part}')
        key_path = tmp_path / f'rank{rank}.keys'
        key_path.write_text('\n'.join(keys) + '\n')
        rank_path = tmp_path / f'rank{rank}.bin'
        with monkeypatch.context() as patch:
            patch.setattr(sys, 'argv', ['rank_cache.py', '--plan', str(donor_root / 'plan.json'),
                                      '--keys', str(key_path), '--mia', str(donor_root),
                                      '--pollard', str(donor_root), '--out', str(rank_path),
                                      '--source-receipt', 'mia=' + str(receipt_path), '--write'])
            unshard.rank_cache.main()
        rank_paths.append(rank_path)
        key_paths.append(key_path)
        receipts.append(rank_path.with_suffix('.receipt.json'))
    return {'template': template, 'template_plan': template / 'plan.json',
            'rank_paths': rank_paths, 'key_paths': key_paths, 'avg_bits': 3.75,
            'tensors': tensors, 'donor': donor, 'receipts': receipts,
            'quant': quant, 'template_plan_data': template_plan}


def export(tiny, **kwargs):
    names = ('template', 'template_plan', 'rank_paths', 'key_paths', 'avg_bits')
    return unshard.export_model(**{n: tiny[n] for n in names}, **kwargs)


def read_tensors(root):
    plan = unshard.read_json(root / 'plan.json')
    result = {}
    for name, spec in plan['tensors'].items():
        with (root / spec['shard']).open('rb') as src:
            header, base = unshard.read_header(src)
            assert header[name] == {k: spec[k] for k in ('dtype', 'shape', 'data_offsets')}
            src.seek(base + spec['data_offsets'][0])
            result[name] = (spec['dtype'], spec['shape'], src.read(spec['bytes']))
    return plan, result


def alter_entry(path, key_substring, mutate):
    raw = bytearray(path.read_bytes())
    end = struct.unpack('<Q', raw[-16:-8])[0]
    index = json.loads(raw[end:-16])
    entry = next(e for e in index if key_substring in e[0])
    mutate(raw, entry)
    path.write_bytes(raw)


def test_round_trip_and_storage_metadata(tiny, tmp_path):
    out = tmp_path / 'export'
    summary = export(tiny, out=out, shard_size=100_000)
    plan, actual = read_tensors(out)
    expected = dict(tiny['donor'])
    assert actual == {n: (d, s, b) for n, (d, s, b) in expected.items()}
    assert list(plan['tensors']) == list(tiny['tensors'])
    assert all(t['source'] == 'mia' for t in plan['tensors'].values())
    assert set(plan['sources']) == {'mia'}
    assert plan['schema'] == tiny['template_plan_data']['schema']
    assert summary['tensors_rebuilt'] + summary['tensors_copied'] == len(actual)
    assert summary['tensors_copied'] == 5  # four unserved MUL1s, plus vision
    assert summary['payload_bytes'] == sum(len(t[2]) for t in actual.values())
    assert summary['shards'] > 1
    assert plan['stored_bytes_by_source'] == {'mia': summary['payload_bytes']}
    assert '48' in summary['trellis_bit_widths_by_source']['rank2']
    assert unshard.read_json(out / 'config.json') == {
        'quantization_config': {'avg_bits': 3.75, 'quant_method': 'exl3'}, 'unchanged': 'preserved'}
    quant = unshard.read_json(out / 'quantization_config.json')
    assert quant['bits'] == tiny['quant']['bits']  # preserve unrelated legacy globals
    for module, record in quant['tensor_storage'].items():
        assert record['bits_per_weight'] == actual[module + '.trellis'][1][-1] / 16
        assert record['mul1_multiplier'] == 0x83DCD12D
        assert record['other'] == 'preserved'
        for name, stored in record['stored_tensors'].items():
            assert stored['shape'] == actual[name][1]
            assert stored['n_bytes'] == len(actual[name][2])
            assert stored['dtype'] == 'torch.' + unshard.TORCH_DTYPES[actual[name][0]]
            assert stored['extra'] == 'retained'
    for asset in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'LICENSE',
                  'encoding/nested/tokens.bin', 'generation_config.json'):
        assert (out / asset).read_bytes() == (tiny['template'] / asset).read_bytes()
    assert not any(p.name.startswith('.export.partial-') for p in tmp_path.iterdir())


def test_check_only_verifies_without_creating_files(tiny, tmp_path):
    before = set(tmp_path.iterdir())
    result = export(tiny, check_only=True, out=tmp_path / 'never-created')
    assert result['status'] == 'CHECKED'
    assert set(tmp_path.iterdir()) == before


def test_reexport_real_rank_cache_matches_all_ranks_and_cleans_scratch(tiny, tmp_path):
    out = tmp_path / 'export'
    export(tiny, out=out)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    result = compare_reexport.compare(plan_path=out / 'plan.json', key_paths=tiny['key_paths'],
                                     scratch_dir=scratch, rank_receipts=tiny['receipts'])
    assert [r['status'] for r in result] == ['MATCH'] * 3
    assert list(scratch.iterdir()) == []


def test_reexport_mismatch_reports_and_cleans(tiny, tmp_path):
    out = tmp_path / 'export'
    export(tiny, out=out)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    result = compare_reexport.compare(plan_path=out / 'plan.json', key_paths=tiny['key_paths'][:1],
                                     scratch_dir=scratch, sha256s=['0' * 64])
    assert result[0]['status'] == 'MISMATCH'
    assert list(scratch.iterdir()) == []


def test_reexport_receipt_wrong_keys_rejected(tiny, tmp_path):
    out = tmp_path / 'export'
    export(tiny, out=out)
    with pytest.raises(ValueError, match='receipt/keys SHA256 differs'):
        compare_reexport.compare(plan_path=out / 'plan.json', key_paths=tiny['key_paths'][1:2],
                                 scratch_dir=tmp_path, rank_receipts=tiny['receipts'][:1])


def test_missing_byte_coverage_fails(tiny, tmp_path):
    # Remove one entire partial row entry, consistently from keys and index.
    path = tiny['rank_paths'][1]
    raw = path.read_bytes()
    end = struct.unpack('<Q', raw[-16:-8])[0]
    index = json.loads(raw[end:-16])
    removed = next(e for e in index if e[0].startswith('rows|') and e[0].endswith('|tr'))
    payload = raw[:removed[3]] + raw[removed[3] + removed[4]:end]
    new_index = []
    for row in index:
        if row == removed:
            continue
        if row[3] > removed[3]:
            row[3] -= removed[4]
        new_index.append(row)
    path.write_bytes(payload + json.dumps(new_index).encode() + struct.pack('<Q', len(payload)) + b'TFDSRK01')
    tiny['key_paths'][1].write_text('\n'.join(e[0] for e in new_index) + '\n')
    out = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='Missing byte coverage: rows.trellis'):
        export(tiny, out=out)
    assert not out.exists()
    assert not any(p.name.startswith('.invalid.partial-') for p in tmp_path.iterdir())


@pytest.mark.parametrize('entry', ['replicated.weight|', 'layers.0.attn.wo_a.slice.8|'])
def test_mismatching_replica_and_wo_a_copy_fail(tiny, entry):
    def corrupt(raw, row):
        raw[row[3]] ^= 1
    alter_entry(tiny['rank_paths'][2], entry, corrupt)
    with pytest.raises(ValueError, match='Overlapping/replicated bytes differ'):
        export(tiny, check_only=True)


@pytest.mark.parametrize(('entry', 'rule'), [
    ('layers.0.attn.wo_b|None|(1408, 2048)|tr', 'copy'),
    ('layers.0.attn.wo_b|None|(1408, 2048)|suh', 'zero'),
    ('layers.0.attn.wq_b|(5376, 8192)|None|tr', 'copy'),
    ('layers.0.attn.wq_b|(5376, 8192)|None|svh', 'copy'),
])
def test_bad_padding_fails(tiny, entry, rule):
    def corrupt(raw, row):
        # These rank2 entries begin inside the virtual padding.
        raw[row[3]] ^= 1
    alter_entry(tiny['rank_paths'][2], entry, corrupt)
    with pytest.raises(ValueError, match=f'Invalid {rule} padding'):
        export(tiny, check_only=True)


def test_float32_cast_must_be_exact(tiny):
    def corrupt(raw, row):
        struct.pack_into('<f', raw, row[3], 1.0001)
    alter_entry(tiny['rank_paths'][0], 'cast.weight|', corrupt)
    with pytest.raises(ValueError, match='Non-exact F32->F16 conversion'):
        export(tiny, check_only=True)


def test_mul1_constant_checked_even_when_unserved(tiny):
    plan = unshard.read_json(tiny['template_plan'])
    spec = plan['tensors']['rows.mul1']
    path = tiny['template'] / spec['shard']
    with path.open('r+b') as file:
        _, base = unshard.read_header(file)
        file.seek(base + spec['data_offsets'][0])
        file.write(b'bad!')
    with pytest.raises(ValueError, match='MUL1 codebook marker differs: rows.mul1'):
        export(tiny, check_only=True)


def test_output_must_not_exist(tiny, tmp_path):
    out = tmp_path / 'existing'
    out.mkdir()
    (out / 'keep').write_text('unchanged')
    with pytest.raises(FileExistsError):
        export(tiny, out=out)
    assert (out / 'keep').read_text() == 'unchanged'


def test_atomic_publication_does_not_replace_racing_empty_directory(tiny, tmp_path, monkeypatch):
    out = tmp_path / 'racing'
    publish = unshard.publish_directory
    def race(stage, destination):
        destination.mkdir()
        publish(stage, destination)
    monkeypatch.setattr(unshard, 'publish_directory', race)
    with pytest.raises(FileExistsError):
        export(tiny, out=out)
    assert out.is_dir() and list(out.iterdir()) == []
    assert not any(p.name.startswith('.racing.partial-') for p in tmp_path.iterdir())


def test_rank_footer_and_key_order_are_checked(tiny):
    raw = tiny['rank_paths'][0].read_bytes()
    tiny['rank_paths'][0].write_bytes(raw[:-1])
    with pytest.raises(ValueError, match='Invalid rank footer'):
        export(tiny, check_only=True)


def test_rank_index_shape_must_match_native_slice(tiny):
    path = tiny['rank_paths'][0]
    raw = path.read_bytes()
    end = struct.unpack('<Q', raw[-16:-8])[0]
    index = json.loads(raw[end:-16])
    row = next(e for e in index if e[0].startswith('dense|') and e[0].endswith('|tr'))
    row[2] = [4, 16, 64]  # same bytes, wrong rectangle
    path.write_bytes(raw[:end] + json.dumps(index).encode() + struct.pack('<Q', end) + b'TFDSRK01')
    with pytest.raises(ValueError, match='Rank index/layout differs'):
        export(tiny, check_only=True)


def test_exact_cast_preserves_signed_zero_and_canonical_nan():
    half = struct.pack('<5e', -0., 2 ** -24, float('inf'), -float('inf'), float('nan'))
    full = b''.join(struct.pack('<f', x[0]) for x in struct.iter_unpack('<e', half))
    assert unshard.narrow_f32(full, 'test') == half
    with pytest.raises(ValueError, match='Non-exact'):
        unshard.narrow_f32(struct.pack('<f', 1e30), 'overflow')


def test_command_line_check_only(tiny, tmp_path):
    args = [sys.executable, str(Path(unshard.__file__)), '--template', str(tiny['template']),
            '--template-plan', str(tiny['template_plan']), '--ranks',
            *map(str, tiny['rank_paths']), '--keys', *map(str, tiny['key_paths']),
            '--avg-bits', '3.75', '--check-only']
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)['status'] == 'CHECKED'
