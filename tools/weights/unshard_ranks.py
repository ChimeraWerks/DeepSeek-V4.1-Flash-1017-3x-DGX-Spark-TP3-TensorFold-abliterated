"""Rebuild an unpadded, standard EXL3 folder from three native TP3 caches.

CPU-only, standard-library-only. Tensor payload memory is one padded tensor plus
bounded I/O/conversion chunks, never a checkpoint or a second logical tensor.
The template supplies ordering, unserved tensors, codebook markers and assets;
rank payloads, not the template, determine served trellis widths.

Example:
  python tools/weights/unshard_ranks.py --template TEMPLATE \
    --template-plan TEMPLATE_PLAN --ranks rank0.bin rank1.bin rank2.bin \
    --keys rank0.keys rank1.keys rank2.keys --avg-bits 3.75 --out EXPORT

--check-only performs the same payload/metadata verification without creating
any output files. Inputs must remain immutable for the duration of either run.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from math import isfinite, prod
from pathlib import Path

TP3_TOOLS = Path(__file__).resolve().parents[2] / 'engine/tensorfold/tools/dsv41-tp3'
sys.path.insert(0, str(TP3_TOOLS))
import rank_cache
from materialize_checkpoint import SIZES, read_header
from source_verify import fingerprint, safe_path
from tp3_layout import extra_name, new_shape, transformation

CHUNK = 1 << 20
MUL1 = struct.pack('<I', 0x83DCD12D)
TORCH_DTYPES = {**rank_cache.DTYPE, 'I8': 'int8', 'F8_E8M0': 'float8_e8m0fnu'}
RANK_SIZES = {v: SIZES[k] for k, v in TORCH_DTYPES.items()}
PLAN_SCHEMA = 'ds41f.tensorfold.checkpoint-plan.v1'


def read_json(path):
    return json.loads(Path(path).read_bytes())


def native_name(key):
    fields = key.split('|')
    if len(fields) == 2:
        name, cast = fields
        if cast not in ('None', 'torch.float16', 'torch.float32'):
            raise ValueError('Invalid cast: ' + key)
        return name
    if len(fields) == 4 and fields[3] in ('tr', 'suh', 'svh'):
        rank_cache.parse_range(fields[1])
        rank_cache.parse_range(fields[2])
        return fields[0] + '.' + {'tr': 'trellis', 'suh': 'suh', 'svh': 'svh'}[fields[3]]
    raise ValueError('Invalid native key: ' + key)


def validate_spec(name, spec):
    shape = spec['shape']
    if (spec['dtype'] not in SIZES or not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)):
        raise ValueError('Invalid tensor descriptor: ' + name)
    lo, hi = spec['data_offsets']
    if (type(lo) is not int or type(hi) is not int or lo < 0
            or hi - lo != prod(shape) * SIZES[spec['dtype']]):
        raise ValueError('Invalid tensor byte extent: ' + name)


class ImmutableFile:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open('rb')
        self.before = fingerprint(self.path)
        self.check_unchanged()

    def check_unchanged(self):
        st = os.fstat(self.file.fileno())
        opened = {'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'ctime_ns': st.st_ctime_ns,
                  'inode': st.st_ino, 'device': st.st_dev}
        if fingerprint(self.path) != self.before or opened != self.before:
            raise ValueError('Input changed during reconstruction: ' + str(self.path))

    def close(self):
        self.file.close()

    def read_at(self, offset, size):
        self.file.seek(offset)
        data = self.file.read(size)
        if len(data) != size:
            raise ValueError('Truncated payload: ' + str(self.path))
        return data


@dataclass
class Entry:
    rank: RankFile
    key: str
    dtype: str
    shape: list
    offset: int
    size: int


class RankFile(ImmutableFile):
    def __init__(self, path, keys, label):
        super().__init__(path)
        self.label = label
        try:
            size = self.before['size']
            if size < 16:
                raise ValueError('Truncated rank footer: ' + str(path))
            tail = self.read_at(size - 16, 16)
            if tail[8:] != b'TFDSRK01':
                raise ValueError('Invalid rank footer: ' + str(path))
            end = struct.unpack('<Q', tail[:8])[0]
            # Match rank_verify.py's existing footer/index-size guard; a bad
            # offset must not turn the complete rank payload into a JSON read.
            if end > size - 16 or size - 16 - end > 128 << 20:
                raise ValueError('Invalid rank index extent: ' + str(path))
            index = json.loads(self.read_at(end, size - 16 - end))
            keys = Path(keys).read_text().splitlines()
            if len(set(keys)) != len(keys):
                raise ValueError('Duplicate native keys: ' + label)
            if not isinstance(index, list) or len(index) != len(keys):
                raise ValueError('Native entry count differs: ' + label)
            self.entries = []
            at = 0
            for key, row in zip(keys, index):
                if not isinstance(row, list) or len(row) != 5 or row[0] != key:
                    raise ValueError('Rank index/key order differs: ' + label)
                _, dtype, shape, offset, n = row
                if (not isinstance(dtype, str) or dtype not in RANK_SIZES
                        or not isinstance(shape, list)
                        or any(type(x) is not int or x < 0 for x in shape)
                        or type(offset) is not int or type(n) is not int
                        or offset != at or n != prod(shape) * RANK_SIZES[dtype]
                        or offset + n > end):
                    raise ValueError('Invalid rank entry extent: ' + key)
                native_name(key)
                self.entries.append(Entry(self, key, dtype, shape, offset, n))
                at += n
            if at != end:
                raise ValueError('Unaccounted rank payload bytes: ' + label)
        except BaseException:
            self.close()
            raise


class Template:
    """Metadata and open handles only; copied payloads are chunked on demand."""
    def __init__(self, root, plan_path, stack):
        self.root = Path(root).resolve()
        self.plan = read_json(plan_path) if plan_path is not None else None
        if self.plan is not None and self.plan.get('schema') != PLAN_SCHEMA:
            raise ValueError('Unknown template plan schema')
        weight_map = read_json(self.root / 'model.safetensors.index.json')['weight_map']
        if self.plan is not None and set(weight_map) != set(self.plan['tensors']):
            raise ValueError('Template plan/index tensor names differ')
        self.files = {}
        headers = {}
        duplicates = []
        for shard in dict.fromkeys(weight_map.values()):
            src = ImmutableFile(safe_path(self.root, shard))
            stack.callback(src.close)
            header, base = read_header(src.file)
            self.files[shard] = src, base
            at = 0
            tensors = {n: t for n, t in header.items() if n != '__metadata__'}
            for name, spec in sorted(tensors.items(), key=lambda item: item[1]['data_offsets'][0]):
                validate_spec(name, spec)
                if spec['data_offsets'][0] != at:
                    raise ValueError('Template shard has gaps or overlaps: ' + shard)
                at = spec['data_offsets'][1]
                if weight_map.get(name) != shard:
                    # Some published folders store a few tensors twice (easthambuilds: 12 Engram tensors of
                    # layers 1 and 14 in two neighbouring shards). Accept the extra copy only if it is
                    # byte-identical to the indexed one; the export writes each tensor once.
                    duplicates.append((name, shard, spec))
            if base + at != src.before['size']:
                raise ValueError('Template shard byte extent differs: ' + shard)
            headers[shard] = tensors
        for name, shard, spec in duplicates:
            home = weight_map.get(name)
            if home not in headers or name not in headers[home]:
                raise ValueError('Template header/index differ: ' + name)
            kept = headers[home][name]
            if (spec['dtype'], spec['shape']) != (kept['dtype'], kept['shape']) or \
                    self._payload(shard, spec) != self._payload(home, kept):
                raise ValueError('Template duplicate differs from indexed copy: ' + name)
        # A downloaded standard folder needs no private plan. Reuse the same
        # header/index/extent validation, taking index order when no plan exists.
        self.headers = headers
        self.tensors = {}
        for name in self.plan['tensors'] if self.plan is not None else weight_map:
            shard = weight_map[name]
            if name not in headers[shard]:
                raise ValueError('Missing template tensor: ' + name)
            spec = headers[shard][name]
            self.tensors[name] = dict(spec, shard=shard,
                                      bytes=spec['data_offsets'][1] - spec['data_offsets'][0])

    def _payload(self, shard, spec):
        src, base = self.files[shard]
        start, end = spec['data_offsets']
        digest = hashlib.sha256()
        for at in range(start, end, CHUNK):
            digest.update(src.read_at(base + at, min(CHUNK, end - at)))
        return digest.digest()

    def chunks(self, name):
        spec = self.tensors[name]
        src, base = self.files[spec['shard']]
        for at in range(0, spec['bytes'], CHUNK):
            yield src.read_at(base + spec['data_offsets'][0] + at,
                              min(CHUNK, spec['bytes'] - at))

    def check_unchanged(self):
        for src, _ in self.files.values():
            src.check_unchanged()


class Coverage:
    """Sorted disjoint byte intervals; payload-size-independent bookkeeping."""
    def __init__(self):
        self.intervals = []

    def put(self, buffer, offset, data, name):
        end = offset + len(data)
        if not data:
            return
        i = bisect_left(self.intervals, (offset,))
        if i and self.intervals[i - 1][1] >= offset:
            i -= 1
        start, stop = offset, end
        j = i
        view = memoryview(buffer)
        while j < len(self.intervals) and self.intervals[j][0] <= stop:
            lo, hi = self.intervals[j]
            a, b = max(offset, lo), min(end, hi)
            if a < b and view[a:b] != memoryview(data)[a - offset:b - offset]:
                raise ValueError(f'Overlapping/replicated bytes differ: {name} at {a}')
            start, stop = min(start, lo), max(stop, hi)
            j += 1
        view[offset:end] = data
        self.intervals[i:j] = [(start, stop)]

    def covers(self, lo, hi):
        i = bisect_left(self.intervals, (lo + 1,)) - 1
        return i >= 0 and self.intervals[i][0] <= lo and hi <= self.intervals[i][1]

    def require_full(self, size, name):
        expected = [(0, size)] if size else []
        if self.intervals != expected:
            raise ValueError('Missing byte coverage: ' + name)


def widen_f16(data):
    """rank_cache's own F16->F32 conversion, for checking cast replicas against raw bytes."""
    return b''.join(struct.pack('<f', x[0]) for x in struct.iter_unpack('<e', data))


def widen(dtype, data):
    """rank_cache's exact widening of an F16 or BF16 source to a float32 rank entry."""
    if dtype == 'BF16':
        return b''.join(b'\x00\x00' + data[i:i + 2] for i in range(0, len(data), 2))
    return widen_f16(data)


def narrow(dtype, data, name):
    """Reverse widen(): F16 by exact float conversion, BF16 by taking the high half (the low half must be zero)."""
    if dtype == 'BF16':
        out = bytearray(len(data) // 2)
        for i in range(0, len(data), 4):
            if data[i:i + 2] != b'\x00\x00':
                raise ValueError('Non-exact F32->BF16 conversion: ' + name)
            out[i // 2:i // 2 + 2] = data[i + 2:i + 4]
        return out
    return narrow_f32(data, name)


def narrow_f32(data, name):
    """Reverse rank_cache's F16->F32 conversion, including signed-zero bits."""
    result = bytearray(len(data) // 2)
    for i, (value,) in enumerate(struct.iter_unpack('<f', data)):
        try:
            struct.pack_into('<e', result, i * 2, value)
            restored = struct.pack('<f', struct.unpack_from('<e', result, i * 2)[0])
        except (OverflowError, struct.error) as exc:
            raise ValueError('Non-exact F32->F16 conversion: ' + name) from exc
        if restored != data[i * 4:i * 4 + 4]:
            raise ValueError('Non-exact F32->F16 conversion: ' + name)
    return result


def prepare_tensors(template, ranks):
    grouped = defaultdict(list)
    histograms = defaultdict(Counter)
    for rank in ranks:
        for entry in rank.entries:
            name = native_name(entry.key)
            if name not in template.tensors:
                original = name.replace('.attn.wo_a.slice.8.', '.attn.wo_a.slice.0.')
                if original not in template.tensors or extra_name(original) != name:
                    raise ValueError('Rank tensor is not in template: ' + name)
                name = original
            grouped[name].append(entry)
            if name.endswith('.trellis'):
                if not entry.shape or entry.shape[-1] <= 0:
                    raise ValueError('Invalid rank trellis width: ' + name)
                histograms[rank.label][entry.shape[-1]] += 1
    specs = {}
    layouts = {}
    for name, original in template.tensors.items():
        spec = dict(original, shape=list(original['shape']))
        entries = grouped.get(name, [])
        if entries and name.endswith('.trellis'):
            widths = {e.shape[-1] for e in entries}
            if len(widths) != 1:
                raise ValueError('Inconsistent rank trellis widths: ' + name)
            spec['shape'][-1] = widths.pop()
        spec['bytes'] = prod(spec['shape']) * SIZES[spec['dtype']]
        specs[name] = spec
        for entry in entries:
            info = rank_cache.layout(entry.key, {name: spec})
            if (entry.dtype != info['dtype'] or entry.shape != info['outshape']
                    or entry.size != info['bytes']):
                raise ValueError('Rank index/layout differs: ' + entry.key)
            layouts[id(entry)] = info
        if not entries and name.endswith('.trellis'):
            histograms['template'][spec['shape'][-1]] += 1
    return specs, grouped, layouts, histograms


def logical_intervals(shape, itemsize, op):
    """Regions retained when dropping padding, in logical row-major order."""
    if op is None:
        yield 0, prod(shape) * itemsize
        return
    inner = prod(shape[op.axis + 1:]) * itemsize
    original, extra = shape[op.axis] * inner, op.count * inner
    for row in range(prod(shape[:op.axis])):
        yield row * (original + extra), original


def check_padding(buffer, shape, itemsize, op, name):
    if op is None:
        return
    inner = prod(shape[op.axis + 1:]) * itemsize
    original, extra = shape[op.axis] * inner, op.count * inner
    view = memoryview(buffer)
    for row in range(prod(shape[:op.axis])):
        start = row * (original + extra)
        for at in range(0, extra, CHUNK):
            n = min(CHUNK, extra - at)
            actual = view[start + original + at:start + original + at + n]
            good = not any(actual) if op.zero else actual == view[start + at:start + at + n]
            if not good:
                rule = 'zero' if op.zero else 'copy'
                raise ValueError(f'Invalid {rule} padding: {name}')


def rebuild_tensor(name, spec, entries, layouts):
    op = transformation(name)
    padded_shape = new_shape(spec['shape'], op) if op else spec['shape']
    itemsize = SIZES[spec['dtype']]
    size = prod(padded_shape) * itemsize
    buffer = bytearray(size)
    coverage = Coverage()
    def is_cast(entry):
        return spec['dtype'] in ('F16', 'BF16') and entry.dtype == 'float32'
    # Raw entries first: a cast replica of bytes already placed is checked by widening those raw bytes the way
    # rank_cache does, so F16 values that narrowing would canonicalize (NaN payloads) are kept bit for bit.
    for entry in sorted(entries, key=is_cast):
        info = layouts[id(entry)]
        at = entry.offset
        cast = is_cast(entry)
        for offset, n in rank_cache.selected_intervals(padded_shape, info['bounds'], itemsize):
            for done in range(0, n, CHUNK):
                take = min(CHUNK, n - done)
                raw = entry.rank.read_at(at, take * (2 if cast else 1))
                at += len(raw)
                lo = offset + done
                if cast and coverage.covers(lo, lo + take):
                    if widen(spec['dtype'], bytes(buffer[lo:lo + take])) != raw:
                        raise ValueError(f'Overlapping/replicated bytes differ: {name} at {lo}')
                    continue
                data = narrow(spec['dtype'], raw, name) if cast else raw
                coverage.put(buffer, lo, data, name)
        if at != entry.offset + entry.size:
            raise ValueError('Rank read byte accounting differs: ' + entry.key)
    coverage.require_full(size, name)
    check_padding(buffer, spec['shape'], itemsize, op, name)
    return buffer


def update_tensor_storage(quant, specs, originals):
    """EXL3 metadata: module -> stored_tensors -> {shape,n_bytes,dtype}.

    quant_config.py defines bits_per_weight as trellis.shape[-1] / 16.
    Preserve all unrelated fields, including the template's legacy global bits.
    """
    storage = quant['tensor_storage']
    if not isinstance(storage, dict):
        raise TypeError('Unknown tensor_storage schema')
    seen = set()
    for module, record in storage.items():
        stored = record['stored_tensors']
        if not isinstance(stored, dict):
            raise TypeError('Unknown stored_tensors schema: ' + module)
        for name, item in stored.items():
            if name not in specs:
                raise ValueError('tensor_storage names absent tensor: ' + name)
            spec = specs[name]
            item.update(shape=spec['shape'], n_bytes=spec['bytes'],
                        dtype='torch.' + TORCH_DTYPES[spec['dtype']])
            seen.add(name)
        if record.get('quant_format') == 'exl3':
            name = module + '.trellis'
            if name not in stored:
                raise ValueError('EXL3 module lacks stored trellis: ' + module)
            bits = specs[name]['shape'][-1] / 16
            record['bits_per_weight'] = int(bits) if bits.is_integer() else bits
    for name, spec in specs.items():
        if spec['shape'] != originals[name]['shape'] and name not in seen:
            raise ValueError('Changed tensor absent from tensor_storage: ' + name)
    return quant


def arrange_shards(specs, shard_size):
    groups = []
    current, size = [], 0
    for name, spec in specs.items():
        if current and size + spec['bytes'] > shard_size:
            groups.append(current)
            current, size = [], 0
        current.append(name)
        size += spec['bytes']
    if current:
        groups.append(current)
    result, headers = {}, {}
    for number, names in enumerate(groups, 1):
        shard = f'model-{number:05d}-of-{len(groups):05d}.safetensors'
        header, offset = {}, 0
        for name in names:
            spec = specs[name]
            header[name] = {'dtype': spec['dtype'], 'shape': spec['shape'],
                            'data_offsets': [offset, offset + spec['bytes']]}
            result[name] = dict(header[name], shard=shard, bytes=spec['bytes'], source='mia')
            offset += spec['bytes']
        headers[shard] = header
    return result, headers


def checkpoint_plan(tensors, *, arm, source):
    """The standalone, unpadded all-mia schema consumed by rank_cache.py."""
    return {
        'schema': PLAN_SCHEMA, 'arm': arm,
        'stage': 'PRE_PADDING_SOURCE_SELECTION', 'serving_ready': False,
        'sources': {'mia': source},
        'stored_bytes_by_source': {'mia': sum(t['bytes'] for t in tensors.values())},
        'tensors': tensors,
    }


def write_json(path, value):
    with Path(path).open('x') as dst:
        json.dump(value, dst, separators=(',', ':'))
        dst.flush()
        os.fsync(dst.fileno())


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_directory(stage, out):
    """Linux no-replace rename: atomic even if an empty target races us."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(out), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(out))
    fsync_directory(out.parent)


def export_model(*, template, template_plan, rank_paths, key_paths, avg_bits,
                 out=None, check_only=False, shard_size=8_000_000_000):
    if len(rank_paths) != 3 or len(key_paths) != 3:
        raise ValueError('Exactly three rank caches and key files are required')
    if not isfinite(avg_bits) or avg_bits <= 0 or shard_size <= 0:
        raise ValueError('avg_bits and shard_size must be positive and finite')
    if not check_only and out is None:
        raise ValueError('An output directory is required unless --check-only')
    out = Path(out).absolute() if out is not None else None
    if out is not None and os.path.lexists(out):
        raise FileExistsError(out)
    stage = None
    try:
        with ExitStack() as stack:
            source = Template(template, template_plan, stack)
            ranks = []
            for number, (path, keys) in enumerate(zip(rank_paths, key_paths)):
                rank = RankFile(path, keys, f'rank{number}')
                stack.callback(rank.close)
                ranks.append(rank)
            specs, grouped, layouts, histograms = prepare_tensors(source, ranks)
            config = read_json(source.root / 'config.json')
            config['quantization_config']['avg_bits'] = avg_bits
            quant = update_tensor_storage(read_json(source.root / 'quantization_config.json'),
                                          specs, source.tensors)
            if 'avg_bits' in quant:
                quant['avg_bits'] = avg_bits
            final_specs, headers = arrange_shards(specs, shard_size)
            # Check all mandatory assets even in check-only mode.
            assets = ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'LICENSE']
            for asset in assets:
                if not (source.root / asset).is_file():
                    raise FileNotFoundError(source.root / asset)
            if not (source.root / 'encoding').is_dir():
                raise FileNotFoundError(source.root / 'encoding')
            if not check_only:
                out.parent.mkdir(parents=True, exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix=f'.{out.name}.partial-', dir=out.parent))
            current_shard, dst = None, None
            rebuilt, copied, payload = 0, 0, 0
            payload_digest = hashlib.sha256()
            try:
                for name, spec in specs.items():
                    shard = final_specs[name]['shard']
                    if stage is not None and shard != current_shard:
                        if dst is not None:
                            dst.flush()
                            os.fsync(dst.fileno())
                            dst.close()
                        dst = (stage / shard).open('xb')
                        header = {'__metadata__': {'format': 'pt'}, **headers[shard]}
                        encoded = json.dumps(header, separators=(',', ':')).encode()
                        encoded += b' ' * (-len(encoded) % 8)
                        dst.write(struct.pack('<Q', len(encoded)))
                        dst.write(encoded)
                        current_shard = shard
                    if grouped.get(name):
                        buffer = rebuild_tensor(name, spec, grouped[name], layouts)
                        if name.endswith('.mul1'):
                            if spec['dtype'] != 'I32' or spec['shape'] != []:
                                raise ValueError('Invalid MUL1 marker descriptor: ' + name)
                            original = b''.join(source.chunks(name))
                            if original != MUL1 or buffer != original:
                                raise ValueError('MUL1 codebook marker differs: ' + name)
                        view = memoryview(buffer)
                        for offset, n in logical_intervals(spec['shape'], SIZES[spec['dtype']], transformation(name)):
                            for done in range(0, n, CHUNK):
                                chunk = view[offset + done:offset + min(n, done + CHUNK)]
                                payload_digest.update(chunk)
                                if dst is not None:
                                    dst.write(chunk)
                                del chunk
                        del view
                        del buffer
                        rebuilt += 1
                    else:
                        if name.endswith('.mul1'):
                            if spec['dtype'] != 'I32' or spec['shape'] != []:
                                raise ValueError('Invalid MUL1 marker descriptor: ' + name)
                            marker = b''.join(source.chunks(name))
                            if marker != MUL1:
                                raise ValueError('MUL1 codebook marker differs: ' + name)
                            payload_digest.update(marker)
                            if dst is not None:
                                dst.write(marker)
                        else:
                            for data in source.chunks(name):
                                payload_digest.update(data)
                                if dst is not None:
                                    dst.write(data)
                        copied += 1
                    payload += spec['bytes']
                if dst is not None:
                    dst.flush()
                    os.fsync(dst.fileno())
            finally:
                if dst is not None:
                    dst.close()
            source.check_unchanged()
            for rank in ranks:
                rank.check_unchanged()
            summary = {
                'status': 'CHECKED' if check_only else 'WRITTEN', 'tensors_rebuilt': rebuilt,
                'tensors_copied': copied, 'payload_bytes': payload, 'shards': len(headers),
                'logical_payload_sha256': payload_digest.hexdigest(),
                'trellis_bit_widths_by_source': {k: {str(w): n for w, n in sorted(v.items())}
                                                for k, v in histograms.items()},
                'histogram_note': 'Stored last-axis widths; bits_per_weight = width/16. '
                                  'Sources are rank caches or template, not unknown donor identities.',
            }
            if stage is not None:
                header_hash = hashlib.sha256(json.dumps({'headers': headers}, separators=(',', ':')).encode()).hexdigest()
                revision = hashlib.sha256((header_hash + payload_digest.hexdigest()).encode()).hexdigest()
                plan = checkpoint_plan(
                    final_specs, arm=source.plan['arm'] + '-UNSHARDED',
                    source={'repo': out.as_uri(), 'revision': 'local-' + revision,
                            'headers_sha256': header_hash})
                write_json(stage / 'plan.json', plan)
                write_json(stage / 'model.safetensors.index.json',
                           {'metadata': {'total_size': payload},
                            'weight_map': {n: t['shard'] for n, t in final_specs.items()}})
                write_json(stage / 'config.json', config)
                write_json(stage / 'quantization_config.json', quant)
                for asset in assets + ['special_tokens_map.json', 'generation_config.json']:
                    src = source.root / asset
                    if src.is_file():
                        shutil.copyfile(src, stage / asset)
                shutil.copytree(source.root / 'encoding', stage / 'encoding')
                for directory, _, files in os.walk(stage):
                    for filename in files:
                        with (Path(directory) / filename).open('rb') as file:
                            os.fsync(file.fileno())
                    fsync_directory(directory)
                source.check_unchanged()
                for rank in ranks:
                    rank.check_unchanged()
                publish_directory(stage, out)
                stage = None
            return summary
    finally:
        if stage is not None:
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--template', required=True, type=Path)
    parser.add_argument('--template-plan', required=True, type=Path)
    parser.add_argument('--ranks', nargs=3, required=True, type=Path)
    parser.add_argument('--keys', nargs=3, required=True, type=Path)
    parser.add_argument('--avg-bits', required=True, type=float)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--shard-size', type=int, default=8_000_000_000, help='Target payload bytes per shard')
    args = parser.parse_args()
    result = export_model(template=args.template, template_plan=args.template_plan,
                          rank_paths=args.ranks, key_paths=args.keys, avg_bits=args.avg_bits,
                          out=args.out, check_only=args.check_only, shard_size=args.shard_size)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
