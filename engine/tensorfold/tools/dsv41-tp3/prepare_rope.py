"""CPU RoPE extension, with bitwise verification of every existing table byte.

Uses the frozen upstream Python arithmetic (including its fixed block shapes).
Writes a NEW directory, never alters the kernel kit or any serving process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PIN = '8b1fe210d5e0d68e0cdb3b14755c135465c9ffa8'
OPS = 'src/tensorfold/families/deepseek_v41/ops.py'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'git-source', 'config', 'original-kit', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--pool', type=int, required=True)
    a = p.parse_args()
    if not 2048 <= a.pool < (1 << 24) - 48:
        raise ValueError('Pool outside exact FP32 position range')
    pinned = subprocess.check_output(['git', '-C', str(a.git_source), 'show', PIN + ':' + OPS])
    if (a.reference / OPS).read_bytes() != pinned:
        raise ValueError('Reference arithmetic differs from pinned source')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(a.reference / 'src'))
    import torch
    from tensorfold.families.deepseek_v41.ops import freqs_cis, rope_block, ROPE_COMPAT, ROPE_BLOCK
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cfg_raw = a.config.read_bytes()
    cfg = json.loads(cfg_raw)['text_config']
    yarn = cfg['rope_scaling']
    half = cfg['qk_rope_head_dim'] // 2
    old = json.loads((a.original_kit / 'rope.json').read_bytes())
    rows = ((a.pool + 49 + 4095) // 4096) * 4096
    if old['half'] != half or rows < old['rows']:
        raise ValueError('This tool extends matching existing RoPE tables only')
    a.out.mkdir(exist_ok=False)
    result = dict(rows=rows, half=half)
    for kind in ('plain', 'compressed'):
        params = (yarn['original_max_position_embeddings'] if kind == 'compressed' else 0,
                  cfg['compress_rope_theta'] if kind == 'compressed' else cfg['rope_theta'],
                  yarn['factor'], yarn['beta_fast'], yarn['beta_slow'])
        names = [f'rope-{kind}-{part}.f32' for part in ('cos', 'sin')]
        from contextlib import ExitStack
        with ExitStack() as stack:
            inputs = [stack.enter_context((a.original_kit / n).open('rb')) for n in names]
            outputs = [stack.enter_context((a.out / n).open('xb')) for n in names]
            old_hashes = [hashlib.sha256(), hashlib.sha256()]
            hashes = [hashlib.sha256(), hashlib.sha256()]
            for start in [0] + list(range(ROPE_COMPAT, rows, ROPE_BLOCK)):
                n = ROPE_COMPAT if start == 0 else ROPE_BLOCK
                f = (freqs_cis.__wrapped__(2 * half, n, *params, device='cpu') if start == 0
                     else rope_block(2 * half, start, n, *params))
                keep = min(n, rows - start)
                check = min(keep, max(0, old['rows'] - start)) * half * 4
                for j, part in enumerate((f.real, f.imag)):
                    data = part[:keep].contiguous().numpy().tobytes()
                    if check:
                        original = inputs[j].read(check)
                        if len(original) != check or original != data[:check]:
                            raise ValueError(f'RoPE reference differs: {names[j]} row {start}')
                        old_hashes[j].update(original)
                    outputs[j].write(data)
                    hashes[j].update(data)
                del f
            for j, name in enumerate(names):
                if inputs[j].read(1) or old_hashes[j].hexdigest() != old[name]:
                    raise ValueError('Original RoPE file or manifest differs: ' + name)
                outputs[j].flush()
                os.fsync(outputs[j].fileno())
                result[name] = hashes[j].hexdigest()
        print('CPU prefix bitwise verified: ' + kind, flush=True)
    with (a.out / 'rope.json').open('x') as f:
        json.dump(result, f, indent=2)
    receipt = dict(status='CPU_REFERENCE_MATCH_NOT_GPU_QUALIFIED', pool=a.pool, rows=rows,
                   prefix_rows_bitwise_verified=old['rows'], reference_commit=PIN,
                   ops_sha256=hashlib.sha256(pinned).hexdigest(),
                   config_sha256=hashlib.sha256(cfg_raw).hexdigest(),
                   torch_version=torch.__version__, gpu_used=False, outputs=result)
    with (a.out / 'RECEIPT.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
