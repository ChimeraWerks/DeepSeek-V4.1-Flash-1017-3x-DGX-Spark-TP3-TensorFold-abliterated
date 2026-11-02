# Copyright (c) 2026 LakyFx. MIT; see LICENSES/LakyFx-MIT.txt.
# Portable launcher retained; the recipe verifies its artifact manifest before start.
"""Print or run one explicitly configured TP3 rank. Does not replace existing services."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent


def validate_rope_rows(rope, context, pool):
    # Match native pool.ropeRows(window, round.max_rows); existing files stay intact.
    window = min(context, pool)
    if rope['rows'] < window + 49:
        raise ValueError('RoPE tables too short for this conversation window')


def command(config, variant, rank, *, benchmark=False, validate_files=False):
    if variant not in {p.stem for p in (HERE/'profiles').glob('*.json')}:
        raise ValueError('Unknown launcher profile')
    if len(config['ranks']) != 3 or config['ranks'][rank].get('rank') != rank:
        raise ValueError('Exactly three explicitly ordered TP3 ranks required')
    profile = json.loads((HERE / 'profiles' / (variant + '.json')).read_text())
    node = config['ranks'][rank]
    root = Path(node['root'])
    binary_root = root / profile.get('binary_subdir', 'bin')
    kit = root / 'kit'
    model = root / profile['model_subdir']
    if rank == 0 and profile.get('head_model_subdir'):
        model = root / profile['head_model_subdir']
    cache = root / profile.get('rank_cache_subdir', 'weights/' + profile['arm'])
    engram = root / profile.get('engram_subdir', 'engram')
    pool = profile['pool_tokens']
    streams = str(profile['parallel'])
    rows = str(profile['round_rows'])
    devices = ','.join(node['rdma_devices'])
    if len(node['rdma_devices']) != 4 or len(node['peer_masks']) != 3 or node['peer_masks'][rank] != 0:
        raise ValueError('Four ordered RDMA devices and a three-entry rank-specific mask required')
    for k in ['master_address', 'nccl_interface', 'nccl_hcas', 'nccl_address_range']:
        if not config.get(k):
            raise ValueError('Operator setting required: ' + k)
    env = {'TF_DS_EXPERIMENTAL_TP3': '1', 'TF_DS_START_TIMEOUT': '1800', 'TF_TP_WORLD': '3',
           'TF_TP_MASTER': config['master_address'], 'TF_TP_PORT': str(config['master_port']),
           'TF_DS_KIT': str(kit), 'TENSORFOLD_CUDA_KERNELS': str(kit / 'aot'),
           'TF_DS_RANK_CACHE': str(cache), 'TF_DS_ENGRAM': str(engram),
           'TF_DS_TOKEN_MAP': str(kit / 'engram-token-map.json'), 'TF_DS_POOL_TOKENS': str(pool),
           'TF_DS_STREAMS': streams, 'TF_DS_ROUND_ROWS': rows, 'TF_DS_GRAPHS': '1',
           'TF_DS_HC_SIDE': '1', 'TF_DS_L2_PREFETCH': '1', 'TF_DS_ENGRAM_AIO': '1',
           'TF_DS_CACHE_SHORT_CONTINUATION': '1', 'TF_DS_TOOL_SEPARATOR_ROUNDTRIP': '1',
           'TF_RDMA_DEVICES': devices, 'TF_RDMA_GID_INDEX': str(node['gid_index']),
           'TF_RDMA_PEER_MASKS': ','.join(map(str, node['peer_masks'])), 'TF_RDMA_TRACE': '0',
           'NCCL_IB_SUBNET_AWARE_ROUTING': '1', 'NCCL_NVLS_ENABLE': '0',
           'NCCL_IB_ADDR_RANGE': config['nccl_address_range'], 'NCCL_IB_MERGE_NICS': '0',
           'NCCL_IB_ROCE_VERSION_NUM': '2', 'NCCL_IB_DISABLE': '0', 'NCCL_DEBUG': 'INFO',
           'NCCL_IB_ADDR_FAMILY': 'AF_INET', 'NCCL_IGNORE_CPU_AFFINITY': '1',
           'NCCL_IB_HCA': ','.join(config['nccl_hcas']), 'NCCL_NET_PLUGIN': 'none',
           'NCCL_CUMEM_ENABLE': '0', 'NCCL_NET': 'IB', 'NCCL_SOCKET_IFNAME': '=' + config['nccl_interface'],
           'TENSORFOLD_NO_UPDATE_CHECK': '1', 'HF_HUB_OFFLINE': '1'}
    if profile.get('round_ms'):
        env['TF_DS_ROUND_MS'] = profile['round_ms']
    if profile.get('adaptive_learning'):
        env['TF_DS_ADAPTIVE_LEARNING'] = '1'
    env.update({key: value.replace('${KIT_DIR}', str(kit))
                for key, value in profile.get('environment', {}).items()})
    env.update(TENSORFOLD_MEMORY_RESERVE_GIB=str(profile['memory_reserve_gib']),
               TF_DS_GRAPH_CAP=str(profile['graph_cap']),
               TF_DS_SPILL_DIR=str(root / 'spill'), TF_DS_SPILL_GIB=str(profile['spill_gib']))
    if os.environ.get('CHIMERA_RDMA_NAP') == '0':
        env['TF_RDMA_IDLE_S'] = '0'
    if rank == 0:
        env.update(profile.get('head_environment', {}))
    if config.get('library_path'):
        env['LD_LIBRARY_PATH'] = config['library_path']
    if rank == 0:
        argv = [str(binary_root / 'tensorfold-native'), 'serve', str(model), '--backend', 'cuda',
                '--host', '127.0.0.1' if benchmark else config['api_host'], '--port', str(config['api_port']),
                '--name', 'deepseek-v4.1-flash', '--context', str(profile['context_tokens']), '--parallel', streams,
                '--reasoning-effort', profile['reasoning'], '--prompt-cache-gib', str(profile['prompt_snapshot_gib_head']), '--no-update-check']
        if profile.get('loop_guard', False):
            argv.append('--loop-guard')
        argv += ['--spill-dir=' + env['TF_DS_SPILL_DIR'], '--spill-gib=' + env['TF_DS_SPILL_GIB'],
                 '--top-k=' + str(profile['top_k'])]
        if benchmark:
            argv.append('--benchmark-control')
    else:
        argv = [str(binary_root / 'tf-dsv41-lanes'), str(model), str(cache), str(rank), '3',
                config['master_address'], str(config['master_port']), str(kit), '--pool', str(pool),
                '--round-rows', rows, '--streams', streams, '--drafts', '1', '--graphs', '1',
                '--side', '1', '--prefetch', '1', '--aio', '1', '--graph-cap', str(profile['graph_cap']),
                '--engram', str(engram),
                '--token-map', str(kit / 'engram-token-map.json'), '--rdma', devices]
    if validate_files:
        # Keep build provenance and binary/template checks fail-closed. v1 has
        # no published digests yet, so the operator must pin the build receipt
        # as well as the artifacts, never reuse another pack's old manifest.
        import hashlib
        checksums = node.get('checksums', {})
        receipt_path = binary_root / 'BUILD.json'
        for path in [receipt_path, binary_root / 'tensorfold-native', binary_root / 'tf-dsv41-lanes',
                     model / 'chat_template.jinja']:
            expected = checksums.get(str(path), '')
            if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                raise ValueError('Require a verified v1 artifact checksum: ' + str(path))
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('v1 artifact differs from its verified checksum: ' + str(path))
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('schema') != 'ds41f.native-build.v1'
                or receipt.get('engine_commit') != profile['engine_commit']
                or receipt.get('adaptive_learning') != profile['adaptive_learning']):
            raise ValueError('Require the verified build receipt for the v1 engine commit')
        for binary in ['tensorfold-native', 'tf-dsv41-lanes']:
            if receipt.get('binaries', {}).get(binary, {}).get('sha256') != checksums[str(binary_root / binary)]:
                raise ValueError('v1 binary differs from its build receipt: ' + binary)
        operator_values = [config['master_address'], config['nccl_interface'], config['nccl_address_range'],
                           *config['nccl_hcas'], *node['rdma_devices']]
        if any('REPLACE_' in value for value in operator_values):
            raise ValueError('Replace every topology placeholder before starting a rank')
        address = ipaddress.ip_address(config['master_address'])
        if any(address in ipaddress.ip_network(cidr) for cidr in ['192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24']):
            raise ValueError('Replace the documentation example address with your actual head address')
        if (not profile['qualified_pool'] and not profile.get('serving_smoke_qualified')
                and not (profile.get('qualification_window') and os.environ.get('CHIMERA_QUALIFICATION_WINDOW') == '1')):
            raise ValueError('This profile has neither recorded pool nor serving smoke qualification')
        for path in [Path(argv[0]), model / 'config.json', model / 'vision.safetensors', cache,
                     kit / 'aot/aot.json', kit / 'engram-token-map.json',
                     engram / profile.get('engram_marker', 'model.safetensors.index.json')]:
            if not path.exists():
                raise FileNotFoundError(path)
        rope = json.loads((kit / 'rope.json').read_text())
        validate_rope_rows(rope, profile['context_tokens'], pool)
        for name in ['rope-plain-cos.f32', 'rope-plain-sin.f32', 'rope-compressed-cos.f32', 'rope-compressed-sin.f32']:
            if (kit / name).stat().st_size != rope['rows'] * rope['half'] * 4:
                raise ValueError('Incomplete RoPE table: ' + name)
        cfg = json.loads((model / 'config.json').read_text())
        validate_model_metadata(cfg, profile)
        if profile.get('source_revision'):
            if cfg['tensorfold_tp3']['original_revision'] != profile['source_revision']:
                raise ValueError('Wrong source model revision')
            source = json.loads((engram / 'SOURCE.json').read_text())
            if (source['rank'] != rank or source['world'] != 3
                    or source['original_revision'] != profile['source_revision']):
                raise ValueError('Wrong Engram rank or source revision')
            for name, metadata in source['files'].items():
                path = engram / name
                if path.parent != engram or path.stat().st_size != metadata['size']:
                    raise ValueError('Wrong Engram file or size: ' + name)
            original = cache / f'rank{rank}of3.bin'
            discovered = [p for p in cache.glob(f'rank{rank}of3-*.bin') if p.is_file() and not p.is_symlink()]
            if (original.is_symlink() or len(discovered) != 1
                    or not original.samefile(discovered[0])):
                raise ValueError('Require one regular rankNof3-*.bin hardlink to the published rank cache')
        elif not list(cache.glob('rank' + str(rank) + '*.bin')):
            raise FileNotFoundError('No exported binary for rank ' + str(rank) + ' in ' + str(cache))
    return argv, env


def validate_model_metadata(cfg, profile):
    # The next label is not an on-disk artifact migration. Bind to the exact
    # original metadata arm; do not accept either label or arbitrary aliases.
    expected_arm = profile.get('artifact_arm', profile['arm'])
    if cfg['tensorfold_tp3']['arm'] != expected_arm or cfg['tensorfold_tp3']['world_size'] != 3:
        raise ValueError('Wrong model metadata for rank cache/profile')


def execution_environment(env):
    """Refuse ambient capacity/identity settings absent from the reviewed plan."""
    unbound = sorted(key for key in os.environ
                     if key.startswith(("TF_DS_", "TF_SPILL_")) and key not in env)
    if unbound:
        raise ValueError("unbound inherited native settings: " + ", ".join(unbound))
    return {**os.environ, **env}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--variant', choices=sorted(p.stem for p in (HERE/'profiles').glob('*.json')), required=True)
    parser.add_argument('--rank', choices=range(3), type=int, required=True)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    argv, env = command(json.loads(args.config.read_text()), args.variant, args.rank,
                        benchmark=args.benchmark, validate_files=args.run)
    if args.run:
        os.execvpe(argv[0], argv, execution_environment(env))
    print(json.dumps({'variant': args.variant, 'rank': args.rank, 'argv': argv, 'environment': env,
                      'shell_command': 'env ' + ' '.join(shlex.quote(k + '=' + v) for k, v in env.items())
                      + ' ' + shlex.join(argv), 'started': False}, indent=2))


if __name__ == '__main__':
    main()
