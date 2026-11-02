"""EXL3-preserving physical TP3 layout, independent of CUDA.

Only whole 128-element Hadamard blocks are copied. The additional attention
group copies group zero; zero appended wo_b input scales suppress its output.
Padded output tokens MUST be masked by both target and draft sampling.
"""
from dataclasses import dataclass
from math import prod
import re

LOGICAL_VOCAB = 129280
PHYSICAL_VOCAB = 129408
HEADS, GROUPS, WORLD = 72, 9, 3
BLOCK = re.compile(r"(?:layers\.[0-9]+|mtp\.[0-9]+)")


@dataclass(frozen=True)
class Append:
    axis: int
    count: int
    zero: bool = False


def transformation(name):
    if name == 'head.trellis':
        return Append(1, 8)
    if name == 'head.svh':
        return Append(0, 128, True)
    if name in ('embed.weight', 'mtp.2.markov_head.embed.weight',
                'mtp.2.markov_head.head.weight'):
        return Append(0, 128, True)
    for prefix in ('layers', 'mtp'):
        if re.fullmatch(prefix + r'\.\d+\.attn\.wq_b\.trellis', name):
            return Append(1, 256)
        if re.fullmatch(prefix + r'\.\d+\.attn\.wq_b\.svh', name):
            return Append(0, 4096)
        if re.fullmatch(prefix + r'\.\d+\.attn\.attn_sink', name):
            return Append(0, 8)
        if re.fullmatch(prefix + r'\.\d+\.attn\.wo_b\.trellis', name):
            return Append(0, 64)
        if re.fullmatch(prefix + r'\.\d+\.attn\.wo_b\.suh', name):
            return Append(0, 1024, True)
    return None


def extra_name(name):
    if re.fullmatch(r'(layers|mtp)\.\d+\.attn\.wo_a\.slice\.0\.(trellis|suh|svh|mul1)', name):
        return name.replace('.slice.0.', '.slice.8.')
    return None


def new_shape(shape, op):
    result = list(shape)
    if not 0 <= op.axis < len(shape) or not 0 < op.count <= shape[op.axis]:
        raise ValueError('Invalid padding geometry')
    result[op.axis] += op.count
    return result


def spans(shape, itemsize, op):
    """Yield (source byte offset or None for zeros, byte count) in output order.

    This can stream large tensors without materializing them in RAM. Original
    data are retained bit for bit; appended copy uses the start of the same
    outer row. Offsets are relative to tensor payload, not safetensors file.
    """
    new_shape(shape, op)
    if itemsize <= 0:
        raise ValueError('Invalid item size')
    inner = prod(shape[op.axis + 1:]) * itemsize
    original = shape[op.axis] * inner
    extra = op.count * inner
    for row in range(prod(shape[:op.axis])):
        yield row * original, original
        yield None if op.zero else row * original, extra


def patch_config(config):
    import copy
    c = copy.deepcopy(config)
    t = c.get('text_config', c)
    expected = {'num_attention_heads': 64, 'o_groups': 8, 'vocab_size': LOGICAL_VOCAB}
    if any(t.get(k) != v for k, v in expected.items()):
        raise ValueError('Expected unpadded DeepSeek V4.1 Flash config')
    t.update(num_attention_heads=HEADS, o_groups=GROUPS, vocab_size=PHYSICAL_VOCAB)
    c['tensorfold_tp3'] = {
        'schema': 1, 'logical_vocab_size': LOGICAL_VOCAB,
        'logical_num_attention_heads': 64, 'logical_o_groups': 8,
        'world_size': WORLD, 'padding_group_source': 0,
        'requires_logical_vocab_mask': True,
    }
    return c
