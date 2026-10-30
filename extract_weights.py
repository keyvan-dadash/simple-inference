#!/usr/bin/env python3
"""Extract one Qwen2.5 attention sublayer from a single-file GGUF v2/v3.

No numpy, gguf-py, llama.cpp or other dependencies. This program copies
Q4_0 tensor bytes AS-IS; it does not dequantize or repack the weights.
"""
import argparse
import json
import math
import os
import struct
from pathlib import Path

# GGUF metadata value encoding: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
META_FORMATS = {
    0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
    7: 'B', 10: 'Q', 11: 'q', 12: 'd',
}
TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 8: 'Q8_0', 10: 'Q2_K',
         11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K'}


def exact(f, size):
    data = f.read(size)
    if len(data) != size:
        raise ValueError('Truncated GGUF file')
    return data


def number(f, fmt):
    return struct.unpack('<' + fmt, exact(f, struct.calcsize(fmt)))[0]


def gguf_string(f, wanted=True):
    n = number(f, 'Q')
    if n > 1 << 33:
        raise ValueError('Unreasonably long GGUF string')
    if wanted:
        return exact(f, n).decode('utf-8')
    f.seek(n, os.SEEK_CUR)
    return None


def value(f, typ, wanted=False, depth=0):
    if depth > 8:
        raise ValueError('GGUF metadata nesting too deep')
    if typ in META_FORMATS:
        fmt = META_FORMATS[typ]
        if wanted:
            return number(f, fmt)
        f.seek(struct.calcsize(fmt), os.SEEK_CUR)
        return None
    if typ == 8:
        return gguf_string(f, wanted)
    if typ == 9:
        inner = number(f, 'I')
        count = number(f, 'Q')
        if count > 100_000_000:
            raise ValueError('Unreasonably long GGUF array')
        # The tokenizer often contains large arrays. Skip them without allocating.
        if not wanted and inner in META_FORMATS:
            f.seek(count * struct.calcsize(META_FORMATS[inner]), os.SEEK_CUR)
            return None
        result = [] if wanted else None
        for _ in range(count):
            item = value(f, inner, wanted, depth + 1)
            if wanted:
                result.append(item)
        return result
    raise ValueError(f'Unsupported GGUF metadata type {typ}')


def align(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def read_index(f):
    if exact(f, 4) != b'GGUF':
        raise ValueError('Expected little-endian GGUF file (magic GGUF)')
    version = number(f, 'I')
    if version not in (2, 3):
        raise ValueError(f'Only GGUF v2/v3 supported; found v{version}')
    ntensors = number(f, 'Q')
    nmeta = number(f, 'Q')
    wanted_meta = {
        'general.architecture', 'general.alignment',
        'qwen2.embedding_length', 'qwen2.attention.head_count',
        'qwen2.attention.head_count_kv', 'qwen2.attention.key_length',
        'qwen2.attention.value_length', 'qwen2.rope.dimension_count',
        'qwen2.rope.freq_base', 'qwen2.attention.layer_norm_rms_epsilon',
    }
    meta = {}
    for _ in range(nmeta):
        key = gguf_string(f)
        typ = number(f, 'I')
        datum = value(f, typ, key in wanted_meta)
        if key in wanted_meta:
            meta[key] = datum
    tensors = {}
    for _ in range(ntensors):
        name = gguf_string(f)
        rank = number(f, 'I')
        if rank < 1 or rank > 8:
            raise ValueError(f'Unexpected rank {rank} for {name}')
        dims = [number(f, 'Q') for _ in range(rank)]
        typ = number(f, 'I')
        off = number(f, 'Q')
        if name in tensors:
            raise ValueError(f'Duplicate tensor {name}')
        tensors[name] = (dims, typ, off)
    multiple = int(meta.get('general.alignment', 32))
    if multiple < 8 or multiple % 8 != 0:
        raise ValueError(f'Unexpected GGUF alignment {multiple}')
    data_start = align(f.tell(), multiple)
    return meta, tensors, data_start


def tensor_size(dims, typ):
    n = math.prod(dims)
    if typ == 2:  # Q4_0
        if n % 32:
            raise ValueError(f'Q4_0 element count {n} not divisible by 32')
        return n // 32 * 18
    if typ == 0:
        return n * 4
    if typ == 1:
        return n * 2
    raise ValueError(f'Unknown tensor type {TYPES.get(typ, typ)}')


def extract(gguf, out, layer):
    names = {
        'q': (f'blk.{layer}.attn_q.weight', 2, [1536, 1536]),
        'k': (f'blk.{layer}.attn_k.weight', 2, [1536, 256]),
        'v': (f'blk.{layer}.attn_v.weight', 2, [1536, 256]),
        'o': (f'blk.{layer}.attn_output.weight', 2, [1536, 1536]),
        'norm': (f'blk.{layer}.attn_norm.weight', None, [1536]),
        'q_bias': (f'blk.{layer}.attn_q.bias', None, [1536]),
        'k_bias': (f'blk.{layer}.attn_k.bias', None, [256]),
        'v_bias': (f'blk.{layer}.attn_v.bias', None, [256]),
    }
    out.mkdir(parents=True, exist_ok=True)
    info = {'source': str(gguf), 'layer': layer, 'tensors': {}}
    with open(gguf, 'rb') as f:
        meta, tensors, data_start = read_index(f)
        info['metadata'] = meta
        if meta.get('general.architecture') not in ('qwen2', None):
            raise ValueError(f"Expected Qwen2 architecture, got {meta['general.architecture']}")
        checks = [('qwen2.embedding_length', 1536), ('qwen2.attention.head_count', 12),
                  ('qwen2.attention.head_count_kv', 2)]
        for key, expected in checks:
            if key in meta and meta[key] != expected:
                raise ValueError(f'{key}={meta[key]} (expected {expected}; this scaffold is for 1.5B)')
        # Validate every required tensor before creating partial output files.
        for short, (name, require_type, expected_dims) in names.items():
            if name not in tensors:
                if short in ('norm', 'q_bias', 'k_bias', 'v_bias'):
                    continue
                raise ValueError(f'Required tensor not found: {name}')
            dims, typ, offset = tensors[name]
            if dims != expected_dims:
                raise ValueError(f'{name} has dims {dims}, expected {expected_dims} (GGUF order)')
            if require_type is not None and typ != require_type:
                raise ValueError(f'{name} is {TYPES.get(typ, typ)}, not Q4_0. '
                                 'Use a Q4_0 GGUF; this extractor never silently dequantizes.')
            if require_type is None and typ not in (0, 1):
                raise ValueError(f'{name} must be F32/F16, got {TYPES.get(typ, typ)}')
            size = tensor_size(dims, typ)
            if offset < 0 or data_start + offset + size > os.fstat(f.fileno()).st_size:
                raise ValueError(f'{name} data range exceeds GGUF file')
        for short, (name, require_type, expected_dims) in names.items():
            suffix = '.q4_0' if require_type == 2 else '.f32'
            destination = out / (short + suffix)
            if name not in tensors:
                if destination.exists():
                    destination.unlink()  # Never accidentally reuse a previous extraction.
                print(f'  optional tensor missing: {name} (using identity norm / zero bias)')
                continue
            dims, typ, offset = tensors[name]
            size = tensor_size(dims, typ)
            f.seek(data_start + offset)
            destination = out / (short + ('.q4_0' if require_type == 2 else '.f32'))
            if require_type == 2 or typ == 0:
                with open(destination, 'wb') as dst:
                    remaining = size
                    while remaining:
                        buf = exact(f, min(1 << 20, remaining))
                        dst.write(buf)
                        remaining -= len(buf)
            else:
                # Small optional scale/bias tensor: normalize F16 to host FP32.
                raw = exact(f, size)
                with open(destination, 'wb') as dst:
                    for (half,) in struct.iter_unpack('<e', raw):
                        dst.write(struct.pack('<f', half))
            info['tensors'][short] = {'name': name, 'gguf_shape': dims,
                'gguf_type': TYPES.get(typ, str(typ)), 'file': destination.name,
                'bytes': destination.stat().st_size}
            print(f'  {name:<34} {TYPES.get(typ,typ):<6} {str(dims):<15} -> {destination}')
    with open(out / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
        f.write('\n')
    print(f'Extracted layer {layer} to {out}/; weights remain packed Q4_0.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gguf', type=Path, help='Local unsplit Q4_0 GGUF file')
    parser.add_argument('--layer', type=int, default=0, help='Layer number, default 0')
    parser.add_argument('--out', type=Path, default=Path('weights'), help='Output directory')
    parser.add_argument('--list', action='store_true', help='List this layer\'s tensors; do not extract')
    args = parser.parse_args()
    try:
        if args.list:
            with open(args.gguf, 'rb') as f:
                meta, tensors, _ = read_index(f)
            for name, (dims, typ, _) in tensors.items():
                if name.startswith(f'blk.{args.layer}.attn_'):
                    print(f'{name:<36} {TYPES.get(typ, typ):<8} {dims}')
            print('model metadata:', {k: v for k, v in meta.items() if k.startswith('qwen2.')})
        else:
            extract(args.gguf, args.out, args.layer)
    except (OSError, ValueError, struct.error) as exc:
        parser.exit(1, f'extract_weights.py: {exc}\n')


if __name__ == '__main__':
    main()
