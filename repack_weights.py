#!/usr/bin/env python3
"""Repack the Q4_0 attention weights into 256-output-channel column-SoA tiles.

No third-party dependencies. Does not dequantize the 4-bit weights.
Only the original FP16 *scale* is expanded losslessly to FP32, so its value
is bitwise identical when used by the existing FP32 dot-product kernel.

All three files are concatenations of tiles of 245760 bytes (240 KiB):
  for each input block of 32 positions in the tile:
    32 packed columns x 128 bytes; low nibble = outputs 0..127,
                                    high nibble = outputs 128..255
    256 FP32 scales (1024 bytes), ordered by output number.
  i.e. 32*128 + 256*4 = 5120 bytes per quantization group.

q.colsoa: 6 tiles, each covers two Q heads (256 rows) and all 1536 inputs.
kv.colsoa: 2 tiles, K[128] in lower half and V[128] in upper half.
o.colsoa: 6 tiles, each covers 256 input columns of W_O, with six
          256-output-row subtiles x 8 quantization groups.

Data is not a GGUF and has no header; the C code and this script agree on
fixed dimensions. manifest_colsoa.json records SHA-256 hashes and layout.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

HIDDEN, HDIM, QK = 1536, 128, 32
BLOCK_BYTES, PACKED_COL_BYTES = 18, 128
GROUP_BYTES = 32 * PACKED_COL_BYTES + 256 * 4  # 5120
TILE_BYTES = 48 * GROUP_BYTES                    # 245760


def read_q4(path: Path, rows: int, cols: int):
    data = path.read_bytes()
    expected = rows * cols // QK * BLOCK_BYTES
    if len(data) != expected:
        raise ValueError(f'{path}: {len(data)} bytes; expected {expected} bytes')
    return data


def nibble(src: bytes, cols: int, row: int, input_col: int):
    block_pos = (row * (cols // QK) + input_col // QK) * BLOCK_BYTES
    k = input_col % QK
    byte = src[block_pos + 2 + k % 16]
    return (byte & 0x0f) if k < 16 else (byte >> 4)


def scale_f32(src: bytes, cols: int, row: int, b: int):
    offset = (row * (cols // QK) + b) * BLOCK_BYTES
    return struct.unpack_from('<e', src, offset)[0]


def pack_tile(out, low_src, high_src, src_cols, low_base, high_base,
              input_base, nblocks):
    # Output-order tiles. Raw bit patterns are kept unchanged for all nibbles.
    for b in range(nblocks):
        group = bytearray(GROUP_BYTES)
        input_block = input_base + b * QK
        for k in range(QK):
            col = input_block + k
            p = k * PACKED_COL_BYTES
            for j in range(HDIM):
                lo = nibble(low_src, src_cols, low_base + j, col)
                hi = nibble(high_src, src_cols, high_base + j, col)
                group[p + j] = lo | (hi << 4)
        for j in range(HDIM):
            struct.pack_into('<f', group, 4096 + 4 * j,
                             scale_f32(low_src, src_cols, low_base + j,
                                       input_block // QK))
            struct.pack_into('<f', group, 4096 + 4 * (j + HDIM),
                             scale_f32(high_src, src_cols, high_base + j,
                                       input_block // QK))
        out.write(group)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path, expected):
    size = path.stat().st_size
    if size != expected:
        raise ValueError(f'{path}: file size {size}; expected {expected}')


def repack(directory: Path):
    q = read_q4(directory / 'q.q4_0', HIDDEN, HIDDEN)
    k = read_q4(directory / 'k.q4_0', 256, HIDDEN)
    v = read_q4(directory / 'v.q4_0', 256, HIDDEN)
    o = read_q4(directory / 'o.q4_0', HIDDEN, HIDDEN)
    raw = {'q': q, 'k': k, 'v': v, 'o': o}
    out_files = {}

    qpath = directory / 'q.colsoa'
    with qpath.open('wb') as f:
        for pair in range(6):
            pack_tile(f, q, q, HIDDEN, pair * 256, pair * 256 + 128, 0, 48)
    verify_file(qpath, 6 * TILE_BYTES)
    out_files['q'] = qpath
    print(f'  Q: 6 two-head column-SoA tiles -> {qpath} ({qpath.stat().st_size:,} B)')

    kvpath = directory / 'kv.colsoa'
    with kvpath.open('wb') as f:
        for group in range(2):
            pack_tile(f, k, v, HIDDEN, group * HDIM, group * HDIM, 0, 48)
    verify_file(kvpath, 2 * TILE_BYTES)
    out_files['kv'] = kvpath
    print(f'  KV: 2 interleaved K/V group tiles -> {kvpath} ({kvpath.stat().st_size:,} B)')

    opath = directory / 'o.colsoa'
    with opath.open('wb') as f:
        for pair in range(6):
            for tile in range(6):
                pack_tile(f, o, o, HIDDEN, tile * 256, tile * 256 + 128,
                          pair * 256, 8)
    verify_file(opath, 6 * TILE_BYTES)
    out_files['o'] = opath
    print(f'  W_O: 6 input-head-pair tiles x 6 output tiles -> {opath} ({opath.stat().st_size:,} B)')

    manifest = {
        'format': 'Q4_0 column-SoA v1',
        'tile_bytes': TILE_BYTES,
        'group_bytes': GROUP_BYTES,
        'packed_column_bytes': PACKED_COL_BYTES,
        'scales': 'FP32 values exactly converted from original FP16 scales',
        'files': {name: {'file': path.name, 'bytes': path.stat().st_size,
                         'sha256': sha256(path)} for name, path in out_files.items()},
        'raw_q4_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
    }
    (directory / 'manifest_colsoa.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('  Manifest:', directory / 'manifest_colsoa.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, default=Path('weights'),
                   help='directory produced by extract_weights.py')
    args = p.parse_args()
    repack(args.weights)


if __name__ == '__main__':
    main()
