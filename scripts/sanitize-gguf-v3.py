#!/usr/bin/env python3
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# sanitize-gguf-v3.py — KV-only sanitizer with a validated parser (2026-08-15).
# Copies tensor_info + tensor_data RAW; rewrites only header+KV; recomputes alignment padding.
# Usage: python3 sanitize-gguf-v3.py <input.gguf> <output.gguf> <config.json>
import json
import mmap
import os
import struct
import sys

FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: 'B', 10: 'Q', 11: 'q', 12: 'd'}
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def parse(path):
    data = mmap.mmap(os.open(path, os.O_RDONLY), 0, access=mmap.ACCESS_READ)
    off = 4  # magic
    version, = struct.unpack_from('<I', data, off); off += 4
    n_tensors, = struct.unpack_from('<Q', data, off); off += 8
    n_kv, = struct.unpack_from('<Q', data, off); off += 8

    def rstr():
        nonlocal off
        l, = struct.unpack_from('<Q', data, off); off += 8
        s = data[off:off + l].decode(errors='replace'); off += l
        return s

    def rt(t):
        nonlocal off
        if t == 8:
            return rstr()
        if t == 9:
            t2, = struct.unpack_from('<I', data, off); off += 4
            n, = struct.unpack_from('<Q', data, off); off += 8
            return (t2, [rt(t2) for _ in range(n)])
        v = struct.unpack_from('<' + FMT[t], data, off)[0]; off += SZ[t]
        return v

    kvs = []
    for _ in range(n_kv):
        k = rstr()
        t, = struct.unpack_from('<I', data, off); off += 4
        kvs.append([k, t, rt(t)])
    kv_end = off
    for _ in range(n_tensors):
        rstr()
        nd, = struct.unpack_from('<I', data, off); off += 4
        off += nd * 8 + 4 + 8
    ti_end = off
    alignment = 32
    for k, t, v in kvs:
        if k == 'general.alignment' and t in (4, 5, 10):
            alignment = v
    data_start = (ti_end + alignment - 1) // alignment * alignment
    return version, n_tensors, kvs, data[kv_end:ti_end], data_start, alignment, len(data)


def wstr(buf, s):
    b = s.encode()
    buf += struct.pack('<Q', len(b)) + b
    return buf


def wt(buf, t, v):
    if t == 8:
        return wstr(buf, v)
    if t == 9:
        t2, arr = v
        buf += struct.pack('<IQ', t2, len(arr))
        for x in arr:
            buf = wt(buf, t2, x)
        return buf
    return buf + struct.pack('<' + FMT[t], v)


def main(inp, outp, cfg_path):
    cfg = json.load(open(cfg_path))
    version, n_tensors, kvs, ti_raw, data_start, alignment, total = parse(inp)
    print(f"[i] orig: v{version} tensors={n_tensors} kv={len(kvs)} alignment={alignment} "
          f"ti_raw={len(ti_raw)}B data_start={data_start} size={total}")

    out = bytearray()
    out += b'GGUF' + struct.pack('<I', version) + struct.pack('<QQ', n_tensors, 0)  # n_kv placeholder
    kept = []
    for kv in kvs:
        k, t, v = kv
        if k in cfg.get('remove', []):
            print(f"[i] remove: {k}")
            continue
        if k in cfg.get('set', {}) and t == 8:
            kv[2] = cfg['set'][k]
            print(f"[i] set: {k}")
        kept.append(kv)
    for k, t, v in kept:
        out = wstr(out, k)
        out += struct.pack('<I', t)
        out = wt(out, t, v)
    out += ti_raw
    pad = (-len(out)) % alignment
    out += b'\x00' * pad
    struct.pack_into('<Q', out, 16, len(kept))  # n_kv: magic(4)+version(4)+n_tensors(8)

    with open(outp, 'wb') as fo:
        fo.write(out)
        with open(inp, 'rb') as fi:
            fi.seek(data_start)
            remaining = total - data_start
            while remaining:
                chunk = fi.read(min(1 << 24, remaining))
                fo.write(chunk)
                remaining -= len(chunk)
    print(f"[✓] written {outp} ({os.path.getsize(outp)} B, orig {total} B)")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
