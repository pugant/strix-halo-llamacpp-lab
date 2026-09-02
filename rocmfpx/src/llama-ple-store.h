// T25: disk-resident PLE n-gram table (qwen4exp per_layer_token_embd).
//
// The table is a read-only sparse gather target: build_ple needs, for each
// token, ple_n_heads rows indexed by the trigram hash. With --ple-disk the
// tensor is never loaded into memory; this store serves dequantized rows
// straight from the GGUF file through a fixed-size LRU cache of whole
// 128-row blocks kept in the on-disk (compressed) format.

#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

#define PLE_STORE_ROWS_PER_BLOCK 128

struct ple_store;

struct ple_store_stats {
    uint64_t hits        = 0; // row lookups served from the cache
    uint64_t misses      = 0; // row lookups that required a disk read
    uint64_t blocks_read = 0;
    uint64_t bytes_read  = 0;
    uint64_t fetch_us    = 0; // cumulative wall time inside ple_store_fetch
    uint64_t fetches     = 0; // number of ple_store_fetch calls
};

// path:     GGUF (or any file) containing the tensor data
// offset:   byte offset of the tensor inside path
// type:     on-disk ggml type (GGML_TYPE_Q5_1 or GGML_TYPE_Q2_0_ROCMFPX in v2)
// head_dim: elements per row (160 for the LEAN table)
// n_rows:   total rows (320'001'536 for the LEAN table)
// cache_bytes: LRU budget, allocated up front, never exceeded
//
// Returns nullptr on failure with a message in err.
ple_store * ple_store_open(const char * path, uint64_t offset,
                           enum ggml_type type, int64_t head_dim,
                           int64_t n_rows, uint64_t cache_bytes,
                           char * err, size_t errlen);

void ple_store_close(ple_store * s);

ple_store_stats ple_store_get_stats(const ple_store * s);

// Fills dst_host with the n requested rows, dequantized to F32, each
// head_dim elements long and written dst_stride_elems apart, preserving the
// order of rows[]. The only call site passes stride == head_dim.
// Thread-safe: concurrent fetches from different slots are fine.
bool ple_store_fetch(ple_store * s, const int32_t * rows, int64_t n,
                     float * dst_host, size_t dst_stride_elems,
                     char * err, size_t errlen);
