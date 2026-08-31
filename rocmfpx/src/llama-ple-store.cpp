#include "llama-ple-store.h"

#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Q5_1 dequantization, copied verbatim (same operation order) from
// ggml/src/ggml-quants.c dequantize_row_q5_1, with GGML_FP16_TO_FP32 replaced
// by the public ggml_fp16_to_fp32() (same conversion) and block fields
// spelled out locally. Validated bit-exact against CPU ggml_get_rows by
// tests/test-ple-store.cpp.
// ---------------------------------------------------------------------------

namespace {

struct block_q5_1_local {
    ggml_fp16_t d;
    ggml_fp16_t m;
    uint8_t     qh[4];
    uint8_t     qs[16];
};

static_assert(sizeof(block_q5_1_local) == 24, "block_q5_1 must be 24 bytes");

void dequant_row_q5_1(const uint8_t * row_bytes, float * dst, int64_t head_dim) {
    const int nb = head_dim / 32;
    for (int b = 0; b < nb; b++) {
        const block_q5_1_local & blk = *(const block_q5_1_local *) (row_bytes + b * sizeof(block_q5_1_local));

        const float d = ggml_fp16_to_fp32(blk.d);
        const float m = ggml_fp16_to_fp32(blk.m);

        uint32_t qh;
        memcpy(&qh, blk.qh, sizeof(qh));

        for (int j = 0; j < 32 / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int x0 = (blk.qs[j] & 0x0F) | xh_0;
            const int x1 = (blk.qs[j] >>   4) | xh_1;

            dst[b * 32 + j + 0 ] = x0 * d + m;
            dst[b * 32 + j + 16] = x1 * d + m;
        }
    }
}

// ---------------------------------------------------------------------------
// LRU cache of whole 128-row blocks, sharded; each shard owns a fixed byte
// budget allocated up front. Rows are kept in the on-disk compressed format.
// ---------------------------------------------------------------------------

constexpr int k_ple_shards = 16;

struct block_entry {
    std::vector<uint8_t> data; // block_bytes (last block may be shorter)
};

struct shard {
    std::mutex mutex;
    std::list<int64_t> lru;                                        // front = most recent
    std::unordered_map<int64_t, std::pair<std::list<int64_t>::iterator, block_entry>> map;
    uint64_t bytes_used = 0;
};

} // namespace

struct ple_store {
    int fd = -1;
    enum ggml_type type = GGML_TYPE_F32;
    int64_t head_dim = 0;
    int64_t n_rows = 0;
    uint64_t row_bytes = 0;
    uint64_t block_rows = PLE_STORE_ROWS_PER_BLOCK;
    uint64_t block_bytes = 0;
    uint64_t tensor_bytes = 0;
    uint64_t file_offset = 0;
    uint64_t shard_budget = 0; // per-shard byte budget, multiple of block_bytes

    shard shards[k_ple_shards];

    ple_store_stats stats;
    mutable std::mutex stats_mutex;

    // serializes the multi-pass fetch plan (single-flight v1)
    std::mutex fetch_mutex;

    shard & shard_for(int64_t block_idx) { return shards[block_idx % k_ple_shards]; }
};

ple_store * ple_store_open(const char * path, uint64_t offset,
                           enum ggml_type type, int64_t head_dim,
                           int64_t n_rows, uint64_t cache_bytes,
                           char * err, size_t errlen) {
    if (type != GGML_TYPE_Q5_1) {
        snprintf(err, errlen, "ple_store: unsupported type %s (v1 supports Q5_1 only)", ggml_type_name(type));
        return nullptr;
    }
    if (head_dim <= 0 || head_dim % 32 != 0) {
        snprintf(err, errlen, "ple_store: head_dim must be a positive multiple of 32, got %lld", (long long) head_dim);
        return nullptr;
    }
    if (n_rows <= 0) {
        snprintf(err, errlen, "ple_store: n_rows must be positive");
        return nullptr;
    }

    ple_store * s = new ple_store();
    s->type = type;
    s->head_dim = head_dim;
    s->n_rows = n_rows;
    s->row_bytes = head_dim / 32 * sizeof(block_q5_1_local); // Q5_1: 24 B per 32 elements
    s->block_bytes = s->row_bytes * s->block_rows;
    s->tensor_bytes = (uint64_t) n_rows * s->row_bytes;
    s->file_offset = offset;
    s->shard_budget = cache_bytes / k_ple_shards / s->block_bytes * s->block_bytes;

    if (s->shard_budget < s->block_bytes) {
        snprintf(err, errlen, "ple_store: cache_bytes %llu too small (need >= %llu)",
                 (unsigned long long) cache_bytes, (unsigned long long) s->block_bytes * k_ple_shards);
        delete s;
        return nullptr;
    }

    s->fd = open(path, O_RDONLY | O_CLOEXEC);
    if (s->fd < 0) {
        snprintf(err, errlen, "ple_store: open %s failed: %s", path, strerror(errno));
        delete s;
        return nullptr;
    }

    struct stat st;
    if (fstat(s->fd, &st) != 0) {
        snprintf(err, errlen, "ple_store: fstat %s failed: %s", path, strerror(errno));
        close(s->fd);
        delete s;
        return nullptr;
    }
    if ((uint64_t) st.st_size < offset + s->tensor_bytes) {
        snprintf(err, errlen, "ple_store: file too small: need offset %llu + %llu bytes, file is %lld",
                 (unsigned long long) offset, (unsigned long long) s->tensor_bytes, (long long) st.st_size);
        close(s->fd);
        delete s;
        return nullptr;
    }

    return s;
}

void ple_store_close(ple_store * s) {
    if (s == nullptr) return;
    if (s->fd >= 0) close(s->fd);
    delete s;
}

ple_store_stats ple_store_get_stats(const ple_store * s) {
    std::lock_guard<std::mutex> lk(s->stats_mutex);
    return s->stats;
}

// read one whole block from disk into data; partial final block is
// allowed (fixture generality; production table divides evenly)
static bool read_block(ple_store * s, int64_t block_idx, std::vector<uint8_t> & data, char * err, size_t errlen) {
    const uint64_t off = s->file_offset + (uint64_t) block_idx * s->block_bytes;
    const uint64_t left = s->tensor_bytes - (uint64_t) block_idx * s->block_bytes;
    const uint64_t want = std::min(s->block_bytes, left);

    data.resize(want);
    size_t done = 0;
    int tries = 0;
    while (done < want) {
        const ssize_t n = pread(s->fd, data.data() + done, want - done, (off_t) (off + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            tries++;
            if (tries > 1) {
                snprintf(err, errlen, "ple_store: pread block %lld failed after retry: %s",
                         (long long) block_idx, strerror(errno));
                return false;
            }
            continue;
        }
        if (n == 0) {
            snprintf(err, errlen, "ple_store: unexpected EOF reading block %lld (file truncated?)",
                     (long long) block_idx);
            return false;
        }
        done += (size_t) n;
    }
    return true;
}

// Ensure a block is cached (idempotent, read-through). Used by both the
// prefetch pass and the gather pass: when a fetch's working set exceeds the
// shard capacity, blocks loaded early can be evicted before their rows are
// consumed, so the gather re-loads on demand instead of failing. The
// budget-never-exceeded contract is preserved (evict-then-insert).
static bool ensure_cached(ple_store * s, int64_t block_idx, char * err, size_t errlen) {
    shard & sh = s->shard_for(block_idx);
    {
        std::lock_guard<std::mutex> sh_lk(sh.mutex);
        auto it = sh.map.find(block_idx);
        if (it != sh.map.end()) {
            sh.lru.splice(sh.lru.begin(), sh.lru, it->second.first); // LRU bump
            return true;
        }
    }

    std::vector<uint8_t> data; // read outside the shard lock
    if (!read_block(s, block_idx, data, err, errlen)) {
        return false;
    }
    {
        // count the read unconditionally: even if the re-check below finds the
        // block already present (future concurrency), the disk read happened
        std::lock_guard<std::mutex> st_lk(s->stats_mutex);
        s->stats.blocks_read += 1;
        s->stats.bytes_read += data.size();
    }

    std::lock_guard<std::mutex> sh_lk(sh.mutex);
    if (sh.map.count(block_idx) == 0) { // re-check under lock
        while (sh.bytes_used + data.size() > s->shard_budget && !sh.lru.empty()) {
            const int64_t victim = sh.lru.back();
            auto vit = sh.map.find(victim);
            sh.bytes_used -= vit->second.second.data.size();
            sh.map.erase(vit);
            sh.lru.pop_back();
        }
        sh.lru.push_front(block_idx);
        sh.map[block_idx] = {sh.lru.begin(), block_entry{std::move(data)}};
        sh.bytes_used += sh.map[block_idx].second.data.size();
    }
    return true;
}

bool ple_store_fetch(ple_store * s, const int32_t * rows, int64_t n,
                     float * dst_host, size_t dst_stride_elems,
                     char * err, size_t errlen) {
    const auto t0 = std::chrono::steady_clock::now();

    if (n <= 0) return true;
    for (int64_t i = 0; i < n; i++) {
        if (rows[i] < 0 || rows[i] >= s->n_rows) {
            snprintf(err, errlen, "ple_store: row index %d out of range [0, %lld)", rows[i], (long long) s->n_rows);
            return false;
        }
    }

    uint64_t rows_hit = 0;
    uint64_t rows_miss = 0;
    std::vector<int64_t> missing;

    // single-flight: the whole fetch is serialized (v1); shard locks still
    // guard against future concurrent callers
    std::lock_guard<std::mutex> flk(s->fetch_mutex);

    // v2#5: group the requested rows by DISTINCT block, in first-occurrence
    // (= row) order, so every later stage locks each shard once per block
    // instead of once per row. Layout is CSR: block k owns row_idx[start[k],
    // start[k+1]). Per-row hit/miss accounting is reconstructed from the
    // group sizes, and the LRU bump order matches v1's row-order bumps for
    // the single-row-per-block case (the pinned thrash trace).
    std::vector<int64_t> blocks;
    std::vector<uint32_t> group_size;
    std::unordered_map<int64_t, uint32_t> block_pos;
    block_pos.reserve((size_t) n * 2);
    for (int64_t i = 0; i < n; i++) {
        const int64_t block_idx = rows[i] / (int64_t) s->block_rows;
        auto it = block_pos.find(block_idx);
        if (it == block_pos.end()) {
            block_pos.emplace(block_idx, (uint32_t) blocks.size());
            blocks.push_back(block_idx);
            group_size.push_back(1);
        } else {
            group_size[it->second]++;
        }
    }
    std::vector<uint32_t> start(group_size.size() + 1, 0);
    for (size_t k = 0; k < group_size.size(); k++) {
        start[k + 1] = start[k] + group_size[k];
    }
    std::vector<uint32_t> fill(group_size.size(), 0);
    std::vector<int64_t> row_idx((size_t) n);
    for (int64_t i = 0; i < n; i++) {
        const int64_t block_idx = rows[i] / (int64_t) s->block_rows;
        const uint32_t k = block_pos[block_idx];
        row_idx[start[k] + fill[k]++] = i;
    }

    // pass 1: classify DISTINCT blocks against the current cache snapshot
    // (stats only), collect missing ones
    for (size_t k = 0; k < blocks.size(); k++) {
        const int64_t block_idx = blocks[k];
        shard & sh = s->shard_for(block_idx);
        std::lock_guard<std::mutex> sh_lk(sh.mutex);
        auto it = sh.map.find(block_idx);
        if (it != sh.map.end()) {
            sh.lru.splice(sh.lru.begin(), sh.lru, it->second.first); // LRU bump
            rows_hit += group_size[k];
        } else {
            rows_miss += group_size[k];
            missing.push_back(block_idx);
        }
    }

    // pass 2: prefetch missing blocks sorted by offset (seek locality).
    // NOTE: blocks inserted here may be evicted again before pass 3 uses them
    // when this fetch's working set exceeds the shard capacity; that is fine.
    if (!missing.empty()) {
        std::sort(missing.begin(), missing.end());
        // v2#1: submit all the reads to the kernel readahead BEFORE consuming
        // them: POSIX_FADV_WILLNEED is non-blocking, so the blocks load
        // concurrently while the first preads are already being served (the
        // single-flight preads stay sequential, but they hit warm pages).
        for (int64_t block_idx : missing) {
            const uint64_t off  = s->file_offset + (uint64_t) block_idx * s->block_bytes;
            const uint64_t left = s->tensor_bytes - (uint64_t) block_idx * s->block_bytes;
            const uint64_t want = std::min(s->block_bytes, left);
            (void) posix_fadvise(s->fd, (off_t) off, (off_t) want, POSIX_FADV_WILLNEED);
        }
        for (int64_t block_idx : missing) {
            if (!ensure_cached(s, block_idx, err, errlen)) {
                return false;
            }
        }
    }

    // pass 3 (v2#5): per DISTINCT block, read-through then dequant the whole
    // group under the shard lock — eviction cannot free the block while the
    // lock is held, and with single-flight there is no real contention.
    for (size_t k = 0; k < blocks.size(); k++) {
        const int64_t block_idx = blocks[k];
        if (!ensure_cached(s, block_idx, err, errlen)) {
            return false;
        }
        shard & sh = s->shard_for(block_idx);
        std::lock_guard<std::mutex> sh_lk(sh.mutex);
        // find() with a loud error: map[] would default-construct an empty
        // entry if the block vanished (future concurrency) and read from
        // nullptr — same cost, failure is explicit instead of silent UB
        auto it = sh.map.find(block_idx);
        if (it == sh.map.end()) {
            snprintf(err, errlen, "ple_store: block %lld missing after ensure_cached", (long long) block_idx);
            return false;
        }
        const uint8_t * base = it->second.second.data.data();
        for (uint32_t j = start[k]; j < start[k + 1]; j++) {
            const int64_t i = row_idx[j];
            const uint64_t in_block = (uint64_t) (rows[i] % (int64_t) s->block_rows);
            dequant_row_q5_1(base + in_block * s->row_bytes, dst_host + i * dst_stride_elems, s->head_dim);
        }
    }

    {
        std::lock_guard<std::mutex> st_lk(s->stats_mutex);
        const auto t1 = std::chrono::steady_clock::now();
        s->stats.fetch_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        s->stats.fetches += 1;
        s->stats.hits += rows_hit;
        s->stats.misses += rows_miss;
    }

    return true;
}
