// state-split-test — deterministic repro + verification for the device
// split-buffer checkpoint restore.
//
// The device-side KV cache state write emits one io.write_tensor per
// (layer x k/v x contiguous cell range), while the read always emits one
// block per (layer x k/v). When the saved sequence state spans more than
// one physical cell range, the restore used to be refused with
// "device checkpoint restore buffer mismatch" (same bytes, different split).
//
// This tool punches a hole in the middle of a filled sequence (guaranteed
// 2 physical ranges), saves the state on device, wipes the sequence and
// restores it.
//
// usage: state-split-test <model.gguf>
// exit 0 = restore OK (pos_min/pos_max verified)
// exit 1 = restore failed (mismatch / wrong positions)
// exit 2 = setup error

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;

    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "[setup] model load failed\n");
        return 2;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 256;
    cp.n_batch   = 256;
    cp.n_seq_max = 1;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "[setup] ctx init failed\n");
        return 2;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    const llama_seq_id seq    = 0;
    const int          n_fill = 64;

    // 1) fill: decode n_fill tokens at pos 0..n_fill-1
    llama_batch batch = llama_batch_init(n_fill, 0, 1);
    batch.n_tokens = n_fill;
    for (int i = 0; i < n_fill; ++i) {
        batch.token[i]   = 1;
        batch.pos[i]     = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i]  = false;
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[setup] decode failed\n");
        return 2;
    }
    fprintf(stderr, "[setup] filled %d pos (pos_min=%d pos_max=%d)\n", n_fill,
            (int) llama_memory_seq_pos_min(mem, seq), (int) llama_memory_seq_pos_max(mem, seq));

    // 2) punch a hole in the middle -> 2 physical cell ranges for seq
    llama_memory_seq_rm(mem, seq, 30, 32);
    fprintf(stderr, "[setup] after seq_rm(30,32): pos_min=%d pos_max=%d\n",
            (int) llama_memory_seq_pos_min(mem, seq), (int) llama_memory_seq_pos_max(mem, seq));

    // 3) device save
    const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

    // expected positions after the hole (model-agnostic: on hybrid memories the
    // recurrent part dominates seq_pos_min, so just round-trip what we have)
    const int pmin_exp = (int) llama_memory_seq_pos_min(mem, seq);
    const int pmax_exp = (int) llama_memory_seq_pos_max(mem, seq);

    const size_t size = llama_state_seq_get_size_ext(ctx, seq, flags);
    std::vector<uint8_t> data(size);

    llama_state_seq_storage * storage = llama_state_seq_storage_init();

    const size_t n1 = llama_state_seq_get_data_ext_storage(ctx, data.data(), size, seq, flags, storage);
    if (n1 != size) {
        fprintf(stderr, "[setup] save size mismatch n1=%zu size=%zu\n", n1, size);
        return 2;
    }
    fprintf(stderr, "[save] %zu bytes on device\n", n1);

    // 4) wipe the sequence
    llama_memory_seq_rm(mem, seq, -1, -1);
    fprintf(stderr, "[wipe] pos_min=%d pos_max=%d\n",
            (int) llama_memory_seq_pos_min(mem, seq), (int) llama_memory_seq_pos_max(mem, seq));

    // 5) restore
    const size_t n2 = llama_state_seq_set_data_ext_storage(ctx, data.data(), n1, seq, flags, storage);
    if (n2 != n1) {
        fprintf(stderr, "RESTORE FAILED (n2=%zu expected=%zu)\n", n2, n1);
        llama_state_seq_storage_free(storage);
        return 1;
    }

    const int pmin = (int) llama_memory_seq_pos_min(mem, seq);
    const int pmax = (int) llama_memory_seq_pos_max(mem, seq);
    fprintf(stderr, "RESTORE OK (pos_min=%d pos_max=%d, expected %d/%d)\n", pmin, pmax, pmin_exp, pmax_exp);

    // round-trip: a second save of the restored state must have the same size
    const size_t n3 = llama_state_seq_get_data_ext_storage(ctx, data.data(), size, seq, flags, storage);
    fprintf(stderr, "[roundtrip] second save: %zu bytes (expected %zu)\n", n3, n1);

    llama_state_seq_storage_free(storage);

    return (pmin == pmin_exp && pmax == pmax_exp && n3 == n1) ? 0 : 1;
}
