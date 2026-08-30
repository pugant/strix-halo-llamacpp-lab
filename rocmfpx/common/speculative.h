#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

struct common_speculative_token_dist {
    llama_tokens ids;
    std::vector<float> probs;
};

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // per-seq active drafter for request routing:
    // NONE = no routing (mono mode) - every implementation may draft the sequence, as before;
    // otherwise only the implementation matching this type drafts the sequence.
    // note: the routing only gates draft() - see the note in common_speculative_process()
    common_speculative_type drafter = COMMON_SPECULATIVE_TYPE_NONE;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;

    // RoPE position that the target model would assign to id_last's successor.
    // Equals n_past for text-only prompts, but they diverge under M-RoPE: an
    // image chunk occupies n_tokens KV slots while advancing position by only
    // n_pos. Only the draft-mtp implementation consumes this; -1 means "caller
    // supplied no M-RoPE-aware position", and consumers fall back to n_past so
    // behaviour is byte-identical to before for every other caller and impl.
    llama_pos   pos_next = -1;

    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    int32_t n_min = -1;
    float   p_min = -1.0f;

    // optional sparse proposal distributions, one per draft token
    std::vector<common_speculative_token_dist> * dists = nullptr;

    float temperature = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;

    // t8 stadio 2 (spec §3): concat round plumbing - when non-null, the pointed
    // tokens are the MTP head (k1' <= concat_k1) already appended to `result` by
    // the MTP arm of this round; the draft-dflash arm conditions its noise block
    // on them (head at n_past+1..n_past+k1', block at n_past+k1'+1..) instead of
    // using the mask placeholder at those positions. Owned by the harness
    // (common_speculative::concat_head): set between the MTP and DFlash arms of
    // the same common_speculative_draft() call and cleared when the round closes.
    // Null outside concat rounds (and on every k1 = 0 boot: zero code path).
    const llama_tokens * concat_head = nullptr;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// types of the implementations instantiated for this context, in priority order
// (spec-route: the per-request routing validates against these, not against the
// request params - the harness may auto-enable types or skip failed ones)
std::vector<enum common_speculative_type> common_speculative_types(const common_speculative * spec);

// spec-route: effective (post-clamp) max draft size of the loaded implementation
// for the given type, as the implementation itself will use it (e.g. MTP clamps
// to its trained head count, DFlash to its block size). -1 = type not loaded or
// no draft-size notion.
int32_t common_speculative_n_max_type(const common_speculative * spec, enum common_speculative_type type);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// true if any implementation requires target post-norm embeddings to be extracted
bool common_speculative_need_embd(common_speculative * spec);

// true if any implementation requires target pre-norm embeddings to be extracted
bool common_speculative_need_embd_pre_norm(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// t8 stadio 2 (spec §6): size of the MTP concat head of the round in flight for
// this sequence (0 = the next common_speculative_accept() will close a plain
// round, not a composed one). Must be read BEFORE the accept call - it consumes
// the per-seq round attribution this readout is derived from.
int32_t common_speculative_concat_head_size(const common_speculative * spec, llama_seq_id seq_id);

// t20 f3 (pi-stack): type of the implementation that closed the last draft
// round for this sequence (COMMON_SPECULATIVE_TYPE_NONE = no round attributed
// yet for this seq). Unlike the concat-head readout above, the impl_last
// attribution is per-round state that survives common_speculative_accept(),
// so this can be read on either side of the accept call closing the round.
enum common_speculative_type common_speculative_round_drafter_type(const common_speculative * spec, llama_seq_id seq_id);

// t8 stadio 2 (spec §6): the configured concat head size (--spec-concat-k1,
// 0 = concat rounds disabled)
int32_t common_speculative_concat_k1(const common_speculative * spec);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
bool common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);
bool common_speculative_state_required(const common_speculative * spec);

// (optional) rewind the internal state to a previously seen position, so that
// processing can resume from there after the target/draft memories rolled back
// (bounded prompt-cache boundary salvage). Returns false when the position is
// not recoverable.
bool common_speculative_rollback_state(common_speculative * spec, llama_seq_id seq_id, llama_pos pos);

// rebase per-sequence positions after the corresponding target/draft contexts shift
void common_speculative_shift_state(common_speculative * spec, llama_seq_id seq_id, llama_pos delta);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

// max number of draft tokens across all enabled speculative types
int32_t common_speculative_n_max(const common_params_speculative * spec);

// derive the draft context params from the base params
common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total   = 0;
    int32_t per_seq = 0;
};

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
