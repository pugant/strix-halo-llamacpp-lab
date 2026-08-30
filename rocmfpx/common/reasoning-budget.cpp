#include "reasoning-budget.h"
#include "common.h"
#include "unicode.h"

#include "log.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_token> tokens;
    size_t pos = 0;

    bool advance(llama_token token) {
        if (tokens.empty()) {
            return false;
        }

        if (token == tokens[pos]) {
            pos++;
            if (pos >= tokens.size()) {
                pos = 0;
                return true;
            }
        } else {
            pos = 0;
            if (token == tokens[0]) {
                pos = 1;
            }
        }
        return false;
    }

    void reset() { pos = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    std::vector<llama_token> forced_tokens;

    // warn window (s1-style mid-budget convergence): warn_tokens is forced once
    // when remaining drops to warn_at, then counting resumes with the residual
    // budget — the model closes by itself (natural end) or the forced end fires
    // at 0 as backstop. Empty warn_tokens = warn disabled.
    std::vector<llama_token> warn_tokens;
    int32_t warn_at = 0;      // warn fires when remaining <= warn_at
    bool warned = false;      // one-shot per reasoning block
    size_t warn_pos = 0;      // next position in warn_tokens to force

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                LOG_INF("reasoning-budget: activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    LOG_INF("reasoning-budget: budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            if (ctx->end_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_DONE;
                LOG_INF("reasoning-budget: deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            if (ctx->vocab != nullptr) {
                const std::string piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    LOG_INF("reasoning-budget: UTF-8 complete, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                // warn window: fire once when the consumed share crosses warn_ratio.
                // Deferred to the next UTF-8-complete token so the forced message
                // never splits a multi-byte sequence mid-way.
                if (!ctx->warned && !ctx->warn_tokens.empty() && utf8_complete
                        && ctx->remaining <= ctx->warn_at && ctx->remaining > 0) {
                    ctx->warned = true;
                    ctx->warn_pos = 0;
                    ctx->end_matcher.reset();
                    ctx->state = REASONING_BUDGET_WARN_FORCING;
                    LOG_INF("reasoning-budget: warn injected at %d%% (used %d/%d)\n",
                        (int) ((float) (ctx->budget - ctx->remaining) * 100.0f / (float) ctx->budget),
                        ctx->budget - ctx->remaining, ctx->budget);
                    break;
                }
                if (ctx->remaining <= 0) {
                    if (utf8_complete) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        LOG_INF("reasoning-budget: budget exhausted, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        LOG_INF("reasoning-budget: budget exhausted, waiting for UTF-8 completion\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_WARN_FORCING:
        {
            // the warn message is part of the generated stream; once fully forced,
            // resume counting — the model keeps the residual budget to close by
            // itself (natural end) before the forced end fires at 0.
            ctx->warn_pos++;
            if (ctx->warn_pos >= ctx->warn_tokens.size()) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->end_matcher.reset();
                LOG_INF("reasoning-budget: warn complete, resuming countdown (remaining %d)\n", ctx->remaining);
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                LOG_INF("reasoning-budget: forced sequence complete, done\n");
            }
            break;
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                ctx->warned = false;
                ctx->warn_pos = 0;
                LOG_INF("reasoning-budget: re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    LOG_INF("reasoning-budget: budget=0, forcing immediately\n");
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state == REASONING_BUDGET_WARN_FORCING) {
        if (ctx->warn_pos >= ctx->warn_tokens.size()) {
            return;
        }
        const llama_token forced = ctx->warn_tokens[ctx->warn_pos];
        for (size_t i = 0; i < cur_p->size; i++) {
            if (cur_p->data[i].id != forced) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
        return;
    }

    if (ctx->state != REASONING_BUDGET_FORCING) {
        // passthrough — don't modify logits
        return;
    }

    if (ctx->force_pos >= ctx->forced_tokens.size()) {
        return;
    }

    const llama_token forced = ctx->forced_tokens[ctx->force_pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->warned = false;
    ctx->warn_pos = 0;
}

// forward declaration for use in clone
static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens, const std::vector<llama_token> & forced_tokens,
        const std::vector<llama_token> & warn_tokens, float warn_ratio,
        int32_t budget, common_reasoning_budget_state initial_state);
// note: internal helper keeps its own param order; the public init() exposes
// warn params AFTER initial_state for positional-caller compatibility

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    struct llama_sampler * clone = common_reasoning_budget_init_state(
        ctx->vocab,
        ctx->start_matcher.tokens,
        ctx->end_matcher.tokens,
        ctx->forced_tokens,
        ctx->warn_tokens,
        1.0f, // warn_at is carried as mutable state below
        ctx->budget,
        ctx->state); // (internal param order: warn before budget/state)

    // carry the full mutable state: the remaining budget, the matchers'
    // positions and the forcing position. common_sampler_clone() snapshots the
    // sampler before every speculative verify round and the server restores the
    // snapshot on each partial rejection: a clone that resets `remaining` to
    // the full budget silently refills the cap on every rollback, so it never
    // fires. This is observable whenever the restore path is taken on every
    // partial rejection (ctx seq_rm type FULL, e.g. any draft-dflash config -
    // ~46% acceptance means most rounds reject), while mono-MTP (PART type,
    // no sampler restore) enforces it fine.
    // The warn state (warned/warn_pos/warn_at) is part of the same mutable
    // state for the same reason: a rollback before the warn must re-arm it
    // (the regenerated stream does not contain the message), a rollback after
    // it must not inject it twice.
    auto * cctx = (common_reasoning_budget_ctx *) clone->ctx;
    cctx->remaining         = ctx->remaining;
    cctx->start_matcher.pos = ctx->start_matcher.pos;
    cctx->end_matcher.pos   = ctx->end_matcher.pos;
    cctx->force_pos         = ctx->force_pos;
    cctx->warned            = ctx->warned;
    cctx->warn_pos          = ctx->warn_pos;
    cctx->warn_at           = ctx->warn_at;

    return clone;
}

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab             * vocab,
        const std::vector<llama_token>       & start_tokens,
        const std::vector<llama_token>       & end_tokens,
        const std::vector<llama_token>       & forced_tokens,
        const std::vector<llama_token>       & warn_tokens,
        float                                  warn_ratio,
        int32_t                                budget,
        common_reasoning_budget_state          initial_state) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    // warn fires when remaining drops to warn_at (the consumed share crosses
    // warn_ratio); disabled with empty warn_tokens or a non-positive window
    int32_t warn_at = 0;
    if (!warn_tokens.empty() && budget > 0 && warn_ratio > 0.0f && warn_ratio < 1.0f) {
        warn_at = (int32_t) ((float) budget * (1.0f - warn_ratio));
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .start_matcher = */ { start_tokens, 0 },
            /* .end_matcher   = */ { end_tokens, 0 },
            /* .forced_tokens = */ forced_tokens,
            /* .warn_tokens   = */ warn_tokens,
            /* .warn_at       = */ warn_at,
            /* .warned        = */ false,
            /* .warn_pos      = */ 0,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        common_reasoning_budget_state    initial_state,
        const std::vector<llama_token> & warn_tokens,
        float                            warn_ratio) {
    return common_reasoning_budget_init_state(vocab, start_tokens, end_tokens, forced_tokens, warn_tokens, warn_ratio, budget, initial_state);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}
