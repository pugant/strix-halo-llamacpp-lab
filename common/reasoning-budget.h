#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,         // waiting for start sequence
    REASONING_BUDGET_COUNTING,     // counting down tokens
    REASONING_BUDGET_WARN_FORCING, // forcing the warn message mid-budget, then back to COUNTING
    REASONING_BUDGET_FORCING,      // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8, // budget exhausted, waiting for UTF-8 completion
    REASONING_BUDGET_DONE,         // passthrough forever
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> [WARN_FORCING -> COUNTING] -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:          passthrough, watching for start_tokens sequence
//   COUNTING:      counting down remaining tokens, watching for natural end_tokens;
//                  when warn_tokens is non-empty and remaining drops to warn_at,
//                  forces warn_tokens once (mid-budget convergence window, s1-style)
//                  and returns to COUNTING with the residual budget left to the model
//   WARN_FORCING:  forces warn_tokens token-by-token (all other logits -> -inf),
//                  then goes back to COUNTING (never terminates the block itself)
//   WAITING_UTF8:  budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:       forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:          passthrough forever
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_tokens   - token sequence that activates counting
//   end_tokens     - token sequence for natural deactivation
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   warn_tokens    - message forced once at warn_at; empty = warn disabled
//   warn_ratio     - fraction of the budget consumed before the warn (0 < r < 1);
//                    warn_at = budget * (1 - warn_ratio); ignored when warn_tokens
//                    is empty
//   initial_state  - initial state
//
struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        const std::vector<llama_token> & warn_tokens = {},
        float                            warn_ratio  = 0.75f,
        common_reasoning_budget_state    initial_state = REASONING_BUDGET_IDLE);

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl);
