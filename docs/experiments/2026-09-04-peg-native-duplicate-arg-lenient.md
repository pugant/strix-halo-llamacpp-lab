# The parameter that came back: leniency for duplicated tool-call args in peg-native

2026-09-04, the lab's production server again. A day after the F4 boundary series
shipped, the joint test with the pi stack hit a different class of failure: an agent
turn that had already generated a 222-line file write — four minutes of decoding —
died with

```
The model produced output that does not match the expected peg-native format
```

and the retry reproduced the identical failure. This note records why that string
exists, what the model actually did, and the fix: the tagged tool-call grammar now
tolerates a duplicated parameter, resolves it last-wins, and *logs the tolerance*
so the signal does not disappear.

## 1. Symptom — a write that never happened

The joint-test session (agent workload, ~183k context, temperature 0.7) reached the
final artifact of its task: a `write` tool call carrying a full TypeScript source
file. The model streamed the whole envelope, the turn ended naturally — and the
client surfaced the error above. On disk: the *previous* version of the file, not
the new one. The tool call had never executed. A manual retry produced the same
malformed envelope, token for token: the slip was reproducible on that generation,
not a one-off sampling accident.

Where does that string come from? Not the client. It is the server's own
`common_chat_peg_parse` throwing when the PEG parser cannot consume the model's
output ([`common/chat.cpp:2416`](../../rocmfpx/common/chat.cpp)) — the client is a
pass-through. The failure is a server-side parse rejection.

## 2. Root cause — a required argument, twice

The tool call the model produced had this shape (content abbreviated; the real one
was 222 lines of source):

```
<tool_call>
<function=write>
<parameter=path>
/workspace/project/src/guard.ts
</parameter>
<parameter=content>
/** ... the whole file ... */
</parameter>
<parameter=path>
/workspace/project/src/guard.ts
</parameter>
</function>
</tool_call>
```

The `path` parameter appears **twice**: once before the content, as the schema
defines, and once *after* the content, as an envelope restart right where
`</function>` belonged. The file content itself was clean — no embedded tags, no
control bytes — the defect was purely the duplicated envelope block.

The grammar could not accept it. The auto-generated tagged parser builds required
parameters as a **fixed sequence, each exactly once, in schema-definition order**
(only optional parameters were repeatable, in any order, after the sequence)
([`common/chat-auto-parser-generator.cpp`](../../rocmfpx/common/chat-auto-parser-generator.cpp),
`build_tool_parser_tag_tagged`). A third `<parameter=path>` has no rule to match,
the whole tool-call rule fails, and the parser throws — taking the entire
generation with it. For a headless run there is no auto-retry: the work is simply
lost.

Two things make this worth fixing rather than shrugging off as model noise:

- **The trigger regime is becoming normal.** Long contexts, mid-range sampling
  temperatures, and multi-thousand-token generations are exactly where an
  "envelope restart" slip appears. In this session it was 1 in 86+ requests; the
  next day's workload will sit in that regime for hours.
- **A latent cousin already existed.** Repeated *optional* parameters were already
  legal in the grammar — and the mapper happily emitted a JSON arguments string
  with a **duplicate key**, silently. Whichever JSON parser sat downstream would
  apply its own duplicate policy. That class was invisible by construction.

## 3. The fix — tolerate, resolve, and log

Two changes, one commit
([`patches/peg-lenient-dup-args/`](../../patches/peg-lenient-dup-args/)):

- **Grammar** (`build_tool_parser_tag_tagged`): the repeat that follows the
  required sequence now admits *any* parameter, required included. The required
  sequence itself is unchanged — order and presence are still enforced up front.
- **Mapper** (`common_chat_peg_mapper::map`): when an argument name is seen again
  for the same tool call, the previous `"name": value` pair is surgically removed
  from the incrementally-built JSON (the other keys and their order are
  preserved), and the new occurrence is re-emitted — **last-wins** semantics. And
  it stays loud:

```
peg-native: leniency-hit: duplicate param 'path' tolerated (last-wins)
```

The marker is the point. The pi-stack team asked for it explicitly when approving
the change: a tolerated duplicate is a *model-behavior signal* — if the slip rate
climbs in a new regime, or a future duplicate carries a **different value** than
the first occurrence, the log is where you notice. Silent leniency would trade a
crash for blindness. The token `leniency-hit` is grep-stable and countable, so the
existing watchers classify it as its own class.

## 4. Validation

Test-first throughout (`tagged_duplicate_required_param` in
[`tests/test-chat-auto-parser.cpp`](../../rocmfpx/tests/test-chat-auto-parser.cpp)):
the test drives the **production** grammar builder — not a hand-rolled copy — with
the tagged syntax this model family uses:

- an identical duplicate parses and yields the same arguments as the clean call;
- a **divergent** duplicate resolves last-wins (equal to a clean call carrying the
  last value);
- the `leniency-hit` marker is asserted by capturing the log output.

Suites: `test-chat-auto-parser` 71/420 and `test-chat-peg-parser` 32/198, zero
failures; the new test was RED against the unfixed code.

The production deployment (`qwen4exp-mtp-vk-optim3`) then met the real workload:
the pi session resumed (a 118,689-token cold prefill — the client's resume
serialization is not a prefix of any disk entry, so the persistent cache could not
serve it) and the very rewrite turn that had died twice completed cleanly — the
third generation of that file happened to be well-formed; the slip is stochastic,
not deterministic. Two hours later the marker earned its keep on its own:

```
13:29:11.252 W peg-native: leniency-hit: duplicate param 'path' tolerated (last-wins)
13:29:11.487 I slot release: task 49642 | stop processing: n_tokens = 85353, truncated = 0
13:29:16.414 I slot launch_slot_: task 49844 …  (session continues, no intervention)
```

The third occurrence of the slip — the same duplicated `path` in a `write` —
tolerated, resolved last-wins, delivered 235 ms later, session uninterrupted.
The two pre-fix occurrences of the same slip had each killed a four-minute turn.

*Measured on the lab's [bare-metal Strix Halo](../../BARE-METAL.md), configuration as of the note's date.*
