# LLM Inference Profiler

An educational profiler that measures what a KV-cache actually buys you. It runs the same model two ways — recomputing attention from scratch versus caching and reusing it — across a multi-turn conversation, and breaks the cost down into prefill and decode.

This project is as much about *how to benchmark GPU inference correctly* as it is about
KV-caches. Getting trustworthy numbers out of Apple Silicon's MPS backend required finding and fixing several measurement traps — each one looked at first like a real performance difference, and turned out to be an artifact of how the benchmark itself was run. The [Measurement Notes](#measurement-notes) section below walks through each one, since the traps generalize to benchmarking any PyTorch model on MPS, not just this comparison.

## What It Does

Compares PyTorch inference across two approaches:

1. **PyTorch (no cache)** — Baseline: re-reads the entire conversation every turn and
   recomputes attention over the whole sequence at every generation step
2. **PyTorch (with cache)** — Manual KV-cache: reuses cached K,V tensors, so each turn
   only processes its new prompt and each decode step only processes one token

Both maintain the same conversation history, so they do genuinely comparable work.

Metrics reported per turn:

- **Prefill (s) / Prefill tok** — Cost of ingesting context before generation starts
- **Decode (s) / Decode tok** — Cost of generating the response, token by token
- **Decode tok/s** — Generation rate, the number that holds up (or doesn't) as context grows
- **Total time / Throughput / Speedup** — Overall per-turn comparison

## Installation

```bash
source .venv/bin/activate      # or wherever your venv lives
pip install -r requirements.txt
```

Tested on Python 3.12 with torch 2.13 and transformers 5.12 on Apple Silicon.
`accelerate` is required because the model is loaded with `device_map`.

## Configuration

Set the model and, for gated models, a Hugging Face token in a `.env` file:

```
MPS_MODEL_PATH=google/gemma-3-1b-it
HF_TOKEN=hf_...
```

`MPS_MODEL_PATH` is required; the app exits if it isn't set. `HF_TOKEN` is only needed
for gated repos — Gemma is gated, so you must accept its terms on the model page first.
Any causal-LM works, e.g. `Qwen/Qwen2.5-0.5B-Instruct` (ungated, no token needed).

`sentencepiece` and `tiktoken` are in the requirements because several tokenizers,
Gemma's included, can't be built without one of them.

## Usage

```bash
python app.py
```

```
prompt> what is the capital of Ghana?

--- Turn 1 ---
Prompt: what is the capital of Ghana?

Running inference comparisons...

  1. PyTorch (no cache)... Generated tokens: 9
0.75s)
  2. PyTorch (with cache)... Generated tokens: 9
(0.42s)

==========================================================================================
COMPARISON RESULTS
==========================================================================================

Method                    Prefill (s)  Prefill tok   Decode (s)   Decode tok   Decode tok/s
------------------------------------------------------------------------------------------
PyTorch (no cache)             0.048s            8       0.703s            9           12.8
PyTorch (with cache)           0.050s            8       0.366s            9           24.6

Method                         Total Time (s)     Throughput (tok/s)
------------------------------------------------------------------------------------------
PyTorch (no cache)                         0.75s             22.6
PyTorch (with cache)                       0.42s             40.8

==========================================================================================
Speedup vs PyTorch (no cache):
==========================================================================================
PyTorch (no cache)             1.00x (+0%)
PyTorch (with cache)           1.80x (+80%)
==========================================================================================
```

By the third turn the gap has widened, because the no-cache path is now re-reading
34 tokens of history that the cached path never touches:

```
Method                    Prefill (s)  Prefill tok   Decode (s)   Decode tok   Decode tok/s
------------------------------------------------------------------------------------------
PyTorch (no cache)             0.077s           34       2.358s           24           10.2
PyTorch (with cache)           0.045s            3       0.989s           24           24.3
```

You'll also see an occasional `[cache-debug] feeding model(shape=..., cache_len_before=...,
mps_mem=(...))` line on turn 2+ — diagnostic logging left in from tracking down the
unresolved anomaly described in [Measurement Notes](#measurement-notes). It's noise for
normal use; useful if that anomaly ever reappears and you want hard data on it.

**Commands:**
- Type prompts to continue the multi-turn conversation
- `reset` — start a new conversation (clears both histories)
- `exit` / `quit` — stop

## What The Numbers Show

Measured over a three-turn conversation with `gemma-3-1b-it` on Apple Silicon (MPS):

| Turn | Prefill tok<br>no-cache → cached | Prefill s<br>no-cache → cached | Decode tok/s<br>no-cache → cached | Speedup |
|------|------|------|------|---------|
| 1 | 8 → 8 | 0.045 → 0.047 | 12.7 → 24.7 | 1.84x |
| 2 | 21 → 4 | 0.077 → 0.050 | 12.2 → 24.8 | 1.98x |
| 3 | 34 → 3 | 0.077 → 0.045 | 10.2 → 24.3 | 2.36x |

Two distinct effects, visible separately:

**Prefill — the cross-turn effect.** Without a cache, every turn re-reads the entire
conversation: 8 → 21 → 34 tokens, and climbing. With one, each turn reads only its new
prompt: 8 → 4 → 3. This gap grows without bound as a conversation gets longer.

**Decode — the within-generation effect.** Without a cache, each generated token requires
a fresh forward pass over the whole sequence, so the rate decays as the response grows:
12.7 → 12.2 → 10.2 tok/s. With one, each step processes a single token against cached
state, and the rate stays flat: ~24.5 tok/s regardless of context length.

Note that turn 1 already shows ~1.8x. That is entirely the decode effect — with no prior
history, both paths prefill the same 8 tokens. Everything above 1.8x in later turns is the
prefill effect compounding on top.

## Measurement Notes

Naive timing on this workload is badly misleading, so the profiler takes some care. Each
of these was found by noticing a number that didn't make sense, then tracking down why —
that process is worth understanding even if you never touch this specific code.

- **MPS compiles a fresh kernel graph the first time it sees a given sequence length** —
  a fresh shape costs noticeably more than a repeat visit to one already compiled. Since
  `no_cache` always runs first in `compare()`, it would systematically pay that tax on
  whatever shape came up that turn, while `cached` got a free ride off the shape
  `no_cache` had *just* compiled a moment earlier — making the cache look better than it
  really is. `warmup_model` (in [pytorch_inference.py](pytorch_inference.py)) fixes the
  general case once at startup, pre-compiling a spread of shapes (1–128) on the shared
  model before either instance is ever timed.
- **That startup warmup doesn't cover every shape a growing conversation reaches**, and a
  fixed list is always a guess. The better fix: since the actual prompt is known the
  moment it's typed, `compare()` computes the *exact* shape each method is about to
  process — `no_cache`'s full accumulated history plus the new prompt, `with_cache`'s new
  prompt alone — and pings precisely those shapes via `wake_gpu` before timing starts.
  No guessing, and it scales to arbitrarily long conversations.
- **Apple Silicon also clocks the GPU down when idle.** The app sits at the `prompt>`
  input between turns for however long you take to type — confirmed empirically to
  inflate whichever method runs first afterward by 2-3x (always `no_cache`). A forward
  pass of *any* shape clears this, which is why `wake_gpu` does double duty: the same
  per-turn pings that pre-compile the exact upcoming shapes also absorb this idle cost,
  since both problems are fixed by "run the GPU right before you time it."
- **These are two separate costs needing two separate fixes** — compilation is paid once
  per shape, idle-ramp is paid once per gap in GPU activity. A fix for one does not cover
  the other; the profiler needed both `warmup_model` (startup, broad) and `wake_gpu`
  (every turn, precise) before per-turn numbers stopped being inconsistent run to run.
- **`torch.mps.synchronize()` at timing boundaries.** MPS queues work asynchronously, so
  an unsynchronized `perf_counter()` measures enqueue time, not compute, and smears the
  prefill/decode split.
- **Token counts come from actual decode steps**, not from re-encoding the output text.
  Re-encoding silently drops special tokens and undercounts real work by several-fold.
- **Generation stops at EOS.** Otherwise the model emits `<end_of_turn>` padding to fill
  `max_tokens`, and you pay for forward passes that produce nothing.
- **Stop tokens are excluded from history**, and BOS is added only on the first turn —
  both otherwise corrupt the context for subsequent turns.
- **The model loads directly onto its target device** (`device_map=get_device()`) rather
  than loading to CPU and copying the full weight set to MPS afterward — a correctness
  no-op, but a pure waste of load time.

Correctness check: with these in place, both implementations produce **byte-identical
output across all turns**, which is what a mathematically exact KV-cache should do.
Measured logit difference between the two paths is ~2.6e-5 against magnitudes of ~16.8.

### A note on the one anomaly that didn't resolve

One odd reading never got fully explained: occasionally, several turns into a real
interactive session, `with_cache`'s prefill spiked to 2-6x its normal cost despite correct,
verified cache state (confirmed directly — the cache was present, the right type, and
exactly the expected length). Kernel-compile tax, idle GPU state, cache validity, MPS
allocator churn, and a no-cache/cached mode-switch cost were all tested directly as
candidate causes and all ruled out — including by faithfully reproducing the exact
reported shapes and token counts through the real code path. It never reproduced in a
controlled script, only in live, longer-running sessions. Likely some accumulated
environmental state (thermal, memory, or OS-level) that a short test process doesn't
build up. Included here deliberately: not every anomaly resolves, and knowing when to
stop chasing one (while leaving instrumentation in place in case it recurs) is also part
of doing this kind of measurement work.

## Known Limitation

Conversation history is built by raw text concatenation rather than the model's chat
template (`<start_of_turn>` / `<end_of_turn>` for Gemma). This is fine for performance
measurement — both paths are affected identically — but it produces lower-quality
responses than the model is capable of.

## Files

- `app.py` — Interactive chat interface
- `inference_comparison.py` — Orchestrates the comparison, formats results
- `pytorch_inference.py` — Both implementations, sharing one loaded model
- `.env` — Model path and HF token
