# LLM Inference Profiler

An interactive profiler that demonstrates why KV-cache optimization matters for LLM inference. Compares three approaches side-by-side through realistic multi-turn conversations.

## What It Does

Compares inference performance across three approaches:
1. **PyTorch (no cache)** — Baseline: recomputes full attention every step
2. **PyTorch (with cache)** — Manual KV-cache: reuses prior attention computation
3. **vLLM** — Optimized serving: automatic caching + kernel optimization

Shows metrics that matter:
- **Time (s)** — Inference latency per turn
- **Throughput (tok/s)** — Tokens generated per second
- **Speedup** — How much faster each approach vs. baseline

## Installation

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install vllm>=0.4.0 transformers>=4.37.0 python-dotenv>=1.0.0
```

## Configuration

**Required:** Set the model path via `.env` file before running:

```
MODEL_PATH=/path/to/your/model
```

Example (Hugging Face model):
```
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
```

The app will fail if `MODEL_PATH` is not set.

## Usage

```bash
python app.py
```

Interactive mode:
```
prompt> What is machine learning?

--- Turn 1 ---
Prompt: What is machine learning?

Running inference comparisons...

  1. PyTorch (no cache)... 12.42s)
  2. PyTorch (with cache)... 3.91s)
  3. vLLM... 1.75s)

==========================================================================================
COMPARISON RESULTS
==========================================================================================

Method                         Time (s)     Throughput (tok/s)
------------------------------------------------------------------------------------------
PyTorch (no cache)                  12.42s              11.0
PyTorch (with cache)                 3.91s              36.0
vLLM                                 1.75s              77.7

==========================================================================================
Speedup vs PyTorch (no cache):
==========================================================================================
PyTorch (no cache)             1.00x (+0%)
PyTorch (with cache)           3.29x (+229%)
vLLM                           7.10x (+610%)
==========================================================================================

prompt> Tell me more about that

--- Turn 2 ---
[Results show maintained speedup with growing context...]

prompt> reset
Conversation reset.

prompt> exit
Goodbye.
```

**Commands:**
- Type prompts to continue multi-turn conversation
- Type `reset` to start a new conversation
- Type `exit` or `quit` to stop

## How It Works

### Why Multi-Turn Conversations?

KV-cache benefit is most visible across multiple turns:

**Turn 1:** All three approaches process prompt and generate output
- PyTorch (no cache): recomputes attention from scratch
- PyTorch (with cache): caches K,V tensors, reuses them during generation
- vLLM: automatic caching + optimized implementation

**Turn 2+:** Cache advantage shows:
- PyTorch (no cache): still recomputes everything (no improvement)
- PyTorch (with cache): reuses cache from turn 1 (faster)
- vLLM: maintains speedup with growing context

### Why These Three?

- **PyTorch (no cache)** = naive implementation (baseline)
- **PyTorch (with cache)** = shows KV-cache benefit in isolation
- **vLLM** = shows real-world performance with optimized implementation

The gap between PyTorch (with cache) and vLLM reveals **how much engineering matters**.

## Key Insights

1. **KV-cache is essential** — 3-4x speedup from caching alone
2. **Implementation quality matters** — Additional 2-3x from optimizations
3. **Multi-turn shows the pattern** — Cache benefit persists across turns
4. **Speed is what matters** — Latency and throughput are the real metrics

## Files

- `app.py` — Interactive chat interface
- `inference_comparison.py` — Orchestrates comparison, displays results
- `pytorch_inference.py` — PyTorch implementations (with/without cache)
- `vllm_inference.py` — vLLM implementation with persistent context
- `.env` — Model path configuration

## Why This Matters

This profiler demonstrates concrete performance differences that matter in production:
- **Batch serving with vLLM is 7-8x faster** than naive implementations
- **KV-cache is non-optional** for multi-turn applications
- **Engineering matters** — same algorithm, different implementations, 2-3x difference

The profiler makes these differences visible and measurable.
