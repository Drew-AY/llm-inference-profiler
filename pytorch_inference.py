"""PyTorch inference implementations (with and without KV-cache)."""

from time import perf_counter

import torch

# MPS compiles a fresh kernel graph the first time it sees a given sequence
# length. Whichever inference method hits a given prompt/decode length first
# in the process pays that compile cost — if it's always the same method
# (e.g. no-cache, which always runs first in a comparison loop), its timing
# looks structurally worse than it actually is. Pre-compiling a spread of
# shapes up front, shared across both instances on the same model, removes
# that bias before either one is ever timed.
WARMUP_SHAPES = (1, 2, 4, 8, 16, 32, 64, 128)


def get_device():
    """Single source of truth for device selection, so the caller can load
    the model directly onto it instead of loading to CPU and copying over."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def warmup_model(model, device):
    """Pre-compile MPS kernels for both prefill (growing input) and
    cached-decode (single-token-with-past_key_values) shapes."""
    with torch.no_grad():
        for shape in WARMUP_SHAPES:
            ids = torch.randint(100, 1000, (1, shape), device=device)
            out = model(ids)
            model(ids[:, :1], past_key_values=out.past_key_values)
    if device.type == "mps":
        torch.mps.synchronize()


def wake_gpu(model, device, shapes=(8,), pings=3):
    """Absorb MPS idle/power-state cost before timing starts, at the exact
    shape(s) about to be measured.

    Two problems, one mechanism: Apple Silicon clocks the GPU down when idle
    (e.g. waiting on user input between turns), and whichever call runs first
    afterward pays a ramp-up tax. Separately, MPS compiles a fresh kernel
    graph the first time it sees a given sequence length. A throwaway
    forward pass before timing absorbs the idle tax regardless of its shape
    — but pinging the *actual* shape(s) about to be measured (the caller's
    real prefill lengths, not a generic dummy) also pre-compiles exactly
    what's needed, so the real calls don't pay a shape-compile tax either.
    """
    if device.type != "mps":
        return
    with torch.no_grad():
        for shape in shapes:
            dummy = torch.randint(100, 1000, (1, shape), device=device)
            for _ in range(pings):
                model(dummy)
    torch.mps.synchronize()


class PyTorchInference:
    """Base class for PyTorch inference."""

    def __init__(self, model, tokenizer):
        self.device = get_device()
        # Cheap no-op if the caller already loaded the model onto this
        # device directly; keeps this class correct standalone either way.
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.stop_token_ids = self._resolve_stop_tokens()

    def _resolve_stop_tokens(self):
        generation_config = getattr(self.model, "generation_config", None)
        eos = getattr(generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        if eos is None:
            return set()
        return {eos} if isinstance(eos, int) else set(eos)

    def _sync(self):
        # MPS queues work asynchronously; without this, timers measure enqueue, not compute
        if self.device.type == "mps":
            torch.mps.synchronize()


class PyTorchNoCacheInference(PyTorchInference):
    """PyTorch inference without KV-cache (recomputes full context every step)."""

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.accumulated_ids = None

    def reset(self):
        """Reset conversation history for new conversation."""
        self.accumulated_ids = None

    def generate(self, prompt, max_tokens=128):
        """Generate text without KV-cache."""
        # BOS belongs only at the start of the conversation, not on every turn
        prompt_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=self.accumulated_ids is None
        ).to(self.device)
        prompt_tokens = prompt_ids.shape[1]

        if self.accumulated_ids is None:
            self.accumulated_ids = prompt_ids.clone()
        else:
            self.accumulated_ids = torch.cat([self.accumulated_ids, prompt_ids], dim=1)

        # No cache to reuse, so prefill reprocesses the entire conversation every turn
        prefill_tokens = self.accumulated_ids.shape[1]

        self._sync()
        prefill_start = perf_counter()
        with torch.no_grad():
            outputs = self.model(self.accumulated_ids)
        self._sync()
        prefill_time = perf_counter() - prefill_start

        gen_start = self.accumulated_ids.shape[1]

        # Decode: every step reprocesses the whole sequence, so cost grows per token
        decode_steps = 0
        decode_start = perf_counter()
        with torch.no_grad():
            for _ in range(max_tokens):
                next_token = torch.argmax(outputs.logits[0, -1, :])
                token_id = next_token.item()
                # Stop tokens are control signals; keeping them would poison later turns
                if token_id in self.stop_token_ids:
                    break
                self.accumulated_ids = torch.cat(
                    [self.accumulated_ids, next_token.view(1, 1)], dim=1
                )
                decode_steps += 1
                if decode_steps >= max_tokens:
                    break
                outputs = self.model(self.accumulated_ids)
        self._sync()
        decode_time = perf_counter() - decode_start

        elapsed = prefill_time + decode_time

        new_token_ids = self.accumulated_ids[0, gen_start:].detach().cpu()
        generated = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        output_tokens = len(new_token_ids)
        print(f"Generated tokens: {output_tokens}")

        return {
            "method": "PyTorch (no cache)",
            "prompt": prompt,
            "output": generated,
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "time": elapsed,
            "throughput": (prompt_tokens + output_tokens) / elapsed if elapsed > 0 else 0,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_steps,
            "decode_throughput": decode_steps / decode_time if decode_time > 0 else 0,
        }


class PyTorchWithCacheInference(PyTorchInference):
    """PyTorch inference with persistent KV-cache across turns."""

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.accumulated_ids = None
        self.past_key_values = None

    def reset(self):
        """Reset cache for new conversation."""
        self.accumulated_ids = None
        self.past_key_values = None

    def generate(self, prompt, max_tokens=128):
        """Generate text with persistent KV-cache."""
        # BOS belongs only at the start of the conversation, not on every turn
        prompt_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=self.accumulated_ids is None
        ).to(self.device)
        prompt_tokens = prompt_ids.shape[1]

        # Prefill: whole prompt on turn 1, only the new prompt tokens on later turns
        self._sync()
        prefill_start = perf_counter()
        with torch.no_grad():
            if self.accumulated_ids is None:
                self.accumulated_ids = prompt_ids.clone()
                outputs = self.model(self.accumulated_ids)
            else:
                self.accumulated_ids = torch.cat([self.accumulated_ids, prompt_ids], dim=1)
                cache_len_before = (
                    self.past_key_values.get_seq_length()
                    if self.past_key_values is not None
                    else None
                )
                mem = (
                    f"alloc={torch.mps.current_allocated_memory()/1e6:.0f}MB "
                    f"driver={torch.mps.driver_allocated_memory()/1e6:.0f}MB"
                    if self.device.type == "mps"
                    else "n/a"
                )
                print(
                    f"[cache-debug] feeding model(shape={tuple(prompt_ids.shape)}, "
                    f"past_key_values={'None' if self.past_key_values is None else type(self.past_key_values).__name__}, "
                    f"cache_len_before={cache_len_before}, mps_mem=({mem}))"
                )
                outputs = self.model(prompt_ids, past_key_values=self.past_key_values)
            self.past_key_values = outputs.past_key_values
        self._sync()
        prefill_time = perf_counter() - prefill_start

        # Everything prefilled so far is context; generated tokens start after it
        gen_start = self.accumulated_ids.shape[1]

        # Decode: one token at a time against the cache
        decode_steps = 0
        decode_start = perf_counter()
        with torch.no_grad():
            for _ in range(max_tokens):
                next_token = torch.argmax(outputs.logits[0, -1, :])
                token_id = next_token.item()
                # Stop tokens are control signals; keeping them would poison later turns
                if token_id in self.stop_token_ids:
                    break
                self.accumulated_ids = torch.cat(
                    [self.accumulated_ids, next_token.view(1, 1)], dim=1
                )
                decode_steps += 1
                if decode_steps >= max_tokens:
                    break
                outputs = self.model(
                    next_token.view(1, 1), past_key_values=self.past_key_values
                )
                self.past_key_values = outputs.past_key_values
        self._sync()
        decode_time = perf_counter() - decode_start

        elapsed = prefill_time + decode_time

        new_token_ids = self.accumulated_ids[0, gen_start:].detach().cpu()
        generated = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        output_tokens = len(new_token_ids)
        print(f"Generated tokens: {output_tokens}")

        return {
            "method": "PyTorch (with cache)",
            "prompt": prompt,
            "output": generated,
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "time": elapsed,
            "throughput": (prompt_tokens + output_tokens) / elapsed if elapsed > 0 else 0,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "prefill_tokens": prompt_tokens,
            "decode_tokens": decode_steps,
            "decode_throughput": decode_steps / decode_time if decode_time > 0 else 0,
        }
