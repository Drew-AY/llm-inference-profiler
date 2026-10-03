"""High-level inference comparison tool."""

import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from pytorch_inference import (
    PyTorchNoCacheInference,
    PyTorchWithCacheInference,
    get_device,
    wake_gpu,
    warmup_model,
)

load_dotenv()

MPS_MODEL_PATH = os.getenv("MPS_MODEL_PATH")


class InferenceComparison:
    """Compare PyTorch inference with and without KV-cache."""

    def __init__(self, mps_model_path=None):
        self.mps_model_path = mps_model_path or MPS_MODEL_PATH

        if not self.mps_model_path:
            raise ValueError("MPS_MODEL_PATH environment variable not set")

        print(f"Loading PyTorch model: {self.mps_model_path}\n")

        # Load PyTorch model and tokenizer once, share across both classes.
        # Load directly onto the target device to avoid loading onto CPU
        self.tokenizer = AutoTokenizer.from_pretrained(self.mps_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.mps_model_path,
            torch_dtype=torch.float32,
            device_map=get_device(),
        )

        # PyTorch classes share the same model
        self.pytorch_no_cache = PyTorchNoCacheInference(self.model, self.tokenizer)
        self.pytorch_with_cache = PyTorchWithCacheInference(self.model, self.tokenizer)

        # Pre-compile MPS kernels across representative shapes once, shared by
        # both instances, so neither one's first real call pays a compile tax
        # that the other benefits from (see warmup_model docstring)
        warmup_model(self.model, self.pytorch_no_cache.device)

    def reset(self):
        """Reset conversation state for starting a new conversation."""
        self.pytorch_no_cache.reset()
        self.pytorch_with_cache.reset()

    def compare(self, prompt, max_tokens=128):
        """Run inference on all approaches and compare."""
        print(f"Prompt: {prompt}\n")
        print("Running inference comparisons...\n")

        # Compute the exact prefill shapes this turn is about to measure, so
        # wake_gpu pre-compiles precisely those instead of a generic guess.
        # no_cache reprocesses its full accumulated history + the new
        # prompt; with_cache only processes the new prompt against its cache.
        no_cache_new_tokens = len(self.tokenizer.encode(
            prompt, add_special_tokens=self.pytorch_no_cache.accumulated_ids is None
        ))
        no_cache_prev_len = (
            self.pytorch_no_cache.accumulated_ids.shape[1]
            if self.pytorch_no_cache.accumulated_ids is not None
            else 0
        )
        no_cache_shape = no_cache_prev_len + no_cache_new_tokens

        with_cache_shape = len(self.tokenizer.encode(
            prompt, add_special_tokens=self.pytorch_with_cache.accumulated_ids is None
        ))

        # Absorb any GPU idle/power-state cost from time spent waiting on
        # user input, at the exact shapes about to be measured, so neither
        # the idle tax nor a shape-compile tax lands on either method below
        wake_gpu(
            self.model,
            self.pytorch_no_cache.device,
            shapes=(no_cache_shape, with_cache_shape),
        )

        results = []

        # PyTorch without cache
        print("  1. PyTorch (no cache)...", end=" ", flush=True)
        result1 = self.pytorch_no_cache.generate(prompt, max_tokens)
        results.append(result1)
        print(f"{result1['time']:.2f}s)")

        # PyTorch with cache
        print("  2. PyTorch (with cache)...", end=" ", flush=True)
        result2 = self.pytorch_with_cache.generate(prompt, max_tokens)
        results.append(result2)
        print(f"({result2['time']:.2f}s)")

        self._print_results(results)
        return results

    def _print_results(self, results):
        """Print comparison results."""
        print("\n" + "=" * 90)
        print("COMPARISON RESULTS")
        print("=" * 90)

        # Prefill/Decode breakdown table
        print(
            f"\n{'Method':<24} {'Prefill (s)':>12} {'Prefill tok':>12} "
            f"{'Decode (s)':>12} {'Decode tok':>12} {'Decode tok/s':>14}"
        )
        print("-" * 90)

        for result in results:
            print(
                f"{result['method']:<24} {result['prefill_time']:>11.3f}s "
                f"{result['prefill_tokens']:>12} "
                f"{result['decode_time']:>11.3f}s "
                f"{result['decode_tokens']:>12} "
                f"{result['decode_throughput']:>14.1f}"
            )

        # Overall metrics table
        print(
            f"\n{'Method':<30} {'Total Time (s)':<18} {'Throughput (tok/s)':<18}"
        )
        print("-" * 90)

        baseline_throughput = results[0]["throughput"]

        for result in results:
            print(
                f"{result['method']:<30} {result['time']:>16.2f}s "
                f"{result['throughput']:>16.1f}"
            )

        print("\n" + "=" * 90)
        print("Speedup vs PyTorch (no cache):")
        print("=" * 90)

        for result in results:
            speedup = result["throughput"] / baseline_throughput if baseline_throughput > 0 else 1
            speedup_pct = (speedup - 1) * 100
            print(f"{result['method']:<30} {speedup:.2f}x ({speedup_pct:+.0f}%)")

        print("=" * 90 + "\n")

