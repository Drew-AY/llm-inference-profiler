"""High-level inference comparison tool."""

import os

from dotenv import load_dotenv
from pytorch_inference import (
    PyTorchNoCacheInference,
    PyTorchWithCacheInference,
)
from vllm_inference import VLLMInference

load_dotenv()

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH")


class InferenceComparison:
    """Compare inference performance across different approaches."""

    def __init__(self, model_path=None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        if not self.model_path:
            raise ValueError("MODEL_PATH environment variable not set")

        print(f"Loading model: {self.model_path}\n")

        self.pytorch_no_cache = PyTorchNoCacheInference(self.model_path)
        self.pytorch_with_cache = PyTorchWithCacheInference(self.model_path)
        self.vllm = VLLMInference(self.model_path)

    def reset(self):
        """Reset cache for starting a new conversation."""
        self.pytorch_with_cache.reset()
        self.vllm.reset()

    def compare(self, prompt, max_tokens=128):
        """Run inference on all approaches and compare."""
        print(f"Prompt: {prompt}\n")
        print("Running inference comparisons...\n")

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
        print(f"{result2['time']:.2f}s)")

        # vLLM
        print("  3. vLLM...", end=" ", flush=True)
        result3 = self.vllm.generate(prompt, max_tokens)
        results.append(result3)
        print(f"({result3['time']:.2f}s)")

        self._print_results(results)
        return results

    def _print_results(self, results):
        """Print comparison results."""
        print("\n" + "=" * 90)
        print("COMPARISON RESULTS")
        print("=" * 90)

        # Metrics table
        print(
            f"\n{'Method':<30} {'Time (s)':<12} {'Throughput (tok/s)':<18}"
        )
        print("-" * 90)

        baseline_throughput = results[0]["throughput"]

        for result in results:
            print(
                f"{result['method']:<30} {result['time']:>10.2f}s "
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

