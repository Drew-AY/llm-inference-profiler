"""PyTorch inference implementations (with and without KV-cache)."""

from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PyTorchInference:
    """Base class for PyTorch inference."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
        )


class PyTorchNoCacheInference(PyTorchInference):
    """PyTorch inference without KV-cache (recomputes each step)."""

    def generate(self, prompt, max_tokens=128):
        """Generate text without KV-cache."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_tokens = len(input_ids[0])

        start = perf_counter()

        generated_ids = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(generated_ids)
                next_token = torch.argmax(outputs.logits[0, -1, :])
                generated_ids = torch.cat(
                    [generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1
                )

        elapsed = perf_counter() - start

        generated = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated = generated[len(prompt) :].strip()  # Remove prompt from output
        output_tokens = len(self.tokenizer.encode(generated))

        return {
            "method": "PyTorch (no cache)",
            "prompt": prompt,
            "output": generated,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "time": elapsed,
            "throughput": (input_tokens + output_tokens) / elapsed if elapsed > 0 else 0,
        }


class PyTorchWithCacheInference(PyTorchInference):
    """PyTorch inference with persistent KV-cache across turns."""

    def __init__(self, model_path):
        super().__init__(model_path)
        self.accumulated_ids = None
        self.past_key_values = None

    def reset(self):
        """Reset cache for new conversation."""
        self.accumulated_ids = None
        self.past_key_values = None

    def generate(self, prompt, max_tokens=128):
        """Generate text with persistent KV-cache."""
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_tokens = len(prompt_ids[0])

        start = perf_counter()

        with torch.no_grad():
            if self.accumulated_ids is None:
                # First turn: process prompt from scratch
                current_ids = prompt_ids.clone()
                outputs = self.model(current_ids)
                self.past_key_values = outputs.past_key_values
            else:
                # Subsequent turns: append new prompt, reuse cache for prior context
                new_prompt_ids = prompt_ids
                self.accumulated_ids = torch.cat(
                    [self.accumulated_ids, new_prompt_ids], dim=1
                )
                # Compute only for new prompt tokens using existing cache
                outputs = self.model(
                    new_prompt_ids, past_key_values=self.past_key_values
                )
                self.past_key_values = outputs.past_key_values
                current_ids = self.accumulated_ids.clone()

            self.accumulated_ids = current_ids.clone()
            generated_ids = current_ids.clone()

            # Generate new tokens
            for _ in range(max_tokens - 1):
                next_token = torch.argmax(outputs.logits[0, -1, :])
                generated_ids = torch.cat(
                    [generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1
                )
                outputs = self.model(
                    next_token.unsqueeze(0).unsqueeze(0),
                    past_key_values=self.past_key_values,
                )
                self.past_key_values = outputs.past_key_values
                self.accumulated_ids = torch.cat(
                    [self.accumulated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1
                )

        elapsed = perf_counter() - start

        generated = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Remove all prior context from output (only show new generation)
        full_text = self.tokenizer.decode(self.accumulated_ids[0], skip_special_tokens=True)
        generated = generated[len(full_text) - len(generated) :].strip()
        output_tokens = len(self.tokenizer.encode(generated))

        return {
            "method": "PyTorch (with cache)",
            "prompt": prompt,
            "output": generated,
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "time": elapsed,
            "throughput": (prompt_tokens + output_tokens) / elapsed if elapsed > 0 else 0,
        }
