"""vLLM inference implementation."""

from transformers import AutoTokenizer
from vllm import LLM


class VLLMInference:
    """vLLM inference with persistent context across turns."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.3,
            max_model_len=2048,
        )
        self.sampling_params = self.llm.get_default_sampling_params()
        self.sampling_params.temperature = 0.0  # Greedy decoding for deterministic output
        self.sampling_params.top_p = 1.0  # Disable nucleus sampling
        self.sampling_params.top_k = -1  # Disable top-k sampling
        self.sampling_params.repetition_penalty = 1.0  # No penalty
        self.sampling_params.max_tokens = 128
        self.accumulated_text = ""

    def reset(self):
        """Reset conversation history."""
        self.accumulated_text = ""

    def generate(self, prompt, max_tokens=128):
        """Generate text with vLLM (persistent context)."""
        from time import perf_counter

        self.sampling_params.max_tokens = max_tokens
        prompt_tokens = len(self.tokenizer.encode(prompt))

        # Build full context: accumulated history + new prompt
        full_context = self.accumulated_text + prompt
        input_tokens = len(self.tokenizer.encode(full_context))

        start = perf_counter()

        # vLLM handles caching internally across the full context
        outputs = self.llm.generate([full_context], self.sampling_params)

        elapsed = perf_counter() - start

        generated = outputs[0].outputs[0].text.strip()
        output_tokens = len(self.tokenizer.encode(generated))

        # Update accumulated text for next turn
        self.accumulated_text = full_context + generated

        return {
            "method": "vLLM",
            "prompt": prompt,
            "output": generated,
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "time": elapsed,
            "throughput": (prompt_tokens + output_tokens) / elapsed if elapsed > 0 else 0,
        }
