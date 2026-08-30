"""
LLM inference Profiler: Compare inference performance across multi-turn conversations.
"""

import argparse

from dotenv import load_dotenv
from inference_comparison import InferenceComparison

load_dotenv()


def interactive_chat():
    """Run interactive multi-turn comparison."""
    comparison = InferenceComparison()

    print("\n" + "=" * 90)
    print("LLM Inference-Profiler - Interactive Comparison")
    print("=" * 90)
    print("Type prompts to compare inference across PyTorch and vLLM")
    print("Multi-turn conversations persist KV-cache for realistic comparison")
    print("Type 'reset' to start a new conversation, 'exit'/'quit' to stop\n")

    turn = 0
    while True:
        try:
            prompt = input("prompt> ").strip()

            if not prompt:
                continue

            if prompt.lower() in {"exit", "quit"}:
                print("\nGoodbye.")
                break

            if prompt.lower() == "reset":
                comparison.reset()
                turn = 0
                print("Conversation reset.\n")
                continue

            turn += 1
            print(f"\n--- Turn {turn} ---")
            comparison.compare(prompt, max_tokens=128)

        except KeyboardInterrupt:
            print("\n\nGoodbye.")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Inference-Profiler: Compare inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py              # Interactive chat mode (recommended)
        """,
    )

    parser.parse_args()
    interactive_chat()


if __name__ == "__main__":
    main()
