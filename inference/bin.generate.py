"""
Generate text from a trained NanoTransformer.

Usage:
  # Interactive mode
  python generate.py --model checkpoints/best

  # Single prompt
  python generate.py --model checkpoints/final --prompt "The universe is"

  # Batch generation
  python generate.py --model checkpoints/best --prompt "Once upon a time" \
      --n 5 --temperature 0.9 --max_tokens 200
"""

import argparse
import sys
import torch
from tokenizers import Tokenizer, decoders

 NanoTransformer


def load_model(path, device="auto"):
    """Load model + tokenizer from a save_pretrained checkpoint."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = NanoTransformer.from_pretrained(path).to(device)
    model.eval()

    tokenizer = Tokenizer.from_file(f"{path}/tokenizer.json")
    # Ensure ByteLevel decoder is set for proper text output
    # (converts Ġ → space, Ã¹ → ù, Ċ → newline, etc.)
    if tokenizer.decoder is None:
        tokenizer.decoder = decoders.ByteLevel()

    info = f"  Loaded model from {path}"
    info += f" | {model.count_params():,} params"
    info += f" | device={device}"
    print(info)

    return model, tokenizer, device


def generate_text(model, tokenizer, prompt, device, max_tokens=200,
                  temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2):
    """Generate text from a prompt string."""
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.token_to_id("<eos>"),
    )
    return tokenizer.decode(output_ids[0].tolist())


def interactive_mode(model, tokenizer, device, args):
    """Interactive REPL for text generation."""
    print(f"\n  🧠 NanoTransformer — Interactive Generation")
    print(f"     temperature={args.temperature}, top_k={args.top_k}, "
          f"top_p={args.top_p}, rep_penalty={args.repetition_penalty}, "
          f"max_tokens={args.max_tokens}")
    print(f"     Type 'quit' to exit, 'config' to change settings\n")

    while True:
        try:
            prompt = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye! 👋")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            print("  Bye! 👋")
            break
        if prompt.lower() == "config":
            print(f"    temperature={args.temperature}, top_k={args.top_k}, "
                  f"top_p={args.top_p}, rep_penalty={args.repetition_penalty}, "
                  f"max_tokens={args.max_tokens}")
            try:
                args.temperature = float(input("    temperature> ") or args.temperature)
                args.top_k = int(input("    top_k> ") or args.top_k)
                args.top_p = float(input("    top_p> ") or args.top_p)
                args.repetition_penalty = float(input("    repetition_penalty> ") or args.repetition_penalty)
                args.max_tokens = int(input("    max_tokens> ") or args.max_tokens)
            except ValueError:
                pass
            continue

        text = generate_text(
            model, tokenizer, prompt, device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"  Model > {text}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate text with NanoTransformer")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--n", type=int, default=1, help="Number of generations")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Penalize repeated tokens (1.0=off, 1.1-1.3 recommended)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model, args.device)

    if args.prompt is None:
        interactive_mode(model, tokenizer, device, args)
    else:
        for i in range(args.n):
            text = generate_text(
                model, tokenizer, args.prompt, device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            if args.n > 1:
                print(f"\n  [{i+1}/{args.n}] {text}")
            else:
                print(f"\n  {text}")


if __name__ == "__main__":
    main()