"""
Real-chat validation for the SFT model.

Loads an SFT checkpoint and runs a battery of ChatML prompts, checking the two
things that decide "usable vs garbage":
  1. Coherence  — does it produce sensible Italian on-topic answers?
  2. Stopping   — does it emit <|im_end|> and stop, or run away to max_tokens?
     (the classic never-stops SFT failure mode)

Run from repo root:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python eval/validate_chat.py --model checkpoints_sft/skylar-medium-chat/best
"""

import argparse
import sys
import torch
from pathlib import Path
from tokenizers import Tokenizer, decoders

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.decoder import NanoTransformer
from utils.chatML import encode_chatml

DEFAULT_SYSTEM = "Sei Skylar, un assistente italiano esperto di normativa bancaria, legale ed europea. Rispondi in modo chiaro e conciso."

# (label, system, user) — covers general knowledge, domain, instruction-following,
# the RAG query-gen candidate task, conversational, and a short stop-stress greeting.
TESTS = [
    ("general",     DEFAULT_SYSTEM, "Cos'è la Costituzione italiana? Rispondi in due frasi."),
    ("banking",     DEFAULT_SYSTEM, "Cosa disciplina la direttiva europea PSD2?"),
    ("instruction", DEFAULT_SYSTEM, "Elenca tre principi fondamentali della Costituzione italiana."),
    ("legal_dora",  DEFAULT_SYSTEM, "Che cos'è il regolamento DORA in ambito finanziario?"),
    ("query_gen",   "Sei un assistente che genera query di ricerca. Data una richiesta, produci 3 brevi query utili a recuperare documenti pertinenti.",
                    "Requisiti di adeguatezza patrimoniale delle banche."),
    ("conversational", DEFAULT_SYSTEM, "Ciao! Mi spieghi a grandi linee come funziona un mutuo?"),
    ("short_fact",  DEFAULT_SYSTEM, "Qual è la capitale d'Italia?"),
    ("stop_stress", DEFAULT_SYSTEM, "Ciao!"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...")
    model = NanoTransformer.from_pretrained(args.model).to(device).eval()
    tok = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    if tok.decoder is None:
        tok.decoder = decoders.ByteLevel()

    im_end = tok.token_to_id("<|im_end|>")
    eos_ids = [i for i in (im_end, tok.token_to_id("<|endoftext|>"), tok.token_to_id("<eos>")) if i is not None]

    n_stopped = 0
    n_total = len(TESTS)
    print("=" * 78)
    for label, system, user in TESTS:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = encode_chatml(messages, tok, add_generation_prompt=True)
        input_ids = torch.tensor([ids], device=device)
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=eos_ids,
                use_cache=True,
            )
        gen_ids = out[0][prompt_len:].tolist()
        stopped = any(t in eos_ids for t in gen_ids)
        # strip trailing eos for clean display
        clean = [t for t in gen_ids if t not in eos_ids]
        text = tok.decode(clean).strip()
        n_new = len(gen_ids)
        n_stopped += int(stopped)
        flag = "✅ STOP" if stopped else "⚠️  RUNAWAY (no <|im_end|>)"
        print(f"\n[{label}]  {flag}  ({n_new} tok)")
        print(f"  Q: {user}")
        print(f"  A: {text}")
    print("\n" + "=" * 78)
    print(f"  Stopped cleanly: {n_stopped}/{n_total}")
    print("  (RUNAWAY on any line = the never-stops bug → SFT needs the im_end boost / more epochs)")
    print("=" * 78)


if __name__ == "__main__":
    main()
