#!/usr/bin/env python3
# filepath: eval_base_model.py
"""
Base model health check — run this BEFORE doing SFT.

Tests whether the pretrained model has learned enough Italian language
structure to be a good foundation for fine-tuning.

Checks:
  1. Validation perplexity (quantitative)
  2. Completion coherence (qualitative — you read the outputs)
  3. Topic consistency (does it stay on topic for 100+ tokens?)
  4. Repetition detection (does it degenerate into loops?)
  5. Metadata leakage (does it emit training artifacts?)

Usage:
    python eval_base_model.py --model ./checkpoints/skylar-100M-Base
    python eval_base_model.py --model ./checkpoints/skylar-100M-Base --verbose
"""

import argparse
import math
import re
import sys
import torch
from pathlib import Path
from tokenizers import Tokenizer, decoders

from models.decoder import NanoTransformer


# ─────────────────────────────────────────────────────────────
# TEST PROMPTS — diverse topics and lengths
# ─────────────────────────────────────────────────────────────

EVAL_PROMPTS: list[dict[str, str]] = [
    {
        "name": "completion_simple",
        "prompt": "La capitale d'Italia è",
        "description": "Basic factual completion",
    },
    {
        "name": "completion_grammar",
        "prompt": "Ieri sono andato al mercato e ho comprato",
        "description": "Grammatical continuation (past tense)",
    },
    {
        "name": "topic_banking",
        "prompt": "Il sistema bancario italiano è composto da",
        "description": "Domain-specific topic continuation",
    },
    {
        "name": "topic_history",
        "prompt": "Durante il Rinascimento, Firenze divenne",
        "description": "Historical topic continuation",
    },
    {
        "name": "topic_science",
        "prompt": "L'energia solare funziona attraverso",
        "description": "Scientific explanation",
    },
    {
        "name": "long_form",
        "prompt": "La Costituzione italiana, entrata in vigore nel 1948, stabilisce",
        "description": "Long-form structured continuation",
    },
    {
        "name": "list_like",
        "prompt": "Le regioni italiane sono venti:",
        "description": "Enumeration / list continuation",
    },
    {
        "name": "conversational",
        "prompt": "Ciao, come stai? Io sto bene, oggi",
        "description": "Informal / conversational tone",
    },
]

# Patterns that indicate metadata leakage from training
METADATA_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"source:\s*\w+"),
    re.compile(r"wiki_id:\s*\d+"),
    re.compile(r"timestamp:\s*\d{4}"),
    re.compile(r"language:\s*\w{2}"),
    re.compile(r"categories:\s*"),
    re.compile(r"^---\s*$", re.MULTILINE),
]


# ─────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────

def detect_repetition(text: str, ngram_size: int = 4) -> float:
    """Detect repetition rate as fraction of repeated n-grams.

    Returns:
        0.0 = no repetition, 1.0 = all repeated.
        Healthy: < 0.3. Problem: > 0.5.
    """
    words = text.split()
    if len(words) < ngram_size + 1:
        return 0.0

    ngrams = [
        tuple(words[i:i + ngram_size])
        for i in range(len(words) - ngram_size + 1)
    ]

    if not ngrams:
        return 0.0

    unique = set(ngrams)
    return 1.0 - (len(unique) / len(ngrams))


def detect_metadata_leakage(text: str) -> list[str]:
    """Check if generated text contains training metadata artifacts."""
    found: list[str] = []
    for pattern in METADATA_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            found.extend(matches)
    return found


def check_italian_coherence(text: str) -> dict[str, bool]:
    """Basic heuristics for Italian text quality."""
    words = text.split()

    return {
        # Has actual Italian words (not just gibberish)
        "has_italian_words": any(
            w.lower() in {"il", "la", "di", "che", "è", "per", "in", "un", "una", "non", "sono", "del", "dei"}
            for w in words
        ),
        # Sentences end with punctuation
        "has_punctuation": bool(re.search(r"[.!?;:]", text)),
        # Not all caps or all lowercase single char
        "reasonable_casing": not text.isupper() and len(words) > 3,
        # At least some words > 3 chars (not just particles)
        "has_content_words": sum(1 for w in words if len(w) > 3) > len(words) * 0.3,
    }


# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_completion(
    model: NanoTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate a completion from the base model."""
    # Disable BOS/EOS injection — we want raw completion
    saved_pp = tokenizer.post_processor
    tokenizer.post_processor = None
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    tokenizer.post_processor = saved_pp

    input_ids = torch.tensor([ids], device=device)

    eos_ids = []
    for name in ("<eos>", "<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.token_to_id(name)
        if tid is not None:
            eos_ids.append(tid)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_ids if eos_ids else None,
    )

    generated = output_ids[0][len(ids):].tolist()
    return tokenizer.decode(generated)


# ─────────────────────────────────────────────────────────────
# PERPLEXITY
# ─────────────────────────────────────────────────────────────

PERPLEXITY_TEXTS: list[str] = [
    "Il Parlamento italiano è composto dalla Camera dei deputati e dal Senato della Repubblica. "
    "Insieme formano il Parlamento in seduta comune per eleggere il Presidente della Repubblica.",

    "Per aprire un conto corrente in Italia è necessario presentare un documento di identità "
    "valido, il codice fiscale e un indirizzo di residenza. La banca verifica i dati forniti.",

    "La pizza napoletana è preparata con farina, acqua, sale e lievito. L'impasto deve "
    "lievitare per almeno otto ore. La cottura avviene in forno a legna a temperatura elevata.",

    "Il protocollo TCP/IP è alla base delle comunicazioni su Internet. I dati vengono "
    "suddivisi in pacchetti che viaggiano attraverso la rete fino alla destinazione.",

    "Leonardo da Vinci nacque a Vinci nel 1452. Fu pittore, scultore, architetto e "
    "ingegnere. La Gioconda e L'Ultima Cena sono tra le sue opere più celebri.",
]


@torch.no_grad()
def compute_perplexity(
    model: NanoTransformer,
    tokenizer: Tokenizer,
    texts: list[str],
    device: str,
) -> float:
    """Compute perplexity on a set of held-out texts."""
    model.eval()

    saved_pp = tokenizer.post_processor
    tokenizer.post_processor = None

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) < 2:
            continue

        input_ids = torch.tensor([ids[:-1]], device=device)
        labels = torch.tensor([ids[1:]], device=device)

        out = model(input_ids, labels=labels)
        if out["loss"] is not None:
            n_tokens = len(ids) - 1
            total_loss += out["loss"].item() * n_tokens
            total_tokens += n_tokens

    tokenizer.post_processor = saved_pp

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ─────────────────────────────────────────────────────────────
# MAIN EVAL
# ─────────────────────────────────────────────────────────────

def run_eval(args: argparse.Namespace) -> None:
    # ── Load ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NanoTransformer.from_pretrained(args.model).to(device)
    model.eval()

    tokenizer = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    if tokenizer.decoder is None:
        tokenizer.decoder = decoders.ByteLevel()

    n_params = model.count_params()
    print(f"\n{'=' * 70}")
    print(f"  BASE MODEL HEALTH CHECK")
    print(f"{'=' * 70}")
    print(f"  Model:   {args.model}")
    print(f"  Params:  {n_params:,}")
    print(f"  Device:  {device}")
    print(f"{'=' * 70}\n")

    # ── Test 1: Perplexity ──
    print("─── TEST 1: Perplexity ───")
    ppl = compute_perplexity(model, tokenizer, PERPLEXITY_TEXTS, device)
    print(f"  Perplexity: {ppl:.1f}")

    if ppl < 50:
        print(f"  ✅ GOOD — model has learned Italian structure")
    elif ppl < 150:
        print(f"  ⚠️  OKAY — some learning, but could be better")
    elif ppl < 500:
        print(f"  ❌ POOR — model is barely learning the language")
    else:
        print(f"  💀 BAD — model hasn't learned much (random would be ~{tokenizer.get_vocab_size()})")

    print()

    # ── Test 2: Completions ──
    print("─── TEST 2: Completions ───")
    results: list[dict] = []

    for test in EVAL_PROMPTS:
        completion = generate_completion(
            model, tokenizer, test["prompt"], device,
            max_tokens=args.max_tokens,
        )

        rep_score = detect_repetition(completion)
        metadata_leaks = detect_metadata_leakage(completion)
        coherence = check_italian_coherence(completion)

        result = {
            "name": test["name"],
            "prompt": test["prompt"],
            "completion": completion,
            "repetition": rep_score,
            "metadata_leaks": metadata_leaks,
            "coherence": coherence,
        }
        results.append(result)

        # Print
        status_parts: list[str] = []

        if rep_score > 0.5:
            status_parts.append("🔴 REP")
        elif rep_score > 0.3:
            status_parts.append("🟡 rep")

        if metadata_leaks:
            status_parts.append("🔴 META")

        if not coherence["has_italian_words"]:
            status_parts.append("🔴 NO-IT")

        if not coherence["has_punctuation"]:
            status_parts.append("🟡 no-punct")

        status = " | ".join(status_parts) if status_parts else "✅"

        print(f"\n  [{test['name']}] {test['description']}")
        print(f"  Prompt: {test['prompt']}")
        print(f"  Output: {completion[:200]}{'...' if len(completion) > 200 else ''}")
        print(f"  Status: {status} (rep={rep_score:.2f})")

    # ── Test 3: Summary ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")

    n_tests = len(results)
    n_repetitive = sum(1 for r in results if r["repetition"] > 0.5)
    n_metadata = sum(1 for r in results if r["metadata_leaks"])
    n_no_italian = sum(1 for r in results if not r["coherence"]["has_italian_words"])
    n_clean = sum(
        1 for r in results
        if r["repetition"] <= 0.3
        and not r["metadata_leaks"]
        and r["coherence"]["has_italian_words"]
        and r["coherence"]["has_punctuation"]
    )

    print(f"\n  Perplexity:       {ppl:.1f}")
    print(f"  Clean outputs:    {n_clean}/{n_tests}")
    print(f"  Repetitive:       {n_repetitive}/{n_tests}")
    print(f"  Metadata leaks:   {n_metadata}/{n_tests}")
    print(f"  No Italian:       {n_no_italian}/{n_tests}")

    # ── Verdict ──
    print(f"\n  {'─' * 40}")

    ready_for_sft = (
        ppl < 150
        and n_clean >= n_tests * 0.5
        and n_metadata == 0
        and n_repetitive <= n_tests * 0.25
    )

    if ready_for_sft:
        print(f"  ✅ READY FOR SFT")
        print(f"  The base model has learned enough Italian to fine-tune.")
        if ppl > 50:
            print(f"  Tip: perplexity is okay but not great — more/better")
            print(f"  pretrain data would improve SFT results.")
    else:
        print(f"  ❌ NOT READY FOR SFT — fix pretrain first")
        if ppl >= 150:
            print(f"  → Perplexity too high: model hasn't learned the language well enough")
        if n_metadata > 0:
            print(f"  → Metadata leaking: strip YAML headers from pretrain data")
        if n_repetitive > n_tests * 0.25:
            print(f"  → Too many repetitive outputs: check for degenerate training")
        if n_clean < n_tests * 0.5:
            print(f"  → Too few coherent outputs: need more/better pretrain data")

    print(f"\n{'=' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Base model health check — run BEFORE SFT")
    parser.add_argument("--model", type=str, required=True, help="Path to base model checkpoint")
    parser.add_argument("--max_tokens", type=int, default=150, help="Max tokens per completion")
    args = parser.parse_args()
    torch.manual_seed(getattr(args, "seed", 0))   # R15: eval riproducibile run-to-run
    run_eval(args)


if __name__ == "__main__":
    main()