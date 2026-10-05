# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
SFT Diagnostic Tool — finds exactly why the chat model gives nonsense.

Checks:
  1. Tokenization consistency (train vs inference)
  2. Loss mask correctness
  3. Catastrophic forgetting (weight drift from base)
  4. Actual model predictions on a test prompt
  5. Embedding health (norms, NaN, dead neurons)

Usage:
  python diagnose_sft_drift.py \
    --base_model checkpoints/skylar-100M-Base \
    --sft_model checkpoints/skylar-100M-Chat-v3/best \
    --sft_data data/distilled_sft_dataset.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo root: runs without PYTHONPATH
from models.decoder import Skylar2ForCausalLM
from utils.chatML import encode_chatml, create_loss_mask, get_chatml_ids


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def load_model(path: str, device: str) -> tuple[Skylar2ForCausalLM, Tokenizer]:
    model = Skylar2ForCausalLM.from_pretrained(path).to(device).eval()
    tok_path = Path(path) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# TEST 1: Tokenization consistency
# ─────────────────────────────────────────────────────────────

def test_tokenization(tokenizer: Tokenizer) -> bool:
    separator("TEST 1: Tokenization Consistency (train vs inference)")

    messages = [
        {"role": "system", "content": "Sei un assistente utile."},
        {"role": "user", "content": "Ciao, come stai?"},
        {"role": "assistant", "content": "Sto bene, grazie!"},
    ]

    # What create_loss_mask produces (used during training)
    train_ids, train_labels = create_loss_mask(messages, tokenizer)

    # What encode_chatml produces (used during inference)
    # Full conversation without generation prompt
    infer_ids_full = encode_chatml(messages, tokenizer, add_generation_prompt=False)

    # With generation prompt (what chat.py sends)
    messages_no_assistant = messages[:-1]
    infer_ids_prompt = encode_chatml(messages_no_assistant, tokenizer, add_generation_prompt=True)

    print(f"  create_loss_mask tokens:  {len(train_ids)}")
    print(f"  encode_chatml (full):     {len(infer_ids_full)}")
    print(f"  encode_chatml (+ prompt): {len(infer_ids_prompt)}")

    # Check: full conversation tokens must match exactly
    match = train_ids == infer_ids_full
    print(f"\n  Token match (train vs infer full): {'✅ MATCH' if match else '❌ MISMATCH'}")

    if not match:
        # Find first divergence
        min_len = min(len(train_ids), len(infer_ids_full))
        for i in range(min_len):
            if train_ids[i] != infer_ids_full[i]:
                ctx_start = max(0, i - 3)
                ctx_end = min(min_len, i + 4)
                print(f"  First divergence at position {i}:")
                print(f"    train: {train_ids[ctx_start:ctx_end]}")
                print(f"    infer: {infer_ids_full[ctx_start:ctx_end]}")
                # Decode around divergence
                print(f"    train decoded: {repr(tokenizer.decode(train_ids[ctx_start:ctx_end]))}")
                print(f"    infer decoded: {repr(tokenizer.decode(infer_ids_full[ctx_start:ctx_end]))}")
                break
        if len(train_ids) != len(infer_ids_full):
            print(f"  Length difference: {len(train_ids)} vs {len(infer_ids_full)}")

    # Check: inference prompt must be a prefix of training tokens
    prefix_len = len(infer_ids_prompt)
    is_prefix = train_ids[:prefix_len] == infer_ids_prompt
    print(f"  Inference prompt is prefix of training: {'✅ YES' if is_prefix else '❌ NO'}")

    if not is_prefix:
        min_len = min(prefix_len, len(train_ids))
        for i in range(min_len):
            if train_ids[i] != infer_ids_prompt[i]:
                print(f"  First prefix divergence at position {i}:")
                print(f"    train:  {train_ids[max(0, i - 2):i + 3]}")
                print(f"    prompt: {infer_ids_prompt[max(0, i - 2):i + 3]}")
                break

    return match and is_prefix


# ─────────────────────────────────────────────────────────────
# TEST 2: Loss mask correctness
# ─────────────────────────────────────────────────────────────

def test_loss_mask(tokenizer: Tokenizer) -> bool:
    separator("TEST 2: Loss Mask Correctness")

    messages = [
        {"role": "system", "content": "Sei un assistente."},
        {"role": "user", "content": "Quanto fa 2+2?"},
        {"role": "assistant", "content": "Fa quattro."},
    ]

    token_ids, labels = create_loss_mask(messages, tokenizer)

    # Find assistant content boundaries
    assistant_content = "Fa quattro."
    assistant_ids = tokenizer.encode(assistant_content, add_special_tokens=False).ids

    # Count masked vs unmasked
    n_total = len(labels)
    n_masked = sum(1 for l in labels if l == -100)
    n_active = n_total - n_masked

    print(f"  Total tokens:    {n_total}")
    print(f"  Masked (-100):   {n_masked}")
    print(f"  Active (loss):   {n_active}")
    print(f"  Assistant content tokens: {len(assistant_ids)}")

    # Get im_end tokens to check they're in loss for assistant
    _, ime_ids = get_chatml_ids(tokenizer)
    expected_active = len(assistant_ids) + len(ime_ids)
    print(f"  Expected active (content + im_end): {expected_active}")
    print(f"  Actual active: {n_active}")

    ok = n_active == expected_active
    print(f"  Loss mask count: {'✅ CORRECT' if ok else '❌ WRONG'}")

    # Show the actual loss mask visually
    print(f"\n  Token-by-token loss mask (first 60 tokens):")
    print(f"  {'Pos':>4} {'ID':>6} {'Label':>6} {'Decoded':<30} {'Status'}")
    print(f"  {'─' * 4} {'─' * 6} {'─' * 6} {'─' * 30} {'─' * 10}")

    for i in range(min(60, len(token_ids))):
        decoded = repr(tokenizer.decode([token_ids[i]]))[1:-1][:28]
        status = "ACTIVE" if labels[i] != -100 else "masked"
        label_str = str(labels[i]) if labels[i] != -100 else "-100"
        marker = "◀ LOSS" if labels[i] != -100 else ""
        print(f"  {i:>4} {token_ids[i]:>6} {label_str:>6} {decoded:<30} {marker}")

    # Verify shift logic (what SFTDataset does)
    shifted_ids = token_ids[:-1]
    shifted_labels = labels[1:]
    # At position i: model sees shifted_ids[i], predicts shifted_labels[i]
    # shifted_labels[i] should be token_ids[i+1] when active
    shift_ok = True
    for i in range(len(shifted_labels)):
        if shifted_labels[i] != -100:
            if shifted_labels[i] != token_ids[i + 1]:
                print(f"  ❌ Shift error at pos {i}: label={shifted_labels[i]} but next token={token_ids[i + 1]}")
                shift_ok = False
                break
    print(f"\n  Causal shift alignment: {'✅ CORRECT' if shift_ok else '❌ BROKEN'}")

    return ok and shift_ok


# ─────────────────────────────────────────────────────────────
# TEST 3: Catastrophic forgetting
# ─────────────────────────────────────────────────────────────

def test_forgetting(base_model: Skylar2ForCausalLM, sft_model: Skylar2ForCausalLM) -> None:
    separator("TEST 3: Catastrophic Forgetting Check")

    base_params = dict(base_model.named_parameters())
    sft_params = dict(sft_model.named_parameters())

    # Check parameter shapes match
    shape_mismatch = []
    for name in base_params:
        if name in sft_params:
            if base_params[name].shape != sft_params[name].shape:
                shape_mismatch.append((name, base_params[name].shape, sft_params[name].shape))

    if shape_mismatch:
        print("  ❌ SHAPE MISMATCHES (embeddings resized?):")
        for name, bs, ss in shape_mismatch:
            print(f"    {name}: base={bs} vs sft={ss}")
        print("\n  ⚠️  This means embeddings were resized during SFT!")
        print("  The old special-token bug is still active in the SFT weights.")
        print("  You need to RE-RUN SFT with the fixed code (sub-token approach).")
        return

    print("  Parameter shapes: ✅ All match (no embedding resize)")

    # Compute weight drift per layer
    print(f"\n  {'Layer':<50} {'L2 Drift':>10} {'Cosine':>8} {'Status'}")
    print(f"  {'─' * 50} {'─' * 10} {'─' * 8} {'─' * 10}")

    total_drift = 0.0
    n_params = 0
    max_drift_name = ""
    max_drift_val = 0.0

    for name in sorted(base_params.keys()):
        if name not in sft_params:
            continue
        bp = base_params[name].float().flatten()
        sp = sft_params[name].float().flatten()

        l2 = (bp - sp).norm().item()
        cos = F.cosine_similarity(bp.unsqueeze(0), sp.unsqueeze(0)).item()
        total_drift += l2
        n_params += 1

        if l2 > max_drift_val:
            max_drift_val = l2
            max_drift_name = name

        # Only print layers with significant drift
        if l2 > 0.01:
            status = "🔴 HIGH" if l2 > 1.0 else ("🟡 MED" if l2 > 0.1 else "🟢 LOW")
            print(f"  {name:<50} {l2:>10.4f} {cos:>8.4f} {status}")

    avg_drift = total_drift / max(n_params, 1)
    print(f"\n  Total L2 drift:  {total_drift:.4f}")
    print(f"  Average drift:   {avg_drift:.6f}")
    print(f"  Max drift layer: {max_drift_name} ({max_drift_val:.4f})")

    if total_drift < 0.001:
        print("\n  ⚠️  SFT weights are nearly IDENTICAL to base model.")
        print("  The SFT training probably didn't converge or LR was too low.")
    elif avg_drift > 5.0:
        print("\n  ⚠️  MASSIVE weight drift — possible catastrophic forgetting.")
        print("  Try lower LR (1e-5 or 5e-6) and fewer steps.")
    else:
        print(f"\n  ✅ Weight drift looks reasonable.")


# ─────────────────────────────────────────────────────────────
# TEST 4: Model predictions
# ─────────────────────────────────────────────────────────────

def test_predictions(sft_model: Skylar2ForCausalLM, tokenizer: Tokenizer,
                     device: str) -> None:
    separator("TEST 4: Model Predictions on Test Prompt")

    test_cases = [
        ("Sei un assistente utile.", "Ciao, come stai?"),
        ("Sei un assistente utile.", "Quanto fa 2+2?"),
        ("Rispondi in italiano.", "Che cos'è il sole?"),
    ]

    for system, user_msg in test_cases:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        token_ids = encode_chatml(messages, tokenizer, add_generation_prompt=True)
        input_ids = torch.tensor([token_ids], device=device)

        print(f"\n  Prompt: \"{user_msg}\"")
        print(f"  Input tokens: {len(token_ids)}")

        # Decode the full input to verify it looks correct
        decoded_input = tokenizer.decode(token_ids)
        # Show last 100 chars to verify generation prompt
        print(f"  Input ends with: ...{repr(decoded_input[-80:])}")

        # Forward pass
        with torch.no_grad():
            out = sft_model(input_ids)
            logits = out["logits"][0, -1, :]  # Last position

        # Top-10 predictions
        probs = F.softmax(logits, dim=-1)
        top_vals, top_ids = probs.topk(10)

        print(f"\n  Top-10 next token predictions:")
        print(f"  {'Rank':>4} {'ID':>6} {'Prob':>8} {'Token'}")
        for rank, (prob, tid) in enumerate(zip(top_vals, top_ids)):
            decoded = repr(tokenizer.decode([tid.item()]))[1:-1]
            print(f"  {rank + 1:>4} {tid.item():>6} {prob.item():>8.4f} {decoded}")

        # Greedy generate 50 tokens
        generated = []
        cur_ids = input_ids.clone()
        for _ in range(50):
            with torch.no_grad():
                out = sft_model(cur_ids[:, -sft_model.config.max_seq_len:])
                next_logit = out["logits"][0, -1, :]
            next_token = next_logit.argmax().unsqueeze(0).unsqueeze(0)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)
            generated.append(next_token.item())
            # Stop on im_end
            decoded_so_far = tokenizer.decode(generated)
            if "<|im_end|>" in decoded_so_far:
                decoded_so_far = decoded_so_far.split("<|im_end|>")[0]
                break

        decoded_gen = tokenizer.decode(generated)
        if "<|im_end|>" in decoded_gen:
            decoded_gen = decoded_gen.split("<|im_end|>")[0]

        print(f"\n  Greedy output: {repr(decoded_gen[:200])}")


# ─────────────────────────────────────────────────────────────
# TEST 5: Embedding health
# ─────────────────────────────────────────────────────────────

def test_embedding_health(model: Skylar2ForCausalLM, label: str) -> None:
    separator(f"TEST 5: Embedding Health ({label})")

    emb = model.token_emb.weight.data

    print(f"  Shape: {emb.shape}")
    print(f"  Dtype: {emb.dtype}")
    print(f"  Mean:  {emb.mean().item():.6f}")
    print(f"  Std:   {emb.std().item():.6f}")
    print(f"  Min:   {emb.min().item():.6f}")
    print(f"  Max:   {emb.max().item():.6f}")

    # Check for NaN/Inf
    n_nan = emb.isnan().sum().item()
    n_inf = emb.isinf().sum().item()
    print(f"  NaN:   {n_nan}")
    print(f"  Inf:   {n_inf}")

    # Check for dead rows (all zeros)
    row_norms = emb.norm(dim=1)
    n_dead = (row_norms < 1e-8).sum().item()
    print(f"  Dead rows (norm < 1e-8): {n_dead}")

    # Check if last rows look different (sign of resize)
    if emb.shape[0] > 40960:
        print(f"\n  ⚠️  Vocab size is {emb.shape[0]} > 40960 — embeddings were resized!")
        print(f"  Last 5 row norms: {row_norms[-5:].tolist()}")
        print(f"  Mean row norm (first 40960): {row_norms[:40960].mean().item():.4f}")
        print(f"  Mean row norm (extra rows):  {row_norms[40960:].mean().item():.4f}")


# ─────────────────────────────────────────────────────────────
# TEST 6: SFT data sanity check
# ─────────────────────────────────────────────────────────────

def test_sft_data(data_path: str, tokenizer: Tokenizer) -> None:
    separator("TEST 6: SFT Data Sanity Check")

    if not Path(data_path).exists():
        print(f"  ⚠ File not found: {data_path}")
        return

    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"  Total examples: {len(examples)}")

    # Check format
    n_valid = 0
    n_no_assistant = 0
    n_empty_assistant = 0
    total_tokens = 0
    total_assistant_tokens = 0
    role_counts = {"system": 0, "user": 0, "assistant": 0}

    for ex in examples:
        msgs = ex.get("messages", [])
        has_assistant = False
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in role_counts:
                role_counts[role] += 1
            if role == "assistant":
                has_assistant = True
                if not content.strip():
                    n_empty_assistant += 1

        if has_assistant:
            n_valid += 1
        else:
            n_no_assistant += 1

        # Tokenize to count
        try:
            ids, labels = create_loss_mask(msgs, tokenizer)
            total_tokens += len(ids)
            total_assistant_tokens += sum(1 for l in labels if l != -100)
        except Exception:
            pass

    print(f"  Valid examples (has assistant): {n_valid}")
    print(f"  No assistant response: {n_no_assistant}")
    print(f"  Empty assistant content: {n_empty_assistant}")
    print(f"  Role counts: {role_counts}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Assistant tokens (in loss): {total_assistant_tokens:,}")
    if total_tokens > 0:
        ratio = total_assistant_tokens / total_tokens * 100
        print(f"  Assistant token ratio: {ratio:.1f}%")
        if ratio < 10:
            print("  ⚠️  Very low assistant ratio — model barely sees supervision signal")

    # Show 3 random examples
    import random
    random.seed(42)
    sample = random.sample(examples, min(3, len(examples)))
    print(f"\n  Sample examples:")
    for i, ex in enumerate(sample):
        print(f"\n  ── Example {i + 1} ──")
        for m in ex["messages"]:
            content_preview = m["content"][:80].replace("\n", " ")
            print(f"    [{m['role']}] {content_preview}...")


# ─────────────────────────────────────────────────────────────
# TEST 7: ChatML sub-token encoding
# ─────────────────────────────────────────────────────────────

def test_chatml_tokens(tokenizer: Tokenizer, base_vocab_size: int) -> bool:
    separator("TEST 7: ChatML Token Encoding")

    ims_id = tokenizer.token_to_id("<|im_start|>")
    ime_id = tokenizer.token_to_id("<|im_end|>")

    print(f"  <|im_start|> token ID: {ims_id}")
    print(f"  <|im_end|> token ID:   {ime_id}")

    ims_ids, ime_ids = get_chatml_ids(tokenizer)
    print(f"\n  get_chatml_ids() returns:")
    print(f"    im_start IDs: {ims_ids} (len={len(ims_ids)})")
    print(f"    im_end IDs:   {ime_ids} (len={len(ime_ids)})")

    # Determine: native (trained with BPE) vs added post-pretrain
    ok = True
    if ims_id is not None and ims_id < base_vocab_size:
        print(f"\n  ✅ <|im_start|> is NATIVE (ID {ims_id} < vocab {base_vocab_size})")
        print(f"     Trained with BPE — model saw it during pretrain.")
    elif ims_id is not None and ims_id >= base_vocab_size:
        print(f"\n  🔴 <|im_start|> was ADDED post-pretrain (ID {ims_id} >= vocab {base_vocab_size})")
        print(f"     Embedding row is random — model never learned this token.")
        ok = False
    else:
        print(f"\n  ℹ️  <|im_start|> encoded as sub-tokens (no single ID)")
        print(f"     Using sub-token approach for ChatML.")

    if ime_id is not None and ime_id < base_vocab_size:
        print(f"  ✅ <|im_end|> is NATIVE (ID {ime_id} < vocab {base_vocab_size})")
    elif ime_id is not None and ime_id >= base_vocab_size:
        print(f"  🔴 <|im_end|> was ADDED post-pretrain (ID {ime_id} >= vocab {base_vocab_size})")
        ok = False
    else:
        print(f"  ℹ️  <|im_end|> encoded as sub-tokens (no single ID)")

    # Verify encoding works: a ChatML fragment must contain the right IDs
    test_ids = tokenizer.encode("<|im_start|>user\nCiao<|im_end|>", add_special_tokens=False).ids
    has_ims = any(t in ims_ids for t in test_ids)
    has_ime = any(t in ime_ids for t in test_ids)
    print(f"\n  Encoding check: '<|im_start|>user\\nCiao<|im_end|>'")
    print(f"    IDs: {test_ids}")
    print(f"    Contains im_start: {'✅' if has_ims else '❌'}")
    print(f"    Contains im_end:   {'✅' if has_ime else '❌'}")

    if not has_ims or not has_ime:
        ok = False

    return ok


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT Diagnostic Tool")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base (pretrained) model")
    parser.add_argument("--sft_model", type=str, required=True,
                        help="Path to SFT (chat) model")
    parser.add_argument("--sft_data", type=str, default="data/sft_train.jsonl",
                        help="Path to SFT training data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  SFT DIAGNOSTIC TOOL")
    print(f"{'=' * 70}")
    print(f"  Base model: {args.base_model}")
    print(f"  SFT model:  {args.sft_model}")
    print(f"  SFT data:   {args.sft_data}")
    print(f"  Device:     {args.device}")

    # Load models
    print(f"\n  Loading base model...")
    base_model, base_tokenizer = load_model(args.base_model, args.device)
    print(f"  Base vocab: {base_model.config.vocab_size}")

    print(f"  Loading SFT model...")
    sft_model, sft_tokenizer = load_model(args.sft_model, args.device)
    print(f"  SFT vocab:  {sft_model.config.vocab_size}")

    # Vocab size check
    if base_model.config.vocab_size != sft_model.config.vocab_size:
        print(f"\n  🔴 VOCAB SIZE MISMATCH: base={base_model.config.vocab_size} vs sft={sft_model.config.vocab_size}")
        print(f"  This confirms the old embedding-resize bug is in the SFT weights.")
        print(f"  You MUST re-run SFT with the fixed code (sub-token approach, no resize).")
        print(f"  Continuing diagnostics anyway...\n")

    # Run all tests
    chatml_ok = test_chatml_tokens(sft_tokenizer, base_model.config.vocab_size)
    tok_ok = test_tokenization(sft_tokenizer)
    mask_ok = test_loss_mask(sft_tokenizer)
    test_forgetting(base_model, sft_model)
    test_predictions(sft_model, sft_tokenizer, args.device)
    test_embedding_health(sft_model, "SFT model")
    test_sft_data(args.sft_data, sft_tokenizer)

    # Summary
    separator("DIAGNOSIS SUMMARY")
    issues = []

    if base_model.config.vocab_size != sft_model.config.vocab_size:
        issues.append("🔴 CRITICAL: Vocab size mismatch — SFT was run with old buggy code (embedding resize)")

    if not chatml_ok:
        issues.append("🔴 CRITICAL: ChatML special tokens not properly encoded")

    if not tok_ok:
        issues.append("🔴 CRITICAL: Tokenization mismatch between training and inference")

    if not mask_ok:
        issues.append("🟡 WARNING: Loss mask may be incorrect")

    if issues:
        print("  Issues found:")
        for issue in issues:
            print(f"    {issue}")
        print()
        if any("Vocab size" in i for i in issues):
            print("  ═══════════════════════════════════════════════════")
            print("  FIX: Re-run SFT from the base model using the")
            print("  fixed code from HANDOFF-23.02 (sub-token approach).")
            print("  The current SFT weights are UNUSABLE because they")
            print("  were trained with resized embeddings.")
            print("  ═══════════════════════════════════════════════════")
    else:
        print("  No critical issues found in tokenization/masks.")
        print("  Check TEST 3 (forgetting) and TEST 4 (predictions) output above.")
        print("  If predictions are still bad, likely causes:")
        print("    - LR too high → catastrophic forgetting")
        print("    - LR too low / too few steps → didn't learn format")
        print("    - SFT data quality issues")
        print("    - Need more epochs on small dataset")


if __name__ == "__main__":
    main()
