#!/usr/bin/env python3
"""
Diagnostica completa della pipeline SFT per il bug <|im_end|>.

Lancia dalla cartella del progetto:
  python diagnose_sft.py \
    --tokenizer ./checkpoints/skylar-100M-Chat/best/tokenizer.json \
    --data data/sft_train.jsonl \
    --base_model ./checkpoints/skylar-100M-Base \
    --sft_model ./checkpoints/skylar-100M-Chat/best \
    --seq_len 512

Controlla:
  1. Token IDs speciali nel tokenizer
  2. Post-processor behavior (BOS/EOS injection involontaria)
  3. Sequenza ChatML raw (prima dello shift)
  4. Sequenza ChatML dopo lo shift (come entra nel modello)
  5. Distribuzione labels nel dataset: quante volte appare im_end vs -100
  6. Logits del modello SFT su un esempio: probabilità assegnata a im_end
  7. Embedding distance: im_end vs altri token speciali vs token normali
  8. Confronto pretrain vs SFT embeddings per im_end
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Diagnose SFT im_end bug")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--data", required=True, help="Path to SFT JSONL")
    parser.add_argument("--base_model", default=None, help="Path to pretrained base model checkpoint")
    parser.add_argument("--sft_model", default=None, help="Path to SFT model checkpoint")
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()

    from tokenizers import Tokenizer, decoders

    tokenizer = Tokenizer.from_file(args.tokenizer)
    if tokenizer.decoder is None:
        tokenizer.decoder = decoders.ByteLevel()

    print("=" * 70)
    print("  DIAGNOSI PIPELINE SFT — Bug <|im_end|>")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────
    # TEST 1: Token IDs speciali
    # ─────────────────────────────────────────────────────────
    print("\n┌─ TEST 1: Token IDs speciali")
    special_tokens = ["<pad>", "<bos>", "<eos>", "<|im_start|>", "<|im_end|>",
                      "<think>", "</think>"]
    for tok in special_tokens:
        tid = tokenizer.token_to_id(tok)
        print(f"  │ {tok:20s} → id={tid}")
        if tid is None:
            print(f"  │ ⚠️  ATTENZIONE: {tok} non trovato nel vocabolario!")

    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    if im_end_id is None:
        print("  └─ ❌ FATAL: <|im_end|> non esiste nel tokenizer. Questo è il bug.")
        sys.exit(1)
    print("  └─ ✅ Token speciali presenti")

    # ─────────────────────────────────────────────────────────
    # TEST 2: Post-processor behavior
    # ─────────────────────────────────────────────────────────
    print("\n┌─ TEST 2: Post-processor — BOS/EOS injection")

    enc_with = tokenizer.encode("hello", add_special_tokens=True).ids
    enc_without = tokenizer.encode("hello", add_special_tokens=False).ids

    print(f"  │ encode('hello', add_special_tokens=True):  {enc_with}")
    print(f"  │ encode('hello', add_special_tokens=False): {enc_without}")

    has_bos_with = bos_id in enc_with if bos_id is not None else False
    has_eos_with = eos_id in enc_with if eos_id is not None else False
    has_bos_without = bos_id in enc_without if bos_id is not None else False
    has_eos_without = eos_id in enc_without if eos_id is not None else False

    print(f"  │ Con special tokens:  BOS={has_bos_with}, EOS={has_eos_with}")
    print(f"  │ Senza special tokens: BOS={has_bos_without}, EOS={has_eos_without}")

    if has_bos_without or has_eos_without:
        print("  └─ ❌ BUG: add_special_tokens=False inietta comunque BOS/EOS!")
        print("       Questo corrompe TUTTE le sequenze SFT!")
    else:
        print("  └─ ✅ Post-processor si disattiva correttamente")

    # ─────────────────────────────────────────────────────────
    # TEST 3: Sequenza ChatML raw (prima dello shift)
    # ─────────────────────────────────────────────────────────
    print("\n┌─ TEST 3: Sequenza ChatML — analisi token-per-token")

    # Import from the project
    sys.path.insert(0, ".")
    try:
        from utils.chatML import create_loss_mask, encode_chatml, get_chatml_ids
    except ImportError:
        print("  └─ ⚠️  Impossibile importare utils.chatML, skip")
        sys.exit(1)

    test_messages = [
        {"role": "user", "content": "ciao"},
        {"role": "assistant", "content": "Ciao!"},
    ]

    token_ids, labels = create_loss_mask(test_messages, tokenizer)

    print(f"  │ Lunghezza sequenza: {len(token_ids)} tokens")
    print(f"  │")
    print(f"  │ Sequenza completa (prima dello shift):")
    print(f"  │ {'Pos':>4s} │ {'TokenID':>8s} │ {'Label':>8s} │ {'Token':20s} │ Note")
    print(f"  │ {'─' * 4}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 20}─┼─{'─' * 20}")

    im_end_in_labels_raw = 0
    for i, (tid, lab) in enumerate(zip(token_ids, labels)):
        decoded = tokenizer.decode([tid])
        note = ""
        if tid == im_start_id:
            note = "← im_start"
        elif tid == im_end_id:
            note = "← im_end"
        elif tid == bos_id:
            note = "← BOS (INATTESO!)"
        elif tid == eos_id:
            note = "← EOS (INATTESO!)"

        if lab == im_end_id:
            note += " 🎯 LABEL=im_end"
            im_end_in_labels_raw += 1

        lab_str = str(lab) if lab != -100 else "-100"
        print(f"  │ {i:4d} │ {tid:8d} │ {lab_str:>8s} │ {repr(decoded):20s} │ {note}")

    print(f"  │")
    print(f"  │ im_end nei labels (raw): {im_end_in_labels_raw}")
    if im_end_in_labels_raw == 0:
        print(f"  └─ ❌ BUG: <|im_end|> non appare MAI nei labels!")
    else:
        print(f"  └─ ✅ <|im_end|> presente nei labels")

    # ─────────────────────────────────────────────────────────
    # TEST 4: Dopo lo shift (come entra nel modello)
    # ─────────────────────────────────────────────────────────
    print("\n┌─ TEST 4: Dopo lo shift causal (come SFTDataset lo passa al modello)")

    shifted_input = token_ids[:-1]
    shifted_labels = labels[1:]

    print(f"  │ input_ids: {len(shifted_input)} tokens")
    print(f"  │ labels:    {len(shifted_labels)} tokens")
    print(f"  │")

    im_end_in_shifted = sum(1 for l in shifted_labels if l == im_end_id)
    trainable_tokens = sum(1 for l in shifted_labels if l != -100)

    print(f"  │ Tokens trainabili (label != -100): {trainable_tokens}")
    print(f"  │ Di cui im_end: {im_end_in_shifted}")
    print(f"  │")

    # Show the last few positions where labels are not -100
    print(f"  │ Ultime posizioni con label trainabile:")
    for i in range(len(shifted_labels) - 1, -1, -1):
        if shifted_labels[i] != -100:
            inp_decoded = tokenizer.decode([shifted_input[i]])
            lab_decoded = tokenizer.decode([shifted_labels[i]])
            is_ime = " 🎯 im_end!" if shifted_labels[i] == im_end_id else ""
            print(f"  │   pos={i}: input={repr(inp_decoded)} → label={repr(lab_decoded)} (id={shifted_labels[i]}){is_ime}")
            if i < len(shifted_labels) - 5:
                break

    if im_end_in_shifted == 0:
        print(f"  └─ ❌ BUG: dopo lo shift, im_end non è nei labels!")
    else:
        print(f"  └─ ✅ im_end presente dopo lo shift")

    # ─────────────────────────────────────────────────────────
    # TEST 5: Distribuzione labels su TUTTO il dataset
    # ─────────────────────────────────────────────────────────
    print("\n┌─ TEST 5: Distribuzione labels su tutto il dataset SFT")

    with open(args.data, "r", encoding="utf-8") as f:
        examples = [json.loads(line.strip()) for line in f if line.strip()]

    total_im_end_labels = 0
    total_trainable = 0
    total_tokens = 0
    truncated_im_end = 0
    examples_without_im_end = 0

    for ex in examples:
        token_ids_ex, labels_ex = create_loss_mask(ex["messages"], tokenizer)
        # Apply same shift as SFTDataset
        inp = token_ids_ex[:-1]
        lab = labels_ex[1:]
        # Apply truncation as SFTDataset
        inp = inp[:args.seq_len]
        lab = lab[:args.seq_len]

        n_ime = sum(1 for l in lab if l == im_end_id)
        n_train = sum(1 for l in lab if l != -100)
        total_im_end_labels += n_ime
        total_trainable += n_train
        total_tokens += len(lab)

        if n_ime == 0:
            examples_without_im_end += 1
            # Check if it was truncated
            full_lab = labels_ex[1:]
            if any(l == im_end_id for l in full_lab):
                truncated_im_end += 1

    print(f"  │ Totale esempi:             {len(examples)}")
    print(f"  │ Totale tokens:             {total_tokens:,}")
    print(f"  │ Tokens trainabili:         {total_trainable:,}")
    print(f"  │ im_end nei labels:         {total_im_end_labels}")
    print(f"  │ Esempi SENZA im_end:       {examples_without_im_end}")
    print(f"  │ Di cui troncati da seq_len: {truncated_im_end}")
    print(f"  │ Ratio im_end/trainabili:   {total_im_end_labels / max(total_trainable, 1):.4%}")

    if total_im_end_labels == 0:
        print(f"  └─ ❌ BUG CRITICO: im_end non appare MAI nei labels del dataset!")
    elif examples_without_im_end > len(examples) * 0.5:
        print(f"  └─ ⚠️  Più della metà degli esempi non ha im_end (possibile troncamento)")
    else:
        print(f"  └─ ✅ im_end presente nel dataset")

    # ─────────────────────────────────────────────────────────
    # TEST 6: Confronto encode_chatml vs create_loss_mask
    # ─────────────────────────────────────────────────────────
    print("\n┌─ TEST 6: Consistenza encode_chatml vs create_loss_mask")

    enc_ids = encode_chatml(test_messages, tokenizer, add_generation_prompt=False)
    mask_ids, _ = create_loss_mask(test_messages, tokenizer)

    if enc_ids == mask_ids:
        print(f"  └─ ✅ Sequenze identiche ({len(enc_ids)} tokens)")
    else:
        print(f"  │ encode_chatml: {len(enc_ids)} tokens")
        print(f"  │ create_loss_mask: {len(mask_ids)} tokens")
        # Find first difference
        for i in range(min(len(enc_ids), len(mask_ids))):
            if enc_ids[i] != mask_ids[i]:
                print(f"  │ Prima differenza a pos {i}: encode={enc_ids[i]}, mask={mask_ids[i]}")
                break
        print(f"  └─ ❌ Sequenze DIVERSE — possibile bug in encode o mask!")

    # ─────────────────────────────────────────────────────────
    # TEST 7: Analisi modello SFT — logits per im_end
    # ─────────────────────────────────────────────────────────
    if args.sft_model:
        print("\n┌─ TEST 7: Analisi logits modello SFT")

        from models.decoder import Skylar2ForCausalLM

        model = Skylar2ForCausalLM.from_pretrained(args.sft_model).to("cuda").eval()

        # Encode a test prompt
        prompt_ids = encode_chatml([
            {"role": "user", "content": "ciao"}
        ], tokenizer, add_generation_prompt=True)

        input_tensor = torch.tensor([prompt_ids], device="cuda")

        with torch.no_grad():
            out = model(input_tensor)
            logits = out["logits"][0, -1, :]  # Last position logits

        probs = torch.softmax(logits, dim=-1)
        top_k = 20

        print(f"  │ Prompt: {len(prompt_ids)} tokens")
        print(f"  │ Logits all'ultima posizione (dove dovrebbe predire il primo token assistente):")
        print(f"  │")

        # Top-k tokens
        top_probs, top_ids = torch.topk(probs, top_k)
        print(f"  │ Top-{top_k} predizioni:")
        for i, (p, tid) in enumerate(zip(top_probs, top_ids)):
            tid = tid.item()
            decoded = tokenizer.decode([tid])
            marker = ""
            if tid == im_end_id:
                marker = " ← im_end"
            elif tid == im_start_id:
                marker = " ← im_start"
            elif tid == bos_id:
                marker = " ← BOS"
            elif tid == eos_id:
                marker = " ← EOS"
            print(f"  │   {i + 1:2d}. id={tid:6d} p={p.item():.6f} {repr(decoded):20s}{marker}")

        # Specifically check im_end probability
        ime_prob = probs[im_end_id].item()
        ime_rank = (probs > probs[im_end_id]).sum().item() + 1
        print(f"  │")
        print(f"  │ <|im_end|> (id={im_end_id}): prob={ime_prob:.8f}, rank={ime_rank}/{probs.shape[0]}")

        # Also check: after generating one assistant token, what's im_end prob?
        print(f"  │")
        print(f"  │ Simulazione generazione — prob di im_end per ogni step:")
        gen_ids = list(prompt_ids)
        for step_i in range(min(15, 100)):
            inp = torch.tensor([gen_ids], device="cuda")
            with torch.no_grad():
                out = model(inp)
                step_logits = out["logits"][0, -1, :]
            step_probs = torch.softmax(step_logits, dim=-1)
            next_token = step_logits.argmax().item()
            ime_p = step_probs[im_end_id].item()
            decoded = tokenizer.decode([next_token])
            print(f"  │   step {step_i}: next={repr(decoded):15s} (id={next_token:5d})  P(im_end)={ime_p:.8f}")
            gen_ids.append(next_token)
            if next_token == im_end_id:
                print(f"  │   → im_end generato!")
                break

        if ime_prob < 1e-6:
            print(f"  └─ ❌ im_end ha probabilità quasi zero — il modello non l'ha imparato")
        else:
            print(f"  └─ ℹ️  im_end ha prob={ime_prob:.8f}")

    # ─────────────────────────────────────────────────────────
    # TEST 8: Embedding analysis — pretrain vs SFT
    # ─────────────────────────────────────────────────────────
    if args.base_model and args.sft_model:
        print("\n┌─ TEST 8: Confronto embedding pretrain vs SFT")

        from models.decoder import Skylar2ForCausalLM

        base = Skylar2ForCausalLM.from_pretrained(args.base_model)
        sft = Skylar2ForCausalLM.from_pretrained(args.sft_model)

        base_emb = base.token_emb.weight.data
        sft_emb = sft.token_emb.weight.data

        tokens_to_check = [
            ("<pad>", pad_id),
            ("<bos>", bos_id),
            ("<eos>", eos_id),
            ("<|im_start|>", im_start_id),
            ("<|im_end|>", im_end_id),
        ]

        print(f"  │ Token embedding changes (L2 distance pretrain → SFT):")
        for name, tid in tokens_to_check:
            if tid is not None:
                dist = (base_emb[tid] - sft_emb[tid]).norm().item()
                base_norm = base_emb[tid].norm().item()
                sft_norm = sft_emb[tid].norm().item()
                print(f"  │   {name:20s} (id={tid:3d}): Δ={dist:.6f}  "
                      f"|base|={base_norm:.4f}  |sft|={sft_norm:.4f}")

        # Compare with average change for normal tokens
        sample_ids = list(range(100, 200))
        dists = [(base_emb[i] - sft_emb[i]).norm().item() for i in sample_ids]
        avg_dist = sum(dists) / len(dists)
        print(f"  │")
        print(f"  │   Token normali (100-199) media Δ: {avg_dist:.6f}")

        ime_dist = (base_emb[im_end_id] - sft_emb[im_end_id]).norm().item()
        if ime_dist < avg_dist * 0.1:
            print(f"  └─ ❌ im_end embedding quasi INVARIATO dal pretrain — non è stato aggiornato!")
        else:
            print(f"  └─ ✅ im_end embedding aggiornato durante SFT")

        del base, sft

    # ─────────────────────────────────────────────────────────
    # TEST 9: Verifica collator — padding non corrompe labels
    # ─────────────────────────────────────────────────────────
    print("\n┌─ TEST 9: Verifica collator padding")

    from train_sft import SFTDataset, collate_fn

    mini_examples = examples[:min(20, len(examples))]
    ds = SFTDataset(mini_examples, tokenizer, args.seq_len)

    if len(ds) > 0:
        # Check a batch
        batch_items = [ds[i] for i in range(min(4, len(ds)))]
        input_ids_batch, labels_batch = collate_fn(batch_items)

        total_ime_in_batch = (labels_batch == im_end_id).sum().item()
        total_pad_in_labels = (labels_batch == pad_id).sum().item()
        total_trainable_batch = (labels_batch != -100).sum().item()

        print(f"  │ Batch shape: input_ids={input_ids_batch.shape}, labels={labels_batch.shape}")
        print(f"  │ im_end nei labels batch: {total_ime_in_batch}")
        print(f"  │ pad_id (0) nei labels batch: {total_pad_in_labels}")
        print(f"  │ Tokens trainabili nel batch: {total_trainable_batch}")

        if total_pad_in_labels > 0:
            print(f"  └─ ❌ ATTENZIONE: pad_id (0) appare nei labels! Il modello impara a predire <pad>!")
        elif total_ime_in_batch == 0:
            print(f"  └─ ❌ im_end non nel batch — campione sfortunato o bug sistematico")
        else:
            print(f"  └─ ✅ Collator corretto")
    else:
        print(f"  └─ ❌ Dataset vuoto!")

    # ─────────────────────────────────────────────────────────
    # TEST 10: Verifica loss su singolo esempio con im_end
    # ─────────────────────────────────────────────────────────
    if args.sft_model and len(ds) > 0:
        print("\n┌─ TEST 10: Loss breakdown su singolo esempio")

        from models.decoder import Skylar2ForCausalLM
        import torch.nn.functional as F

        model = Skylar2ForCausalLM.from_pretrained(args.sft_model).to("cuda").eval()

        # Find an example that has im_end in labels
        test_sample = None
        for i in range(len(ds)):
            s = ds[i]
            if (s["labels"] == im_end_id).any():
                test_sample = s
                break

        if test_sample is not None:
            inp = test_sample["input_ids"].unsqueeze(0).to("cuda")
            lab = test_sample["labels"].unsqueeze(0).to("cuda")

            with torch.no_grad():
                out = model(inp, labels=lab)
                logits = out["logits"]

            # Find positions where label == im_end
            ime_positions = (lab[0] == im_end_id).nonzero(as_tuple=True)[0]
            print(f"  │ im_end label positions: {ime_positions.tolist()}")

            for pos in ime_positions:
                pos = pos.item()
                pos_logits = logits[0, pos, :]
                pos_probs = torch.softmax(pos_logits, dim=-1)
                ime_prob = pos_probs[im_end_id].item()
                top_prob, top_id = pos_probs.max(dim=-1)
                top_decoded = tokenizer.decode([top_id.item()])

                per_token_loss = F.cross_entropy(
                    pos_logits.unsqueeze(0), lab[0, pos:pos + 1]
                ).item()

                input_token = tokenizer.decode([inp[0, pos].item()])
                print(f"  │ Pos {pos}: input={repr(input_token):15s} → "
                      f"P(im_end)={ime_prob:.6f}, "
                      f"top=id {top_id.item()} {repr(top_decoded)} p={top_prob.item():.4f}, "
                      f"loss={per_token_loss:.4f}")

            print(f"  └─ ℹ️  Se P(im_end) ≈ 0, il modello non ha converguto su questo token")
        else:
            print(f"  └─ ❌ Nessun esempio nel dataset ha im_end nei labels (dopo shift+truncation)!")

    print("\n" + "=" * 70)
    print("  DIAGNOSI COMPLETATA")
    print("=" * 70)


if __name__ == "__main__":
    main()