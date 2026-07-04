"""
Converte DATA-AI_Conversation_ITA_full.json → sft_train.jsonl
con system prompt variati per evitare memorizzazione.

Usage:
python sft_data_shaffle.py \
  --input data/DATA-AI_Conversation_ITA_full.json \
  --mix data/sft_train.jsonl \
  --mix_output data/sft_train_mixed.jsonl
"""

import json
import random
import hashlib
import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT VARIATI
# ─────────────────────────────────────────────────────────────
# 30% nessun system, 30% generico variato, 20% Skylar, 20% istruzioni specifiche

SYSTEM_NONE = None  # 30% — nessun system prompt

SYSTEM_GENERIC = [
    "Sei un assistente utile e disponibile. Rispondi in italiano.",
    "Rispondi in modo chiaro, conciso e utile.",
    "Sei un assistente AI. Aiuta l'utente al meglio delle tue capacità.",
    "Sei un assistente conversazionale. Rispondi in italiano in modo naturale.",
    "Rispondi alle domande dell'utente in modo accurato e cordiale.",
    "Sei un assistente intelligente. Fornisci risposte utili e ben strutturate.",
    "Aiuta l'utente con le sue domande. Sii chiaro e diretto.",
    "Sei un assistente AI amichevole. Rispondi in italiano.",
    "Rispondi in modo naturale e conversazionale.",
    "Sei un assistente disponibile. Rispondi con precisione e gentilezza.",
]

SYSTEM_SKYLAR = [
    "Sei Skylar, assistente AI. Rispondi in italiano in modo naturale e utile.",
    "Sei Skylar. Aiuta l'utente con competenza e cordialità.",
    "Sei Skylar, un assistente conversazionale. Parla in italiano.",
    "Sei Skylar. Rispondi in modo chiaro e professionale.",
    "Sei Skylar, assistente intelligente. Sii utile e preciso.",
]

SYSTEM_SPECIFIC = [
    "Rispondi in modo conciso, massimo 3 paragrafi.",
    "Rispondi in modo dettagliato e approfondito.",
    "Rispondi in modo semplice, come se parlassi a un bambino di 10 anni.",
    "Sei un esperto. Rispondi con competenza tecnica ma in modo comprensibile.",
    "Rispondi con un tono amichevole e informale.",
    "Rispondi con un tono professionale e formale.",
    "Quando possibile, fornisci esempi pratici nella tua risposta.",
    "Struttura la risposta con punti chiave se l'argomento è complesso.",
]


def pick_system_prompt():
    """Sceglie un system prompt con la distribuzione corretta."""
    r = random.random()
    if r < 0.30:
        return None                             # 30% nessun system
    elif r < 0.60:
        return random.choice(SYSTEM_GENERIC)    # 30% generico
    elif r < 0.80:
        return random.choice(SYSTEM_SKYLAR)     # 20% Skylar
    else:
        return random.choice(SYSTEM_SPECIFIC)   # 20% istruzioni specifiche


def convert_example(item):
    """Converte un singolo {"prompt", "response"} → {"messages": [...]}."""
    messages = []

    system = pick_system_prompt()
    if system is not None:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": item["prompt"]})
    messages.append({"role": "assistant", "content": item["response"]})

    return {"messages": messages}


def load_input(filepath):
    """Carica il JSON. Supporta sia {"train": [...]} che lista diretta [...]."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # {"train": [...], "test": [...], ...}
        examples = []
        for split_name, split_data in data.items():
            if isinstance(split_data, list):
                print(f"  Split '{split_name}': {len(split_data)} esempi")
                examples.extend(split_data)
        return examples
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Formato non riconosciuto: {type(data)}")


def save_jsonl(examples, filepath):
    """Salva in formato JSONL."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Converti dataset italiano → JSONL per SFT")
    parser.add_argument("--input", type=str, default="data/DATA-AI_Conversation_ITA_full.json")
    parser.add_argument("--output", type=str, default="data/sft_ita_converted.jsonl")
    parser.add_argument("--mix", type=str, default=None,
                        help="JSONL esistente da mixare (es. il tuo dataset bancario)")
    parser.add_argument("--mix_output", type=str, default="data/sft_train_mixed.jsonl",
                        help="Output del mix finale")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Carica e converti ──
    print(f"\n  📂 Caricamento: {args.input}")
    raw_examples = load_input(args.input)
    print(f"  Totale esempi raw: {len(raw_examples)}")

    # Filtra esempi vuoti o troppo corti
    valid = [ex for ex in raw_examples
             if ex.get("prompt", "").strip() and ex.get("response", "").strip()]
    skipped_empty = len(raw_examples) - len(valid)
    if skipped_empty > 0:
        print(f"  ⚠ Scartati {skipped_empty} esempi vuoti/invalidi")

    # ── Deduplicazione ──
    # Dedup su prompt+response: rimuove solo se ENTRAMBI sono identici.
    # Stesso prompt con risposta diversa = esempio valido, lo teniamo.
    seen_hashes = set()
    deduped = []
    for ex in valid:
        turns = ex["prompt"].strip().lower() + "|" + ex["response"].strip().lower()
        h = hashlib.md5(turns.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(ex)
    n_dupes = len(valid) - len(deduped)
    if n_dupes > 0:
        print(f"  🧹 Rimossi {n_dupes} duplicati ({len(deduped)} unici rimasti)")
    valid = deduped

    # Filtra risposte troppo corte (< 10 char) — probabilmente spazzatura
    before = len(valid)
    valid = [ex for ex in valid if len(ex["response"].strip()) >= 10]
    skipped_short = before - len(valid)
    if skipped_short > 0:
        print(f"  ⚠ Scartati {skipped_short} esempi con risposta troppo corta (<10 char)")

    # Converti
    converted = [convert_example(ex) for ex in valid]
    print(f"  ✅ Convertiti: {len(converted)} conversazioni")

    # Stats system prompts
    n_none = sum(1 for c in converted if len(c["messages"]) == 2)
    n_with = len(converted) - n_none
    print(f"     Con system: {n_with} ({n_with/len(converted)*100:.0f}%) | "
          f"Senza system: {n_none} ({n_none/len(converted)*100:.0f}%)")

    # Salva dataset convertito
    save_jsonl(converted, args.output)
    print(f"  💾 Salvato: {args.output}")

    # ── Mix con dataset esistente (opzionale) ──
    all_examples = list(converted)

    if args.mix and Path(args.mix).exists():
        print(f"\n  🔀 Mixaggio con: {args.mix}")
        with open(args.mix, "r", encoding="utf-8") as f:
            existing = [json.loads(line) for line in f if line.strip()]
        print(f"     Esistenti: {len(existing)}")
        all_examples.extend(existing)

        # Dedup anche sul mix: hash di tutti i turn user+assistant (ignora system)
        seen = set()
        deduped_mix = []
        for ex in all_examples:
            turns = "|".join(
                m["role"] + ":" + m["content"].strip().lower()
                for m in ex["messages"] if m["role"] in ("user", "assistant")
            )
            h = hashlib.md5(turns.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                deduped_mix.append(ex)
        n_cross_dupes = len(all_examples) - len(deduped_mix)
        if n_cross_dupes > 0:
            print(f"     🧹 Rimossi {n_cross_dupes} duplicati cross-dataset")
        all_examples = deduped_mix

        print(f"     Totale mixato: {len(all_examples)}")

    # ── Shuffle e salva ──
    random.shuffle(all_examples)

    out_path = args.mix_output if args.mix else args.output
    save_jsonl(all_examples, out_path)

    print(f"\n  📊 Risultato finale:")
    print(f"     Totale: {len(all_examples)} → {out_path}")

    # ── Mostra un esempio ──
    sample = random.choice(all_examples)
    print(f"\n  📝 Esempio random:")
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:80]
        print(f"     [{role}] {content}...")

    print()


if __name__ == "__main__":
    main()