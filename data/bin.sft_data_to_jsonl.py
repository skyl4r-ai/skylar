# filepath: convert_to_jsonl.py
# !/usr/bin/env python3
"""
Converte dataset ChatML .txt → JSONL per LoRA fine-tuning.

Ogni riga JSONL:
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Uso:
    python convert_to_jsonl.py                          # converte tutti i .txt in datasets/
    python convert_to_jsonl.py --dir ./datasets          # specifica directory
    python convert_to_jsonl.py --output train.jsonl      # specifica output
    python convert_to_jsonl.py --no-shuffle              # disabilita shuffle
    python convert_to_jsonl.py --validate train.jsonl    # valida JSONL esistente
"""

import sys
import json
import math
import random
import hashlib
from pathlib import Path
from collections import Counter


# ══════════════════════════════════════
# Parsing
# ══════════════════════════════════════

def parse_conversation(block: str) -> list[dict] | None:
    """
    Parsa un singolo blocco conversazione ChatML.
    Splitta su <|im_start|> e per ogni turno estrae role e content.
    """
    messages = []
    turns = block.split("<|im_start|>")

    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue

        # Rimuovi <|im_end|> e tutto ciò che viene dopo
        if "<|im_end|>" in turn:
            turn = turn.split("<|im_end|>")[0].strip()

        # Prima riga = role, resto = content
        newline_pos = turn.find("\n")
        if newline_pos == -1:
            continue

        role = turn[:newline_pos].strip().lower()
        content = turn[newline_pos + 1:].strip()

        if role not in ("system", "user", "assistant", "tool"):
            continue
        if not content:
            continue

        messages.append({"role": role, "content": content})

    return messages if messages else None


def parse_txt_file(filepath: Path) -> list[tuple[list[dict], str]]:
    """
    Parsa un file .txt splittando le conversazioni su '<|im_start|>system'.
    Ritorna lista di (messages, source_label).
    """
    text = filepath.read_text(encoding="utf-8")
    results = []

    # Split su <|im_start|>system — ogni pezzo è una conversazione
    parts = text.split("<|im_start|>system")

    for idx, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # Ricostruisci il blocco completo
        block = "<|im_start|>system\n" + part
        messages = parse_conversation(block)

        if messages:
            source = f"{filepath.name}[{idx}]"
            results.append((messages, source))

    return results


# ══════════════════════════════════════
# Validazione
# ══════════════════════════════════════

def validate_conversation(messages: list[dict], source: str = "") -> list[str]:
    """Valida una conversazione, ritorna lista di problemi (vuota = ok)."""
    issues = []

    if not messages:
        issues.append("Conversazione vuota")
        return issues

    roles = [m["role"] for m in messages]

    if roles[0] != "system":
        issues.append(f"Non inizia con system (inizia con {roles[0]})")

    if "assistant" not in roles:
        issues.append("Nessuna risposta assistant")

    if not any(r in roles for r in ["user", "tool"]):
        issues.append("Nessun input user/tool")

    if roles[-1] != "assistant":
        issues.append(f"Non finisce con assistant (finisce con {roles[-1]})")

    for i, msg in enumerate(messages):
        if not msg["content"]:
            issues.append(f"Messaggio {i} ({msg['role']}) vuoto")

    return issues


def compute_hash(messages: list[dict]) -> str:
    """Hash per dedup basato su contenuto non-system."""
    content = ""
    for msg in messages:
        if msg["role"] != "system":
            content += f"{msg['role']}:{msg['content'][:200]}|"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ══════════════════════════════════════
# Shuffle aggressivo
# ══════════════════════════════════════

def deep_shuffle(items: list, seed: int | None = None) -> list:
    """
    Shuffle aggressivo multi-pass. Numero di pass calcolato su len(items)
    per garantire distribuzione realmente casuale anche su dataset piccoli.

    Strategia:
      - pass base = ceil(log2(n)) + 3  (minimo 5 pass)
      - ogni pass usa un seed derivato diverso
      - verifica finale: nessun elemento rimane nella posizione originale
        (Fisher-Yates derangement check); se fallisce, ripete
    """
    n = len(items)
    if n <= 1:
        return items

    rng = random.Random(seed)
    result = list(items)

    num_passes = max(5, math.ceil(math.log2(n)) + 3)

    for p in range(num_passes):
        pass_seed = rng.randint(0, 2 ** 63)
        pass_rng = random.Random(pass_seed)
        # Fisher-Yates shuffle con RNG dedicato per pass
        for i in range(n - 1, 0, -1):
            j = pass_rng.randint(0, i)
            result[i], result[j] = result[j], result[i]

    # Verifica: almeno il 90% degli elementi deve aver cambiato posizione
    # Se non è così, continua a shufflare (max 50 tentativi extra)
    for _ in range(50):
        same_pos = sum(1 for i in range(n) if id(result[i]) == id(items[i]))
        displacement_ratio = 1.0 - (same_pos / n)
        if displacement_ratio >= 0.9 or n < 5:
            break
        extra_seed = rng.randint(0, 2 ** 63)
        extra_rng = random.Random(extra_seed)
        for i in range(n - 1, 0, -1):
            j = extra_rng.randint(0, i)
            result[i], result[j] = result[j], result[i]

    return result


# ══════════════════════════════════════
# Conversione
# ══════════════════════════════════════

def convert(datasets_dir: Path, output_path: Path, shuffle: bool = True):
    """Converte tutti i .txt in un JSONL validato e deduplicated."""
    txt_files = sorted(datasets_dir.glob("*.txt"))

    if not txt_files:
        print("  Nessun file .txt trovato in", datasets_dir)
        return

    print()
    print("  ═══════════════════════════════════════")
    print("  Conversione ChatML → JSONL")
    print("  ═══════════════════════════════════════")
    print(f"  Sorgente:  {datasets_dir}")
    print(f"  File .txt: {len(txt_files)}")
    print(f"  Shuffle:   {'✅ attivo' if shuffle else '❌ disattivo'}")
    print()

    all_convs = []
    all_issues = []
    stats = {"parsed": 0, "valid": 0, "invalid": 0, "duplicates": 0}

    for filepath in txt_files:
        parsed = parse_txt_file(filepath)
        file_valid = 0
        file_invalid = 0

        for messages, source in parsed:
            stats["parsed"] += 1
            issues = validate_conversation(messages, source)

            if issues:
                stats["invalid"] += 1
                file_invalid += 1
                for issue in issues:
                    all_issues.append((source, issue))
            else:
                stats["valid"] += 1
                file_valid += 1
                all_convs.append((messages, source))

        status = "✅" if file_invalid == 0 else "⚠️"
        print(f"  {status} {filepath.name}: {len(parsed)} trovate, {file_valid} valide, {file_invalid} scartate")

    # Dedup
    seen = set()
    unique = []
    for messages, source in all_convs:
        h = compute_hash(messages)
        if h not in seen:
            seen.add(h)
            unique.append(messages)
        else:
            stats["duplicates"] += 1

    # Shuffle aggressivo prima di scrivere
    if shuffle and len(unique) > 1:
        unique = deep_shuffle(unique)
        num_passes = max(5, math.ceil(math.log2(len(unique))) + 3)
        print(f"\n  🔀 Shuffle: {num_passes} pass su {len(unique)} conversazioni")

    # Scrivi JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for messages in unique:
            line = json.dumps({"messages": messages}, ensure_ascii=False)
            f.write(line + "\n")

    # Report
    print()
    print("  ═══════════════════════════════════════")
    print("  RISULTATI")
    print("  ═══════════════════════════════════════")
    print(f"  Conversazioni trovate:  {stats['parsed']}")
    print(f"  Valide:                 {stats['valid']}")
    print(f"  Invalide (scartate):    {stats['invalid']}")
    print(f"  Duplicati (rimossi):    {stats['duplicates']}")
    print(f"  ─────────────────────────────────────")
    print(f"  DATASET FINALE:         {len(unique)} conversazioni")
    print(f"  Output:                 {output_path}")

    # Distribuzione ruoli
    role_counts = Counter()
    turn_counts = Counter()
    for messages in unique:
        for msg in messages:
            role_counts[msg["role"]] += 1
        turn_counts[len(messages)] += 1

    print()
    print(f"  Distribuzione ruoli:")
    for role in ["system", "user", "assistant", "tool"]:
        if role in role_counts:
            print(f"    {role:12s}: {role_counts[role]:>6d}")

    print()
    print(f"  Turni per conversazione:")
    for turns in sorted(turn_counts.keys()):
        print(f"    {turns} turni: {turn_counts[turns]:>4d}")

    # Distribuzione system prompt (per capire il mix di task)
    system_prompts = Counter()
    for messages in unique:
        sys_msg = messages[0]["content"][:80]
        system_prompts[sys_msg] += 1

    print()
    print(f"  Categorie (per system prompt):")
    for prompt, count in system_prompts.most_common(15):
        print(f"    [{count:>3d}x] {prompt}...")

    # Issues
    if all_issues:
        print()
        print(f"  ⚠️  {len(all_issues)} problemi trovati:")
        issue_types = Counter(issue for _, issue in all_issues)
        for issue, count in issue_types.most_common(10):
            print(f"    [{count:>3d}x] {issue}")

    # File size
    size_kb = output_path.stat().st_size / 1024
    print(f"\n  File size: {size_kb:.1f} KB")
    print()


# ══════════════════════════════════════
# Validazione JSONL
# ══════════════════════════════════════

def validate(jsonl_path: Path):
    """Valida un file JSONL esistente."""
    print()
    print("  ═══════════════════════════════════════")
    print(f"  Validazione: {jsonl_path.name}")
    print("  ═══════════════════════════════════════")

    if not jsonl_path.exists():
        print(f"  ERRORE: file non trovato")
        return False

    total = 0
    valid = 0
    issues = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append((line_num, f"JSON invalido: {e}"))
                continue

            if "messages" not in data:
                issues.append((line_num, "Manca 'messages'"))
                continue

            if not isinstance(data["messages"], list):
                issues.append((line_num, "'messages' non è una lista"))
                continue

            conv_issues = validate_conversation(data["messages"], f"riga {line_num}")
            if conv_issues:
                for ci in conv_issues:
                    issues.append((line_num, ci))
            else:
                valid += 1

    print(f"  Righe totali:  {total}")
    print(f"  Valide:        {valid}")
    print(f"  Con problemi:  {total - valid}")

    if issues:
        print(f"\n  Problemi:")
        issue_types = Counter(issue for _, issue in issues)
        for issue, count in issue_types.most_common(10):
            print(f"    [{count:>3d}x] {issue}")
    else:
        print(f"\n  ✅ Tutte le {valid} conversazioni sono valide!")

    print()
    return len(issues) == 0


# ══════════════════════════════════════
# Main
# ══════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Conversione ChatML .txt → JSONL")
    parser.add_argument("--dir", type=str, default="./data/source/chat", help="Directory con i .txt")
    parser.add_argument("--output", type=str, default="./data/sft_train.jsonl", help="File JSONL output")
    parser.add_argument("--no-shuffle", action="store_true", default=False,
                        help="Disabilita shuffle (default: shuffle attivo)")
    parser.add_argument("--validate", type=str, default=None, help="Valida un JSONL esistente")
    args = parser.parse_args()

    if args.validate:
        ok = validate(Path(args.validate))
        sys.exit(0 if ok else 1)
    else:
        convert(Path(args.dir), Path(args.output), shuffle=not args.no_shuffle)


if __name__ == "__main__":
    main()
