"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Bits-per-byte su un set di byte congelato — l'unico ponte fra modelli con
vocabolari diversi.

## Il problema che risolve

Il 980M gira su vocab 48128, il v2 girerà su 64000. La loro **perplexity non è
confrontabile**: la cross-entropy misura bit per *token*, e un token del vocab 64k
copre in media più byte di uno del 48k. Un modello che sembra migliore in ppl può
esserlo solo perché ha token più grossi. Sono due unità di misura diverse.

La bits-per-byte normalizza sul denominatore comune che i due condividono — i byte
del testo originale:

    bpb = (Σ log2 P(token))⁻¹ / n_byte  =  CE_nats · n_token / (ln2 · n_byte)

Su uno **stesso set di byte congelato** è confrontabile fra qualunque coppia di
modelli, tokenizzatori e vocabolari. È la metrica che usano i paper quando cambiano
tokenizer, ed è la ragione per cui va calcolata sul 980M **prima** di iniziare il
run nuovo: dopo, non c'è più un termine di paragone.

## Il set

Va **congelato una volta** e riusato per sempre, con slice separate: se COBOL,
IT-legal e general code stanno in un numero unico, un modello che migliora su uno
e peggiora sull'altro sembra fermo.

    # 1. congela il set (una volta sola)
    python eval/bin.bits_per_byte.py --build --out eval/frozen_bytes

    # 2. misura il baseline, PRIMA del run nuovo
    python eval/bin.bits_per_byte.py --model Skyl4r-Ai/Skylar-980M-Cobol-Base \\
        --set eval/frozen_bytes

    # 3. la stessa riga sul modello v2, a run finito
"""

import argparse
import glob
import json
import math
import os
import sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Le slice restano separate: un numero unico nasconde i compromessi.
SLICES = {
    "cobol_real": "projects/skylar-cobol/data/real/*.jsonl",
    "cobol_synth": "projects/skylar-cobol/data/synth/pretrain/*.jsonl",
    "code_general": "projects/skylar-cobol/data/code/code_v3_Python_*.jsonl",
    "it_legal": "projects/skylar-cobol/data/text/*.jsonl",
}
FIELDS = ("program", "text", "content", "code")


def build_set(out_dir, per_slice_kb=512, repo="."):
    """Congela N KB di byte per slice. Da rifare MAI: il confronto vale solo a set fisso."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for name, pattern in SLICES.items():
        files = sorted(glob.glob(os.path.join(repo, pattern)))
        if not files:
            print(f"  [skip] {name}: nessun file per {pattern}")
            continue
        buf, budget = [], per_slice_kb * 1024
        for f in files:
            if budget <= 0:
                break
            for line in open(f, errors="ignore"):
                if budget <= 0:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = next((rec[k] for k in FIELDS if isinstance(rec.get(k), str)), None)
                if not txt or len(txt) < 200:
                    continue
                buf.append(txt)
                budget -= len(txt.encode("utf-8"))
        blob = "\n\n".join(buf)
        path = os.path.join(out_dir, f"{name}.txt")
        with open(path, "w") as fh:
            fh.write(blob)
        manifest[name] = {"bytes": len(blob.encode("utf-8")), "docs": len(buf)}
        print(f"  {name:14} {manifest[name]['bytes']:>9,} byte  {len(buf):>5} documenti")
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\nset congelato in {out_dir} — NON rigenerarlo, o i confronti passati "
          f"smettono di valere")


@torch.no_grad()
def bits_per_byte(model, tok, text, device, seq_len=2048, stride=1024):
    """
    bpb con finestra scorrevole: ogni token è predetto avendo davanti almeno
    `seq_len - stride` token di contesto, così il numero non dipende da dove
    cadono i tagli.
    """
    n_bytes = len(text.encode("utf-8"))
    ids = tok.encode(text).ids
    if len(ids) < 2:
        return None
    total_nats, n_pred = 0.0, 0
    for start in range(0, len(ids) - 1, stride):
        chunk = ids[start:start + seq_len + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], device=device)
        y = torch.tensor([chunk[1:]], device=device)
        # Solo le posizioni con contesto sufficiente contano, tranne nella prima
        # finestra dove non c'è alternativa.
        skip = 0 if start == 0 else (seq_len - stride)
        logits = model(x)["logits"].float()
        ce = torch.nn.functional.cross_entropy(
            logits[0, skip:], y[0, skip:], reduction="sum")
        total_nats += ce.item()
        n_pred += y.shape[1] - skip
    return {
        "bpb": total_nats / math.log(2) / n_bytes,
        "bytes": n_bytes,
        "tokens": n_pred,
        "bytes_per_token": n_bytes / max(n_pred, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="congela il set di byte")
    ap.add_argument("--out", default="eval/frozen_bytes")
    ap.add_argument("--set", default="eval/frozen_bytes")
    ap.add_argument("--model", help="path o repo HF del modello da misurare")
    ap.add_argument("--tokenizer", default=None, help="default: tokenizer.json del modello")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.build:
        build_set(args.out)
        return
    if not args.model:
        ap.error("serve --model (o --build)")

    from models.decoder import Skylar2ForCausalLM
    model = Skylar2ForCausalLM.from_pretrained(args.model).to(args.device).eval()
    tok_path = args.tokenizer or os.path.join(args.model, "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)

    print(f"\n{args.model}")
    print(f"vocab {tok.get_vocab_size()}  ·  set {args.set}\n")
    print(f"  {'slice':<14} {'bpb':>8} {'byte/token':>11} {'byte':>10}")
    rows = {}
    for path in sorted(glob.glob(os.path.join(args.set, "*.txt"))):
        name = os.path.basename(path)[:-4]
        r = bits_per_byte(model, tok, open(path).read(), args.device, seq_len=args.seq_len)
        if r:
            rows[name] = r
            print(f"  {name:<14} {r['bpb']:>8.4f} {r['bytes_per_token']:>11.3f} {r['bytes']:>10,}")
    if rows:
        avg = sum(r["bpb"] for r in rows.values()) / len(rows)
        print(f"  {'MEDIA':<14} {avg:>8.4f}")
        print("\nLa bpb e' confrontabile fra vocabolari diversi; la perplexity NO.")


if __name__ == "__main__":
    main()
