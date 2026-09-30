# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Embedding inference CLI for SkylarEmbedder.

Encodes texts to L2-normalized vectors and (optionally) ranks a list of
documents against a query by cosine similarity — a mini retrieval demo.

  # similarity demo:
  .venv/bin/python inference/bin.embed.py --model checkpoints_embed/skylar-mp-embed \
      --query "Cos'è il TAEG?" \
      --docs "Il TAEG è il costo totale annuo di un finanziamento." \
             "L'IBAN identifica un conto corrente." \
             "Il bonifico SEPA trasferisce denaro in euro."

  # raw vector for one text:
  .venv/bin/python inference/bin.embed.py --model <dir> --text "ciao mondo" --show_vector
"""
import argparse, sys
from pathlib import Path
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.embedder import SkylarEmbedder


def encode(model, tok, texts, device, max_len=128, pad_id=0):
    ids = [tok.encode(t, add_special_tokens=False).ids[:max_len] for t in texts]
    m = max(1, max(len(x) for x in ids))
    input_ids = torch.full((len(ids), m), pad_id, dtype=torch.long)
    attn = torch.zeros((len(ids), m), dtype=torch.long)
    for i, x in enumerate(ids):
        if x:
            input_ids[i, :len(x)] = torch.tensor(x)
            attn[i, :len(x)] = 1
    with torch.no_grad():
        out = model(input_ids.to(device), attention_mask=attn.to(device))
    return out["embeddings"].float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--query", default=None)
    ap.add_argument("--docs", nargs="*", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--show_vector", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = SkylarEmbedder.from_pretrained(args.model).to(args.device).eval()
    tok = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    pad_id = tok.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0
    dim = model.config.d_model
    print(f"Embedder loaded | dim={dim} | pool={getattr(model.config,'pool_strategy','mean')}")

    if args.text:
        v = encode(model, tok, [args.text], args.device, pad_id=pad_id)[0]
        print(f"||v|| = {v.norm():.4f} (should be ~1.0)")
        if args.show_vector:
            print(v[:16].tolist(), "...")
        return

    if args.query and args.docs:
        q = encode(model, tok, [args.query], args.device, pad_id=pad_id)
        d = encode(model, tok, args.docs, args.device, pad_id=pad_id)
        sims = (q @ d.t())[0]                       # cosine (already normalized)
        order = sims.argsort(descending=True)
        print(f"\nQuery: {args.query}\n" + "-" * 60)
        for rank, i in enumerate(order.tolist(), 1):
            print(f"  {rank}. cos={sims[i]:.3f}  {args.docs[i]}")
        return

    print("Niente da fare: passa --text, oppure --query + --docs.")


if __name__ == "__main__":
    main()
