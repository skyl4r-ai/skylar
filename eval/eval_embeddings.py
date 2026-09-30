# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Intrinsic retrieval evaluation for SkylarEmbedder.

Corpus = the 30 domain concept passages. Eval queries use phrasings DIFFERENT
from the training templates (held-out surface forms), so this measures whether
the embedder generalizes meaning, not memorizes lexical patterns.

Metrics: Recall@1, Recall@5, MRR. Random Recall@1 over 30 docs ≈ 3.3%.

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python eval/eval_embeddings.py --model checkpoints_embed/skylar-mp-embed
"""
import argparse, sys
from pathlib import Path
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.embedder import SkylarEmbedder
from data.gen_contrastive_data import CONCEPTS

# eval query templates — intentionally NOT in data/gen_contrastive_data.Q_TEMPLATES
EVAL_TEMPLATES = [
    "spiegami cosa indica {t}",
    "qual è il significato di {t}",
    "vorrei informazioni su {t}",
    "in parole semplici, {t} cosa vuol dire",
    "a cosa fa riferimento {t} in ambito bancario",
]


def encode(model, tok, texts, device, max_len=128, pad_id=0, bs=64):
    embs = []
    for s in range(0, len(texts), bs):
        chunk = texts[s:s + bs]
        ids = [tok.encode(t, add_special_tokens=False).ids[:max_len] for t in chunk]
        m = max(1, max(len(x) for x in ids))
        input_ids = torch.full((len(ids), m), pad_id, dtype=torch.long)
        attn = torch.zeros((len(ids), m), dtype=torch.long)
        for i, x in enumerate(ids):
            if x:
                input_ids[i, :len(x)] = torch.tensor(x)
                attn[i, :len(x)] = 1
        with torch.no_grad():
            out = model(input_ids.to(device), attention_mask=attn.to(device))
        embs.append(out["embeddings"].float().cpu())
    return torch.cat(embs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = SkylarEmbedder.from_pretrained(args.model).to(args.device).eval()
    tok = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    pad_id = tok.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    passages = [p for _, p in CONCEPTS]
    doc_emb = encode(model, tok, passages, args.device, pad_id=pad_id)   # (30, D)

    queries, gold = [], []
    for gi, (term, _) in enumerate(CONCEPTS):
        for tpl in EVAL_TEMPLATES:
            queries.append(tpl.format(t=term))
            gold.append(gi)
    q_emb = encode(model, tok, queries, args.device, pad_id=pad_id)      # (Q, D)

    sims = q_emb @ doc_emb.t()                                            # (Q, 30)
    ranks = sims.argsort(dim=1, descending=True)
    r1 = r5 = mrr = 0.0
    N = len(queries)
    for i in range(N):
        order = ranks[i].tolist()
        pos = order.index(gold[i])                                       # 0-based rank
        r1 += int(pos == 0)
        r5 += int(pos < 5)
        mrr += 1.0 / (pos + 1)
    print("=" * 56)
    print(f"  Embedding retrieval ({len(passages)} docs, {N} queries)")
    print("-" * 56)
    print(f"  Recall@1 : {r1/N:.3f}   (random ~{1/len(passages):.3f})")
    print(f"  Recall@5 : {r5/N:.3f}")
    print(f"  MRR      : {mrr/N:.3f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
