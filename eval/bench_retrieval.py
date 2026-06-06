"""
Real Italian retrieval benchmark — apples-to-apples, same pool, same metrics.

Task: SQuAD-it (crux82/squad_it). Query = question, corpus = the unique
contexts, gold = the question's own context. test split = 7609 queries over
1988 contexts. This is OPEN-DOMAIN Italian Wikipedia text — an honest test of
whether an embedder GENERALIZES beyond its training domain.

Every encoder is run through the identical search + metric code, so the numbers
are directly comparable across models of different sizes / tokenizers:
  Skylar-embed (236M, from-scratch IT)  vs  BAAI/bge-m3 (568M, multilingual SOTA)
  vs  intfloat/multilingual-e5-base (278M, same size class)  [optional, --models]

Metrics (single relevant doc per query): Recall@{1,5,10}, MRR@10, nDCG@10.

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python eval/bench_retrieval.py \
      --models "skylar=checkpoints_embed/skylar-mp-embed,bge-m3" \
      --split test
"""
import argparse, math, os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ───────────────────────── datasets ─────────────────────────
def load_squad_it(split="test", max_q=None):
    from datasets import load_dataset
    ds = load_dataset("crux82/squad_it", split=split)
    ctx_list, ctx_id, queries = [], {}, []
    for ex in ds:
        c = ex["context"]
        if c not in ctx_id:
            ctx_id[c] = len(ctx_list)
            ctx_list.append(c)
        queries.append((ex["question"], ctx_id[c]))
        if max_q and len(queries) >= max_q:
            break
    return queries, ctx_list


def load_concepts(split="test", max_q=None):
    """Small IN-DOMAIN Italian banking/legal probe: 30 concept→definition passages,
    queried with held-out phrasings (NOT in the contrastive training templates, and
    none of the compared models trained on this exact set → fair for all)."""
    from data.gen_contrastive_data import CONCEPTS
    eval_tpl = [
        "spiegami cosa indica {t}",
        "qual è il significato di {t}",
        "a cosa fa riferimento {t} in ambito bancario",
        "in parole semplici, {t} cosa vuol dire",
        "definizione di {t} nel settore finanziario",
    ]
    corpus = [p for _, p in CONCEPTS]
    queries = []
    for gi, (term, _) in enumerate(CONCEPTS):
        for tpl in eval_tpl:
            queries.append((tpl.format(t=term), gi))
            if max_q and len(queries) >= max_q:
                return queries, corpus
    return queries, corpus


DATASETS = {"squad_it": load_squad_it, "concepts": load_concepts}


# ───────────────────────── encoders ─────────────────────────
class SkylarEnc:
    """SkylarEmbedder — mean-pool + L2-norm already inside the model."""
    def __init__(self, path, device, max_len=512):
        from models.embedder import SkylarEmbedder
        from tokenizers import Tokenizer
        self.model = SkylarEmbedder.from_pretrained(path).to(device).eval()
        self.tok = Tokenizer.from_file(f"{path}/tokenizer.json")
        self.pad = self.tok.token_to_id("<pad>") or 0
        self.device, self.max_len = device, max_len
        self.params = sum(p.numel() for p in self.model.parameters())

    def encode(self, texts, bs=64, is_query=False):
        embs = []
        for s in range(0, len(texts), bs):
            chunk = texts[s:s + bs]
            ids = [self.tok.encode(t, add_special_tokens=False).ids[:self.max_len] for t in chunk]
            m = max(1, max(len(x) for x in ids))
            input_ids = torch.full((len(ids), m), self.pad, dtype=torch.long)
            attn = torch.zeros((len(ids), m), dtype=torch.long)
            for i, x in enumerate(ids):
                if x:
                    input_ids[i, :len(x)] = torch.tensor(x)
                    attn[i, :len(x)] = 1
            with torch.no_grad():
                out = self.model(input_ids.to(self.device), attention_mask=attn.to(self.device))
            embs.append(out["embeddings"].float().cpu())
        return torch.cat(embs, 0)


class HFEnc:
    """Off-the-shelf HF encoder (XLM-R / BERT). cls pooling for bge, mean for e5."""
    def __init__(self, name, device, pooling="cls", max_len=512, q_prefix="", d_prefix=""):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device, self.pooling, self.max_len = device, pooling, max_len
        self.q_prefix, self.d_prefix = q_prefix, d_prefix
        self.params = sum(p.numel() for p in self.model.parameters())

    def encode(self, texts, bs=32, is_query=False):
        pre = self.q_prefix if is_query else self.d_prefix
        embs = []
        for s in range(0, len(texts), bs):
            chunk = [pre + t for t in texts[s:s + bs]]
            enc = self.tok(chunk, padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            h = out.last_hidden_state
            if self.pooling == "cls":
                v = h[:, 0]
            else:
                mask = enc["attention_mask"][:, :, None].float()
                v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs.append(F.normalize(v, p=2, dim=-1).float().cpu())
        return torch.cat(embs, 0)


def build_encoder(spec, device):
    """spec: 'skylar=PATH' | 'bge-m3' | 'e5=intfloat/multilingual-e5-base' | 'e5-small'"""
    name, _, arg = spec.partition("=")
    name = name.strip()
    if name == "skylar":
        return f"Skylar-embed", SkylarEnc(arg or "checkpoints_embed/skylar-mp-embed", device)
    if name == "bge-m3":
        return "bge-m3", HFEnc("BAAI/bge-m3", device, pooling="cls")
    if name == "e5":
        model = arg or "intfloat/multilingual-e5-base"
        return model.split("/")[-1], HFEnc(model, device, pooling="mean",
                                           q_prefix="query: ", d_prefix="passage: ")
    raise ValueError(f"unknown encoder spec: {spec}")


# ───────────────────────── search + metrics ─────────────────────────
def search(q_emb, d_emb, device, topk=10, qbs=256):
    d = d_emb.to(device)
    idxs = []
    for s in range(0, q_emb.shape[0], qbs):
        qs = q_emb[s:s + qbs].to(device)
        sims = qs @ d.t()
        idxs.append(sims.topk(min(topk, d.shape[0]), dim=1).indices.cpu())
    return torch.cat(idxs, 0)


def metrics(top_idx, gold, ks=(1, 5, 10)):
    Q = len(gold)
    rec = {k: 0 for k in ks}; mrr = ndcg = 0.0
    for i in range(Q):
        row = top_idx[i].tolist()
        g = gold[i]
        if g in row:
            rank = row.index(g)            # 0-based within top-10
            for k in ks:
                if rank < k:
                    rec[k] += 1
            mrr += 1.0 / (rank + 1)
            ndcg += 1.0 / math.log2(rank + 2)
    out = {f"R@{k}": rec[k] / Q for k in ks}
    out["MRR@10"] = mrr / Q
    out["nDCG@10"] = ndcg / Q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="skylar=checkpoints_embed/skylar-mp-embed,bge-m3")
    ap.add_argument("--dataset", default="squad_it", choices=list(DATASETS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="cap #queries (smoke)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    queries, corpus = DATASETS[args.dataset](args.split, args.limit)
    qtexts = [q for q, _ in queries]
    gold = [g for _, g in queries]
    print(f"{args.dataset}[{args.split}] : {len(qtexts)} queries over {len(corpus)} contexts "
          f"(random R@1 ≈ {1/len(corpus):.4f})")
    print("=" * 86)
    hdr = f"{'model':<26}{'params':>8}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR@10':>9}{'nDCG@10':>9}{'sec':>7}"
    print(hdr); print("-" * 86)

    results = {}
    for spec in args.models.split(","):
        spec = spec.strip()
        if not spec:
            continue
        label, enc = build_encoder(spec, args.device)
        t0 = time.time()
        d_emb = enc.encode(corpus, is_query=False)
        q_emb = enc.encode(qtexts, is_query=True)
        top = search(q_emb, d_emb, args.device)
        m = metrics(top, gold)
        dt = time.time() - t0
        results[label] = m
        pm = enc.params / 1e6
        print(f"{label:<26}{pm:>7.0f}M{m['R@1']:>8.3f}{m['R@5']:>8.3f}{m['R@10']:>8.3f}"
              f"{m['MRR@10']:>9.3f}{m['nDCG@10']:>9.3f}{dt:>7.0f}")
        del enc
        if args.device == "cuda":
            torch.cuda.empty_cache()
    print("=" * 86)


if __name__ == "__main__":
    main()
