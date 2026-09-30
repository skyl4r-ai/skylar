# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Train SkylarSparseEncoder (SPLADE-style learned sparse retrieval).

Same (query, positive) pairs as the dense embedder, but the representation is a
sparse vocab vector and the score is a DOT product (not cosine). Loss = InfoNCE
with in-batch negatives + FLOPS regularization that drives sparsity (most vocab
dims → 0). Reuses the pretrained decoder backbone via from_decoder — NO re-pretrain.

Pairs with the dense embedder for hybrid retrieval (Qdrant), BGE-M3 style.

  # real (after dense pretrain finishes):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python training/bin.sparse.py \
      --base_model checkpoints/skylar-mp-base/final \
      --data .datasets/embed/contrastive_it.jsonl --epochs 3 --bf16 --batch_size 96 \
      --out_dir checkpoints_embed/skylar-mp-sparse
  # smoke (CPU):
  .venv/bin/python training/bin.sparse.py --preset test \
      --tokenizer .datasets/tokenized/tokenizer.json \
      --data .datasets/embed/contrastive_it.jsonl --device cpu --max_steps 6 --batch_size 8
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.config import get_config
from models.sparse_encoder import SkylarSparseEncoder, flops_regularizer


class PairDS(Dataset):
    def __init__(self, path):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    self.rows.append((o["query"], o["positive"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collate(tok, max_len, pad_id):
    def enc(texts):
        ids = [tok.encode(t, add_special_tokens=False).ids[:max_len] for t in texts]
        m = max(1, max(len(x) for x in ids))
        input_ids = torch.full((len(ids), m), pad_id, dtype=torch.long)
        attn = torch.zeros((len(ids), m), dtype=torch.long)
        for i, x in enumerate(ids):
            if x:
                input_ids[i, :len(x)] = torch.tensor(x)
                attn[i, :len(x)] = 1
        return input_ids, attn

    def collate(batch):
        return enc([b[0] for b in batch]), enc([b[1] for b in batch])
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out_dir", default="checkpoints_embed/skylar-sparse")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--lambda_q", type=float, default=0.01, help="FLOPS reg weight (query)")
    ap.add_argument("--lambda_d", type=float, default=0.008, help="FLOPS reg weight (doc)")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vocab_size", type=int, default=32768)
    args = ap.parse_args()

    dev = args.device
    if args.base_model:
        print(f"Init sparse encoder from decoder {args.base_model}")
        model = SkylarSparseEncoder.from_decoder(args.base_model)
        tok = Tokenizer.from_file(f"{args.base_model}/tokenizer.json")
    else:
        assert args.preset and args.tokenizer
        model = SkylarSparseEncoder(get_config(args.preset, vocab_size=args.vocab_size))
        tok = Tokenizer.from_file(args.tokenizer)
    model = model.to(dev).train()
    pad_id = tok.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    ds = PairDS(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    collate_fn=make_collate(tok, args.max_len, pad_id))
    total = args.max_steps or len(dl) * args.epochs
    print(f"params={model.count_params()/1e6:.1f}M | pairs={len(ds)} | steps={total} | bs={args.batch_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(s):
        if s < args.warmup_steps:
            return args.lr * (s + 1) / args.warmup_steps
        prog = (s - args.warmup_steps) / max(1, total - args.warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    # FLOPS reg is ramped quadratically over the first part of training (SPLADE trick)
    def reg_scale(s):
        return min(1.0, (s / max(1, total * 0.3)) ** 2)

    amp = (args.bf16 and dev == "cuda")
    step = 0; t0 = time.time(); done = False
    for ep in range(args.epochs):
        for (q_ids, q_m), (p_ids, p_m) in dl:
            q_ids, q_m = q_ids.to(dev), q_m.to(dev)
            p_ids, p_m = p_ids.to(dev), p_m.to(dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                q = model(q_ids, attention_mask=q_m)["sparse"].float()
                p = model(p_ids, attention_mask=p_m)["sparse"].float()
                scores = q @ p.t()                                 # (B, B) dot product
                labels = torch.arange(q.shape[0], device=dev)
                ce = F.cross_entropy(scores, labels)
                rs = reg_scale(step)
                reg = args.lambda_q * rs * flops_regularizer(q) + args.lambda_d * rs * flops_regularizer(p)
                loss = ce + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0 or step == total - 1:
                acc = (scores.argmax(1) == labels).float().mean().item()
                l0 = (q > 0).float().sum(1).mean().item()          # avg non-zeros / query
                print(f"step {step:5d}/{total} | loss {loss.item():.4f} (ce {ce.item():.3f}) | "
                      f"acc {acc:.3f} | L0 {l0:.0f} | lr {lr_at(step):.2e}")
            step += 1
            if step >= total:
                done = True; break
        if done:
            break

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save(str(out / "tokenizer.json"))
    print(f"✅ Saved sparse encoder -> {out}")


if __name__ == "__main__":
    main()
