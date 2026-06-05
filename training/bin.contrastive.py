"""
Contrastive trainer for SkylarEmbedder (InfoNCE / in-batch negatives).

Turns the pretrained decoder into a dense text embedder — the E5-Mistral /
GTE-Qwen / LLM2Vec recipe:
  1. pretrain decoder on next-token (cheap, lots of data)   ← done elsewhere
  2. init bidirectional embedder from decoder weights (from_decoder)
  3. contrastive fine-tune with InfoNCE on (query, positive) pairs ← THIS

Loss: for a batch of B (query, positive) pairs, embed both sides (mean-pool,
L2-normalize), build the B×B cosine-similarity matrix scaled by 1/temperature,
and apply cross-entropy where the positive of query i sits on the diagonal.
Every other positive in the batch is an in-batch negative — so a larger batch
gives more negatives and a better signal. Optionally symmetric (q→p and p→q).

Note (LLM2Vec): the decoder was trained causally; running it bidirectionally is
slightly out-of-distribution. Contrastive tuning adapts it; a short MNTP step
first would help further (future). Works well enough directly for v1.

  # real (init from the trained base):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python training/bin.contrastive.py \
      --base_model checkpoints/skylar-mp-base/final \
      --data .datasets/embed/contrastive_it.jsonl --epochs 1 --bf16 \
      --out_dir checkpoints_embed/skylar-mp-embed
  # smoke (CPU, tiny):
  .venv/bin/python training/bin.contrastive.py --preset test \
      --tokenizer .datasets/tokenized/tokenizer.json \
      --data .datasets/embed/contrastive_it.jsonl --device cpu --max_steps 5 --batch_size 8
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.config import get_config
from models.embedder import SkylarEmbedder


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
    def encode_batch(texts):
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
        q = [b[0] for b in batch]
        p = [b[1] for b in batch]
        return encode_batch(q), encode_batch(p)
    return collate


def info_nce(q_emb, p_emb, temperature, symmetric=True):
    # q_emb, p_emb already L2-normalized (B, D)
    logits = (q_emb @ p_emb.t()) / temperature          # (B, B)
    labels = torch.arange(q_emb.shape[0], device=q_emb.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.t(), labels))
    acc = (logits.argmax(dim=1) == labels).float().mean().item()   # in-batch retrieval acc
    return loss, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base_model", default=None, help="decoder ckpt → from_decoder()")
    ap.add_argument("--preset", default=None, help="build fresh embedder (smoke) instead of from_decoder")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json (needed with --preset)")
    ap.add_argument("--pool", default="mean", choices=["mean", "cls", "last"])
    ap.add_argument("--out_dir", default="checkpoints_embed/skylar-embed")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vocab_size", type=int, default=32768)
    args = ap.parse_args()

    dev = args.device
    # ── model ──
    if args.base_model:
        print(f"Init embedder from decoder {args.base_model} (pool={args.pool})")
        model = SkylarEmbedder.from_decoder(args.base_model, pool_strategy=args.pool)
        tok = Tokenizer.from_file(f"{args.base_model}/tokenizer.json")
    else:
        assert args.preset and args.tokenizer, "need --preset and --tokenizer without --base_model"
        cfg = get_config(args.preset, vocab_size=args.vocab_size)
        cfg.pool_strategy = args.pool
        model = SkylarEmbedder(cfg)
        tok = Tokenizer.from_file(args.tokenizer)
    model = model.to(dev).train()
    pad_id = tok.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    ds = PairDS(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    collate_fn=make_collate(tok, args.max_len, pad_id))
    steps_per_epoch = len(dl)
    total = args.max_steps or steps_per_epoch * args.epochs
    print(f"params={model.count_params()/1e6:.1f}M | pairs={len(ds)} | steps={total} | bs={args.batch_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    amp = (args.bf16 and dev == "cuda")
    step = 0; t0 = time.time(); done = False
    for ep in range(args.epochs):
        for (q_ids, q_m), (p_ids, p_m) in dl:
            q_ids, q_m = q_ids.to(dev), q_m.to(dev)
            p_ids, p_m = p_ids.to(dev), p_m.to(dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                q_emb = model(q_ids, attention_mask=q_m)["embeddings"]
                p_emb = model(p_ids, attention_mask=p_m)["embeddings"]
                loss, acc = info_nce(q_emb.float(), p_emb.float(), args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0 or step == total - 1:
                rate = (step + 1) / (time.time() - t0)
                print(f"step {step:5d}/{total} | loss {loss.item():.4f} | inbatch_acc {acc:.3f} | lr {lr_at(step):.2e} | {rate:.1f} it/s")
            step += 1
            if step >= total:
                done = True; break
        if done:
            break

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save(str(out / "tokenizer.json"))
    print(f"✅ Saved embedder -> {out}")


if __name__ == "__main__":
    main()
