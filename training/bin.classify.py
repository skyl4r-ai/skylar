# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Train SkylarClassifier — BERT-style sequence classification (non-generative).

Reuses the pretrained decoder backbone (from_decoder: bidirectional on a dense
model, causal with the last token on Skylar 2) + a linear head. Reads {"text","label"} JSONL, cross-entropy, reports accuracy on a held-out
split. No re-pretrain — same base as everything else.

  # real:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python training/bin.classify.py \
      --base_model checkpoints/skylar-mp-base/final \
      --data .datasets/cls/intent_it.jsonl --num_labels 5 --epochs 3 --bf16 \
      --out_dir checkpoints_cls/skylar-mp-intent
  # smoke (CPU):
  .venv/bin/python training/bin.classify.py --preset test \
      --tokenizer .datasets/tokenized/tokenizer.json \
      --data .datasets/cls/intent_it.jsonl --num_labels 5 --device cpu --max_steps 20 --batch_size 16
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.config import get_config
from models.classifier import SkylarClassifier


class TextDS(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]["text"], int(self.rows[i]["label"])


def make_collate(model, tok, max_len, pad_id):
    def collate(batch):
        labels = torch.tensor([b[1] for b in batch])
        # right padding, <eos> appended when the model pools the last token (Skylar 2)
        input_ids, attn = model.tokenize(tok, [b[0] for b in batch], max_len, pad_id)
        return input_ids, attn, labels
    return collate


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    correct = n = 0
    for ids, attn, labels in dl:
        out = model(ids.to(dev), attention_mask=attn.to(dev))
        pred = out["logits"].argmax(-1).cpu()
        correct += (pred == labels).sum().item(); n += len(labels)
    model.train()
    return correct / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--num_labels", type=int, required=True)
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--pool", default=None, choices=["mean", "cls", "last"], help="default: last (an appended <eos>) on Skylar 2, which reads causally; mean on dense models")
    ap.add_argument("--out_dir", default="checkpoints_cls/skylar-cls")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="gradient checkpointing: recomputes the blocks in the backward to save memory "
                         "(needed for Skylar 2 at batch 32 × 256 on 24 GB)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vocab_size", type=int, default=32768)
    args = ap.parse_args()

    dev = args.device
    if args.base_model:
        model = SkylarClassifier.from_decoder(args.base_model, args.num_labels, args.pool)
        tok = Tokenizer.from_file(f"{args.base_model}/tokenizer.json")
    else:
        assert args.preset and args.tokenizer
        cfg = get_config(args.preset, vocab_size=args.vocab_size)
        cfg.num_labels = args.num_labels; cfg.pool_strategy = args.pool
        model = SkylarClassifier(cfg)
        tok = Tokenizer.from_file(args.tokenizer)
    model = model.to(dev).train()
    if args.grad_ckpt:
        model.gradient_checkpointing = True
        print("gradient checkpointing: ON")
    pad_id = tok.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    collate = make_collate(model, tok, args.max_len, pad_id)
    dl = DataLoader(TextDS(train_rows), batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    vdl = DataLoader(TextDS(val_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    total = args.max_steps or len(dl) * args.epochs
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M | train={len(train_rows)} val={len(val_rows)} | steps={total}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(s):
        if s < args.warmup_steps:
            return args.lr * (s + 1) / args.warmup_steps
        prog = (s - args.warmup_steps) / max(1, total - args.warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    amp = (args.bf16 and dev == "cuda")
    step = 0; done = False
    for ep in range(args.epochs):
        for ids, attn, labels in dl:
            ids, attn, labels = ids.to(dev), attn.to(dev), labels.to(dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(ids, attention_mask=attn, labels=labels)
                loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0 or step == total - 1:
                print(f"step {step:5d}/{total} | loss {loss.item():.4f} | lr {lr_at(step):.2e}")
            step += 1
            if step >= total:
                done = True; break
        if done:
            break

    acc = evaluate(model, vdl, dev)
    print(f"✅ val accuracy: {acc:.3f} ({len(val_rows)} held-out)")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save(str(out / "tokenizer.json"))
    print(f"✅ Saved classifier -> {out}")


if __name__ == "__main__":
    main()
