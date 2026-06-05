"""
Reference-free preference optimization for the chat policy: ORPO (default) or SimPO.

Why not DPO: DPO needs a frozen reference copy of the model in memory (≈2× the
weights). For a small model trained on limited preference data, reference-free
methods are simpler, lighter, and competitive:

  ORPO (Hong et al. 2024) — single stage, no reference model. Loss = SFT NLL on
    the chosen response + λ · odds-ratio term that raises chosen over rejected.
    The SFT term keeps generation quality while preferences are learned.
        L = -mean_logp(y_w) + λ·(-log σ( log_odds(y_w) - log_odds(y_l) ))
        log_odds(y) = logP(y) - log(1 - P(y)),  P(y) = exp(mean_logp(y))

  SimPO (Meng et al. 2024) — reference-free, length-normalized reward with margin:
        L = -log σ( β·mean_logp(y_w) - β·mean_logp(y_l) - γ )

Response masking reuses utils.chatML.create_loss_mask (assistant content + closing
<|im_end|>) — identical to SFT, so this composes cleanly after bin.sft.py.

  # real (after SFT):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python training/bin.preference.py \
      --base_model checkpoints_sft/skylar-mp-chat/best \
      --data .datasets/pref/preference_it.jsonl --loss orpo --epochs 1 --lr 8e-6 --bf16 \
      --out_dir checkpoints_pref/skylar-mp-orpo
  # smoke (CPU):
  .venv/bin/python training/bin.preference.py --preset test \
      --tokenizer .datasets/tokenized/tokenizer.json \
      --data .datasets/pref/preference_it.jsonl --device cpu --max_steps 5 --batch_size 4
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.config import get_config
from models.decoder import NanoTransformer
from utils.chatML import create_loss_mask


class PrefDS(Dataset):
    def __init__(self, path):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def build_seq(ex, key, tok, max_len):
    msgs = [{"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex[key]}]
    ids, labels = create_loss_mask(msgs, tok)
    ids, labels = ids[:max_len], labels[:max_len]
    return ids, labels


def make_collate(tok, max_len, pad_id):
    def pad(seqs, fill):
        m = max(1, max(len(s) for s in seqs))
        out = torch.full((len(seqs), m), fill, dtype=torch.long)
        for i, s in enumerate(seqs):
            if s:
                out[i, :len(s)] = torch.tensor(s)
        return out

    def collate(batch):
        cw = [build_seq(b, "chosen", tok, max_len) for b in batch]
        cl = [build_seq(b, "rejected", tok, max_len) for b in batch]
        w_ids = pad([a for a, _ in cw], pad_id); w_lab = pad([b for _, b in cw], -100)
        l_ids = pad([a for a, _ in cl], pad_id); l_lab = pad([b for _, b in cl], -100)
        return (w_ids, w_lab), (l_ids, l_lab)
    return collate


def mean_logp(model, ids, labels):
    """Per-sequence mean log-prob over response tokens (labels != -100)."""
    out = model(ids)
    logits = out["logits"] if isinstance(out, dict) else out.logits
    logits = logits[:, :-1, :].float()                 # predicts positions 1..T-1
    tgt = ids[:, 1:]
    mask = (labels[:, 1:] != -100).float()
    logp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(1) / mask.sum(1).clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--loss", default="orpo", choices=["orpo", "simpo"])
    ap.add_argument("--lambda_or", type=float, default=0.5, help="ORPO odds-ratio weight")
    ap.add_argument("--beta", type=float, default=2.0, help="SimPO reward scale")
    ap.add_argument("--gamma", type=float, default=0.8, help="SimPO target margin")
    ap.add_argument("--out_dir", default="checkpoints_pref/skylar-pref")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vocab_size", type=int, default=32768)
    args = ap.parse_args()

    dev = args.device
    if args.base_model:
        print(f"Load policy {args.base_model} (loss={args.loss})")
        model = NanoTransformer.from_pretrained(args.base_model)
        tok = Tokenizer.from_file(f"{args.base_model}/tokenizer.json")
    else:
        assert args.preset and args.tokenizer
        model = NanoTransformer(get_config(args.preset, vocab_size=args.vocab_size))
        tok = Tokenizer.from_file(args.tokenizer)
    model = model.to(dev).train()
    pad_id = tok.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    ds = PrefDS(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    collate_fn=make_collate(tok, args.max_len, pad_id))
    total = args.max_steps or len(dl) * args.epochs
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M | pairs={len(ds)} | steps={total}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    def lr_at(s):
        if s < args.warmup_steps:
            return args.lr * (s + 1) / args.warmup_steps
        prog = (s - args.warmup_steps) / max(1, total - args.warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    amp = (args.bf16 and dev == "cuda")
    step = 0; t0 = time.time(); done = False
    for ep in range(args.epochs):
        for (w_ids, w_lab), (l_ids, l_lab) in dl:
            w_ids, w_lab = w_ids.to(dev), w_lab.to(dev)
            l_ids, l_lab = l_ids.to(dev), l_lab.to(dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                lp_w = mean_logp(model, w_ids, w_lab)
                lp_l = mean_logp(model, l_ids, l_lab)
                if args.loss == "orpo":
                    P_w = lp_w.exp().clamp(max=1 - 1e-4)
                    P_l = lp_l.exp().clamp(max=1 - 1e-4)
                    log_odds = (lp_w - torch.log1p(-P_w)) - (lp_l - torch.log1p(-P_l))
                    l_or = -F.logsigmoid(log_odds).mean()
                    l_sft = -lp_w.mean()
                    loss = l_sft + args.lambda_or * l_or
                else:  # simpo
                    loss = -F.logsigmoid(args.beta * lp_w - args.beta * lp_l - args.gamma).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0 or step == total - 1:
                margin = (lp_w - lp_l).mean().item()
                acc = (lp_w > lp_l).float().mean().item()
                print(f"step {step:5d}/{total} | loss {loss.item():.4f} | margin {margin:+.3f} | pref_acc {acc:.3f} | lr {lr_at(step):.2e}")
            step += 1
            if step >= total:
                done = True; break
        if done:
            break

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save(str(out / "tokenizer.json"))
    print(f"✅ Saved preference-tuned policy -> {out}")


if __name__ == "__main__":
    main()
