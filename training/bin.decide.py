# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Train SkylarDecider (models/decider.py, beta): typed, calibrated decisions on the decoder's trunk. No re-pretrain:
the same base as everything else, a fresh decision head, a fine-tune.

Data: the schema of Jev's API and of the open typed-decisions dataset (huggingface.co/datasets/LocalLLaMA/typed-decisions,
Apache-2.0), as parquet or JSONL, one CASE per row: {"id", "state", "questions": {qid: {"type", "instructions",
"criteria": {key: description}}}, "gold": {qid: {"probabilities": {key: p}, "label": key}}} ("questions" and "gold"
may be JSON strings). The gold can be a teacher's distribution or a one-hot of a verified label.

Steps:
  1. a share of the training CASES (--calib_frac) is kept out of training: the temperature is fitted there, never on
     what the model trained on (calibrating on trained items made the ECE 3.7x worse, arXiv 2609.33843 §8.1);
  2. fine-tune with a proper scoring rule (--loss ce = log score, or brier), choice options shuffled per example;
  3. one temperature on the held-out cases;
  4. the test set: accuracy, log loss, Brier, ECE, signed confidence gap, AURC, per type -> <out>/metrics.json.

    python training/bin.decide.py --base_model <decoder_dir> --train <train.parquet|jsonl> --test <test...> \
        --epochs 4 --lr 2.5e-5 --head_lr 1e-4 --batch_size 16 --bf16 --out <out_dir>
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.decider import SkylarDecider, TYPES, decision_metrics, option_list   # noqa: E402


def load_cases(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd
        rows = pd.read_parquet(path).to_dict("records")
    else:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        for k in ("questions", "gold"):
            if isinstance(r.get(k), str):
                r[k] = json.loads(r[k])
    return rows


def flatten(cases):
    """Cases -> one example per question: (case id, state, question, options keys, target distribution, label)."""
    out = []
    for c in cases:
        for qid, q in c["questions"].items():
            g = c["gold"][qid]
            keys = [k for k, _ in option_list(q)]
            if "probabilities" in g:
                tgt = [float(g["probabilities"].get(k, 0.0)) for k in keys]
            else:
                tgt = [1.0 if k == str(g["label"]) else 0.0 for k in keys]
            s = sum(tgt)
            tgt = [t / s for t in tgt] if s > 0 else [1.0 / len(keys)] * len(keys)
            label = str(g.get("label", keys[max(range(len(keys)), key=tgt.__getitem__)]))
            out.append({"case": c.get("id"), "state": c["state"], "q": q, "keys": keys, "target": tgt,
                        "label": keys.index(label) if label in keys else max(range(len(keys)), key=tgt.__getitem__)})
    return out


def shuffled(ex, rng):
    """A choice question with its options (and their targets) in a random order: the order must not be a feature."""
    if ex["q"]["type"] != "choice":
        return ex
    opts = option_list(ex["q"])
    perm = list(range(len(opts)))
    rng.shuffle(perm)
    q = dict(ex["q"], criteria={opts[i][0]: opts[i][1] for i in perm})
    return dict(ex, q=q, keys=[ex["keys"][i] for i in perm], target=[ex["target"][i] for i in perm],
                label=perm.index(ex["label"]))


def batches(model, tok, examples, bs, max_len, pad_id, rng=None):
    order = list(range(len(examples)))
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), bs):
        exs = [examples[j] if rng is None else shuffled(examples[j], rng) for j in order[i:i + bs]]
        enc = [model.encode(tok, e["state"], e["q"], max_len) for e in exs]
        ids, att, pos, types = model.collate(enc, pad_id)
        K = pos.shape[1]
        tgt = torch.tensor([e["target"] + [0.0] * (K - len(e["target"])) for e in exs])
        labels = torch.tensor([e["label"] for e in exs])
        yield ids, att, pos, types, tgt, labels


@torch.no_grad()
def predict(model, tok, examples, bs, max_len, pad_id, dev, bf16):
    model.eval()
    L, T, Y, TY = [], [], [], []
    for ids, att, pos, types, tgt, labels in batches(model, tok, examples, bs, max_len, pad_id):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = model(ids.to(dev), att.to(dev), pos.to(dev), types.to(dev))
        K = 255
        lg = torch.full((len(labels), K), -1e4)
        lg[:, :out["logits"].shape[1]] = out["logits"].float().cpu()
        tt = torch.zeros((len(labels), K))
        tt[:, :tgt.shape[1]] = tgt
        L.append(lg); T.append(tt); Y.append(labels); TY.append(types)
    model.train()
    return torch.cat(L), torch.cat(T), torch.cat(Y), torch.cat(TY)


def report(logits, targets, labels, types, temperature):
    probs = (logits / temperature).softmax(-1)
    res = {"all": decision_metrics(probs, targets, labels)}
    for i, t in enumerate(TYPES):
        sel = types == i
        if sel.any():
            res[t] = decision_metrics(probs[sel], targets[sel], labels[sel])
    return res


def main():
    ap = argparse.ArgumentParser(description="Train SkylarDecider: typed, calibrated decisions (beta)")
    ap.add_argument("--base_model", default=None, help="a decoder checkpoint: its trunk + a fresh decision head")
    ap.add_argument("--init", default=None, help="or a SkylarDecider to continue")
    ap.add_argument("--tokenizer", default=None, help="default: <base_model or init>/tokenizer.json")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--calib_frac", type=float, default=0.15, help="share of training CASES kept for the temperature")
    ap.add_argument("--calib_on", choices=("label", "soft"), default="label",
                    help="fit the temperature to the gold label (confidence = accuracy, the default) or to the soft "
                         "gold distribution (it then reproduces the teacher's own uncertainty)")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=2.5e-5, help="the trunk")
    ap.add_argument("--head_lr", type=float, default=1e-4, help="the decision head")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06, help="share of the steps")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--head_layers", type=int, default=2)
    ap.add_argument("--loss", choices=("ce", "brier"), default="ce")
    ap.add_argument("--freeze_trunk", action="store_true", help="train only the head")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    src = args.init or args.base_model
    if not src:
        sys.exit("--base_model or --init")
    tok = Tokenizer.from_file(args.tokenizer or str(Path(src) / "tokenizer.json"))
    pad_id = tok.token_to_id("<pad>") or 0
    model = SkylarDecider.from_pretrained(args.init) if args.init else \
        SkylarDecider.from_decoder(args.base_model, head_layers=args.head_layers)
    dev = torch.device(args.device)
    model.to(dev)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    bf16 = args.bf16 and dev.type == "cuda"

    cases = load_cases(args.train)
    rng.shuffle(cases)
    n_cal = max(1, int(round(len(cases) * args.calib_frac))) if args.calib_frac > 0 else 0
    cal_ex, train_ex = flatten(cases[:n_cal]), flatten(cases[n_cal:])
    test_ex = flatten(load_cases(args.test))
    print(f"[decide] {src}: {model.count_params() / 1e6:.1f}M params ({'causal' if model.causal else 'bidirectional'}) | "
          f"train {len(train_ex)} questions, calibration {len(cal_ex)} ({n_cal} cases), test {len(test_ex)}", flush=True)

    head = [p for n, p in model.named_parameters() if n.startswith("decision.")]
    trunk = [p for n, p in model.named_parameters() if not n.startswith("decision.")]
    if args.freeze_trunk:
        for p in trunk:
            p.requires_grad_(False)
    groups = [{"params": head, "lr": args.head_lr}]
    if not args.freeze_trunk:
        groups.append({"params": trunk, "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.wd, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(train_ex) / args.batch_size / args.grad_accum)
    total = max(1, int(steps_per_epoch * args.epochs))
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = max(1, int(total * args.warmup))
    base_lrs = [g["lr"] for g in opt.param_groups]

    model.train()
    step, t0, run = 0, time.time(), 0.0
    while step < total:
        micro = 0
        for ids, att, pos, types, tgt, labels in batches(model, tok, train_ex, args.batch_size, args.max_len, pad_id, rng):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                out = model(ids.to(dev), att.to(dev), pos.to(dev), types.to(dev), targets=tgt.to(dev), loss=args.loss)
            (out["loss"] / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            f = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * f
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run = 0.98 * run + 0.02 * out["loss"].item() if step > 1 else out["loss"].item()
            if step % 25 == 0 or step == total:
                print(f"  step {step:5d}/{total}  loss {run:.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
            if step >= total:
                break

    # the temperature on the held-out cases, then the test
    temperature = 1.0
    if cal_ex:
        lg, tt, yy, _ = predict(model, tok, cal_ex, args.batch_size, args.max_len, pad_id, dev, bf16)
        if args.calib_on == "label":
            tt = torch.nn.functional.one_hot(yy, lg.shape[1]).float()
        temperature = SkylarDecider.fit_temperature(lg, tt)
    model.config.decision_temperature = temperature
    lg, tt, yy, ty = predict(model, tok, test_ex, args.batch_size, args.max_len, pad_id, dev, bf16)
    res = {"raw": report(lg, tt, yy, ty, 1.0), "calibrated": report(lg, tt, yy, ty, temperature),
           "temperature": temperature, "train_questions": len(train_ex), "calib_questions": len(cal_ex),
           "test_questions": len(test_ex), "steps": step, "minutes": (time.time() - t0) / 60, "args": vars(args)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    shutil.copy2(args.tokenizer or str(Path(src) / "tokenizer.json"), out / "tokenizer.json")
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    a, c = res["raw"]["all"], res["calibrated"]["all"]
    print(f"[decide] test {c['n']} questions: accuracy {a['accuracy']:.4f} | T={temperature:.3f} | ECE {a['ece']:.3f} -> "
          f"{c['ece']:.3f} | signed gap {a['signed_gap']:+.3f} -> {c['signed_gap']:+.3f} | log loss {a['log_loss']:.4f} -> "
          f"{c['log_loss']:.4f} | AURC {c['aurc']:.4f}", flush=True)
    for t in TYPES:
        if t in res["calibrated"]:
            m = res["calibrated"][t]
            print(f"  {t:6s} n={m['n']:5d} accuracy {m['accuracy']:.4f} ECE {m['ece']:.3f} AURC {m['aurc']:.4f}")
    print(f"[decide] written {out}", flush=True)


if __name__ == "__main__":
    main()
