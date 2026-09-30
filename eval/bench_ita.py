# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Real public Italian benchmarks for the Skylar2ForCausalLM base model.

Likelihood-based multiple-choice scoring (the standard lm-eval method): for each
candidate continuation we sum the model's log-prob over the continuation tokens
given the context, then pick argmax. Two metrics, as in lm-eval-harness:
  acc       — argmax of total log-prob
  acc_norm  — argmax of log-prob / continuation length (in characters)

Tasks (public, citable, Italian):
  xcopa_it      — causal commonsense, 2 choices  (random 50%)   [xcopa/it]
  hellaswag_it  — sentence completion, 4 choices (random 25%)   [alexandrainst/m_hellaswag/it]
  belebele_it   — reading comprehension, 4 choices (random 25%) [facebook/belebele/ita_Latn]

We prepend <bos> to the context because the model was pretrained on <bos>…<eos>
document wrapping — this keeps the prompt in-distribution.

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python eval/bench_ita.py --model checkpoints/skylar-mp-base/final \
      --tasks xcopa_it,hellaswag_it,belebele_it
"""
import argparse, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.decoder import Skylar2ForCausalLM


# ── task builders: return list of (context:str, candidates:[str], gold:int) ──
def build_xcopa(limit=None):
    from datasets import load_dataset
    ds = load_dataset("xcopa", "it", split="test")
    conn = {"cause": "perché", "effect": "quindi"}
    out = []
    for ex in ds:
        prem = ex["premise"].strip()
        if prem.endswith("."):
            prem = prem[:-1]
        ctx = f"{prem} {conn[ex['question']]}"
        cands = []
        for c in (ex["choice1"], ex["choice2"]):
            c = c.strip()
            cands.append(" " + c[0].lower() + c[1:] if c else c)
        out.append((ctx, cands, int(ex["label"])))
        if limit and len(out) >= limit:
            break
    return out


def build_hellaswag(limit=None):
    from datasets import load_dataset
    ds = load_dataset("alexandrainst/m_hellaswag", "it", split="val")
    out = []
    for ex in ds:
        ctx = ex["ctx"].strip()
        cands = [" " + e.strip() for e in ex["endings"]]
        gold = int(ex["label"])
        out.append((ctx, cands, gold))
        if limit and len(out) >= limit:
            break
    return out


def build_belebele(limit=None):
    from datasets import load_dataset
    ds = load_dataset("facebook/belebele", "ita_Latn", split="test")
    out = []
    for ex in ds:
        ctx = f"{ex['flores_passage'].strip()}\nDomanda: {ex['question'].strip()}\nRisposta:"
        cands = [" " + ex[f"mc_answer{i}"].strip() for i in range(1, 5)]
        gold = int(ex["correct_answer_num"]) - 1
        out.append((ctx, cands, gold))
        if limit and len(out) >= limit:
            break
    return out


BUILDERS = {"xcopa_it": build_xcopa, "hellaswag_it": build_hellaswag, "belebele_it": build_belebele}
RANDOM = {"xcopa_it": 0.5, "hellaswag_it": 0.25, "belebele_it": 0.25}


@torch.no_grad()
def score(model, tok, bos_id, ctx, cont, device, max_len):
    """Return (sum_logprob, n_cont_tokens, n_cont_chars) for cont given ctx."""
    ctx_ids = ([bos_id] if bos_id is not None else []) + tok.encode(ctx, add_special_tokens=False).ids
    cont_ids = tok.encode(cont, add_special_tokens=False).ids
    if not cont_ids:
        return -1e9, 1, 1
    ids = ctx_ids + cont_ids
    ids = ids[-max_len:]                       # left-truncate long contexts
    n_cont = min(len(cont_ids), len(ids) - 1)  # cont tokens still present
    x = torch.tensor([ids], device=device)
    out = model(x)
    logits = out["logits"] if isinstance(out, dict) else out.logits
    logp = F.log_softmax(logits[0].float(), dim=-1)
    # token at position i is predicted by logits at i-1
    total = 0.0
    start = len(ids) - n_cont
    for i in range(start, len(ids)):
        total += logp[i - 1, ids[i]].item()
    return total, n_cont, max(1, len(cont))


def run_task(model, tok, bos_id, task, device, max_len, limit):
    data = BUILDERS[task](limit=limit)
    correct = correct_norm = 0
    for ctx, cands, gold in data:
        scores, norms = [], []
        for c in cands:
            lp, nt, nc = score(model, tok, bos_id, ctx, c, device, max_len)
            scores.append(lp)
            norms.append(lp / nc)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        pred_norm = max(range(len(norms)), key=lambda i: norms[i])
        correct += int(pred == gold)
        correct_norm += int(pred_norm == gold)
    n = len(data)
    return n, correct / n, correct_norm / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="xcopa_it,hellaswag_it,belebele_it")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task (smoke)")
    args = ap.parse_args()

    print(f"Loading {args.model} on {args.device} ...")
    model = Skylar2ForCausalLM.from_pretrained(args.model).to(args.device).eval()
    tok = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    bos_id = tok.token_to_id("<bos>")

    print("=" * 70)
    print(f"{'task':<16}{'n':>6}{'acc':>9}{'acc_norm':>11}{'random':>9}")
    print("-" * 70)
    results = {}
    for task in args.tasks.split(","):
        task = task.strip()
        if task not in BUILDERS:
            print(f"{task:<16}  unknown, skipped"); continue
        n, acc, accn = run_task(model, tok, bos_id, task, args.device, args.max_len, args.limit)
        results[task] = (n, acc, accn)
        best = max(acc, accn)
        gold = "  ⭐" if best > RANDOM[task] + 0.05 else ""
        print(f"{task:<16}{n:>6}{acc:>9.3f}{accn:>11.3f}{RANDOM[task]:>9.2f}{gold}")
    print("=" * 70)
    print("acc_norm (length-normalized) è la metrica headline per hellaswag/belebele;")
    print("xcopa usa acc. ⭐ = >5pt sopra il random baseline.")


if __name__ == "__main__":
    main()
