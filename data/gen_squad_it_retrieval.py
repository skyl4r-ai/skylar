# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Build a REAL Italian retrieval training set from SQuAD-it (crux82/squad_it).

Each row → {"query": question, "positive": context, "group": <context-id>}.
The `group` field tells the contrastive trainer that two questions sharing the
same gold context must NOT be in-batch negatives of each other (SQuAD has ~27
questions per context, so this matters a lot).

We also assert the TRAIN and TEST contexts are disjoint, so eval on the test
split (eval/bench_retrieval.py) is a clean held-out measurement.

  .venv/bin/python data/gen_squad_it_retrieval.py -o .datasets/embed/squad_it_train.jsonl
"""
import argparse, json
from pathlib import Path


def main():
    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=".datasets/embed/squad_it_train.jsonl")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    train = load_dataset("crux82/squad_it", split=args.split)
    test = load_dataset("crux82/squad_it", split="test")
    test_ctx = set(ex["context"] for ex in test)

    ctx_id = {}
    rows, seen, leak = [], set(), 0
    for ex in train:
        c = ex["context"]
        if c in test_ctx:
            leak += 1                      # context appears in test → skip (no leakage)
            continue
        q = ex["question"].strip()
        key = (q, c)
        if key in seen:
            continue
        seen.add(key)
        gid = ctx_id.setdefault(c, len(ctx_id))
        rows.append({"query": q, "positive": c, "group": f"ctx{gid}"})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train rows={len(rows)} | unique contexts={len(ctx_id)} | "
          f"avg q/ctx={len(rows)/max(1,len(ctx_id)):.1f} | skipped (in test)={leak}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
