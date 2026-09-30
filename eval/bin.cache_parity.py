"""
Cache parity on a TRAINED checkpoint: greedy decode with the KV/recurrent cache vs full recompute.

Why a separate tool from gate G9 (`bin.gate_arch_v2.py`): at random init a deep model is chaotic —
on `1B_D` the full recompute alone differs by ~80% between fp32 and bf16, so a cache check on random
weights cannot tell a bug from noise. On trained weights the distribution is peaked and a correct cache
agrees to ~1e-5 in fp32. Run it on the first real checkpoint of every new run:

    python eval/bin.cache_parity.py --ckpt <out>/last
    python eval/bin.cache_parity.py --ckpt <out>/last --prompt "       IDENTIFICATION DIVISION." --steps 50

Pass: fp32 argmax identical at every step. The bf16 line is informational.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.decoder import Skylar2ForCausalLM  # noqa: E402


def run(model, ids, steps, dtype):
    m = model.to(dtype)
    worst, agree = 0.0, 0
    with torch.no_grad():
        out = m(ids, use_cache=True)
        cache, seq = out["kv_cache"], ids.clone()
        nxt = out["logits"][:, -1:].argmax(-1)
        for _ in range(steps):
            seq = torch.cat([seq, nxt], 1)
            o = m(nxt, kv_cache=cache, use_cache=True)
            cache = o["kv_cache"]
            lc = o["logits"][0, -1].float()
            lf = m(seq)["logits"][0, -1].float()
            worst = max(worst, ((lc - lf).abs().max() / lf.abs().max()).item())
            agree += int(lc.argmax() == lf.argmax())
            nxt = lf.argmax().view(1, 1)        # always follow the full recompute
    return worst, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None, help="default: <ckpt>/tokenizer.json")
    ap.add_argument("--prompt", default="       IDENTIFICATION DIVISION.\n       PROGRAM-ID. HELLO.\n")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer or str(Path(args.ckpt) / "tokenizer.json"))
    bos = tok.token_to_id("<bos>")
    model = Skylar2ForCausalLM.from_pretrained(args.ckpt).cuda().eval()
    ids = torch.tensor([([bos] if bos is not None else []) + tok.encode(args.prompt).ids], device="cuda")

    ok = True
    for dt in (torch.float32, torch.bfloat16):
        worst, agree = run(model, ids, args.steps, dt)
        tag = str(dt).replace("torch.", "")
        if dt is torch.float32:
            ok = agree == args.steps
        print(f"{tag:9} max relative logit error {worst:.2e} · argmax equal {agree}/{args.steps}")
    print("PASS" if ok else "FAIL — cache and full recompute diverge in fp32")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
