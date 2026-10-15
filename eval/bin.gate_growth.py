"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Gates for the width growth (models/growth.py, beta): a model grown k times wider must compute the same function
as its source before any training, on a dense Skylar 1 model and on a Skylar 2 hybrid (KDA 3:1, AttnRes, GatedNorm,
output gate, SiTU). Each gate costs seconds:

  W1  same logits and loss, with and without the symmetry-breaking noise, on plain and on packed batches
      (document_ids: the KDA state and the short conv reset at each document); tied and untied heads, biases,
      full-rank gates
  W2  greedy generation with the cache (attention KV, KDA state, short-conv state) gives the same tokens
  W3  the noise breaks the symmetry: after one Muon + AdamW step the copies have moved apart; without it they
      stay identical (why the noise is not optional)
  W4  save_pretrained / from_pretrained of the grown model: same logits, same shape (d_head, heads, KDA heads)
  W5  the 990M (1B_D with every Skylar 2 flag, vocab 64k) grown 2x: the parameter count, on the meta device
  W6  bf16 on the GPU with the fla kernels: the grown model is as close to the source in fp32 as the source in
      bf16 is (skipped without CUDA)

    python eval/bin.gate_growth.py

Exit code 0 = all passed, 1 = at least one broken.
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.config import get_config                     # noqa: E402
from models.decoder import Skylar2ForCausalLM               # noqa: E402
from models.growth import grow_width, grown_config, compare_logits, copy_similarity   # noqa: E402
from training.optim import build_optimizer                 # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
RESULTS = []
V = 512
TOL_CPU = 1e-5            # relative to the largest logit, fp32 on the CPU (measured ~1e-6)
# GQA with 3 q heads per kv head, as on the 990M (12/4): the copies of the q heads must read the copies of their kv
HYBRID = dict(kda_ratio="3:1", attn_res=True, attn_res_mode="block", attn_res_block_size=2, gated_norm=16,
              attn_out_gate="perhead", hidden_act="situ_glu", n_heads=6, n_kv_heads=2, d_head=32)
VARIANTS = {
    "dense": {},
    "hybrid": HYBRID,
    "hybrid untied+bias+fullrank": dict(HYBRID, tie_weights=False, bias=True, attn_out_gate="fullrank",
                                        kda_gate="fullrank"),
}


def report(gate, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{PASS if ok else FAIL}] {gate:<52} {detail}")


def source_model(kw, seed=0):
    """A small model whose every parameter is non-trivial: the AttnRes queries start at 0 and the norms at 1, and a
    clone of zeros or ones would hide a wrong split."""
    torch.manual_seed(seed)
    m = Skylar2ForCausalLM(get_config("test", vocab_size=V, dropout=0.0, **kw)).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


def batch(seed=1, B=2, T=64):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, V, (B, T), generator=g)
    cut = (T * 29) // 64                                   # two documents, the cut away from the chunk edges
    doc = torch.cat([torch.zeros(B, cut, dtype=torch.long), torch.ones(B, T - cut, dtype=torch.long)], 1)
    return ids, doc


def w1():
    ids, doc = batch()
    for name, kw in VARIANTS.items():
        src = source_model(kw)
        for snr in (None, 10.0):
            big = grow_width(src, 2, noise_snr_db=snr).eval()
            worst = max(compare_logits(src, big, ids)["max_rel"],
                        compare_logits(src, big, ids, document_ids=doc)["max_rel"])
            r = compare_logits(src, big, ids)
            report(f"W1 {name}, noise {'off' if snr is None else f'{snr:g} dB'}", worst < TOL_CPU,
                   f"max|dlogit|/max|logit| {worst:.1e}, loss {r['loss_source']:.6f} = {r['loss_grown']:.6f}")


@torch.no_grad()
def greedy(model, prompt, n):
    out = model(prompt, use_cache=True)
    cache, ids = out["kv_cache"], [int(out["logits"][0, -1].argmax())]
    for _ in range(n - 1):
        out = model(torch.tensor([[ids[-1]]]), kv_cache=cache, use_cache=True)
        cache = out["kv_cache"]
        ids.append(int(out["logits"][0, -1].argmax()))
    return ids


def w2():
    src = source_model(HYBRID)
    big = grow_width(src, 2).eval()
    prompt = batch(seed=3, B=1, T=12)[0]
    a, b = greedy(src, prompt, 24), greedy(big, prompt, 24)
    report("W2 greedy with the cache, hybrid", a == b, f"{sum(x == y for x, y in zip(a, b))}/24 tokens equal")


def w3():
    ids, doc = batch(seed=5, T=48)
    for snr, want in ((None, "identical"), (10.0, "apart")):
        big = grow_width(source_model(HYBRID), 2, noise_snr_db=snr).train()
        opt, _ = build_optimizer(big, 1e-3, 0.0, "muon")
        big(ids, labels=ids, document_ids=doc)["loss"].backward()
        opt.step()
        cos = copy_similarity(big, 2)
        ok = (cos > 1 - 1e-6) if want == "identical" else (cos < 1 - 1e-5)
        report(f"W3 one Muon step, noise {'off' if snr is None else f'{snr:g} dB'}: copies {want}", ok,
               f"cosine between the embedding copies {cos:.7f}")


def w4():
    src = source_model(HYBRID)
    big = grow_width(src, 2).eval()
    ids, doc = batch(seed=7)
    with tempfile.TemporaryDirectory() as t:
        big.save_pretrained(t)
        back = Skylar2ForCausalLM.from_pretrained(t).eval()
    c, s = back.config, src.config
    shape_ok = (c.d_model == 2 * s.d_model and c.n_heads == 2 * s.n_heads and c.n_kv_heads == 2 * s.n_kv_heads
                and c.kda_heads == 2 * s.kda_heads and c.d_head == s.d_head and c.d_ff == 2 * s.d_ff
                and c.layer_types == s.layer_types)
    r = compare_logits(big, back, ids, document_ids=doc)
    report("W4 save / from_pretrained of the grown model", shape_ok and r["max_abs"] == 0.0,
           f"d_model {c.d_model}, heads {c.n_heads}/{c.n_kv_heads}, KDA {c.kda_heads}, d_head {c.d_head}, "
           f"max|dlogit| {r['max_abs']:.1e}")


def w5():
    flags = dict(kda_ratio="3:1", attn_res=True, attn_res_mode="block", attn_res_block_size=8, gated_norm=16,
                 attn_out_gate="perhead", hidden_act="situ_glu")
    cfg = get_config("1B_D", vocab_size=64000, **flags)
    with torch.device("meta"):
        small = sum(p.numel() for p in Skylar2ForCausalLM(cfg).parameters())
        big = sum(p.numel() for p in Skylar2ForCausalLM(grown_config(cfg, 2)).parameters())
    g = grown_config(cfg, 2)
    report("W5 990M grown 2x (meta device)", abs(big / 1e9 - 3.866) < 0.002,
           f"{small / 1e9:.3f}B -> {big / 1e9:.3f}B: d_model {g.d_model}, heads {g.n_heads}/{g.n_kv_heads}, "
           f"KDA {g.kda_heads}, d_ff {g.d_ff}, {g.n_layers} layers")


def w6():
    if not torch.cuda.is_available():
        print("  [skip] W6 bf16 on the GPU: no CUDA")
        return
    src = source_model(HYBRID)
    big = grow_width(src, 2)
    ids, doc = (t.cuda() for t in batch(seed=9, T=128))
    src32 = src.cuda().float().eval()
    ref = src32(ids, document_ids=doc)["logits"].float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        own = src32(ids, document_ids=doc)["logits"].float()
        grown = big.cuda().float().eval()(ids, document_ids=doc)["logits"].float()
    e_src, e_big = (own - ref).abs().max().item(), (grown - ref).abs().max().item()
    report("W6 bf16 on the GPU (fla kernels)", e_big <= 3 * e_src + 1e-3,
           f"max|dlogit| vs fp32: source in bf16 {e_src:.1e}, grown in bf16 {e_big:.1e}")


def main():
    print("growth gates (models/growth.py)")
    for gate in (w1, w2, w3, w4, w5, w6):
        gate()
    n, ok = len(RESULTS), sum(RESULTS)
    print(f"  {ok}/{n} growth gates passed")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
