"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Gates for the encoders built on the decoder's backbone: the dense embedder, the sparse (SPLADE) encoder and the
classifier (models/encoder_base.py). Each one asks «is it broken?» and costs seconds, on a dense Skylar 1 model and on
a Skylar 2 hybrid (KDA 3:1, AttnRes, GatedNorm, output gate, SiTU):

  E1  from_decoder loads the whole trunk of a decoder checkpoint into each encoder, nothing skipped
  E2  the encoder's states are the decoder's own (same code): equal to decoder(..., return_hidden=True)
  E3  reading mode: the hybrid reads causally and pools an appended <eos>; a dense model reads bidirectionally;
      bidirectional on KDA layers is refused
  E4  right padding never changes a text's vector (each text alone = the same text in a padded batch)
  E5  save_pretrained / from_pretrained give back the same outputs, for all three
  E6  gradient checkpointing gives the same loss and gradients (AttnRes keeps a depth state across blocks)
  E7  they learn: in-batch retrieval, a classification rule, sparse retrieval under the FLOPS regularizer
  E8  bf16 autocast on the GPU

    python eval/bin.gate_encoders.py            # on the GPU if there is one, else on the CPU (the KDA in PyTorch)

Exit code 0 = all passed, 1 = at least one broken.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.config import get_config                     # noqa: E402
from models.decoder import Skylar2ForCausalLM               # noqa: E402
from models.embedder import SkylarEmbedder                 # noqa: E402
from models.sparse_encoder import SkylarSparseEncoder, flops_regularizer   # noqa: E402
from models.classifier import SkylarClassifier             # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
RESULTS = []
STEPS = 1
V = 512
HYBRID = dict(kda_ratio="3:1", attn_res=True, attn_res_mode="block", attn_res_block_size=2, gated_norm=16,
              attn_out_gate="perhead", hidden_act="situ_glu")


def report(gate, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{PASS if ok else FAIL}] {gate:<44} {detail}")


def make_decoders(tmp, dev):
    """A dense and a hybrid decoder on the test preset, saved: what from_decoder reads."""
    paths = {}
    for name, kw in [("dense", {}), ("hybrid", HYBRID)]:
        torch.manual_seed(0)
        m = Skylar2ForCausalLM(get_config("test", vocab_size=V, **kw))
        p = Path(tmp) / name
        m.save_pretrained(p)
        paths[name] = str(p)
    return paths


def batch(dev, lengths, seed=0):
    """Right-padded random ids (pad 0, <eos> 2 at the end of each row) and the mask."""
    g = torch.Generator().manual_seed(seed)
    T = max(lengths)
    ids = torch.zeros(len(lengths), T, dtype=torch.long)
    mask = torch.zeros(len(lengths), T, dtype=torch.long)
    for i, n in enumerate(lengths):
        ids[i, :n - 1] = torch.randint(5, V, (n - 1,), generator=g)
        ids[i, n - 1] = 2
        mask[i, :n] = 1
    return ids.to(dev), mask.to(dev)


def e1_e3(paths, dev):
    built = {}
    for name in ("dense", "hybrid"):
        try:
            built[name] = (SkylarEmbedder.from_decoder(paths[name]).to(dev).eval(),
                           SkylarSparseEncoder.from_decoder(paths[name]).to(dev).eval(),
                           SkylarClassifier.from_decoder(paths[name], num_labels=3).to(dev).eval())
            report(f"E1 from_decoder, {name}", True, "embedder, sparse, classifier: every trunk weight loaded")
        except Exception as e:
            report(f"E1 from_decoder, {name}", False, f"{type(e).__name__}: {str(e)[:90]}")
            return None
    e, h = built["dense"][0], built["hybrid"][0]
    ok = (not e.causal and e.pool_strategy == "mean" and h.causal and h.pool_strategy == "last" and h.append_eos)
    try:
        cfg = get_config("test", vocab_size=V, **HYBRID)
        cfg.encoder_attention = "bidirectional"
        SkylarEmbedder(cfg)
        refused = False
    except ValueError:
        refused = True
    report("E3 reading mode", ok and refused,
           f"dense: {'causal' if e.causal else 'bidirectional'}/{e.pool_strategy}; hybrid: "
           f"{'causal' if h.causal else 'bidirectional'}/{h.pool_strategy}, eos={h.append_eos}; "
           f"bidirectional hybrid refused={refused}")
    return built


def e2_same_trunk(paths, built, dev):
    ids, mask = batch(dev, [24, 24])
    for name, kw in [("hybrid", {}), ("dense, causal", {"encoder_attention": "causal"})]:
        src = paths["hybrid" if name == "hybrid" else "dense"]
        dec = Skylar2ForCausalLM.from_pretrained(src).to(dev).eval()
        enc = SkylarEmbedder.from_decoder(src, **kw).to(dev).eval()
        with torch.no_grad():
            a = dec(ids, return_hidden=True)["hidden"]
            b, _ = enc.hidden_states(ids, mask)
        report(f"E2 same states as the decoder, {name}", torch.equal(a, b),
               "identical" if torch.equal(a, b) else f"max|diff| {(a - b).abs().max().item():.3g}")


def e4_padding(built, dev):
    lengths = [9, 17, 30]
    ids, mask = batch(dev, lengths, seed=3)
    for name in ("dense", "hybrid"):
        enc = built[name][0]
        for pool in (["mean"] if name == "dense" else ["last", "mean"]):
            enc.config.pool_strategy = pool
            with torch.no_grad():
                together = enc(ids, attention_mask=mask)["embeddings"]
                alone = torch.cat([enc(ids[i:i + 1, :n], attention_mask=mask[i:i + 1, :n])["embeddings"]
                                   for i, n in enumerate(lengths)])
            d = (together - alone).abs().max().item()
            # on the GPU the KDA Triton kernels work in chunks whose numerics depend on the sequence length: the
            # decoder itself moves a token's state by ~2e-4 when more tokens follow, padding or real (measured
            # 10/2026). The exact CPU path shows the logic: ~1e-7.
            tol = 1e-3 if (dev == "cuda" and name == "hybrid") else 1e-5
            report(f"E4 padding-invariant, {name}, {pool}", d < tol, f"max|diff| {d:.2e} (tolerance {tol:g})")
        enc.config.pool_strategy = None


def e5_roundtrip(built, dev, tmp):
    ids, mask = batch(dev, [12, 20], seed=5)
    for enc, key in zip(built["hybrid"], ("embeddings", "sparse", "logits")):
        p = Path(tmp) / f"rt_{type(enc).__name__}"
        enc.save_pretrained(p)
        back = type(enc).from_pretrained(p).to(dev).eval()
        with torch.no_grad():
            a, b = enc(ids, attention_mask=mask)[key], back(ids, attention_mask=mask)[key]
        report(f"E5 save/load, hybrid {type(enc).__name__}", torch.equal(a, b) and back.causal,
               "identical" if torch.equal(a, b) else f"max|diff| {(a - b).abs().max().item():.3g}")


def e6_checkpointing(paths, dev):
    ids, mask = batch(dev, [16, 22, 22, 9], seed=7)
    out = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        enc = SkylarEmbedder.from_decoder(paths["hybrid"]).to(dev).train()
        enc.gradient_checkpointing = ckpt
        emb = enc(ids, attention_mask=mask)["embeddings"]
        target = torch.randn(emb.shape, generator=torch.Generator().manual_seed(1)).to(dev)
        loss = ((emb - F.normalize(target, dim=-1)) ** 2).sum()
        loss.backward()
        out.append((loss.detach(), {n: p.grad.detach().clone() for n, p in enc.named_parameters()
                                    if p.grad is not None}))
    (l0, g0), (l1, g1) = out
    worst = max((g0[n] - g1[n]).abs().max().item() / (g0[n].abs().max().item() + 1e-12) for n in g0)
    report("E6 checkpointing: same loss and grads, hybrid", torch.allclose(l0, l1, atol=1e-5) and worst < 1e-3
           and set(g0) == set(g1) and l0.item() > .1,
           f"loss {l0.item():.4f}/{l1.item():.4f}, {len(g0)} grads, worst relative diff {worst:.1e}")


def e7_learning(paths, dev):
    torch.manual_seed(0)
    # retrieval: a query is a text, its positive the same text with a fifth of the tokens replaced
    enc = SkylarEmbedder.from_decoder(paths["hybrid"]).to(dev).train()
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
    accs = []
    for step in range(120 * STEPS):
        q, qm = batch(dev, [20] * 16, seed=100 + step)
        p = q.clone()
        noise = (torch.rand(p.shape, device=dev) < .2) & (qm.bool()) & (p != 2)
        p[noise] = torch.randint(5, V, (int(noise.sum()),), device=dev)
        a, b = enc(q, attention_mask=qm)["embeddings"], enc(p, attention_mask=qm)["embeddings"]
        logits = a @ b.t() / 0.05
        loss = F.cross_entropy(logits, torch.arange(16, device=dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        accs.append((logits.argmax(1) == torch.arange(16, device=dev)).float().mean().item())
    report("E7 embedder learns retrieval, hybrid", sum(accs[-10:]) / 10 > .9,
           f"in-batch accuracy {sum(accs[:5]) / 5:.2f} → {sum(accs[-10:]) / 10:.2f}")
    # classification: the label is the token at position 3 modulo 3
    cls = SkylarClassifier.from_decoder(paths["hybrid"], num_labels=3).to(dev).train()
    opt = torch.optim.AdamW(cls.parameters(), lr=1e-3)
    accs = []
    for step in range(150 * STEPS):
        x, m = batch(dev, [16] * 32, seed=300 + step)
        y = torch.randint(0, 3, (32,), generator=torch.Generator().manual_seed(400 + step)).to(dev)
        pos = torch.randint(0, 14, (32,), generator=torch.Generator().manual_seed(450 + step)).to(dev)
        x[torch.arange(32, device=dev), pos] = 10 + y          # the class is a marker somewhere in the text
        out = cls(x, attention_mask=m, labels=y)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        accs.append((out["logits"].argmax(1) == y).float().mean().item())
    report("E7 classifier finds a marker, hybrid", sum(accs[-10:]) / 10 > .9,
           f"accuracy {sum(accs[:5]) / 5:.2f} → {sum(accs[-10:]) / 10:.2f}")
    # sparse: retrieval with SPLADE vectors under the FLOPS regularizer
    sp = SkylarSparseEncoder.from_decoder(paths["hybrid"]).to(dev).train()
    opt = torch.optim.AdamW(sp.parameters(), lr=1e-3)
    accs, nnz = [], []
    for step in range(120 * STEPS):
        q, qm = batch(dev, [20] * 16, seed=500 + step)
        p = q.clone()
        noise = (torch.rand(p.shape, device=dev) < .2) & (qm.bool()) & (p != 2)
        p[noise] = torch.randint(5, V, (int(noise.sum()),), device=dev)
        a, b = sp(q, attention_mask=qm)["sparse"], sp(p, attention_mask=qm)["sparse"]
        logits = a @ b.t()
        loss = F.cross_entropy(logits, torch.arange(16, device=dev)) + 1e-4 * (flops_regularizer(a) +
                                                                               flops_regularizer(b))
        opt.zero_grad()
        loss.backward()
        opt.step()
        accs.append((logits.argmax(1) == torch.arange(16, device=dev)).float().mean().item())
        nnz.append((a > 0).float().sum(1).mean().item())
    ok = sum(accs[-10:]) / 10 > .9 and bool((a >= 0).all())
    report("E7 sparse learns retrieval, hybrid", ok,
           f"accuracy {sum(accs[:5]) / 5:.2f} → {sum(accs[-10:]) / 10:.2f}, non-zero terms {nnz[0]:.0f} → "
           f"{nnz[-1]:.0f} of {V}, all ≥ 0")


def e8_bf16(built, dev):
    if dev != "cuda":
        print("  [ -- ] E8 bf16 autocast                              only on a GPU")
        return
    ids, mask = batch(dev, [14, 25], seed=9)
    ok = True
    for enc, key in zip(built["hybrid"], ("embeddings", "sparse", "logits")):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ok &= bool(torch.isfinite(enc(ids, attention_mask=mask)[key].float()).all())
    report("E8 bf16 autocast, hybrid", ok, "finite outputs from all three")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=1, help="multiply the E7 training steps (a slower, surer check)")
    a = ap.parse_args()
    global STEPS
    STEPS = a.steps
    dev = a.device
    print(f"encoder gates on {dev}")
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_decoders(tmp, dev)
        built = e1_e3(paths, dev)
        if built is not None:
            e2_same_trunk(paths, built, dev)
            e4_padding(built, dev)
            e5_roundtrip(built, dev, tmp)
            e6_checkpointing(paths, dev)
            e7_learning(paths, dev)
            e8_bf16(built, dev)
    n = len(RESULTS)
    print(f"\n  {sum(RESULTS)}/{n} encoder gates passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
