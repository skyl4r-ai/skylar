"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Gates for the typed-decision model (models/decider.py, beta), on a dense Skylar 1 model and a Skylar 2 hybrid, with
a word-level tokenizer built on the spot and a synthetic task (the color, the size and «is it big?» of an item):

  D1  from_decoder loads the whole trunk; a fresh head; the trunk's states are the decoder's own
  D2  the input: one closing <|im_end|> read per option, noul read as (false, true), score levels in order, a long state
      truncated while the question and the options stay whole
  D3  padding: a question's logits do not change when it shares a batch with a question with more options
  D4  the probabilities sum to 1 over the real options and are 0 on the padding
  D5  it learns: choice, noul and score on the synthetic task, dense and hybrid
  D6  one temperature: fitted on over-confident logits it brings the log loss and the ECE down
  D7  decide(): Jev's schema and confidence formulas; save / from_pretrained gives the same answers, temperature included
  D8  bf16 on the GPU (skipped without CUDA)

    python eval/bin.gate_decider.py

Exit code 0 = all passed, 1 = at least one broken.
"""

import json
import random
import sys
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.config import get_config                     # noqa: E402
from models.decoder import Skylar2ForCausalLM               # noqa: E402
from models.decider import SkylarDecider, decision_metrics  # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
RESULTS = []
V = 512
HYBRID = dict(kda_ratio="3:1", attn_res=True, attn_res_mode="block", attn_res_block_size=2, gated_norm=16,
              attn_out_gate="perhead", hidden_act="situ_glu")
COLORS = ("red", "green", "blue", "yellow")
SHAPES = ("box", "ball", "cone")


def report(gate, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{PASS if ok else FAIL}] {gate:<52} {detail}")


def tokenizer():
    words = ["<pad>", "<bos>", "<eos>", "<|im_start|>", "<|im_end|>", "<unk>",
             "state", "question", "option", "choice", "score", "noul", "false", "true", "level", "item", "is", "the",
             "which", "color", "size", "shape", "how", "big", "small", "it", "of", "padding", "x",
             *COLORS, *SHAPES, *[str(i) for i in range(10)], "{", "}", "[", "]", ":", ",", '"', '":', '",', '{"', '"}',
             '":"', '","', "?", "."]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


QUESTIONS = {
    "color": {"type": "choice", "instructions": "which color is the item ?",
              "criteria": {c: f"the item is {c}" for c in COLORS}},
    "big": {"type": "noul", "instructions": "the item is big .",
            "criteria": {"true": "it is big", "false": "it is small"}},
    "size": {"type": "score", "instructions": "how big is the item ?",
             "criteria": {str(i): f"level {i}" for i in range(5)}},
}


def case(rng):
    c, s, h = rng.choice(COLORS), rng.randrange(5), rng.choice(SHAPES)
    gold = {"color": {"label": c}, "big": {"label": "true" if s >= 3 else "false"}, "size": {"label": str(s)}}
    return {"state": {"color": c, "size": s, "shape": h}, "questions": QUESTIONS, "gold": gold}


def examples(model, tok, cases):
    """Flattened questions: (encoded, target one-hot, label index)."""
    out = []
    for c in cases:
        for qid, q in c["questions"].items():
            ids, marks, ty, keys = model.encode(tok, c["state"], q, max_len=256)
            lab = keys.index(c["gold"][qid]["label"])
            out.append(((ids, marks, ty, keys), [1.0 if i == lab else 0.0 for i in range(len(keys))], lab))
    return out


def batch(model, exs):
    ids, att, pos, types = model.collate([e[0] for e in exs])
    K = pos.shape[1]
    tgt = torch.tensor([e[1] + [0.0] * (K - len(e[1])) for e in exs])
    return ids, att, pos, types, tgt, torch.tensor([e[2] for e in exs])


def make_decoders(tmp):
    paths = {}
    for name, kw in [("dense", {}), ("hybrid", HYBRID)]:
        torch.manual_seed(0)
        m = Skylar2ForCausalLM(get_config("test", vocab_size=V, dropout=0.0, **kw))
        m.save_pretrained(Path(tmp) / name)
        paths[name] = str(Path(tmp) / name)
    return paths


def d1_d4(paths, tok):
    for name, p in paths.items():
        dec = Skylar2ForCausalLM.from_pretrained(p).eval()
        m = SkylarDecider.from_decoder(p).eval()
        ids = torch.randint(10, 40, (2, 24))
        h_dec = dec(ids, return_hidden=True)["hidden"] if m.causal else None
        h = m.hidden_states(ids)[0]
        same = True if h_dec is None else torch.allclose(h, h_dec, atol=1e-5)
        report(f"D1 from_decoder, {name}", same and any(n.startswith("decision.") for n, _ in m.named_parameters()),
               f"{'causal' if m.causal else 'bidirectional'}, {m.count_params() / 1e6:.2f}M, trunk = decoder: {same}")
    m = SkylarDecider.from_decoder(paths["hybrid"]).eval()
    im_e = tok.token_to_id("<|im_end|>")
    ids, marks, ty, keys = m.encode(tok, {"color": "red", "size": 3}, QUESTIONS["big"], max_len=256)
    _, _, _, skeys = m.encode(tok, {"color": "red"}, QUESTIONS["size"], max_len=256)
    long_state = {"color": "red", "shape": " ".join(["x"] * 400)}
    lids, lmarks, _, _ = m.encode(tok, long_state, QUESTIONS["color"], max_len=96)
    ok = (all(ids[k] == im_e for k in marks) and keys == ["false", "true"] and skeys == [str(i) for i in range(5)]
          and len(lids) <= 96 and len(lmarks) == 4 and all(lids[k] == im_e for k in lmarks))
    report("D2 input: markers, noul and score order, truncation", ok,
           f"noul {keys}, score {skeys}, long state cut to {len(lids)} tokens with its 4 options whole")
    rng = random.Random(0)
    for name, p in paths.items():
        m = SkylarDecider.from_decoder(p).eval()
        exs = examples(m, tok, [case(rng)])
        two = [e for e in exs if len(e[1]) == 2][0]
        five = [e for e in exs if len(e[1]) == 5][0]
        with torch.no_grad():
            alone = m(*batch(m, [two])[:4])["logits"][0, :2]
            mixed = m(*batch(m, [two, five])[:4])["logits"][0, :2]
            pr = m.probabilities(m(*batch(m, [two, five])[:4])["logits"])
        diff = (alone - mixed).abs().max().item()
        report(f"D3 padding of options and tokens, {name}", diff < 1e-4, f"max|dlogit| {diff:.1e}")
        sums_ok = torch.allclose(pr.sum(-1), torch.ones(2), atol=1e-5) and pr[0, 2:].abs().max().item() < 1e-6
        report(f"D4 probabilities on the real options, {name}", sums_ok,
               f"sums {pr.sum(-1).tolist()}, padding max {pr[0, 2:].abs().max().item():.1e}")


def d5(paths, tok):
    rng = random.Random(1)
    train = [case(rng) for _ in range(600)]
    test = [case(rng) for _ in range(150)]
    for name, p in paths.items():
        torch.manual_seed(0)
        m = SkylarDecider.from_decoder(p).train()
        tr, te = examples(m, tok, train), examples(m, tok, test)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)            # 2e-3 diverges on the hybrid from scratch
        r = random.Random(2)
        for step in range(300):
            out = m(*batch(m, r.sample(tr, 32))[:5])
            out["loss"].backward()
            opt.step()
            opt.zero_grad()
        m.eval()
        accs = {}
        with torch.no_grad():
            for ty, kind in enumerate(("choice", "score", "noul")):
                sub = [e for e in te if e[0][2] == ty]
                b = batch(m, sub)
                pred = m(*b[:4])["logits"].argmax(-1)
                accs[kind] = (pred == b[5]).float().mean().item()
        ok = accs["choice"] > 0.9 and accs["noul"] > 0.9 and accs["score"] > 0.6
        report(f"D5 learns the synthetic task, {name}", ok,
               ", ".join(f"{k} {v:.2f}" for k, v in accs.items()) + f" (chance 0.25 / 0.20 / 0.50)")


def d6():
    g = torch.Generator().manual_seed(0)
    n, k = 4000, 4
    true_logits = torch.randn(n, k, generator=g)
    labels = torch.multinomial(true_logits.softmax(-1), 1, generator=g).squeeze(-1)
    targets = torch.nn.functional.one_hot(labels, k).float()
    over = true_logits * 3.0                                                 # three times too sure
    t = SkylarDecider.fit_temperature(over, targets)
    before = decision_metrics(over.softmax(-1), targets, labels)
    after = decision_metrics((over / t).softmax(-1), targets, labels)
    ok = 2.5 < t < 3.5 and after["log_loss"] < before["log_loss"] and after["ece"] < before["ece"]
    report("D6 one temperature on over-confident logits", ok,
           f"T {t:.2f} (true 3), log loss {before['log_loss']:.3f} -> {after['log_loss']:.3f}, ECE "
           f"{before['ece']:.3f} -> {after['ece']:.3f}")


def d7(paths, tok):
    m = SkylarDecider.from_decoder(paths["hybrid"]).eval()
    m.config.decision_temperature = 1.7
    state = {"color": "blue", "size": 4, "shape": "cone"}
    a = m.decide(tok, state, QUESTIONS)
    pc = list(a["color"]["probabilities"].values())
    ps = list(a["size"]["probabilities"].values())
    exp_conf = (max(pc) - 0.25) / 0.75
    mean = sum(i * p for i, p in enumerate(ps))
    shape_ok = (a["color"]["choice"] in COLORS and abs(a["color"]["confidence"] - exp_conf) < 1e-9
                and abs(a["size"]["score"] - mean) < 1e-9 and 0 <= a["big"]["noul"] <= 1
                and abs(a["big"]["confidence"] - abs(2 * a["big"]["noul"] - 1)) < 1e-9)
    with tempfile.TemporaryDirectory() as t:
        m.save_pretrained(t)
        back = SkylarDecider.from_pretrained(t).eval()
        b = back.decide(tok, state, QUESTIONS)
    same = back.temperature == 1.7 and all(
        abs(a[q]["probabilities"][k] - b[q]["probabilities"][k]) < 1e-6 for q in a for k in a[q]["probabilities"])
    report("D7 decide(): schema, confidences, save / load", shape_ok and same,
           f"choice {a['color']['choice']} conf {a['color']['confidence']:.3f}, score {a['size']['score']:.2f}, "
           f"noul {a['big']['noul']:.3f}, T after reload {back.temperature}")


def d8(paths, tok):
    if not torch.cuda.is_available():
        print("  [skip] D8 bf16 on the GPU: no CUDA")
        return
    m = SkylarDecider.from_decoder(paths["hybrid"]).cuda().eval()
    b = batch(m, examples(m, tok, [case(random.Random(5)) for _ in range(4)]))
    args = [x.cuda() for x in b[:4]]
    with torch.no_grad():
        ref = m.probabilities(m(*args)["logits"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            low = m.probabilities(m(*args)["logits"])
    diff = (ref - low).abs().max().item()
    report("D8 bf16 on the GPU (fla kernels)", diff < 0.05, f"max|dp| {diff:.1e}")


def main():
    print("decider gates (models/decider.py)")
    tok = tokenizer()
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_decoders(tmp)
        d1_d4(paths, tok)
        d5(paths, tok)
        d6()
        d7(paths, tok)
        d8(paths, tok)
    n, ok = len(RESULTS), sum(RESULTS)
    print(f"  {ok}/{n} decider gates passed")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
