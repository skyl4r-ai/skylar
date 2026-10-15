"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

SkylarDecider: typed, calibrated decisions on the decoder's trunk (beta, 11/10/2026).

A "System One" model (TypeSafe AI's Jev; the open Laya, arXiv 2609.33843): it does not generate text. It reads a state
(text or JSON) and answers typed questions with a probability distribution over fixed options:

    choice  one option out of 2-255                      -> probabilities, the argmax, a confidence
    score   a position on an ordered rubric of 2-10 levels -> probabilities, the expected level, a confidence
    noul    a statement is true or false                   -> P(true)

Questions and answers use the schema of Jev's API and of the open `typed-decisions` dataset: `{"type", "instructions",
"criteria": {key: description}}`, the gold as `{"probabilities": {key: p}, "label": key}`.

How it reads: one sequence per question, in ChatML blocks the pretrained model has seen,

    <|im_start|>state\n{state}<|im_end|><|im_start|>question\n[{type}] {instructions}<|im_end|>
    <|im_start|>option\n{key}: {description}<|im_end|>   ... one block per option

and the decision reads the trunk's state at each option's closing <|im_end|>: on a causal hybrid that token has read
the state, the question and the option itself (Laya, a bidirectional encoder, puts the options first and reads a
[MASK] in front of each). A small head then lets the options see each other (a type embedding plus 2 bidirectional
layers over the option vectors, as in Laya), and one scorer shared by all types gives one logit per option: a masked
softmax over the question's options handles any number of them.

Calibration is a single temperature fitted after training on cases kept out of it (`fit_temperature`): on Laya one
temperature takes the held-out ECE from 0.204 to 0.037, per-type or per-option-count tables add nothing, and isotonic
regression overfits (arXiv 2609.33843, Table 4).
"""

import json
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder_base import SkylarEncoderBase

logger = logging.getLogger(__name__)

TYPES = ("choice", "score", "noul")
NEG = -1e4                       # the logit of a padding option: finite, so a soft target of 0 gives 0, not NaN


def option_list(question):
    """[(key, description)] in the order the model reads them: the rubric levels sorted by their number."""
    crit = question.get("criteria")
    if crit is None and question["type"] == "noul":          # optional for a noul, as in Jev's API
        crit = {"false": "the statement is false", "true": "the statement is true"}
    if isinstance(crit, dict):
        items = list(crit.items())
    elif crit and all(isinstance(c, str) for c in crit):     # a rubric as a list of levels: keys "0", "1", ...
        items = [(str(i), c) for i, c in enumerate(crit)]
    else:
        items = [(str(k), str(v)) for k, v in crit]
    if question["type"] == "score" and all(str(k).lstrip("-").isdigit() for k, _ in items):
        items.sort(key=lambda kv: int(kv[0]))
    if question["type"] == "noul":
        keys = [k for k, _ in items]
        if sorted(keys) != ["false", "true"]:
            raise ValueError(f"a noul question has the criteria 'false' and 'true', got {keys}")
        items.sort(key=lambda kv: kv[0] != "false")                     # false first: P(true) is p[1]
    return [(str(k), str(v)) for k, v in items]


class DecisionHead(nn.Module):
    """Option vectors (B, K, D) -> one logit per option (B, K)."""

    def __init__(self, d_model, n_heads, n_layers=2, dropout=0.0):
        super().__init__()
        self.type_emb = nn.Embedding(len(TYPES), d_model)
        self.layers = None
        if n_layers:
            layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.layers = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.scorer = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, options, option_mask, types):
        h = options + self.type_emb(types)[:, None, :]
        if self.layers is not None:
            h = self.layers(h, src_key_padding_mask=~option_mask)
        return self.scorer(self.norm(h)).squeeze(-1).masked_fill(~option_mask, NEG)


class SkylarDecider(SkylarEncoderBase):
    """(state, typed question) -> a calibrated distribution over the question's options."""

    _head_prefixes = ("decision.",)

    def __init__(self, config) -> None:
        super().__init__(config)
        d = config.d_model
        heads = config.n_heads if d % config.n_heads == 0 else 1
        self.decision = DecisionHead(d, heads, int(getattr(config, "decision_head_layers", 2)),
                                     float(getattr(config, "dropout", 0.0)))
        self.post_init()

    @property
    def temperature(self):
        return float(getattr(self.config, "decision_temperature", 1.0) or 1.0)

    # ── input ────────────────────────────────────────────────────────────────────────────────────
    def encode(self, tokenizer, state, question, max_len=1024, option_max=48, question_max=192):
        """One question -> (ids, positions of the options' closing <|im_end|>, type index, option keys).
        The question and the options are kept whole (each option capped at `option_max` tokens); the state gets the
        room that is left and, when it does not fit, keeps its beginning."""
        im_s, im_e = tokenizer.token_to_id("<|im_start|>"), tokenizer.token_to_id("<|im_end|>")
        if im_s is None or im_e is None:
            raise ValueError("the tokenizer has no <|im_start|> / <|im_end|>")

        def enc(text):
            return tokenizer.encode(text, add_special_tokens=False).ids

        options = option_list(question)
        q_ids = [im_s] + enc(f"question\n[{question['type']}] {question['instructions']}")[:question_max] + [im_e]
        o_ids = [[im_s] + enc(f"option\n{k}: {v}")[:option_max] + [im_e] for k, v in options]
        state_txt = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        room = max_len - len(q_ids) - sum(map(len, o_ids)) - 2
        if room < 8:
            raise ValueError(f"max_len {max_len}: the question and its {len(options)} options leave no room for the state")
        ids = [im_s] + enc("state\n" + state_txt)[:room] + [im_e] + q_ids
        marks = []
        for o in o_ids:
            ids += o
            marks.append(len(ids) - 1)
        return ids, marks, TYPES.index(question["type"]), [k for k, _ in options]

    @staticmethod
    def collate(items, pad_id=0):
        """[(ids, marks, type, ...)] -> right-padded input_ids, attention_mask, marker positions (-1 = no option), types."""
        T = max(len(it[0]) for it in items)
        K = max(len(it[1]) for it in items)
        ids = torch.full((len(items), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(items), T), dtype=torch.long)
        pos = torch.full((len(items), K), -1, dtype=torch.long)
        for i, it in enumerate(items):
            ids[i, :len(it[0])] = torch.tensor(it[0])
            att[i, :len(it[0])] = 1
            pos[i, :len(it[1])] = torch.tensor(it[1])
        types = torch.tensor([it[2] for it in items], dtype=torch.long)
        return ids, att, pos, types

    # ── forward ──────────────────────────────────────────────────────────────────────────────────
    def forward(self, input_ids, attention_mask, marker_pos, types, targets=None, loss="ce"):
        """Logits (B, K), padding options at NEG. `targets` (B, K): a distribution per question (one-hot or a
        teacher's); the loss is a proper scoring rule: the log score (cross-entropy) or the Brier score."""
        x, _ = self.hidden_states(input_ids, attention_mask)
        mask = marker_pos >= 0
        idx = marker_pos.clamp(min=0)
        vec = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        logits = self.decision(vec.float(), mask, types).float()
        out = {"logits": logits, "loss": None}
        if targets is not None:
            t = targets.float().masked_fill(~mask, 0.0)
            if loss == "brier":
                out["loss"] = (logits.softmax(-1) - t).pow(2).sum(-1).mean()
            else:
                out["loss"] = -(t * logits.log_softmax(-1)).sum(-1).mean()
        return out

    def probabilities(self, logits):
        return (logits / self.temperature).softmax(-1)

    # ── the API: a state and its questions ───────────────────────────────────────────────────────
    @torch.no_grad()
    def decide(self, tokenizer, state, questions, max_len=1024, device=None):
        """`questions` = {id: {"type", "instructions", "criteria": {key: description}}} (Jev's schema). Returns
        {id: {"type", "probabilities": {key: p}, and "choice" | "score" | "noul", "confidence"}}. The confidences
        follow Jev's documented formulas: choice (p_max - 1/n)/(1 - 1/n); score 1 - sum_i p_i|i - m| / MAD of the
        uniform rubric (floored at 0); noul |2p - 1|."""
        dev = device or next(self.parameters()).device
        qids = list(questions)
        enc = [self.encode(tokenizer, state, questions[q], max_len) for q in qids]
        ids, att, pos, types = self.collate(enc, pad_id=tokenizer.token_to_id("<pad>") or 0)
        probs = self.probabilities(self(ids.to(dev), att.to(dev), pos.to(dev), types.to(dev))["logits"]).cpu()
        out = {}
        for i, q in enumerate(qids):
            keys, kind = enc[i][3], questions[q]["type"]
            p = probs[i, :len(keys)].tolist()
            r = {"type": kind, "probabilities": dict(zip(keys, p))}
            n = len(p)
            if kind == "choice":
                r["choice"] = keys[max(range(n), key=p.__getitem__)]
                r["confidence"] = (max(p) - 1 / n) / (1 - 1 / n)
            elif kind == "score":
                m = sum(i * pi for i, pi in enumerate(p))
                mad_unif = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
                r["score"] = m
                r["confidence"] = max(0.0, 1 - sum(pi * abs(i - m) for i, pi in enumerate(p)) / mad_unif)
            else:
                r["noul"] = p[1]
                r["confidence"] = abs(2 * p[1] - 1)
            out[q] = r
        return out

    # ── calibration ──────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def fit_temperature(logits, targets, steps=200):
        """The single temperature that minimises the log loss of `logits` (N, K) against `targets` (N, K) on data
        the model was NOT trained on. Returns T (a float)."""
        logits, targets = logits.detach().float(), targets.detach().float()
        log_t = torch.zeros((), requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            nll = -(targets * (logits / log_t.exp()).log_softmax(-1)).sum(-1).mean()
            nll.backward()
            return nll

        opt.step(closure)
        t = log_t.detach().exp().item()
        return t if math.isfinite(t) else 1.0

    # ── loading ──────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_decoder(cls, decoder_path, head_layers=2, pool_strategy=None, encoder_attention="auto"):
        """A pretrained decoder's trunk + a fresh decision head."""
        return cls._from_decoder(decoder_path, decision_head_layers=head_layers, decision_temperature=1.0,
                                 pool_strategy=pool_strategy, encoder_attention=encoder_attention)


@torch.no_grad()
def decision_metrics(probs, targets, labels, n_bins=15):
    """Accuracy, log loss, Brier, ECE (equal-width bins on the top probability), the signed gap (mean confidence -
    accuracy: < 0 = under-confident, the sign decides which way a confidence gate fails, arXiv 2609.33843 §4.2) and
    the area under the risk-coverage curve (AURC: how well the confidence orders right before wrong; lower is
    better). `probs`/`targets` (N, K) with padding options at 0, `labels` (N,) the gold option index."""
    probs, targets = probs.float(), targets.float()
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    n = len(labels)
    ece = torch.zeros(())
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            ece += sel.float().mean() * (conf[sel].mean() - correct[sel].mean()).abs()
    order = conf.argsort(descending=True)
    risk = (1 - correct[order]).cumsum(0) / torch.arange(1, n + 1)
    return {"n": n, "accuracy": correct.mean().item(),
            "log_loss": -(targets * probs.clamp_min(1e-12).log()).sum(-1).mean().item(),
            "brier": (probs - targets).pow(2).sum(-1).mean().item(),
            "ece": ece.item(), "signed_gap": (conf.mean() - correct.mean()).item(), "aurc": risk.mean().item()}
