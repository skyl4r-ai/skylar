"""
Generic rejection-sampling fine-tuning (RFT / STaR) — the cheap, high-value step BETWEEN SFT and
GRPO. Sample N candidate completions per record, KEEP ONLY the ones a PLUGGABLE reward verifies
(reward >= threshold), dedup, and emit them as NEW SFT pairs to re-SFT on. The framework's
"poor man's RL"; a project supplies reward_fn / build_messages / make_pair (e.g. COBOL: compile+run).

Why: captures much of GRPO's gain at a fraction of the complexity/instability, and is the natural
precursor for code because the verifier already exists. The model teaches itself from its OWN
correct solutions -> no teacher API cost.

Callbacks (domain-agnostic core):
  build_messages(record) -> list[{"role","content"}]       # ChatML messages for the sampler
  reward_fn(record, completion_text) -> (float, dict)
  make_pair(record, kept_code) -> dict                     # the SFT record to emit
  extract_code(completion_text) -> str                     # optional; default identity
  dedup_key(code) -> hashable                              # optional; default the code itself

  from training.rft import rft_over_records, LocalGenerator
  gen = LocalGenerator(model_dir)
  pairs, stats = rft_over_records(records, gen, reward_fn, make_pair, build_messages, n=8)
"""
from pathlib import Path


class LocalGenerator:
    """Lazily-loaded local NanoTransformer sampler (batch-1 generate). Framework-native: reuses
    models.decoder + utils.chatML. Takes ChatML `messages` and returns `n` decoded completions."""
    def __init__(self, model_dir, max_new_tokens=900, temperature=0.8, top_k=50):
        self.model_dir, self.max_new_tokens = model_dir, max_new_tokens
        self.temperature, self.top_k = temperature, top_k
        self._loaded = False

    def _load(self):
        import torch
        from models.decoder import NanoTransformer
        from tokenizers import Tokenizer
        from utils.chatML import encode_chatml
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NanoTransformer.from_pretrained(self.model_dir).to(self.dev).eval()
        self.tok = Tokenizer.from_file(str(Path(self.model_dir) / "tokenizer.json"))
        self.enc = encode_chatml
        self.eos = self.tok.token_to_id("<|im_end|>")
        self._loaded = True

    def sample(self, messages, n):
        import torch
        if not self._loaded:
            self._load()
        ids = self.enc(messages, self.tok, add_generation_prompt=True)
        x = torch.tensor([ids], device=self.dev)
        outs = []
        amp = (self.dev == "cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            for _ in range(n):                          # batch-1 generate (NanoTransformer limit)
                out = self.model.generate(x, max_new_tokens=self.max_new_tokens,
                                          temperature=self.temperature, top_k=self.top_k,
                                          eos_token_id=self.eos)
                outs.append(self.tok.decode(out[0].tolist()[len(ids):]))
        return outs


def rft_over_records(records, gen, reward_fn, make_pair, build_messages, *, n=8,
                     keep_per_problem=2, extract_code=None, dedup_key=None, threshold=1.0,
                     log_every=50, log=print):
    """gen.sample(messages, n) -> [raw completion str]. Returns (pairs, stats). Domain-agnostic:
    keeps completions with reward_fn(rec, c)[0] >= threshold, dedups by dedup_key(extract_code(c))."""
    extract_code = extract_code or (lambda c: c)
    dedup_key = dedup_key or (lambda code: code)
    pairs, n_solved = [], 0
    for i, rec in enumerate(records):
        cands = gen.sample(build_messages(rec), n)
        winners, seen = [], set()
        for c in cands:
            if reward_fn(rec, c)[0] < threshold:         # keep only verified-correct
                continue
            code = extract_code(c)
            key = dedup_key(code)
            if key in seen:
                continue
            seen.add(key)
            winners.append(code)
            if len(winners) >= keep_per_problem:
                break
        if winners:
            n_solved += 1
            pairs.extend(make_pair(rec, w) for w in winners)
        if (i + 1) % log_every == 0:
            log(f"  [{i+1}/{len(records)}] solved {n_solved} -> {len(pairs)} pairs")
    return pairs, {"records": len(records), "solved": n_solved, "pairs": len(pairs)}
