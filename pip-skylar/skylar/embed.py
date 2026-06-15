"""Skylar embeddings — high-level API for SkylarEmbedder models.

    import skylar
    e = skylar.load_embedder("Sophia-AI/Skylar-236M-Embed")
    vecs = e.encode(["la banca centrale", "il tasso di interesse"])   # L2-normalized
    sim  = e.similarity("mutuo a tasso fisso", "prestito immobiliare")
"""
import torch
from tokenizers import Tokenizer

from .embedder import SkylarEmbedder
from .core import _resolve_tokenizer

DEFAULT_EMBED_MODEL = "Sophia-AI/Skylar-236M-Embed"


class SkylarEmbed:
    """A loaded SkylarEmbedder + tokenizer with a friendly encode API."""

    def __init__(self, model, tokenizer, device, max_len=512):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.max_len = max_len

    @classmethod
    def load(cls, model=DEFAULT_EMBED_MODEL, device=None, max_len=512):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = SkylarEmbedder.from_pretrained(model).to(device).eval()
        tok = Tokenizer.from_file(_resolve_tokenizer(model))
        return cls(net, tok, device, max_len=max_len)

    def _batch(self, texts):
        enc = [self.tok.encode(t, add_special_tokens=False).ids[:self.max_len] for t in texts]
        maxL = max((len(e) for e in enc), default=1)
        ids = torch.zeros(len(enc), maxL, dtype=torch.long)
        mask = torch.zeros(len(enc), maxL, dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            mask[i, :len(e)] = 1
        return ids.to(self.device), mask.to(self.device)

    def encode(self, texts, batch_size=32):
        """Encode str | list[str] -> list of L2-normalized vectors (list[list[float]])."""
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        ids, mask = self._batch(texts)
        with torch.no_grad():
            embs = self.model.encode(ids, attention_mask=mask, batch_size=batch_size)
        out = embs.float().cpu().tolist()
        return out[0] if single else out

    def similarity(self, a, b):
        """Cosine similarity between two texts (embeddings are L2-normalized -> dot)."""
        va, vb = self.encode([a, b])
        return float(sum(x * y for x, y in zip(va, vb)))

    def rank(self, query, docs):
        """Return docs sorted by similarity to query: list of (doc, score) desc."""
        qv = self.encode(query)
        dvs = self.encode(list(docs))
        scored = [(d, float(sum(x * y for x, y in zip(qv, dv)))) for d, dv in zip(docs, dvs)]
        return sorted(scored, key=lambda t: t[1], reverse=True)


def load_embedder(model=DEFAULT_EMBED_MODEL, device=None, max_len=512):
    """Convenience: skylar.load_embedder(...) -> SkylarEmbed instance."""
    return SkylarEmbed.load(model, device=device, max_len=max_len)


def is_embedder(model):
    """Best-effort: does this repo/dir hold a SkylarEmbedder? (reads config architectures)."""
    try:
        from transformers import PretrainedConfig
        cfg = PretrainedConfig.from_pretrained(model)
        arch = getattr(cfg, "architectures", None) or []
        return any("Embedder" in a for a in arch)
    except Exception:
        return False
