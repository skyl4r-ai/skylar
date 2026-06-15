"""Skylar — loader + high-level API.

    import skylar
    m = skylar.load("Sophia-AI/Skylar-236M-Chat")   # auto-download from HF
    print(m.generate("Dove ha sede la Banca d'Italia?",
                     system="Rispondi solo dal contesto fornito."))

Works with a HuggingFace repo id OR a local checkpoint directory.
"""
import os
import torch
from tokenizers import Tokenizer

from .decoder import NanoTransformer
from .chatml import encode_chatml

DEFAULT_MODEL = "Sophia-AI/Skylar-236M-Chat"
COBOL_SYSTEM = "Sei un esperto programmatore COBOL."


def _resolve_tokenizer(model):
    """Return a path to tokenizer.json, from a local dir or an HF repo id."""
    if os.path.isdir(model):
        p = os.path.join(model, "tokenizer.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"tokenizer.json non trovato in {model}")
        return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model, "tokenizer.json")


class Skylar:
    """A loaded Skylar model + tokenizer with a small, friendly generation API."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self._imend = tokenizer.token_to_id("<|im_end|>")

    @classmethod
    def load(cls, model=DEFAULT_MODEL, device=None, dtype=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = NanoTransformer.from_pretrained(model).to(device).eval()
        if dtype is not None:
            net = net.to(dtype)
        tok = Tokenizer.from_file(_resolve_tokenizer(model))
        return cls(net, tok, device)

    # -- internals ---------------------------------------------------------
    def _encode(self, prompt, system):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return encode_chatml(msgs, self.tok, add_generation_prompt=True)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=(self.device == "cuda"))

    # -- public API --------------------------------------------------------
    def generate(self, prompt, system=None, max_new_tokens=512,
                 temperature=0.0, top_k=40, top_p=0.9, repetition_penalty=1.0, seed=None):
        """Return the assistant completion for one user turn (greedy by default).

        system defaults to None (no system message) — pass your own to steer the model."""
        ids = self._encode(prompt, system)
        x = torch.tensor([ids], device=self.device)
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad(), self._autocast():
            out = self.model.generate(
                x, max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                eos_token_id=self._imend)
        return self.tok.decode(out[0].tolist()[len(ids):])

    def complete_cobol(self, stub, entry_point=None, max_new_tokens=900, temperature=0.0):
        """Complete a COBOLEval-style stub into a full, compilable COBOL program."""
        from .cobol import complete_cobol
        return complete_cobol(self, stub, entry_point=entry_point,
                              max_new_tokens=max_new_tokens, temperature=temperature)

    def stream(self, prompt, system=None, max_new_tokens=512,
               temperature=0.0, top_k=40, top_p=0.9, repetition_penalty=1.0, seed=None):
        """Yield the completion incrementally (text deltas). system defaults to None."""
        ids = self._encode(prompt, system)
        x = torch.tensor([ids], device=self.device)
        if seed is not None:
            torch.manual_seed(seed)
        acc, prev = [], ""
        with torch.no_grad(), self._autocast():
            for tid in self.model.generate_streaming(
                    x, max_new_tokens=max_new_tokens, temperature=temperature,
                    top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                    eos_token_id=self._imend):
                acc.append(tid)
                text = self.tok.decode(acc)
                if len(text) > len(prev):
                    yield text[len(prev):]
                    prev = text


def load(model=DEFAULT_MODEL, device=None, dtype=None):
    """Convenience: skylar.load(...) -> Skylar instance."""
    return Skylar.load(model, device=device, dtype=dtype)
