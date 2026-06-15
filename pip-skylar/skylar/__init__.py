"""Skylar — local, sovereign, from-scratch LLMs (chat, embeddings, and a COBOL specialist)."""
from .core import Skylar, load, DEFAULT_MODEL, COBOL_SYSTEM
from .config import NanoTransformerConfig
from .decoder import NanoTransformer
from .embed import SkylarEmbed, load_embedder, is_embedder, DEFAULT_EMBED_MODEL
from .embedder import SkylarEmbedder
from .chatml import encode_chatml

__version__ = "0.2.3"
__all__ = ["Skylar", "load", "NanoTransformer", "NanoTransformerConfig",
           "SkylarEmbed", "load_embedder", "is_embedder", "SkylarEmbedder",
           "encode_chatml", "DEFAULT_MODEL", "DEFAULT_EMBED_MODEL", "COBOL_SYSTEM",
           "__version__"]


def _register_auto():
    """Make AutoConfig/AutoModelForCausalLM aware of the custom arch, so
    `AutoModelForCausalLM.from_pretrained(repo)` works after `import skylar`.
    Best-effort: never break import if transformers internals change."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        AutoConfig.register("nano-transformer", NanoTransformerConfig)
        AutoModelForCausalLM.register(NanoTransformerConfig, NanoTransformer)
    except Exception:
        pass


_register_auto()
