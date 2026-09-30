"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | 2026
=================================================================

Skylar 2 decoder-only Transformer — PyTorch + HuggingFace: `Skylar2ForCausalLM` (text only, hence
`ForCausalLM`: it predicts the next token; `ForConditionalGeneration` is for encoder-decoder and
multimodal models). With every Skylar 2 option off it is the v1 decoder, bit for bit.

`NanoTransformer` is the name of the checkpoints saved before 30/09/2026 (Skylar-236M, Skylar-980M-Cobol):
same code, kept so they load unchanged. Both names are registered with transformers, so
`AutoModelForCausalLM.from_pretrained(path)` opens either, once this module is imported.

Every component is written explicitly (no nn.TransformerDecoder),
so you can see, modify, and learn from every line.

Supports:
  - FlexAttention (PyTorch 2.5+) for zero-overhead document masking
    during packed sequence training — same technique used by LLaMA,
    Qwen, Mistral, and OLMo at scale.
  - µP (Maximal Update Parameterization) for HP transfer from small
    proxy models to large targets — used by Cerebras, EleutherAI,
    DeepSeek. Tune HPs on a 50M proxy, transfer to 4B.
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, __version__ as _tf_version

from models.config import NanoTransformerConfig, Skylar2Config
from models.layers.block import TransformerBlock
from models.layers.norm import make_norm
from models.layers.attention import (
    HAS_FLEX_ATTENTION,
    build_cu_seqlens,
    create_document_block_mask,
    make_packing_mask,
)
from models.layers.attn_res import AttnResMixer, DepthState
from models.layers.kv_cache import validate_kv_cache
from models.mtp import MTPModule

logger = logging.getLogger(__name__)


class Skylar2ForCausalLM(PreTrainedModel):
    """
    Skylar 2 decoder-only Transformer (docs/PAPER_V2.md; with the options off, the v1 decoder of docs/PAPER.md).

    Modern architecture choices:
      - RMSNorm (instead of LayerNorm)
      - RoPE (instead of sinusoidal/learned positional embeddings)
      - SwiGLU FFN (instead of GELU)
      - GQA — Grouped Query Attention (fewer KV heads → less memory)
      - KV-Cache for fast generation
      - Flash Attention (via PyTorch scaled_dot_product_attention)
      - FlexAttention document masking for packed training (zero overhead)
      - Optional weight tying
      - µP for HP transfer across model widths

    Document masking (packed training):
      Pass document_ids to forward() and the model automatically:
      - Uses FlexAttention block_mask if available (O(T) memory, zero overhead)
      - Falls back to dense attention_mask if not (O(T²) memory)
      - Does nothing during generation (kv_cache mode)

    µP (Maximal Update Parameterization):
      Set config.mup_base_d_model to the proxy model width. Then:
      - Hidden weight init scales as 0.02 / √(width_mult)
      - Attention logits scale as 1/d_head (not 1/√d_head)
      - LM head output scales by 1/width_mult
      - Use mup_param_groups() for per-parameter LR scaling in optimizer
      Net effect: HPs tuned on proxy transfer to any target width.

    HuggingFace compatible:
      - model.save_pretrained("path")
      - Skylar2ForCausalLM.from_pretrained("path")   (also opens checkpoints saved as NanoTransformer)
    """

    config_class = Skylar2Config
    supports_gradient_checkpointing = True
    _tied_weights_keys = (
        {"lm_head.weight": "token_emb.weight"}
        if int(_tf_version.split(".")[0]) >= 5
        else ["lm_head.weight"]
    )

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # layer_idx: ogni blocco deve sapere dove si trova, o non può sapere se è
        # ricorrente o full-attention (config.layer_types) né indicizzare la cache.
        self.blocks = nn.ModuleList([TransformerBlock(config, layer_idx=i)
                                     for i in range(config.n_layers)])

        # AttnRes ha 2L+1 punti di applicazione: due per blocco più questo, che
        # sceglie da quale profondità legge la testa di output.
        self.attn_res = bool(getattr(config, "attn_res", False))
        self.attn_res_mode = getattr(config, "attn_res_mode", "block")
        self.attn_res_block_size = getattr(config, "attn_res_block_size", 8)
        if self.attn_res:
            window = None
            if self.attn_res_mode == "window":
                w = getattr(config, "attn_res_block", None)
                window = None if not w else 2 * w + 1
            self.res_final = AttnResMixer(config.d_model, window=window,
                                          rms_plus_eps=window is not None)
        self.ln_f = make_norm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # MTP: teste ausiliarie che predicono piu' avanti di un passo. Si SCARTANO a
        # fine pretrain, quindi il costo deployato e' zero. Spente di default.
        self.mtp = MTPModule(config) if getattr(config, "mtp_layers", 0) else None

        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        # µP: output logit scaling
        # Divides lm_head output by width_mult so that logit magnitude
        # stays constant as model width grows. When µP is off, this is 1.0.
        self._mup_output_alpha = 1.0 / config.mup_width_mult

        # ── Weight initialization ── v4|v5
        self.post_init()

        # Residual projection scaled by depth (GPT-2 style)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        if config.mup_base_d_model is not None:
            residual_std /= math.sqrt(config.mup_width_mult)
        for pn, p in self.named_parameters():
            if pn.endswith("W_o.weight") or pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        """
        Weight initialization with µP scaling.

        SP (standard): all weights init to N(0, 0.02)
        µP:
          - Embedding:       N(0, 0.02)  — unchanged
          - Hidden weights:  N(0, 0.02 / √width_mult)  — scales down with width
          - 1D params:       unchanged (norms, biases)
        """
        std = 0.02

        if isinstance(module, nn.Linear):

            if self.config.tie_weights and module is self.lm_head:
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                return

            if self.config.mup_base_d_model is not None:
                std /= math.sqrt(self.config.mup_width_mult)
            nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Embedding init unchanged in µP — it's an "input" parameter
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.token_emb

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, **kwargs):
        """Ensure lm_head shares weights with token_emb after loading."""
        super().tie_weights(**kwargs)
        if self.config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def mup_param_groups(self, base_lr, weight_decay):
        """
        Build optimizer parameter groups with µP per-parameter LR scaling.

        Returns standard groups when µP is disabled.

        µP rules:
          - Embedding params:       lr = base_lr, wd = 0
          - 1D params (norms):      lr = base_lr, wd = 0
          - Hidden 2D weights:      lr = base_lr / width_mult, wd = weight_decay

        With tied weights, lm_head.weight IS token_emb.weight, so it
        naturally gets the embedding LR — correct for µP.
        """
        wm = self.config.mup_width_mult

        embedding_params = []
        decay_params = []
        nodecay_params = []

        embedding_ids = {id(self.token_emb.weight)}
        if self.config.tie_weights:
            embedding_ids.add(id(self.lm_head.weight))

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in embedding_ids:
                embedding_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        groups = [
            {"params": embedding_params, "lr": base_lr, "weight_decay": 0.0,
             "name": "embedding"},
            {"params": decay_params, "lr": base_lr / wm, "weight_decay": weight_decay,
             "name": "hidden"},
            {"params": nodecay_params, "lr": base_lr, "weight_decay": 0.0,
             "name": "nodecay"},
        ]

        return [g for g in groups if g["params"]]

    @property
    def _has_recurrent_layers(self):
        return any(getattr(b, "layer_type", "attention") == "kda" for b in self.blocks)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Opens checkpoints of both names. One saved before 30/09/2026 (model_type "nano-transformer")
        goes through the legacy classes — same code — so transformers does not warn about a type mismatch."""
        if cls is Skylar2ForCausalLM and "config" not in kwargs:
            try:
                saved, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path)
            except Exception:
                saved = {}
            if saved.get("model_type") == NanoTransformerConfig.model_type:
                return NanoTransformer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def _validate_kv_cache(self, kv_cache, batch_size, x_device):
        # La validazione conosce solo la tupla (k, v) rank-4 dell'attention. In un
        # modello ibrido la cache e' ETEROGENEA: i layer KDA portano
        # (stato_ricorrente, stati_conv), di forma diversa. Validarli con la regola
        # dell'attention darebbe un errore su una cache perfettamente valida.
        if self._has_recurrent_layers:
            if len(kv_cache) != len(self.blocks):
                raise ValueError(f"cache con {len(kv_cache)} voci per {len(self.blocks)} layer")
            attn_only = [(c, b) for c, b in zip(kv_cache, self.blocks)
                         if getattr(b, "layer_type", "attention") != "kda"]
            if attn_only:
                validate_kv_cache([c for c, _ in attn_only], [b for _, b in attn_only],
                                  batch_size, x_device)
            return
        validate_kv_cache(kv_cache, self.blocks, batch_size, x_device)

    def forward(self, input_ids, labels=None, kv_cache=None, document_ids=None,
                attention_mask=None, use_cache=False, return_hidden=False):
        """
        Args:
            input_ids:      (B, T) token indices
            labels:         (B, T) target token indices (optional, for training)
            kv_cache:       list of (k, v) tuples per layer (for generation)
            document_ids:   (B, T) document IDs for packed training (optional).
                            Enables automatic document masking via FlexAttention.
            attention_mask: (B, 1, T, T) dense additive mask (backward compat).
            use_cache: Bool if use cache.
            return_hidden:  also return 'hidden', the (B, T, d_model) state the head reads (after the
                            final norm): what an embedder pools.

        Returns:
            dict with 'logits', 'loss' (if labels), 'kv_cache' (and 'hidden' if return_hidden)
        """
        B, T = input_ids.shape
        # F3: attention_mask, if passed by a caller, MUST be the 4D additive form (B,1,T,T).
        # A 2D HF-style padding mask (B,T) would silently disable causality and act as a bias.
        if attention_mask is not None and attention_mask.dim() != 4:
            raise ValueError(
                f"Skylar2ForCausalLM.forward expects a 4D additive attention_mask (B,1,T,T), got "
                f"{attention_mask.dim()}D. Build it as (1-mask)[:,None,None,:]*min_val, or pass "
                f"document_ids for packed training.")
        x = self.drop(self.token_emb(input_ids))

        if kv_cache is not None:
            self._validate_kv_cache(kv_cache, batch_size=B, x_device=x.device)

        # ── Build attention mask from document_ids ──
        block_mask = None
        if document_ids is not None and kv_cache is None:
            if HAS_FLEX_ATTENTION:
                block_mask = create_document_block_mask(
                    document_ids, self.config.n_heads, device=input_ids.device,
                )
            else:
                if input_ids.is_cuda and torch.is_autocast_enabled():
                    # PyTorch 2.x compat
                    if hasattr(torch, "get_autocast_dtype"):
                        mask_dtype = torch.get_autocast_dtype("cuda")
                    else:
                        mask_dtype = torch.get_autocast_gpu_dtype()
                else:
                    mask_dtype = x.dtype

                attention_mask = make_packing_mask(document_ids, dtype=mask_dtype)

        collect_cache = bool(use_cache or kv_cache is not None)
        new_cache = [] if collect_cache else None

        # ── cu_seqlens per i layer ricorrenti ──
        # I layer KDA non vedono la block_mask: per loro un confine di documento non
        # è una maschera, è un punto in cui lo STATO va azzerato. Senza questo, lo
        # stato scorre da un programma COBOL al successivo — e la loss non lo mostra.
        cu_seqlens = None
        if document_ids is not None and kv_cache is None and self._has_recurrent_layers:
            cu_seqlens = build_cu_seqlens(document_ids)

        depth = (DepthState(x, self.attn_res_mode, self.attn_res_block_size)
                 if self.attn_res else None)

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            if self.gradient_checkpointing and self.training and kv_cache is None:
                # ⚠️ Gli argomenti erano passati PER POSIZIONE: aggiungendo parametri
                # alla firma del blocco si finiva a passare `cu_seqlens` dove il
                # blocco si aspetta `use_cache`, senza nessun errore. Con la closure
                # il legame è per nome e il problema non si ripresenta mai più.
                #
                # ⚠️ AttnRes: il blocco FA AVANZARE lo stato di profondità. Nel backward il
                # checkpoint riesegue il blocco: se riceve lo stato vero, a quel punto è già
                # avanzato di tutti i layer successivi e il ricalcolo legge sorgenti
                # sbagliate. Il blocco lavora quindi su una FOTOGRAFIA presa ora, e i
                # tensori nuovi (sorgenti chiuse, somma parziale) escono come output.
                def _run(inp, _b=block, _c=layer_cache,
                         _f=depth.freeze() if depth is not None else None):
                    st = None if _f is None else DepthState.thaw(
                        _f, self.attn_res_mode, self.attn_res_block_size)
                    out, cache = _b(inp, kv_cache=_c, block_mask=block_mask,
                                    attention_mask=attention_mask, use_cache=False,
                                    cu_seqlens=cu_seqlens, depth=st)
                    if st is None:
                        return out, cache
                    return (out, cache, *st.new_tensors(len(_f[0])))
                res = torch.utils.checkpoint.checkpoint(_run, x, use_reentrant=False)
                x, cache_i = res[0], res[1]
                if depth is not None:
                    depth.adopt(list(res[2:]), pushes=2)
            else:
                x, cache_i = block(
                    x,
                    kv_cache=layer_cache,
                    block_mask=block_mask,
                    attention_mask=attention_mask,
                    use_cache=collect_cache,
                    cu_seqlens=cu_seqlens,
                    depth=depth,
                )

            if collect_cache:
                new_cache.append(cache_i)

        if self.attn_res and self.mtp is None:
            # La lettura finale sceglie da quale profondità legge la testa, con ln_f
            # piegata nel kernel fuso (e, se gated, il suo gate applicato dopo).
            x_pre = None
            x = self.res_final.mix(depth.sources(), self.ln_f)
        else:
            if self.attn_res:
                x = self.res_final.mix(depth.sources())
            x_pre = x             # serve alle teste MTP, che normalizzano per conto loro
            x = self.ln_f(x)
        logits = self.lm_head(x)

        # µP: scale output logits to keep magnitude stable across widths
        if self._mup_output_alpha != 1.0:
            logits = logits * self._mup_output_alpha

        loss = None
        loss_parts = {}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            if self.mtp is not None and len(self.mtp):
                # `x_pre` e' lo stato PRIMA di ln_f: la testa MTP applica la sua
                # normalizzazione, come nel trunk.
                loss_parts["ce"] = loss.detach()
                aux, detail = self.mtp.loss(x_pre, self.token_emb, input_ids, labels,
                                            self.ln_f, self.lm_head)
                if aux is not None:
                    loss = loss + aux
                    loss_parts.update(detail)

        out = {"logits": logits, "loss": loss, "kv_cache": new_cache, "loss_parts": loss_parts}
        if return_hidden:
            out["hidden"] = x
        return out

    def count_params(self, non_embedding=False):
        """Count parameters (optionally excluding embedding)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_emb.weight.numel()
        return n

    # ─────────────────────────────────────────────────────────────
    # Static generation
    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200, temperature=0.8, top_k=50,
                 top_p=0.9, repetition_penalty=1.0, eos_token_id=None, use_cache=True):
        """
        Autoregressive generation with KV-cache, top-k, top-p, temperature,
        and repetition penalty.

        Args:
            input_ids:          (1, T) starting tokens
            max_new_tokens:     how many tokens to generate
            temperature:        sampling temperature (lower = more deterministic)
            top_k:              keep only top-k logits
            top_p:              nucleus sampling threshold
            repetition_penalty: penalize already-generated tokens (1.0 = no penalty,
                                >1.0 = discourage repeats, 1.1-1.3 recommended)
            eos_token_id:       stop generation at this token (int or list of ints)
            use_cache:          use KV-cache for faster generation

        Returns:
            (1, T + generated) generated token ids
        """
        assert input_ids.shape[0] == 1, f"generate() supports batch_size=1, got {input_ids.shape[0]}"

        if eos_token_id is None:
            eos_set = set()
        elif isinstance(eos_token_id, int):
            eos_set = {eos_token_id}
        else:
            eos_set = set(eos_token_id)

        max_seq_len = self.config.max_seq_len
        prompt_len = input_ids.shape[1]

        if prompt_len > max_seq_len:
            logger.warning(
                f"Prompt length ({prompt_len}) exceeds max_seq_len ({max_seq_len}). "
                f"Truncating to last {max_seq_len} tokens."
            )
            input_ids = input_ids[:, -max_seq_len:]
            prompt_len = max_seq_len

        max_gen = min(max_new_tokens, max_seq_len - prompt_len)
        if max_gen < max_new_tokens:
            logger.warning(
                f"Clamping max_new_tokens from {max_new_tokens} to {max_gen} "
                f"(max_seq_len={max_seq_len}, prompt={prompt_len})"
            )
        if max_gen <= 0:
            logger.warning("No room to generate — prompt fills entire context window.")
            return input_ids

        was_training = self.training
        self.eval()
        kv_cache = None

        for _ in range(max_gen):
            if use_cache and kv_cache is not None:
                idx_input = input_ids[:, -1:]
            else:
                idx_input = input_ids[:, -max_seq_len:]
                kv_cache = None

            out = self.forward(idx_input, kv_cache=kv_cache, use_cache=use_cache)
            logits = out["logits"][:, -1, :]

            if use_cache:
                kv_cache = out["kv_cache"]

            if repetition_penalty != 1.0:
                seen_tokens = input_ids[0].unique()
                score = logits[0, seen_tokens]
                logits[0, seen_tokens] = torch.where(
                    score > 0,
                    score / repetition_penalty,
                    score * repetition_penalty,
                )

            # DOPO (fix)
            if temperature <= 0:
                # Greedy: prendi il token col logit più alto, zero casualità
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = -float("inf")

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[mask] = -float("inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() in eos_set:
                break

        if was_training:
            self.train()
        return input_ids

    # ─────────────────────────────────────────────────────────────
    # Streaming generation
    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_streaming(self, input_ids, max_new_tokens=200, temperature=0.8, top_k=50,
                           top_p=0.9, repetition_penalty=1.0, eos_token_id=None, use_cache=True):
        """
        Autoregressive generation that yields one token ID at a time.

        Same logic as generate() — KV-cache, top-k, top-p, temperature,
        repetition penalty — but yields instead of accumulating.

        Args:
            Same as generate().

        Yields:
            int: token ID at each step. Stops after eos or max_new_tokens.
        """
        assert input_ids.shape[0] == 1, f"generate_streaming() supports batch_size=1, got {input_ids.shape[0]}"

        if eos_token_id is None:
            eos_set = set()
        elif isinstance(eos_token_id, int):
            eos_set = {eos_token_id}
        else:
            eos_set = set(eos_token_id)

        max_seq_len = self.config.max_seq_len
        prompt_len = input_ids.shape[1]

        if prompt_len > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]
            prompt_len = max_seq_len

        max_gen = min(max_new_tokens, max_seq_len - prompt_len)
        if max_gen <= 0:
            return

        was_training = self.training
        self.eval()
        kv_cache = None

        for _ in range(max_gen):
            if use_cache and kv_cache is not None:
                idx_input = input_ids[:, -1:]
            else:
                idx_input = input_ids[:, -max_seq_len:]
                kv_cache = None

            out = self.forward(idx_input, kv_cache=kv_cache, use_cache=use_cache)
            logits = out["logits"][:, -1, :]

            if use_cache:
                kv_cache = out["kv_cache"]

            if repetition_penalty != 1.0:
                seen_tokens = input_ids[0].unique()
                score = logits[0, seen_tokens]
                logits[0, seen_tokens] = torch.where(
                    score > 0,
                    score / repetition_penalty,
                    score * repetition_penalty,
                )

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = -float("inf")

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[mask] = -float("inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            token_id = next_token.item()

            yield token_id

            if token_id in eos_set:
                break

        if was_training:
            self.train()


class NanoTransformer(Skylar2ForCausalLM):
    """Name of the checkpoints saved before 30/09/2026 (config.json: model_type "nano-transformer",
    architectures ["NanoTransformer"]): Skylar-236M and Skylar-980M-Cobol. Same code as Skylar2ForCausalLM."""
    config_class = NanoTransformerConfig


def _register_with_transformers():
    """AutoConfig and AutoModelForCausalLM learn both names, so AutoModelForCausalLM.from_pretrained(path)
    opens old and new checkpoints alike after `import models.decoder`."""
    from transformers import AutoConfig, AutoModelForCausalLM
    for cfg, model in ((Skylar2Config, Skylar2ForCausalLM), (NanoTransformerConfig, NanoTransformer)):
        try:
            AutoConfig.register(cfg.model_type, cfg)
            AutoModelForCausalLM.register(cfg, model)
        except ValueError:          # already registered: the module was imported twice, or by the pip package
            pass


_register_with_transformers()
