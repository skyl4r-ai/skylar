"""
Prose quality filter for pre-training corpus construction.

Three-layer approach to ensure only clean, flowing prose enters
the training data — no tables, no random numbers, no garbage.

Layer 1 — Semantic Chunking
────────────────────────────
  Split documents at paragraph / section boundaries, evaluate each
  chunk independently.  Only prose-quality chunks survive.

Layer 2 — Heuristic Prose Scoring
──────────────────────────────────
  Fast CPU-only checks per chunk:
    • Numeric density    — ratio of digit chars to alpha chars
    • Number-word ratio  — fraction of tokens that are numbers
    • Table detection    — aligned columns, short uniform lines
    • Sentence structure — does it contain proper sentences?
    • Reference/code     — legal references, codes, footnotes

Layer 3 — GPU Perplexity Scoring (optional)
───────────────────────────────────────────
  Small Italian causal LM on CUDA (e.g. xglm-564M).
  Chunks with perplexity outside [low, high] bounds are dropped:
    • Very low  PPL → repetitive / boilerplate
    • Very high PPL → random / incoherent / numeric noise

Final step — Prose Flattening
─────────────────────────────
  Collapse all whitespace so the document is a single flowing
  stream of clean text.  No random newlines, no column alignment,
  no orphan spaces.  Paragraphs separated by a single newline.
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ProseConfig:
    """Thresholds for prose quality filtering."""

    # ── Chunking ──────────────────────────────────────────────────
    min_chunk_words: int = 15
    min_chunk_chars: int = 80

    # ── Numeric density ───────────────────────────────────────────
    max_digit_ratio: float = 0.15  # digit chars / alpha chars
    max_number_word_ratio: float = 0.25  # number tokens / total tokens

    # ── Table detection ───────────────────────────────────────────
    max_short_line_ratio: float = 0.50  # lines < 40 chars / total lines
    max_tabular_line_ratio: float = 0.30  # lines matching table patterns
    min_mean_line_length: float = 40.0  # mean chars per non-empty line

    # ── Sentence structure ────────────────────────────────────────
    min_sentence_ratio: float = 0.30  # lines ending with .!? / total
    max_orphan_ratio: float = 0.40  # lines with ≤ 3 words / total

    # ── Reference / code noise ────────────────────────────────────
    max_ref_ratio: float = 0.40  # lines with legal refs / total
    max_paren_ratio: float = 0.25  # parenthetical density

    # ── GPU perplexity (optional) ─────────────────────────────────
    perplexity_enabled: bool = False
    perplexity_model: str = "facebook/xglm-564M"
    perplexity_min: float = 5.0  # below = too repetitive
    perplexity_max: float = 1500.0  # above = incoherent noise
    perplexity_max_tokens: int = 256  # tokens per chunk for scoring
    perplexity_device: str = "cuda"


# ──────────────────────────────────────────────────────────────────────
# Compiled patterns
# ──────────────────────────────────────────────────────────────────────

# Section / paragraph boundaries
_SECTION_SPLIT = re.compile(
    r"\n\s*\n"  # double newline
    r"|\n(?=Articol[oi]\s+\d)"  # Italian article headers
    r"|\n(?=Capitolo\s+[IVXLCDM\d])"
    r"|\n(?=Sezione\s+\d)"
    r"|\n(?=CAPO\s+[IVXLCDM])"
    r"|\n(?=Allegat[oi]\s+[IVXLCDMA-Z\d])"
    r"|\n(?=Tabella\s+[IVXLCDM\d])"
    r"|\n(?=TITOLO\s+[IVXLCDM])"
    r"|\n(?=PARTE\s+[IVXLCDM])",
    re.IGNORECASE,
)

# Table-like line patterns
_TABLE_LINE = re.compile(
    r"^\s*"
    r"(?:"
    r"[\d.,]+\s+[\d.,]+"  # columns of numbers
    r"|[A-Z]{2,5}\s+[\d.,]+"  # code + number
    r"|[\d.,]+\s*[%€$£¥]"  # number + unit
    r"|[—\-\|│┃]+\s*$"  # separator lines
    r"|(?:\(\s*\d+\s*\))"  # footnote markers
    r"|(?:\w+\s+){0,2}[\d.,]+\s+[\d.,]+\s+[\d.,]+"  # word num num num
    r")"
    r"\s*$",
    re.MULTILINE,
)

# Standalone number line (quotas, page refs, codes)
_NUMBER_LINE = re.compile(
    r"^\s*[\d.,\s\-%/()]+\s*$",
    re.MULTILINE,
)

# Legal / regulatory reference NOISE — citation-heavy lines, not prose
# Only matches lines that are predominantly citations/codes, not
# prose that happens to mention "articolo 1" inline.
_LEGAL_REF = re.compile(
    r"(?i)"
    r"(?:GU\s*L\s*\d+\s*del\s*\d+"  # GU L 318 del 5.12.2007
    r"|pag\.\s*\d+"  # pag. 1
    r"|regolamento\s*\([A-Z]+\)\s*n\.\s*\d+/\d+"  # regolamento (CE) n. 1308/2013
    r"|decisione\s*\([A-Z]+\)\s*\d{4}/\d+"  # decisione (UE) 2019/848
    r"|[A-Z]{2,5}/[A-Z\d.*]+/[A-Z\d.*]+"  # COI/T.15/NC, WHG/2AC4.
    r"|zona[e]?\s+(?:NAFO|CIEM)\s*\d+"  # zona NAFO 3M
    r"|TAC\s+(?:analitico|non\s+pertinente))"  # TAC analitico
)

# Parenthetical noise — (1), (2), (*), etc.
_PAREN_NOISE = re.compile(r"\(\s*[\d*†‡]+\s*\)")

# Sentence-ending punctuation
_SENTENCE_END = re.compile(r"[.!?;:]\s*$")

# Footnote patterns — ONLY standalone footnote lines at bottom of documents
# NOT numbered paragraphs like "(1) L'accordo internazionale..."
_FOOTNOTE = re.compile(
    r"^\s*\(\s*\d+\s*\)\s+(?="  # (N) followed by...
    r"(?:GU\s|Regolamento|Decisione|Direttiva|Cfr|V\.|Ib|Op)\b"  # reference keywords
    r")",
    re.MULTILINE | re.IGNORECASE,
)

# Lines that are just codes/identifiers
_CODE_LINE = re.compile(
    r"^\s*[A-Z]{2,10}[/\-]\S+\s*$",
    re.MULTILINE,
)


# ══════════════════════════════════════════════════════════════════════
#  CHUNK SCORER
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ChunkScore:
    """Diagnostic scores for a single chunk."""

    text: str
    n_words: int = 0
    digit_ratio: float = 0.0
    number_word_ratio: float = 0.0
    short_line_ratio: float = 0.0
    tabular_line_ratio: float = 0.0
    mean_line_length: float = 0.0
    sentence_ratio: float = 0.0
    orphan_ratio: float = 0.0
    ref_ratio: float = 0.0
    paren_ratio: float = 0.0
    is_prose: bool = False
    reject_reasons: list[str] = field(default_factory=list)


def _normalize_chunk_lines(text: str) -> list[str]:
    """Merge PDF soft line-breaks into logical sentences for scoring.

    PDF extractors break lines at column width (30-50 chars).  Raw
    scoring on those lines yields catastrophically wrong metrics
    (mean_line_length=32, short_line_ratio=1.0) that kill legitimate
    prose.

    This heuristic merges continuation lines back into sentence-level
    units:
      • If a line does NOT end with sentence punctuation → join with
        the next line (soft break from PDF layout).
      • Paragraph boundaries (double newlines) are preserved.
      • Result: logical lines ≈ sentences, not PDF column widths.
    """
    _SENT_END = frozenset(".!?;:")
    paragraphs = re.split(r"\n\s*\n", text)
    logical_lines: list[str] = []

    for para in paragraphs:
        raw_lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not raw_lines:
            continue

        buffer: list[str] = [raw_lines[0]]
        for line in raw_lines[1:]:
            prev = buffer[-1]
            # Previous line doesn't end with sentence punctuation
            # → continuation (soft line-break from PDF layout)
            if prev and prev[-1] not in _SENT_END:
                buffer[-1] = prev + " " + line
            else:
                buffer.append(line)

        logical_lines.extend(buffer)

    return logical_lines


def _score_chunk(text: str, cfg: ProseConfig) -> ChunkScore:
    """Compute heuristic prose-quality scores for a text chunk.

    Line-based metrics use logical lines (soft PDF breaks merged)
    so that column width doesn't affect quality scoring.

    Verdict uses a two-tier system:
      • HARD signals (tables, pure numbers): reject on their own.
      • SOFT signals (sentences, orphans, refs, line length):
        need ≥2 simultaneous failures to reject.
    This prevents legal preambles from being killed by a single
    slightly-over-threshold metric.
    """
    score = ChunkScore(text=text)

    words = text.split()
    score.n_words = len(words)

    if score.n_words < cfg.min_chunk_words or len(text) < cfg.min_chunk_chars:
        score.reject_reasons.append("too_short")
        return score

    # ── EARLY EXIT: Word-salad / table extraction garbage ─────────
    # Table-heavy PDF pages produce incoherent text with extreme word
    # repetition and no sentence structure.  Catch this before any
    # expensive scoring.
    if score.n_words >= 30:
        lower_words = [w.lower().strip(".,;:()\"'") for w in words]
        _unique = len(set(w for w in lower_words if len(w) > 1))
        _consec = sum(
            1 for i in range(1, len(words))
            if words[i].lower() == words[i - 1].lower()
        )
        # Unique word ratio < 40% = extreme repetition (clean prose ≈ 80%)
        if _unique / score.n_words < 0.40:
            score.reject_reasons.append(
                f"word_salad_unique:{_unique / score.n_words:.2f}"
            )
            return score
        # >20% consecutive duplicates = column/cell extraction noise
        if _consec / score.n_words > 0.20:
            score.reject_reasons.append(
                f"word_salad_consec:{_consec / score.n_words:.2f}"
            )
            return score
        # Data type codes (an..35, n1, a2, 9x) = form specification garbage
        _dtypes = len(re.findall(
            r"\b(?:an?\d|n\d|an?\.\.\d+|[0-9]{1,3}[xX])\b", text
        ))
        if _dtypes >= 3:
            score.reject_reasons.append(f"form_spec_dtypes:{_dtypes}")
            return score

    # Use LOGICAL lines (PDF breaks merged) for line-based metrics
    lines = _normalize_chunk_lines(text)
    n_lines = len(lines) or 1

    # Track hard vs soft failures separately
    hard_failures: list[str] = []
    soft_failures: list[str] = []

    # ── Numeric density (character-level, unaffected by lines) ─────
    n_alpha = sum(1 for ch in text if ch.isalpha())
    n_digit = sum(1 for ch in text if ch.isdigit())

    if n_alpha > 0:
        score.digit_ratio = n_digit / n_alpha
    else:
        score.digit_ratio = 1.0 if n_digit > 0 else 0.0

    # Number tokens (words that are purely numeric, possibly with .,%)
    num_tokens = sum(
        1 for w in words
        if re.fullmatch(r"[\d.,/%€$£¥°()*\-+≤≥<>]+", w)
    )
    score.number_word_ratio = num_tokens / score.n_words

    if score.digit_ratio > cfg.max_digit_ratio:
        # Always soft — real tables are caught by tabular_line_ratio.
        # Legal text has high digit density from regulation numbers
        # (e.g. "regolamento (UE) 2022/2554") which is NOT noise.
        soft_failures.append(f"digit_ratio:{score.digit_ratio:.2f}")
    if score.number_word_ratio > cfg.max_number_word_ratio:
        if score.number_word_ratio > cfg.max_number_word_ratio * 2:
            hard_failures.append(f"num_words:{score.number_word_ratio:.2f}")
        else:
            soft_failures.append(f"num_words:{score.number_word_ratio:.2f}")

    # ── Table detection (on logical lines) ─────────────────────────
    short_lines = sum(1 for ln in lines if len(ln.strip()) < 40)
    score.short_line_ratio = short_lines / n_lines

    tabular_lines = sum(1 for ln in lines if _TABLE_LINE.match(ln))
    number_lines = sum(1 for ln in lines if _NUMBER_LINE.match(ln))
    score.tabular_line_ratio = (tabular_lines + number_lines) / n_lines

    score.mean_line_length = sum(len(ln.strip()) for ln in lines) / n_lines

    if score.tabular_line_ratio > cfg.max_tabular_line_ratio:
        hard_failures.append(f"tabular:{score.tabular_line_ratio:.2f}")  # HARD
    if score.short_line_ratio > cfg.max_short_line_ratio:
        soft_failures.append(f"short_lines:{score.short_line_ratio:.2f}")
    if score.mean_line_length < cfg.min_mean_line_length:
        soft_failures.append(f"mean_line_len:{score.mean_line_length:.0f}")

    # ── Sentence structure (on logical lines) ──────────────────────
    # Count lines ending with sentence punctuation (.!?;:)
    # Also give half-credit for comma/paren endings (legal preambles)
    sentence_count = 0.0
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        last_ch = stripped[-1]
        if last_ch in ".!?;:":
            sentence_count += 1.0
        elif last_ch in ",)":
            sentence_count += 0.5  # partial credit for legal clauses
    score.sentence_ratio = sentence_count / n_lines

    orphan_lines = sum(1 for ln in lines if len(ln.split()) <= 3)
    score.orphan_ratio = orphan_lines / n_lines

    if score.sentence_ratio < cfg.min_sentence_ratio:
        soft_failures.append(f"no_sentences:{score.sentence_ratio:.2f}")
    if score.orphan_ratio > cfg.max_orphan_ratio:
        soft_failures.append(f"orphans:{score.orphan_ratio:.2f}")

    # ── Reference / code noise (on logical lines) ──────────────────
    # Only count a line as "ref noise" if the reference patterns
    # dominate — a prose line that mentions "regolamento (UE) n. 575"
    # inline should NOT count.
    ref_noise_lines = 0
    for ln in lines:
        ref_matches = list(_LEGAL_REF.finditer(ln))
        if not ref_matches:
            continue
        # Sum of matched character spans vs total line length
        matched_chars = sum(m.end() - m.start() for m in ref_matches)
        if matched_chars / max(len(ln), 1) > 0.30:
            ref_noise_lines += 1
    score.ref_ratio = ref_noise_lines / n_lines

    paren_count = len(_PAREN_NOISE.findall(text))
    score.paren_ratio = paren_count / score.n_words if score.n_words else 0

    if score.ref_ratio > cfg.max_ref_ratio:
        soft_failures.append(f"refs:{score.ref_ratio:.2f}")
    if score.paren_ratio > cfg.max_paren_ratio:
        soft_failures.append(f"parens:{score.paren_ratio:.2f}")

    # ── Verdict (two-tier) ────────────────────────────────────────
    # Hard failures → instant reject (tables, extreme numeric density)
    # Soft failures → need ≥2 to reject (structure, refs, line length)
    all_failures = hard_failures + soft_failures
    score.reject_reasons = all_failures

    if hard_failures:
        score.is_prose = False
    elif len(soft_failures) >= 2:
        score.is_prose = False
    else:
        score.is_prose = True

    return score


# ══════════════════════════════════════════════════════════════════════
#  PROSE FLATTENER
# ══════════════════════════════════════════════════════════════════════

# Patterns for cleanup
_MULTI_NL = re.compile(r"\n{3,}")


def flatten_to_prose(text: str) -> str:
    """Minimal cleanup preserving original text structure.

    Rules
    ─────
      • Normalize CR → LF
      • Cap consecutive blank lines at 2
      • Strip trailing whitespace per line
      • Preserve indentation, markdown formatting, list structure

    The content keeps its original form (markdown tables, indented
    lists, code blocks, etc.) so that the output is readable and
    structurally faithful to the source.
    """
    if not text:
        return ""

    # Normalize CR
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip trailing whitespace per line, preserve leading
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)

    # Cap consecutive blank lines at 2 (i.e. max one empty line between blocks)
    text = _MULTI_NL.sub("\n\n", text)

    return text.strip()


def _merge_small_chunks(chunks: list[str], min_words: int = 15) -> list[str]:
    """Merge adjacent chunks that are too small to evaluate.

    Prevents over-splitting: consecutive small paragraphs
    (like numbered "considerando" items in EU regulations) get
    merged into one evaluable block.
    """
    if not chunks:
        return []

    merged: list[str] = []
    buffer: list[str] = []
    buffer_words: int = 0

    for chunk in chunks:
        n_words = len(chunk.split())

        if buffer_words >= min_words and n_words >= min_words:
            # Buffer is big enough AND current chunk is big enough
            # → flush buffer, start fresh
            merged.append("\n\n".join(buffer))
            buffer = [chunk]
            buffer_words = n_words
        elif buffer_words >= min_words * 3 and n_words < min_words:
            # Buffer is already very large, small chunk might be noise
            # → flush buffer, start new buffer with the small chunk
            merged.append("\n\n".join(buffer))
            buffer = [chunk]
            buffer_words = n_words
        else:
            # Accumulate into buffer
            buffer.append(chunk)
            buffer_words += n_words

    if buffer:
        merged.append("\n\n".join(buffer))

    return [m for m in merged if m.strip()]


# ══════════════════════════════════════════════════════════════════════
#  SEMANTIC CHUNKER + FILTER
# ══════════════════════════════════════════════════════════════════════


@dataclass
class FilterStats:
    """Aggregate stats for prose filtering."""

    total_chunks: int = 0
    kept_chunks: int = 0
    rejected_chunks: int = 0
    total_docs_in: int = 0
    total_docs_out: int = 0
    empty_after_filter: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def record_reject(self, reason: str) -> None:
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1


class ProseFilter:
    """Full prose quality pipeline: chunk → score → filter → flatten.

    Takes a raw document string, returns clean prose or empty string.
    """

    def __init__(self, cfg: ProseConfig | None = None) -> None:
        self.cfg = cfg or ProseConfig()
        self.stats = FilterStats()
        self._perplexity_scorer: PerplexityScorer | None = None

        if self.cfg.perplexity_enabled:
            self._perplexity_scorer = PerplexityScorer(
                model_name=self.cfg.perplexity_model,
                device=self.cfg.perplexity_device,
                max_tokens=self.cfg.perplexity_max_tokens,
            )

    def process(self, text: str) -> str:
        """Process a document through the full prose pipeline.

        Returns:
            Clean prose string, or empty string if nothing survives.
        """
        self.stats.total_docs_in += 1

        if not text or not text.strip():
            self.stats.empty_after_filter += 1
            return ""

        # ── 1. Clean up obvious non-prose noise ──────────────────
        text = _FOOTNOTE.sub("", text)
        text = _CODE_LINE.sub("", text)

        # ── 2. Semantic chunking ─────────────────────────────────
        chunks = _SECTION_SPLIT.split(text)
        chunks = [c.strip() for c in chunks if c and c.strip()]

        if not chunks:
            self.stats.empty_after_filter += 1
            return ""

        # ── 3. Merge adjacent small chunks ────────────────────────
        # Prevents over-splitting: numbered paragraphs like
        # "(1) L'accordo...\n\n(2) A norma..." should stay together.
        merged_chunks = _merge_small_chunks(
            chunks,
            min_words=self.cfg.min_chunk_words,
        )

        # ── 4. Score and filter each chunk ────────────────────────
        surviving: list[str] = []
        for chunk in merged_chunks:
            self.stats.total_chunks += 1

            score = _score_chunk(chunk, self.cfg)

            if not score.is_prose:
                self.stats.rejected_chunks += 1
                for reason in score.reject_reasons:
                    category = reason.split(":")[0]
                    self.stats.record_reject(category)
                continue

            # ── 4b. GPU perplexity check (optional) ──────────────
            if self._perplexity_scorer is not None:
                ppl = self._perplexity_scorer.score(chunk)
                if ppl is not None:
                    if ppl < self.cfg.perplexity_min:
                        self.stats.rejected_chunks += 1
                        self.stats.record_reject("low_perplexity")
                        continue
                    if ppl > self.cfg.perplexity_max:
                        self.stats.rejected_chunks += 1
                        self.stats.record_reject("high_perplexity")
                        continue

            self.stats.kept_chunks += 1
            surviving.append(chunk)

        if not surviving:
            self.stats.empty_after_filter += 1
            return ""

        # ── 5. Flatten to clean prose ─────────────────────────────
        merged = "\n\n".join(surviving)
        result = flatten_to_prose(merged)

        if result and result.strip():
            self.stats.total_docs_out += 1
            return result

        self.stats.empty_after_filter += 1
        return ""

    def summary(self) -> list[tuple[str, int]]:
        """Return rejection reason counts sorted by frequency."""
        return sorted(
            self.stats.reject_reasons.items(),
            key=lambda x: x[1],
            reverse=True,
        )


# ══════════════════════════════════════════════════════════════════════
#  GPU PERPLEXITY SCORER (optional)
# ══════════════════════════════════════════════════════════════════════


class PerplexityScorer:
    """Compute perplexity using a causal language model on GPU.

    Loads the model lazily on first call.  Designed for Italian text
    using multilingual models that cover Italian well.

    Recommended models (by size / quality):
      • ``sapienzanlp/Minerva-350M-base-v1.0`` — 350M, 50% Italian, fastest
      • ``facebook/xglm-564M``   — 1B, 50% Italian, best balance
      • ``sapienzanlp/Minerva-3B-base-v1.0``   — 3B, 50% Italian, best quality

    The scorer tokenizes the chunk, computes the cross-entropy loss,
    and returns ``exp(loss)`` = perplexity.
    """

    def __init__(
            self,
            model_name: str = "facebook/xglm-564M",
            device: str = "cuda",
            max_tokens: int = 256,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._disabled = False

    def _load(self) -> None:
        """Lazy-load model and tokenizer.

        On failure, sets ``_disabled = True`` and logs the error
        so that subsequent ``score()`` calls return ``None``
        instead of crashing the pipeline.
        """
        if self._model is not None or self._disabled:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading perplexity model: %s", self._model_name)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, use_fast=False,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16,
            ).to(self._device)
            self._model.eval()

            logger.info(
                "Perplexity model loaded on %s (%.0f M params)",
                self._device,
                sum(p.numel() for p in self._model.parameters()) / 1e6,
            )

        except ImportError:
            logger.error(
                "torch/transformers not installed — GPU perplexity disabled. "
                "Install with: pip install torch transformers"
            )
            self._disabled = True
        except Exception as exc:
            logger.error(
                "Failed to load perplexity model: %s — "
                "perplexity scoring disabled, pipeline continues without it",
                exc,
            )
            self._disabled = True

    def score(self, text: str) -> float | None:
        """Compute perplexity for a text chunk.

        Returns:
            Perplexity (float).  Lower = more predictable/fluent.
            Returns ``None`` if the model could not be loaded.
            Returns ``float('inf')`` on scoring error.
        """
        self._load()

        if self._model is None:
            return None

        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_tokens,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()

            return math.exp(loss)

        except Exception as exc:
            logger.warning("Perplexity scoring failed: %s", exc)
            return float("inf")
