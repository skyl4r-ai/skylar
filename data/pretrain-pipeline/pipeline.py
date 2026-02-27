"""
Data quality pipeline for pre-training corpus construction.

Filters
───────
  • PII removal      — email, phone, codice fiscale, IBAN, IP, credit cards
  • Quality          — length, repetition, compression ratio, boilerplate, unicode
  • Spam / blacklist — URL density, domain blacklist, keyword / regex patterns
  • Exact dedup      — xxhash-based exact document hashing
  • Near-dedup       — MinHash LSH (datasketch) with configurable Jaccard threshold

Every filter is a standalone callable so the pipeline is fully composable.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import xxhash
from datasketch import MinHash, MinHashLSH

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PipelineConfig:
    """All knobs for the filtering pipeline.

    Load defaults, then optionally override from a YAML file via
    ``PipelineConfig.from_yaml(path)``.
    """

    # ── PII ───────────────────────────────────────────────────────
    pii_enabled: bool = True
    pii_email: bool = True
    pii_phone: bool = True
    pii_codice_fiscale: bool = True
    pii_partita_iva: bool = True
    pii_iban: bool = True
    pii_ip_address: bool = True
    pii_credit_card: bool = True

    # ── Quality ───────────────────────────────────────────────────
    quality_enabled: bool = True
    min_doc_chars: int = 100
    max_doc_chars: int = 500_000
    min_word_count: int = 20
    max_word_count: int = 100_000
    max_uppercase_ratio: float = 0.4
    max_special_char_ratio: float = 0.3
    min_mean_word_length: float = 2.0
    max_mean_word_length: float = 20.0
    max_line_dup_ratio: float = 0.5
    max_paragraph_dup_ratio: float = 0.3
    min_compression_ratio: float = 0.10
    max_compression_ratio: float = 0.90
    max_boilerplate_ratio: float = 0.5
    max_bullet_ratio: float = 0.7
    max_ellipsis_ratio: float = 0.3

    # ── Spam / blacklist ──────────────────────────────────────────
    spam_enabled: bool = True
    max_url_density: float = 0.10
    blacklist_domains_path: str | None = None
    blacklist_keywords_path: str | None = None

    # ── Exact dedup ───────────────────────────────────────────────
    exact_dedup_enabled: bool = True

    # ── Near dedup ────────────────────────────────────────────────
    near_dedup_enabled: bool = True
    near_dedup_threshold: float = 0.80
    near_dedup_num_perm: int = 128
    near_dedup_ngram_size: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load config from a YAML file, merging onto defaults."""
        import yaml  # lazy — not needed if user sticks with defaults

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────
# Filter protocol
# ──────────────────────────────────────────────────────────────────────


class FilterResult:
    """Outcome of running a single filter on a document."""

    __slots__ = ("keep", "text", "reason")

    def __init__(self, keep: bool, text: str | None = None, reason: str = "") -> None:
        self.keep = keep
        self.text = text  # mutated text (PII scrub) or None
        self.reason = reason

    @staticmethod
    def accept(text: str | None = None) -> FilterResult:
        return FilterResult(keep=True, text=text)

    @staticmethod
    def reject(reason: str) -> FilterResult:
        return FilterResult(keep=False, reason=reason)


class DocFilter(Protocol):
    """Protocol for any document filter."""

    name: str

    def __call__(self, text: str) -> FilterResult: ...


# ══════════════════════════════════════════════════════════════════════
#  1 — PII FILTER
# ══════════════════════════════════════════════════════════════════════

# Pre-compiled patterns (Italian-focused + universal)
_RE_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
)
_RE_PHONE_IT = re.compile(
    r"(?<!\d)"
    r"(?:\+39[\s\-]?)?"
    r"(?:"
    r"3[0-9]{2}[\s\-.]?[0-9]{3}[\s\-.]?[0-9]{4}"  # cellulare
    r"|0[0-9]{1,3}[\s\-.]?[0-9]{4,8}"  # fisso
    r")"
    r"(?!\d)"
)
_RE_CODICE_FISCALE = re.compile(
    r"\b[A-Z]{6}[0-9]{2}[A-EHLMPRST][0-9]{2}[A-Z][0-9]{3}[A-Z]\b",
    re.IGNORECASE,
)
_RE_PARTITA_IVA = re.compile(r"\b(?:IT)?[0-9]{11}\b")
_RE_IBAN_IT = re.compile(
    r"\bIT\s?[0-9]{2}\s?[A-Z]\s?(?:[0-9]{5}\s?){2}[0-9A-Z]{12}\b",
    re.IGNORECASE,
)
_RE_IP = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
_RE_CREDIT_CARD = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?"  # Visa
    r"|5[1-5][0-9]{14}"  # Mastercard
    r"|3[47][0-9]{13}"  # Amex
    r"|6(?:011|5[0-9]{2})[0-9]{12}"  # Discover
    r")\b"
)

_PII_PATTERNS: dict[str, tuple[re.Pattern[str], str]] = {
    "pii_email": (_RE_EMAIL, "<EMAIL>"),
    "pii_phone": (_RE_PHONE_IT, "<PHONE>"),
    "pii_codice_fiscale": (_RE_CODICE_FISCALE, "<CF>"),
    "pii_partita_iva": (_RE_PARTITA_IVA, "<PIVA>"),
    "pii_iban": (_RE_IBAN_IT, "<IBAN>"),
    "pii_ip_address": (_RE_IP, "<IP>"),
    "pii_credit_card": (_RE_CREDIT_CARD, "<CC>"),
}


class PIIFilter:
    """Scrub personally identifiable information, replacing with tokens."""

    name: str = "pii"

    def __init__(self, cfg: PipelineConfig) -> None:
        self._active: list[tuple[re.Pattern[str], str]] = []
        for key, (pattern, replacement) in _PII_PATTERNS.items():
            if getattr(cfg, key, False):
                self._active.append((pattern, replacement))

    def __call__(self, text: str) -> FilterResult:
        for pattern, replacement in self._active:
            text = pattern.sub(replacement, text)
        return FilterResult.accept(text)


# ══════════════════════════════════════════════════════════════════════
#  2 — QUALITY FILTER
# ══════════════════════════════════════════════════════════════════════

_RE_BULLET = re.compile(r"^[\s]*[\-\*•▪▸►◆◇→‣⁃]", re.MULTILINE)
_RE_ELLIPSIS = re.compile(r"\.{3,}|…")
_RE_BOILERPLATE_IT = re.compile(
    r"(?i)"
    r"(?:cookie\s*polic|privacy\s*polic|termini\s*(?:di\s*)?(?:servizio|uso)|"
    r"accett[ao]\s*(?:i\s*)?cookie|informativa\s*(?:sulla\s*)?privacy|"
    r"iscriviti\s*alla\s*newsletter|tutti\s*i\s*diritti\s*riservati|"
    r"copyright\s*©|powered\s*by|all\s*rights\s*reserved|"
    r"clicca\s*(?:qui|per)|leggi\s*(?:tutto|di\s*più|anche)|"
    r"condividi\s*(?:su|questo)|articoli?\s*correlat[io]|"
    r"lascia\s*un\s*commento|tag(?:s)?:|categori[ae]:|"
    r"©\s*\d{4}|ultimo\s*aggiornamento|"
    r"segui(?:ci)?\s*su|(?:facebook|twitter|instagram|linkedin|youtube|tiktok))"
)
_RE_URL = re.compile(r"https?://\S+", re.IGNORECASE)


def _compression_ratio(text: str) -> float:
    """Ratio of compressed / original byte length. Lower = more redundant."""
    raw = text.encode("utf-8")
    if len(raw) == 0:
        return 0.0
    compressed = zlib.compress(raw, level=6)
    return len(compressed) / len(raw)


def _line_dup_ratio(lines: list[str]) -> float:
    """Fraction of lines that appear more than once."""
    if not lines:
        return 0.0
    counts = Counter(lines)
    dup_count = sum(c - 1 for c in counts.values() if c > 1)
    return dup_count / len(lines)


def _paragraph_dup_ratio(text: str) -> float:
    """Fraction of paragraphs (double-newline separated) that are duplicated."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) < 2:
        return 0.0
    counts = Counter(paragraphs)
    dup_count = sum(c - 1 for c in counts.values() if c > 1)
    return dup_count / len(paragraphs)


class QualityFilter:
    """Heuristic quality gate — length, repetition, compression, boilerplate."""

    name: str = "quality"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def __call__(self, text: str) -> FilterResult:
        c = self.cfg

        # ── Length ────────────────────────────────────────────────
        n_chars = len(text)
        if n_chars < c.min_doc_chars:
            return FilterResult.reject(f"too_short:{n_chars}")
        if n_chars > c.max_doc_chars:
            return FilterResult.reject(f"too_long:{n_chars}")

        words = text.split()
        n_words = len(words)
        if n_words < c.min_word_count:
            return FilterResult.reject(f"few_words:{n_words}")
        if n_words > c.max_word_count:
            return FilterResult.reject(f"many_words:{n_words}")

        # ── Character-level ratios ────────────────────────────────
        alpha = sum(1 for ch in text if ch.isalpha())
        if alpha > 0:
            upper_ratio = sum(1 for ch in text if ch.isupper()) / alpha
            if upper_ratio > c.max_uppercase_ratio:
                return FilterResult.reject(f"uppercase:{upper_ratio:.2f}")

        special = sum(
            1 for ch in text if not ch.isalnum() and not ch.isspace()
        )
        if n_chars > 0 and special / n_chars > c.max_special_char_ratio:
            return FilterResult.reject(f"special_chars:{special / n_chars:.2f}")

        # ── Word-level stats ──────────────────────────────────────
        mean_wl = sum(len(w) for w in words) / n_words if n_words else 0.0
        if mean_wl < c.min_mean_word_length:
            return FilterResult.reject(f"short_words:{mean_wl:.1f}")
        if mean_wl > c.max_mean_word_length:
            return FilterResult.reject(f"long_words:{mean_wl:.1f}")

        # ── Line/paragraph repetition ─────────────────────────────
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            ld = _line_dup_ratio(lines)
            if ld > c.max_line_dup_ratio:
                return FilterResult.reject(f"line_dup:{ld:.2f}")

        pd = _paragraph_dup_ratio(text)
        if pd > c.max_paragraph_dup_ratio:
            return FilterResult.reject(f"para_dup:{pd:.2f}")

        # ── Compression ratio (perplexity proxy) ──────────────────
        cr = _compression_ratio(text)
        if cr < c.min_compression_ratio:
            return FilterResult.reject(f"low_entropy:{cr:.2f}")
        if cr > c.max_compression_ratio:
            return FilterResult.reject(f"high_entropy:{cr:.2f}")

        # ── Boilerplate lines ─────────────────────────────────────
        if lines:
            bp_count = sum(
                1 for ln in lines if _RE_BOILERPLATE_IT.search(ln)
            )
            bp_ratio = bp_count / len(lines)
            if bp_ratio > c.max_boilerplate_ratio:
                return FilterResult.reject(f"boilerplate:{bp_ratio:.2f}")

        # ── Bullet / ellipsis heavy ───────────────────────────────
        if lines:
            bullet_count = sum(
                1 for ln in lines if _RE_BULLET.match(ln)
            )
            if bullet_count / len(lines) > c.max_bullet_ratio:
                return FilterResult.reject(f"bullet_heavy:{bullet_count / len(lines):.2f}")

        ellipsis_count = len(_RE_ELLIPSIS.findall(text))
        if n_words > 0 and ellipsis_count / n_words > c.max_ellipsis_ratio:
            return FilterResult.reject(
                f"ellipsis:{ellipsis_count / n_words:.2f}"
            )

        return FilterResult.accept()


# ══════════════════════════════════════════════════════════════════════
#  3 — SPAM / BLACKLIST FILTER
# ══════════════════════════════════════════════════════════════════════

_DEFAULT_SPAM_KEYWORDS: set[str] = {
    "casino", "slot machine", "scommesse online", "viagra",
    "cialis", "forex trading", "guadagna subito",
    "lavoro da casa facile", "bitcoin gratis", "trading automatico",
    "prestito immediato", "dimagrire velocemente", "offerta esclusiva",
    "clicca qui per vincere", "congratulazioni hai vinto",
    "eredità milionaria", "principe nigeriano",
}


def _load_lines(path: str | Path | None) -> set[str]:
    """Load non-empty, stripped, lowercase lines from a text file."""
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    return {
        ln.strip().lower()
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    }


def _extract_domains(text: str) -> list[str]:
    """Extract domain names from all URLs in text."""
    domains: list[str] = []
    for m in _RE_URL.finditer(text):
        url = m.group(0)
        # Rough extraction — covers http(s)://domain.tld/...
        parts = url.split("/")
        if len(parts) >= 3:
            domain = parts[2].split(":")[0].lower()
            domains.append(domain)
    return domains


class SpamFilter:
    """Block documents with excessive URLs, blacklisted domains, or spam keywords."""

    name: str = "spam"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._blacklist_domains: set[str] = _load_lines(cfg.blacklist_domains_path)
        self._blacklist_keywords: set[str] = (
            _load_lines(cfg.blacklist_keywords_path) or _DEFAULT_SPAM_KEYWORDS
        )

    def __call__(self, text: str) -> FilterResult:
        words = text.split()
        n_words = len(words) or 1

        # ── URL density ───────────────────────────────────────────
        url_count = len(_RE_URL.findall(text))
        if url_count / n_words > self.cfg.max_url_density:
            return FilterResult.reject(f"url_density:{url_count / n_words:.2f}")

        # ── Domain blacklist ──────────────────────────────────────
        if self._blacklist_domains:
            domains = _extract_domains(text)
            for d in domains:
                if d in self._blacklist_domains:
                    return FilterResult.reject(f"blacklist_domain:{d}")

        # ── Keyword spam ──────────────────────────────────────────
        text_lower = text.lower()
        for kw in self._blacklist_keywords:
            if kw in text_lower:
                return FilterResult.reject(f"spam_keyword:{kw}")

        return FilterResult.accept()


# ══════════════════════════════════════════════════════════════════════
#  4 — EXACT DEDUP
# ══════════════════════════════════════════════════════════════════════


class ExactDedup:
    """O(1) lookup exact deduplication via xxhash-64."""

    name: str = "exact_dedup"

    def __init__(self) -> None:
        self._seen: set[int] = set()

    def __call__(self, text: str) -> FilterResult:
        h = xxhash.xxh64_intdigest(text.encode("utf-8"))
        if h in self._seen:
            return FilterResult.reject("exact_dup")
        self._seen.add(h)
        return FilterResult.accept()

    @property
    def n_hashes(self) -> int:
        return len(self._seen)

    def ram_bytes(self) -> int:
        """Approximate RAM usage (set of int64 ≈ 50 bytes/entry in CPython)."""
        return self.n_hashes * 50


# ══════════════════════════════════════════════════════════════════════
#  5 — NEAR-DEDUP  (MinHash LSH)
# ══════════════════════════════════════════════════════════════════════


def _word_ngrams(text: str, n: int) -> list[str]:
    """Generate word-level n-grams as space-joined strings."""
    words = text.lower().split()
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


class NearDedup:
    """Approximate near-duplicate detection using MinHash LSH.

    Memory: the LSH index stores hash bands only — roughly 2 KB per
    document at num_perm=128.  For 10 M docs ≈ 20 GB; reduce
    ``num_perm`` to 64 for ≈ 10 GB, or stream in two passes for
    extreme scale.
    """

    name: str = "near_dedup"

    def __init__(self, cfg: PipelineConfig) -> None:
        self._num_perm = cfg.near_dedup_num_perm
        self._ngram_size = cfg.near_dedup_ngram_size
        self._lsh = MinHashLSH(
            threshold=cfg.near_dedup_threshold,
            num_perm=self._num_perm,
        )
        self._counter: int = 0

    def _make_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self._num_perm)
        for ng in _word_ngrams(text, self._ngram_size):
            m.update(ng.encode("utf-8"))
        return m

    def __call__(self, text: str) -> FilterResult:
        mh = self._make_minhash(text)

        # Query BEFORE insert — if any existing doc is similar, reject.
        if self._lsh.query(mh):
            return FilterResult.reject("near_dup")

        key = f"d{self._counter}"
        self._counter += 1
        try:
            self._lsh.insert(key, mh)
        except ValueError:
            # Extremely rare hash collision inside LSH bands — skip insert,
            # keep the document (it already passed the query).
            pass

        return FilterResult.accept()

    @property
    def n_entries(self) -> int:
        return self._counter


# ══════════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════


@dataclass
class PipelineStats:
    """Aggregate stats keyed by filter name → rejection reason → count."""

    total_seen: int = 0
    total_kept: int = 0
    rejected_by: dict[str, dict[str, int]] = field(default_factory=dict)

    def record_reject(self, filter_name: str, reason: str) -> None:
        bucket = self.rejected_by.setdefault(filter_name, {})
        bucket[reason] = bucket.get(reason, 0) + 1

    @property
    def total_rejected(self) -> int:
        return self.total_seen - self.total_kept

    def rejection_counts(self) -> dict[str, int]:
        """Per-filter total rejections."""
        return {
            name: sum(reasons.values())
            for name, reasons in self.rejected_by.items()
        }


class Pipeline:
    """Composable document processing pipeline.

    Filters are applied in order.  *Mutating* filters (PII) modify the
    text and pass it along; *gate* filters drop the document on failure.
    """

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self.stats = PipelineStats()
        self._filters: list[DocFilter] = []
        self._build()

    def _build(self) -> None:
        c = self.cfg

        if c.pii_enabled:
            self._filters.append(PIIFilter(c))  # type: ignore[arg-type]

        if c.quality_enabled:
            self._filters.append(QualityFilter(c))  # type: ignore[arg-type]

        if c.spam_enabled:
            self._filters.append(SpamFilter(c))  # type: ignore[arg-type]

        if c.exact_dedup_enabled:
            self._filters.append(ExactDedup())  # type: ignore[arg-type]

        if c.near_dedup_enabled:
            self._filters.append(NearDedup(c))  # type: ignore[arg-type]

    def process(self, text: str) -> tuple[bool, str]:
        """Run *text* through all filters.

        Returns:
            ``(keep, cleaned_text)`` — if *keep* is False the document
            should be discarded.
        """
        self.stats.total_seen += 1
        current = text

        for filt in self._filters:
            result = filt(current)
            if not result.keep:
                self.stats.record_reject(filt.name, result.reason)
                return False, current
            if result.text is not None:
                current = result.text

        self.stats.total_kept += 1
        return True, current

    def summary_table(self) -> list[tuple[str, int, float]]:
        """Return list of *(filter_name, rejected_count, pct_of_total)*."""
        total = self.stats.total_seen or 1
        rows: list[tuple[str, int, float]] = []
        for name, reasons in self.stats.rejected_by.items():
            count = sum(reasons.values())
            rows.append((name, count, count / total * 100))
        return rows

    def top_reasons(self, n: int = 10) -> list[tuple[str, str, int]]:
        """Top *n* rejection reasons across all filters."""
        flat: list[tuple[str, str, int]] = []
        for name, reasons in self.stats.rejected_by.items():
            for reason, count in reasons.items():
                flat.append((name, reason, count))
        flat.sort(key=lambda x: x[2], reverse=True)
        return flat[:n]
