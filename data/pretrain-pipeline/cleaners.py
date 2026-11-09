# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Text cleaning and normalization for pre-training corpus construction.

Five main components
────────────────────
  • MinimalNormalizer  — lightweight text cleanup for text that is already
                        reasonably clean (e.g. Docling output).  Only does:
                        encoding fix (ftfy), NFKC, zero-width/control char
                        removal, multi-space collapse, newline normalization.
                        Does NOT touch content structure, legal text, page
                        numbers, hyphenation, list markers, or short lines.
  • TextNormalizer     — (legacy) aggressive text cleanup.  Kept for backward
                        compatibility but no longer used by default.
  • PDFCleaner         — PDF-specific post-extraction cleanup (header/footer
                        detection across pages, hyphenation repair, paragraph
                        merge for reflowed text).
  • ColumnMergeCleaner — detects and strips text corrupted by PDF two-column
                        merge (EUR-Lex, Gazzetta Ufficiale).  Scores each
                        paragraph for corruption signals and removes broken
                        chunks while preserving clean prose.
  • RepetitionScorer   — sliding-window n-gram repetition ratio.  Used as an
                        additional quality signal beyond zlib compression.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import ftfy

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class CleanerConfig:
    """Knobs for text normalization."""

    max_consecutive_newlines: int = 2
    min_line_chars: int = 3
    collapse_leader_dots: bool = True
    fix_hyphenation: bool = True
    remove_page_numbers: bool = True
    unicode_normalize: bool = True
    fix_encoding: bool = True
    # PDF-specific
    header_footer_max_lines: int = 2
    header_footer_min_pages: float = 0.5  # fraction of pages a line must appear in
    paragraph_merge: bool = True
    # Repetition
    repetition_ngram_size: int = 10
    max_repetition_ratio: float = 0.20


# ──────────────────────────────────────────────────────────────────────
# Compiled patterns
# ──────────────────────────────────────────────────────────────────────

_ZERO_WIDTH = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e"
    r"\u2060\u2061-\u2064\ufeff\ufffe\u00ad]"
)

# Repeated punctuation (only dots — markdown syntax like ###, ---, ~~~, *** preserved)
_REPEATED_DOTS = re.compile(r"\.{4,}")

# Whitespace
_MULTI_NEWLINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)

# Page numbers (standalone lines)
_PAGE_NUM_LINE = re.compile(
    r"^[\s]*"
    r"(?:"
    r"(?:pag(?:ina)?\.?\s*)?[0-9]{1,5}"
    r"|[—\-–]\s*[0-9]{1,5}\s*[—\-–]"
    r"|[ivxlcdm]+"
    r")"
    r"[\s]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Hyphenation at line breaks
_HYPHEN_BREAK = re.compile(r"(\w{2,})-\s*\n\s*(\w{2,})")

# Leader dots (TOC style)
_LEADER_DOTS = re.compile(r"[.\s·]{5,}\s*\d+\s*$", re.MULTILINE)

# Bullet/list markers at line start (for normalization, not removal)
_LIST_MARKER = re.compile(r"^[\s]*[•▪▸►◆◇→‣⁃■□●○]\s*", re.MULTILINE)

# Common PDF artifacts
_FORM_FEED = re.compile(r"\f")
_TAB_RUNS = re.compile(r"\t{2,}")


# ══════════════════════════════════════════════════════════════════════
#  MINIMAL NORMALIZER (default for Docling output)
# ══════════════════════════════════════════════════════════════════════

# Collapse 2+ spaces into one (but NOT leading whitespace — preserves indentation)
_MULTI_SPACES = re.compile(r"(?<=\S) {2,}")


class MinimalNormalizer:
    """Lightweight text normalizer for already-clean extractor output.

    Docling and modern extractors produce reasonably clean text.
    This normalizer only fixes encoding-level issues without
    touching document structure or content:

      1. ftfy encoding repair + NFKC unicode normalization
      2. Remove zero-width / invisible characters
      3. Remove non-printable control chars (keep newline, tab)
      4. Collapse multiple spaces between words into one
      5. Normalize CR → LF
      6. Cap consecutive blank lines at 2
      7. Strip trailing whitespace per line
    """

    def __call__(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        # ── 1. Encoding repair + NFKC ─────────────────────────────
        text = ftfy.fix_text(text, normalization="NFKC")

        # ── 2. Zero-width / invisible characters ──────────────────
        text = _ZERO_WIDTH.sub("", text)

        # ── 3. Non-printable control chars (keep \n \t) ───────────
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _FORM_FEED.sub("\n", text)
        text = "".join(
            ch for ch in text
            if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
        )

        # ── 4. Collapse multiple spaces between words ─────────────
        text = _MULTI_SPACES.sub(" ", text)

        # ── 5. Trailing whitespace per line ────────────────────────
        text = _TRAILING_SPACES.sub("", text)

        # ── 6. Cap consecutive blank lines ─────────────────────────
        text = _MULTI_NEWLINES.sub("\n\n", text)

        return text.strip()


# ══════════════════════════════════════════════════════════════════════
#  TEXT NORMALIZER (legacy — aggressive, no longer used by default)
# ══════════════════════════════════════════════════════════════════════


class TextNormalizer:
    """Universal multi-stage text normalization.

    Applied to all documents regardless of source format.
    """

    def __init__(self, cfg: CleanerConfig | None = None) -> None:
        self.cfg = cfg or CleanerConfig()

    def __call__(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        c = self.cfg

        # ── 1. Encoding repair ────────────────────────────────────
        if c.fix_encoding:
            text = ftfy.fix_text(
                text,
                normalization="NFKC" if c.unicode_normalize else "NFC",
            )
        elif c.unicode_normalize:
            text = unicodedata.normalize("NFKC", text)

        # ── 2. Control / zero-width characters ────────────────────
        text = _ZERO_WIDTH.sub("", text)
        text = _FORM_FEED.sub("\n", text)
        text = _TAB_RUNS.sub("\t", text)

        # Remove non-printable control chars (keep \n \t \r)
        text = "".join(
            ch for ch in text
            if ch in ("\n", "\t", "\r") or not unicodedata.category(ch).startswith("C")
        )

        # ── 3. Page numbers (standalone lines) ────────────────────
        if c.remove_page_numbers:
            text = _PAGE_NUM_LINE.sub("", text)

        # ── 4. Leader dots (TOC) ─────────────────────────────────
        if c.collapse_leader_dots:
            text = _LEADER_DOTS.sub("", text)

        # ── 5. Hyphenation repair ─────────────────────────────────
        if c.fix_hyphenation:
            text = _HYPHEN_BREAK.sub(r"\1\2", text)

        # ── 6. Repeated punctuation → single symbol ──────────────
        # Only collapse dots (4+); preserve markdown syntax:
        # ### headings, --- rules, ~~~ fences, *** separators, etc.
        text = _REPEATED_DOTS.sub("…", text)

        # ── 7. Whitespace normalization ───────────────────────────
        text = _TRAILING_SPACES.sub("", text)
        text = _MULTI_NEWLINES.sub("\n\n", text)

        # ── 8. Remove artifact lines ─────────────────────────────
        lines = text.split("\n")
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            # Keep empty lines (paragraph boundaries) and lines ending in punctuation
            if not stripped:
                cleaned.append("")
                continue
            if len(stripped) < c.min_line_chars:
                # Keep if it ends with sentence punctuation (might be "Sì." or "No.")
                if stripped and stripped[-1] in '.!?:;\u00bb\u201d\u2019"':
                    cleaned.append(line)
                continue
            cleaned.append(line)

        text = "\n".join(cleaned)

        # ── 9. Normalize list markers ─────────────────────────────
        text = _LIST_MARKER.sub("- ", text)

        # ── 10. Final collapse ────────────────────────────────────
        text = _MULTI_NEWLINES.sub("\n\n", text)

        return text.strip()


# ══════════════════════════════════════════════════════════════════════
#  PDF CLEANER
# ══════════════════════════════════════════════════════════════════════


class PDFCleaner:
    """PDF-specific post-extraction cleaner.

    Operates on a list of per-page text strings *before* they are
    concatenated and passed to TextNormalizer.

    Performs
    ───────
      1. Header/footer detection (repeated first/last N lines across pages).
      2. Paragraph-aware line merging (joins soft line-breaks within
         paragraphs while preserving intentional paragraph boundaries).
    """

    def __init__(self, cfg: CleanerConfig | None = None) -> None:
        self.cfg = cfg or CleanerConfig()

    def clean_pages(self, pages: list[str]) -> str:
        """Take a list of per-page text and return a single cleaned string."""
        if not pages:
            return ""

        n_pages = len(pages)

        # ── 1. Detect headers/footers ─────────────────────────────
        hf_lines = self._detect_header_footer(pages)

        # ── 2. Strip detected lines & concatenate ─────────────────
        page_texts: list[str] = []
        for page in pages:
            lines = page.split("\n")
            filtered = [ln for ln in lines if ln.strip() not in hf_lines]
            page_texts.append("\n".join(filtered))

        text = "\n\n".join(page_texts)

        # ── 3. Paragraph-aware merge ──────────────────────────────
        if self.cfg.paragraph_merge:
            text = self._merge_paragraphs(text)

        return text

    def _detect_header_footer(self, pages: list[str]) -> set[str]:
        """Find lines that repeat at the top/bottom of many pages.

        Strategy: for each page, take the first N and last N non-empty
        lines.  Lines appearing in ≥ threshold fraction of pages are
        classified as header/footer.
        """
        n = self.cfg.header_footer_max_lines
        threshold = self.cfg.header_footer_min_pages
        n_pages = len(pages)

        if n_pages < 3:
            return set()

        min_count = max(2, int(n_pages * threshold))
        candidate_counter: Counter[str] = Counter()

        for page in pages:
            lines = [ln.strip() for ln in page.split("\n") if ln.strip()]
            if not lines:
                continue

            top = lines[:n]
            bottom = lines[-n:] if len(lines) > n else []

            # Use a set per page to avoid double-counting
            seen: set[str] = set()
            for ln in top + bottom:
                # Normalize page numbers in the line for matching
                normalized = re.sub(r"\d+", "#", ln)
                if normalized not in seen:
                    seen.add(normalized)
                    candidate_counter[normalized] += 1

        hf_normalized: set[str] = set()
        for normalized, count in candidate_counter.items():
            if count >= min_count and len(normalized.strip()) < 200:
                hf_normalized.add(normalized)

        # Map back: a line matches if its normalized form is in the set
        result: set[str] = set()
        for page in pages:
            for ln in page.split("\n"):
                normalized = re.sub(r"\d+", "#", ln.strip())
                if normalized in hf_normalized:
                    result.add(ln.strip())

        return result

    @staticmethod
    def _merge_paragraphs(text: str) -> str:
        """Merge soft line-breaks within paragraphs.

        Heuristic: a line NOT ending in sentence-final punctuation
        followed by a line starting with a lowercase letter →
        these are the same paragraph, join with a space.
        """
        _SENTENCE_END = frozenset(".!?;:\u00bb\u201d\u2019\")")
        lines = text.split("\n")
        merged: list[str] = []
        buffer: list[str] = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                # Empty line = paragraph boundary
                if buffer:
                    merged.append(" ".join(buffer))
                    buffer = []
                merged.append("")
                continue

            if not buffer:
                buffer.append(stripped)
                continue

            prev = buffer[-1]
            # If previous line ends with sentence punctuation OR current
            # starts with uppercase/number/bullet → new sentence in same para
            # If previous line does NOT end with punctuation AND current starts
            # with lowercase → continuation of same sentence (soft break)
            prev_ends_sentence = prev and prev[-1] in _SENTENCE_END
            curr_starts_lower = stripped and stripped[0].islower()

            if not prev_ends_sentence and curr_starts_lower:
                # Soft line break — same paragraph, same sentence
                buffer.append(stripped)
            elif prev_ends_sentence and curr_starts_lower:
                # New sentence in same paragraph
                buffer.append(stripped)
            else:
                # Likely same paragraph but new sentence
                buffer.append(stripped)

        if buffer:
            merged.append(" ".join(buffer))

        return "\n".join(merged)


# ══════════════════════════════════════════════════════════════════════
#  COLUMN MERGE CLEANER
# ══════════════════════════════════════════════════════════════════════

# --- Gazette page headers embedded inline (nearly 100 % signal) ------
_GAZETTE_INLINE = re.compile(
    r"(?:"
    r"L\s+\d{1,4}/\d{1,4}\s+IT\s+Gazzetta\s+ufficiale"
    r"|GU\s+L\s+del\s+\d{1,2}\.\d{1,2}\.\d{4}\s+IT"
    r"|Gazzetta\s+ufficiale\s+dell[ae]\s+"
    r"(?:Comunità|Unione)\s+europe[ae]\s+\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{4}"
    r")",
    re.IGNORECASE,
)

# --- Structural legal markers displaced to mid-sentence (not at start)
# Must follow lowercase text / comma / semicolon to be "displaced".
_DISPLACED_MARKER = re.compile(
    r"(?<=[a-zà-úA-Z,;:)\d])\s{1,3}"
    r"(?:"
    r"(?:Articolo|Article)\s+\d+"
    r"|DECIDE\s*:"
    r"|HA\s+ADOTTATO\s+(?:IL|LA)\s+PRESENTE"
    r"|IL\s+CONSIGLIO\s+DELL"
    r"|IL\s+PARLAMENTO\s+EUROPEO"
    r"|Fatto\s+a\s+[A-ZÀ-Ú][a-zà-ú]+"
    r"|Per\s+il\s+Consiglio"
    r"|Il\s+presidente\s+del\s+Consiglio"
    r"|vista\s+la\s+proposta\s+della\s+Commissione"
    r"|visto\s+il\s+(?:trattato|parere)"
    r"|considerando\s+quanto\s+segue"
    r"|considerando\s+che"
    r")",
)

# --- Cross-column broken hyphenation -----------------------------------
# word-<space>nextword  OR  word-nextword (no space, fused)
_BROKEN_COL_HYPHEN = re.compile(
    r"\b([a-zà-ú]{2,10})-\s*([a-zà-ú]{3,})\b"
)

# Known Italian prefixes that legitimately hyphenate (won't false-positive)
_LEGIT_HYPHEN_PREFIXES: frozenset[str] = frozenset({
    "anti", "auto", "bi", "co", "contro", "de", "dis", "ex", "extra",
    "foto", "geo", "iper", "infra", "inter", "intra", "macro", "mega",
    "meta", "micro", "mini", "multi", "neo", "non", "omni", "para",
    "pluri", "poli", "post", "pre", "pro", "proto", "pseudo", "quasi",
    "radio", "re", "retro", "ri", "semi", "socio", "sovra", "sub",
    "super", "sur", "tecno", "tele", "trans", "tri", "ultra", "uni",
    "vice",
})

# Known legitimate hyphenated Italian words (prefix-continuation)
# These are common words that happen to contain a hyphen prefix match
_LEGIT_COMPOUNDS: frozenset[str] = frozenset({
    "controcomando", "controllare", "contromisura",
})

# --- Fused punctuation-word (column boundary ate the newline) ----------
# "rica;particolare", "europea,l'articolo", "Comunità,gnare"
_FUSED_PUNCT_WORD = re.compile(
    r"[a-zà-ú][;,.]\s*[a-zà-ú]{2,}"   # lowercase;lowercase fused
    r"(?!\s*(?:che|di|in|per|con|a|da|su|tra|fra)\b)"  # skip normal punct
)

# --- Orphan word fragments (truncated by column boundary) ──────────────
# Short fragments ending with a space before unrelated text.
# "impe della", "gnare la", "mate- rie" where "impe", "gnare" aren't words
_COMMON_IT_SHORT: frozenset[str] = frozenset({
    # 2-4 letter Italian words that are NOT fragments
    "è", "il", "la", "le", "lo", "di", "in", "un", "da", "su", "se",
    "al", "ai", "ci", "ne", "vi", "si", "ma", "ed", "od", "ha", "ho",
    "fa", "va", "no", "me", "te", "tu", "io",
    "che", "chi", "con", "del", "dei", "gli", "per", "più", "non", "nel",
    "tra", "fra", "una", "uno", "due", "tre", "già", "poi", "qui", "ora",
    "sia", "suo", "sua", "mio", "mia", "tuo", "tua", "cui", "ciò",
    "alle", "alla", "allo", "agli", "come", "dopo", "dove", "fino",
    "ogni", "pure", "solo", "sono", "quel", "quei", "alle", "anno",
    "caso", "modo", "fase", "rete", "tipo", "base", "fine", "dati",
    "vita", "pari", "tale", "cosa", "nome", "dato", "nota", "zone",
    "voce", "atto", "area", "ente", "capo", "tali", "sola", "pena",
    "cura", "essa", "esso", "esse", "essi", "tali",
})

# --- Abrupt ALLCAPS header mid-sentence --------------------------------
_ALLCAPS_MID = re.compile(
    r"(?<=[a-zà-ú,.;:)\d])\s{1,3}"
    r"(?:[A-ZÀ-Ú]{4,}\s+){2,}"  # ≥2 consecutive ALLCAPS words
)

# --- Sudden capitalized clause mid-sentence (not after period/colon) ---
# Detects column merge where a new clause from the adjacent column starts
# with a capital letter mid-sentence without proper punctuation.
# "europea e la Il presidente del Consiglio è autorizzato"
_SUDDEN_CAPITAL_CLAUSE = re.compile(
    r"(?<=[a-zà-ú])\s+"           # after lowercase
    r"(?:"
    r"(?:Il|La|Le|Lo|I|Gli|Un|Una|Uno)\s+"   # Italian article
    r"[a-zà-ú]"                              # followed by lowercase = new clause
    r")"
)

# --- Interleaved numbered paragraphs (numbers jump) --------------------
_NUMBERED_PARA = re.compile(r"\((\d+)\)")


@dataclass(slots=True)
class ColumnMergeConfig:
    """Thresholds for column merge detection."""

    max_merge_score: float = 0.08  # per-paragraph: signals / words
    max_corrupted_ratio: float = 0.40  # corrupted paragraphs / total
    min_paragraph_words: int = 10  # skip scoring very short paragraphs
    strip_gazette_lines: bool = True  # remove gazette headers regardless


@dataclass
class MergeStats:
    """Diagnostic stats for column merge cleaning."""

    paragraphs_total: int = 0
    paragraphs_corrupted: int = 0
    paragraphs_clean: int = 0
    gazette_lines_stripped: int = 0
    docs_in: int = 0
    docs_out: int = 0
    docs_rejected: int = 0


class ColumnMergeCleaner:
    """Detect and strip text corrupted by PDF two-column merge.

    When PDF extractors fail to detect a two-column layout, they
    read left-to-right across both columns, interleaving unrelated
    text. The result is syntactically broken:

        "visto il trattato che istituisce la Comunità europea, in
         partico- simultaneamente per garantire un'applicazione"

    This cleaner scores each paragraph for merge corruption signals
    and strips corrupted paragraphs while preserving clean ones.

    Signals detected
    ────────────────
      1. Gazette page headers embedded inline (≈100% reliable)
      2. Structural legal markers displaced to mid-sentence
      3. Cross-column broken hyphenation (word- wrongcontinuation)
      4. Abrupt ALLCAPS headers appearing mid-sentence
      5. Interleaved numbered paragraph sequences
    """

    def __init__(self, cfg: ColumnMergeConfig | None = None) -> None:
        self.cfg = cfg or ColumnMergeConfig()
        self.stats = MergeStats()

    def clean(self, text: str) -> str:
        """Score paragraphs for column merge corruption.

        Returns text with corrupted paragraphs removed.  If the
        document is predominantly corrupted (above threshold),
        returns empty string to reject the whole document.
        """
        if not text or not text.strip():
            return ""

        self.stats.docs_in += 1

        # ── 0. Strip inline gazette page headers regardless ────────
        if self.cfg.strip_gazette_lines:
            text, n_stripped = self._strip_gazette_lines(text)
            self.stats.gazette_lines_stripped += n_stripped

        # ── 1. Split into paragraphs (double newline or section) ───
        paragraphs = re.split(r"\n\s*\n", text)
        if not paragraphs:
            return ""

        clean_paras: list[str] = []
        n_scored = 0
        n_corrupted = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            words = para.split()
            n_words = len(words)

            # Very short paragraphs: keep as-is (not enough signal)
            if n_words < self.cfg.min_paragraph_words:
                clean_paras.append(para)
                self.stats.paragraphs_total += 1
                self.stats.paragraphs_clean += 1
                continue

            # ── Score for merge corruption ──────────────────────
            score = self._score_paragraph(para, n_words)
            self.stats.paragraphs_total += 1
            n_scored += 1

            if score <= self.cfg.max_merge_score:
                clean_paras.append(para)
                self.stats.paragraphs_clean += 1
            else:
                n_corrupted += 1
                self.stats.paragraphs_corrupted += 1

        # ── 2. Check if document is predominantly corrupted ────────
        if n_scored > 0:
            local_ratio = n_corrupted / n_scored

            if local_ratio > self.cfg.max_corrupted_ratio and len(clean_paras) < 2:
                self.stats.docs_rejected += 1
                return ""

        if not clean_paras:
            self.stats.docs_rejected += 1
            return ""

        self.stats.docs_out += 1
        return "\n\n".join(clean_paras)

    def _score_paragraph(self, text: str, n_words: int) -> float:
        """Compute merge corruption score for a single paragraph.

        Returns score in [0, ∞) — higher = more corrupted.
        Normalized by word count so longer paragraphs aren't penalized.

        Signals (weight):
          1. Gazette headers inline          (5.0 each)
          2. Displaced structural markers     (3.0 each)
          3. Cross-column broken hyphenation  (3.0 each)
          4. ALLCAPS header mid-sentence      (2.0 each)
          5. Interleaved numbered paragraphs  (2.0 per jump)
          6. Sudden capitalized clause        (2.0 each)
          7. Orphan word fragments            (1.5 each)
        """
        signals = 0.0

        # ── Signal 1: Gazette headers inline ───────────────────────
        gazette_hits = len(_GAZETTE_INLINE.findall(text))
        signals += gazette_hits * 5.0

        # ── Signal 2: Displaced structural markers ─────────────────
        displaced_hits = len(_DISPLACED_MARKER.findall(text))
        signals += displaced_hits * 3.0

        # ── Signal 3: Cross-column broken hyphenation ──────────────
        for match in _BROKEN_COL_HYPHEN.finditer(text):
            prefix = match.group(1).lower()
            continuation = match.group(2).lower()

            # Skip legitimate hyphen prefixes
            if prefix in _LEGIT_HYPHEN_PREFIXES:
                continue

            # Skip known legitimate compounds
            if prefix + continuation in _LEGIT_COMPOUNDS:
                continue

            # Column break: prefix is NOT a common word ending
            # and the combined form doesn't look like a real word
            if not prefix.endswith(("zione", "mente", "enza", "anza", "enza", "ibile")):
                signals += 3.0

        # ── Signal 4: ALLCAPS header mid-sentence ──────────────────
        allcaps_hits = len(_ALLCAPS_MID.findall(text))
        signals += allcaps_hits * 2.0

        # ── Signal 5: Interleaved numbered paragraphs ──────────────
        numbers = [int(m.group(1)) for m in _NUMBERED_PARA.finditer(text)]
        if len(numbers) >= 3:
            jumps = sum(
                1 for i in range(1, len(numbers))
                if abs(numbers[i] - numbers[i - 1]) > 2
            )
            signals += jumps * 2.0

        # ── Signal 6: Sudden capitalized clause mid-sentence ───────
        # "europea e la Il presidente del Consiglio è autorizzato"
        sudden_caps = len(_SUDDEN_CAPITAL_CLAUSE.findall(text))
        signals += sudden_caps * 2.0

        # ── Signal 7: Orphan word fragments ────────────────────────
        # Words that look like truncated fragments from column breaks
        # e.g. "impe", "gnare", "partico", "esclu" appearing as tokens
        words = text.split()
        for w in words:
            clean_w = w.strip(".,;:()\"'")
            if (
                2 <= len(clean_w) <= 6
                and clean_w.isalpha()
                and clean_w.lower() not in _COMMON_IT_SHORT
                and not clean_w[0].isupper()  # skip proper nouns / sentence starts
                and clean_w.endswith(("-", ))  # trailing hyphen = obvious fragment
            ):
                signals += 1.5

        # Also check for fused words across column boundary
        # "d'Ame-visto", "rica;particolare", "alvisto"
        # Look for word;word or word-Capitalword patterns
        fused_count = 0
        for m in re.finditer(r"[a-zà-ú];[a-zà-ú]", text):
            fused_count += 1
        for m in re.finditer(r"[a-zà-ú]-[A-ZÀ-Ú][a-zà-ú]", text):
            # "Ame-visto" — hyphen joining unrelated words
            before = text[max(0, m.start() - 5):m.start() + 1]
            if "'" in before or before.strip().startswith(("l'", "d'", "un'")):
                fused_count += 1
        signals += fused_count * 2.5

        return signals / max(n_words, 1)

    @staticmethod
    def _strip_gazette_lines(text: str) -> tuple[str, int]:
        """Remove standalone Gazette page header/footer lines.

        These are always noise in extracted EU PDFs:
          "10.8.2010 IT Gazzetta ufficiale dell'Unione europea L 209/3"
          "L 396/46 IT Gazzetta ufficiale dell'Unione europea 31.12.2004"
          "GU L del 28.5.2025 IT"
        """
        gazette_line = re.compile(
            r"^\s*"
            r"(?:"
            r"(?:\d{1,2}\.\d{1,2}\.\d{4}\s+)?IT\s+Gazzetta\s+ufficiale"
            r"|L\s+\d{1,4}/\d{1,4}\s+IT\s+Gazzetta\s+ufficiale"
            r"|GU\s+L\s+del\s+\d{1,2}\.\d{1,2}\.\d{4}\s+IT"
            r")"
            r".*$",
            re.MULTILINE | re.IGNORECASE,
        )
        result, count = gazette_line.subn("", text)
        return result, count

    def summary(self) -> dict[str, int | float]:
        """Return stats for reporting."""
        s = self.stats
        return {
            "paragraphs_total": s.paragraphs_total,
            "paragraphs_corrupted": s.paragraphs_corrupted,
            "paragraphs_clean": s.paragraphs_clean,
            "gazette_lines_stripped": s.gazette_lines_stripped,
            "docs_in": s.docs_in,
            "docs_out": s.docs_out,
            "docs_rejected": s.docs_rejected,
        }


# ══════════════════════════════════════════════════════════════════════
#  REPETITION SCORER
# ══════════════════════════════════════════════════════════════════════


class RepetitionScorer:
    """Detect documents with excessive internal repetition.

    Uses a sliding-window word n-gram approach: if too many n-grams
    appear more than once, the document is considered repetitive.
    This catches patterns that zlib compression ratio misses
    (e.g. paraphrased repetition, template-generated text).
    """

    def __init__(self, ngram_size: int = 10) -> None:
        self.ngram_size = ngram_size

    def score(self, text: str) -> float:
        """Return repetition ratio in [0, 1].

        0 = no repeated n-grams, 1 = fully repeated.
        """
        words = text.lower().split()
        n = self.ngram_size

        if len(words) < n:
            return 0.0

        ngrams: list[str] = []
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i : i + n]))

        if not ngrams:
            return 0.0

        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return repeated / len(ngrams)

    def is_repetitive(self, text: str, threshold: float = 0.20) -> bool:
        """Check if text exceeds repetition threshold."""
        return self.score(text) > threshold
