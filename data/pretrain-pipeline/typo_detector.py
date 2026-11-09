# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Lightweight Italian typo / OCR error detector for corpus QA.

Does NOT correct anything — only logs suspicious words to a CSV
file for post-run human review.  Zero risk of corrupting good text.

Detection strategy
──────────────────
  1. Load a spell-checker backend (spaCy word vectors or spylls/hunspell).
  2. For each document, extract words that:
     - Are ≥ 5 chars (short words have too many false positives)
     - Are pure alpha (no digits, punctuation, codes)
     - Fail the spell-check (not recognized in any inflected form)
     - Are NOT uppercase (acronyms, legal codes)
     - Are NOT likely proper nouns (capitalized)
  3. Log suspicious words with source file and context to CSV.

The detector is optional — disabled by default, enabled with
``--typo-log`` CLI flag.

Backend priority
────────────────
  1. **spaCy it_core_news_lg** (already required for Presidio PII) —
     uses word vectors, ``vocab[word].is_oov`` handles all inflected forms.
  2. **spylls** (pure-Python hunspell) — reads system ``.dic`` + ``.aff``
     files and applies affix rules for proper morphological spell-checking.
  3. Disabled with warning if neither is available.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# ── Patterns to skip ─────────────────────────────────────────────────
_SKIP_WORD = re.compile(
    r"^(?:"
    r"[A-ZÀ-Ú]{2,}"          # ALLCAPS (acronyms)
    r"|[A-ZÀ-Ú][a-zà-ú]+$"  # Capitalized (proper nouns)
    r"|.*\d.*"                 # Contains digits
    r"|.{1,4}$"                # Too short
    r"|.{30,}$"                # Too long (URLs, codes)
    r")$"
)


# ── Spell-checker backends ───────────────────────────────────────────

class _SpellBackend(Protocol):
    """Common interface for spell-check backends."""
    def is_known(self, word: str) -> bool: ...


class _SpacyBackend:
    """Uses spaCy word vectors — is_oov handles all inflected forms."""

    def __init__(self) -> None:
        import spacy
        self._nlp = spacy.load(
            "it_core_news_lg",
            disable=["parser", "ner", "tagger", "lemmatizer"],
        )
        logger.info(
            "Typo detector: using spaCy it_core_news_lg word vectors "
            "(%d vectors)", len(self._nlp.vocab.vectors),
        )

    def is_known(self, word: str) -> bool:
        return not self._nlp.vocab[word.lower()].is_oov


class _SpyllsBackend:
    """Uses spylls (pure-Python hunspell) with .dic + .aff affix rules."""

    def __init__(self, dic_path: Path) -> None:
        from spylls.hunspell import Dictionary
        # spylls expects path without extension — it loads both .dic and .aff
        base = str(dic_path).removesuffix(".dic").removesuffix(".aff")
        self._dict = Dictionary.from_files(base)
        logger.info("Typo detector: using spylls hunspell from %s", base)

    def is_known(self, word: str) -> bool:
        return self._dict.lookup(word)


# ── Main detector ────────────────────────────────────────────────────

class TypoDetector:
    """Detect suspicious words and log them to CSV.

    Uses spaCy word vectors or spylls hunspell for morphology-aware
    spell-checking.  Words not recognized are logged with source and
    context for human review.
    """

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._backend: _SpellBackend | None = None
        self._writer = None
        self._fh = None
        self._total_checked: int = 0
        self._total_suspicious: int = 0
        self._disabled: bool = False

    def _load_backend(self) -> bool:
        """Try to load a spell-check backend. Returns True on success."""
        if self._backend is not None:
            return True

        # Strategy 1: spaCy (already required for Presidio PII detection)
        try:
            self._backend = _SpacyBackend()
            return True
        except Exception as exc:
            logger.debug("spaCy backend unavailable: %s", exc)

        # Strategy 2: spylls with system hunspell files
        hunspell_paths = [
            Path("/usr/share/hunspell/it_IT.dic"),
            Path("/usr/share/myspell/it_IT.dic"),
        ]
        for path in hunspell_paths:
            if path.exists():
                try:
                    self._backend = _SpyllsBackend(path)
                    return True
                except Exception as exc:
                    logger.debug("spylls backend failed for %s: %s", path, exc)

        logger.warning(
            "Typo detector: no spell-check backend available — disabled. "
            "Ensure spaCy it_core_news_lg is installed (required for Presidio) "
            "or install spylls + hunspell-it."
        )
        self._disabled = True
        return False

    def _ensure_writer(self) -> None:
        """Lazy-open CSV writer."""
        if self._writer is not None:
            return
        self._fh = open(self._output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["source", "word", "context"])

    def check(self, text: str, source: str = "") -> int:
        """Check text for suspicious words. Returns count of suspects.

        Does not modify the text in any way.
        """
        if self._disabled:
            return 0

        if not self._load_backend():
            return 0

        self._ensure_writer()

        suspects = 0
        words = text.split()
        self._total_checked += len(words)

        for i, raw_word in enumerate(words):
            # Strip punctuation from edges
            word = raw_word.strip(".,;:!?\"'«»()[]{}—–-…")

            if not word or not word.isalpha():
                continue

            if _SKIP_WORD.match(word):
                continue

            if self._backend.is_known(word):
                continue

            # Extract context (5 words around the suspect)
            start = max(0, i - 3)
            end = min(len(words), i + 4)
            context = " ".join(words[start:end])

            self._writer.writerow([source, word, context])
            suspects += 1

        self._total_suspicious += suspects
        return suspects

    @property
    def total_checked(self) -> int:
        return self._total_checked

    @property
    def total_suspicious(self) -> int:
        return self._total_suspicious

    @property
    def suspicious_ratio(self) -> float:
        if self._total_checked == 0:
            return 0.0
        return self._total_suspicious / self._total_checked

    def close(self) -> None:
        """Flush and close CSV file."""
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None
