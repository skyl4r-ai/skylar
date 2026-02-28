"""
Incremental processing journal for the pre-training pipeline.

Tracks which source files have already been processed via content hashing
(xxhash64). On subsequent runs, unchanged files are loaded from cache
instead of being re-extracted and re-filtered.

Cache layout
────────────
  {output}/.cache/journal.json          — master index
  {output}/.cache/docs/{cache_id}.jsonl  — one file per source, each line
                                           {"text": "...", "source": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import xxhash

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_HASH_CHUNK_SIZE = 64 * 1024  # 64 KB


def hash_file(path: Path) -> str:
    """Compute xxhash64 hex digest of a file using streaming reads."""
    h = xxhash.xxh64()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Journal entry
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class JournalEntry:
    """Single entry in the journal — one per source file."""

    hash: str
    cache_id: str
    n_docs: int
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "hash": self.hash,
            "cache_id": self.cache_id,
            "n_docs": self.n_docs,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> JournalEntry:
        return cls(
            hash=d["hash"],
            cache_id=d["cache_id"],
            n_docs=d["n_docs"],
            timestamp=d["timestamp"],
        )


# ──────────────────────────────────────────────────────────────────────
# Journal
# ──────────────────────────────────────────────────────────────────────


class Journal:
    """Manages the incremental processing cache.

    Parameters
    ----------
    output_dir : Path
        Root output directory.  Cache lives under ``{output_dir}/.cache/``.
    enabled : bool
        If False, the journal is a no-op (all lookups miss, nothing saved).
    """

    def __init__(self, output_dir: Path, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._cache_dir = output_dir / ".cache"
        self._docs_dir = self._cache_dir / "docs"
        self._journal_path = self._cache_dir / "journal.json"
        self._entries: dict[str, JournalEntry] = {}  # abs_path → entry

        if self.enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._docs_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    # ── Load / save ────────────────────────────────────────────────

    def _load(self) -> None:
        """Load journal from disk if it exists."""
        if not self._journal_path.exists():
            return
        try:
            data = json.loads(self._journal_path.read_text(encoding="utf-8"))
            for abs_path, entry_dict in data.items():
                self._entries[abs_path] = JournalEntry.from_dict(entry_dict)
            logger.info("Journal loaded: %d entries", len(self._entries))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Corrupt journal, starting fresh: %s", exc)
            self._entries.clear()

    def save(self) -> None:
        """Atomically persist the journal (write .tmp then os.replace)."""
        if not self.enabled:
            return
        data = {path: entry.to_dict() for path, entry in self._entries.items()}
        tmp_path = self._journal_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self._journal_path)

    # ── Lookup ─────────────────────────────────────────────────────

    def lookup(self, file_path: Path, file_hash: str) -> list[tuple[str, str]] | None:
        """Check if a source file is cached and unchanged.

        Returns
        -------
        list of (text, source) tuples if cache hit, None if miss.
        """
        if not self.enabled:
            return None

        abs_key = str(file_path.resolve())
        entry = self._entries.get(abs_key)
        if entry is None or entry.hash != file_hash:
            return None

        # Load cached documents
        cache_path = self._docs_dir / f"{entry.cache_id}.jsonl"
        if not cache_path.exists():
            # Cache file missing — treat as miss
            del self._entries[abs_key]
            return None

        docs: list[tuple[str, str]] = []
        try:
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                docs.append((obj["text"], obj["source"]))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Corrupt cache for %s: %s", file_path.name, exc)
            del self._entries[abs_key]
            return None

        return docs

    # ── Store ──────────────────────────────────────────────────────

    def store(
        self,
        file_path: Path,
        file_hash: str,
        docs: list[tuple[str, str]],
    ) -> None:
        """Cache processed documents for a source file and save journal.

        Parameters
        ----------
        file_path : Path
            Original source file.
        file_hash : str
            xxhash64 hex digest of the source file.
        docs : list of (text, source)
            Documents extracted and filtered from this source.
        """
        if not self.enabled:
            return

        abs_key = str(file_path.resolve())

        # Remove old cache file if entry exists
        old_entry = self._entries.get(abs_key)
        if old_entry is not None:
            old_cache = self._docs_dir / f"{old_entry.cache_id}.jsonl"
            old_cache.unlink(missing_ok=True)

        cache_id = uuid.uuid4().hex[:12]
        cache_path = self._docs_dir / f"{cache_id}.jsonl"

        # Write cache file
        lines: list[str] = []
        for text, source in docs:
            lines.append(json.dumps({"text": text, "source": source}, ensure_ascii=False))
        cache_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

        # Update journal entry
        self._entries[abs_key] = JournalEntry(
            hash=file_hash,
            cache_id=cache_id,
            n_docs=len(docs),
            timestamp=time.time(),
        )

        # Progressive save — persist after every file
        self.save()

    # ── Stats ──────────────────────────────────────────────────────

    @property
    def n_entries(self) -> int:
        return len(self._entries)
