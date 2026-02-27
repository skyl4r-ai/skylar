#!/usr/bin/env python3
# filepath: corpus_diagnostic.py
"""
Corpus diagnostic tool for NanoTransformer pretraining data.

Analyzes:
- Document count and distribution by source
- Metadata overhead (bytes wasted on headers vs actual content)
- Token-level statistics per source
- Detects imbalanced corpus and suggests resampling weights

Usage:
    python corpus_diagnostic.py --data_dir ./data/pretrain/
    python corpus_diagnostic.py --file ./data/pretrain/train.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentStats:
    """Statistics for a single document."""
    source: str
    title: str
    total_chars: int
    metadata_chars: int
    content_chars: int
    estimated_tokens: int  # rough: chars / 4 for Italian


@dataclass
class SourceBucket:
    """Aggregated statistics for a source type."""
    count: int = 0
    total_chars: int = 0
    metadata_chars: int = 0
    content_chars: int = 0
    estimated_tokens: int = 0
    titles: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_METADATA_BLOCK_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.MULTILINE | re.DOTALL,
)

_YAML_KV_RE = re.compile(r"^(\w+)\s*:\s*(.*)$", re.MULTILINE)

_BOS = "<bos>"
_EOS = "<eos>"

# ChatML tokens considered structural (not content)
_CHATML_STRUCTURAL = re.compile(
    r"<\|im_start\|>(?:system|user|assistant)\n|<\|im_end\|>"
)


def parse_metadata(text: str) -> tuple[dict[str, str], int]:
    """Extract YAML-like metadata block and return (metadata_dict, chars_consumed)."""
    match = _METADATA_BLOCK_RE.match(text)
    if not match:
        return {}, 0

    block = match.group(1)
    metadata: dict[str, str] = {}
    for kv_match in _YAML_KV_RE.finditer(block):
        metadata[kv_match.group(1).strip()] = kv_match.group(2).strip()

    return metadata, match.end()


def estimate_tokens(char_count: int) -> int:
    """Rough token estimate for Italian text (~4 chars/token for BPE)."""
    return max(1, char_count // 4)


def split_documents(stream: TextIO) -> list[str]:
    """Split a stream into documents delimited by <bos>...<eos>."""
    content = stream.read()
    documents: list[str] = []

    parts = content.split(_BOS)
    for part in parts:
        if not part.strip():
            continue

        eos_idx = part.find(_EOS)
        if eos_idx != -1:
            doc_text = part[:eos_idx].strip()
        else:
            doc_text = part.strip()

        if doc_text:
            documents.append(doc_text)

    return documents


def analyze_document(raw: str) -> DocumentStats:
    """Analyze a single document for metadata overhead."""
    total_chars = len(raw)

    metadata, meta_end = parse_metadata(raw)
    source = metadata.get("source", "unknown")
    title = metadata.get("title", "(no title)")

    # Count metadata chars: the YAML block itself
    metadata_chars = meta_end

    # For chat documents, also count ChatML structural tokens as "overhead"
    content_text = raw[meta_end:]
    if source == "chat":
        structural_chars = sum(
            len(m.group()) for m in _CHATML_STRUCTURAL.finditer(content_text)
        )
        metadata_chars += structural_chars

    content_chars = total_chars - metadata_chars

    return DocumentStats(
        source=source,
        title=title,
        total_chars=total_chars,
        metadata_chars=metadata_chars,
        content_chars=content_chars,
        estimated_tokens=estimate_tokens(total_chars),
    )


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------

def aggregate(docs: list[DocumentStats]) -> dict[str, SourceBucket]:
    """Aggregate per-document stats into per-source buckets."""
    buckets: dict[str, SourceBucket] = defaultdict(SourceBucket)

    for doc in docs:
        b = buckets[doc.source]
        b.count += 1
        b.total_chars += doc.total_chars
        b.metadata_chars += doc.metadata_chars
        b.content_chars += doc.content_chars
        b.estimated_tokens += doc.estimated_tokens
        if len(b.titles) < 5:
            b.titles.append(doc.title)

    return dict(buckets)


def print_report(
    buckets: dict[str, SourceBucket],
    total_docs: int,
) -> None:
    """Print a human-readable diagnostic report."""
    grand_total_chars = sum(b.total_chars for b in buckets.values())
    grand_total_tokens = sum(b.estimated_tokens for b in buckets.values())
    grand_metadata_chars = sum(b.metadata_chars for b in buckets.values())

    print("\n" + "=" * 78)
    print("  CORPUS DIAGNOSTIC REPORT")
    print("=" * 78)

    print(f"\n  Total documents:        {total_docs:>12,}")
    print(f"  Total chars:            {grand_total_chars:>12,}")
    print(f"  Est. total tokens:      {grand_total_tokens:>12,}")
    print(f"  Metadata overhead:      {grand_metadata_chars:>12,} chars "
          f"({grand_metadata_chars / max(grand_total_chars, 1) * 100:.1f}%)")

    print("\n" + "-" * 78)
    print(f"  {'SOURCE':<20} {'DOCS':>8} {'%DOCS':>7} {'EST.TOKENS':>12} "
          f"{'%TOKENS':>8} {'META%':>7}")
    print("-" * 78)

    sorted_buckets = sorted(
        buckets.items(), key=lambda x: x[1].estimated_tokens, reverse=True
    )

    for source, b in sorted_buckets:
        doc_pct = b.count / max(total_docs, 1) * 100
        tok_pct = b.estimated_tokens / max(grand_total_tokens, 1) * 100
        meta_pct = b.metadata_chars / max(b.total_chars, 1) * 100

        print(f"  {source:<20} {b.count:>8,} {doc_pct:>6.1f}% "
              f"{b.estimated_tokens:>12,} {tok_pct:>7.1f}% {meta_pct:>6.1f}%")

    # -----------------------------------------------------------------------
    # Warnings
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  DIAGNOSTICS")
    print("=" * 78)

    warnings: list[str] = []

    # Check metadata overhead
    meta_pct_total = grand_metadata_chars / max(grand_total_chars, 1) * 100
    if meta_pct_total > 5:
        warnings.append(
            f"[CRITICAL] Metadata overhead is {meta_pct_total:.1f}% of corpus. "
            f"Strip metadata headers before training. "
            f"On 100M params this wastes significant capacity."
        )

    # Check imbalance
    if "chat" in buckets and grand_total_tokens > 0:
        chat_pct = buckets["chat"].estimated_tokens / grand_total_tokens * 100
        if chat_pct < 10:
            warnings.append(
                f"[CRITICAL] Chat data is only {chat_pct:.1f}% of tokens. "
                f"Model will default to dominant source pattern at inference. "
                f"Upsample chat to ≥15-20% or use a separate SFT stage with "
                f"proper loss masking."
            )

    # Check if any single source dominates
    for source, b in sorted_buckets:
        pct = b.estimated_tokens / max(grand_total_tokens, 1) * 100
        if pct > 70:
            warnings.append(
                f"[WARNING] Source '{source}' dominates at {pct:.1f}% of tokens. "
                f"Consider downsampling or mixing in more diverse data."
            )
            break

    # Check for EU/legal content specifically
    legal_keywords = {"regulation", "regolamento", "directive", "direttiva", "legal"}
    legal_tokens = 0
    for source, b in buckets.items():
        if any(kw in source.lower() for kw in legal_keywords):
            legal_tokens += b.estimated_tokens
        # Also check titles for legal content
        for title in b.titles:
            if any(kw in title.lower() for kw in legal_keywords):
                legal_tokens += b.estimated_tokens // b.count
    if legal_tokens > grand_total_tokens * 0.15:
        warnings.append(
            f"[WARNING] Estimated {legal_tokens:,} legal/regulatory tokens "
            f"({legal_tokens / max(grand_total_tokens, 1) * 100:.1f}%). "
            f"This repetitive, formulaic text gets memorized fast on small models."
        )

    if not warnings:
        print("\n  No critical issues detected.\n")
    else:
        for i, w in enumerate(warnings, 1):
            print(f"\n  {i}. {w}")
        print()

    # -----------------------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("  RECOMMENDATIONS")
    print("=" * 78)

    print("""
  1. STRIP METADATA from pretraining docs:
     Remove '---\\nsource: ...\\n---' headers, wiki_id, timestamp, categories.
     Keep only: '# Title\\n\\nContent...'
     This alone can recover 5-15% of model capacity.

  2. REBALANCE CORPUS:
     For a 100M model targeting chat, aim for:
       - Wiki/knowledge: 40-50%
       - Books/articles:  20-30%
       - Chat/dialogue:   15-25%  (critical!)
       - Code/JSON:        5-10%
     Currently chat is likely <5%.

  3. SFT STRATEGY for small models:
     - LR: 1e-5 to 5e-5 (NOT the pretraining LR)
     - Epochs: 3-5 max on 2.5k examples
     - Warmup: 10% of steps
     - Loss masking: compute loss ONLY on assistant tokens
     - Weight decay: 0.01-0.1

  4. PRETRAIN FIX (if re-pretraining):
     - Increase chat data in pretrain corpus to ≥15%
     - Strip all metadata
     - Use document-level shuffling per epoch
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze pretraining corpus distribution and metadata overhead",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data_dir",
        type=Path,
        help="Directory containing .txt training files with <bos>...<eos> docs",
    )
    group.add_argument(
        "--file",
        type=Path,
        help="Single training file with <bos>...<eos> docs",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional: save raw stats as JSON",
    )
    args = parser.parse_args()

    # Collect files
    if args.data_dir:
        files = sorted(args.data_dir.glob("*.txt"))
        if not files:
            log.error("no_txt_files", directory=str(args.data_dir))
            sys.exit(1)
        log.info("scanning_directory", directory=str(args.data_dir), file_count=len(files))
    else:
        if not args.file.exists():
            log.error("file_not_found", path=str(args.file))
            sys.exit(1)
        files = [args.file]

    # Parse all documents
    all_docs: list[DocumentStats] = []
    for fpath in files:
        log.info("parsing_file", path=str(fpath))
        with open(fpath, encoding="utf-8") as f:
            raw_docs = split_documents(f)
            for raw in raw_docs:
                all_docs.append(analyze_document(raw))

    if not all_docs:
        log.error("no_documents_found")
        sys.exit(1)

    log.info("analysis_complete", total_documents=len(all_docs))

    # Aggregate and report
    buckets = aggregate(all_docs)
    print_report(buckets, total_docs=len(all_docs))

    # Optional JSON export
    if args.output_json:
        export = {
            source: {
                "count": b.count,
                "total_chars": b.total_chars,
                "metadata_chars": b.metadata_chars,
                "content_chars": b.content_chars,
                "estimated_tokens": b.estimated_tokens,
                "metadata_pct": round(b.metadata_chars / max(b.total_chars, 1) * 100, 2),
                "sample_titles": b.titles,
            }
            for source, b in buckets.items()
        }
        args.output_json.write_text(json.dumps(export, indent=2, ensure_ascii=False))
        log.info("exported_json", path=str(args.output_json))


if __name__ == "__main__":
    main()