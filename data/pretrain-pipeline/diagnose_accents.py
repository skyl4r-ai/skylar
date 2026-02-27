#!/usr/bin/env python3
"""Diagnose where Italian accented characters are lost in the pipeline.

Usage:
    python diagnose_accents.py /path/to/problematic.pdf

Runs extraction at each stage and reports where accents disappear.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

ITALIAN_ACCENTED = set("àèéìòùÀÈÉÌÒÙ")


def count_accents(text: str) -> dict[str, int]:
    """Count occurrences of each Italian accented character."""
    counts = {}
    for ch in ITALIAN_ACCENTED:
        n = text.count(ch)
        if n:
            counts[ch] = n
    return counts


def show_context(text: str, char: str, n: int = 3) -> None:
    """Show first N occurrences of char in context."""
    shown = 0
    for i, ch in enumerate(text):
        if ch == char:
            start = max(0, i - 20)
            end = min(len(text), i + 20)
            ctx = text[start:end].replace("\n", "\\n")
            print(f"    ...{ctx}...")
            shown += 1
            if shown >= n:
                break


def find_missing_accent_spots(text: str) -> list[str]:
    """Find patterns that suggest a missing accent (double space, letter+space oddly)."""
    import re
    # "Comunit  europea" — double space where accent should be
    spots = re.findall(r"\w{3,}  \w{3,}", text)
    return spots[:10]


def report(label: str, text: str) -> None:
    """Print accent report for a pipeline stage."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {label}")
    print(f"{'='*60}")
    print(f"  Length: {len(text)} chars")

    accents = count_accents(text)
    if accents:
        print(f"  Accented chars found: {accents}")
    else:
        print("  ⚠ NO Italian accented characters found!")

    # Check for combining marks (NFD decomposed accents)
    combining = sum(1 for ch in text if unicodedata.category(ch) == "Mn")
    if combining:
        print(f"  Combining marks (Mn): {combining} — accents may be in decomposed form")

    # Check for suspicious double spaces (accent dropped)
    spots = find_missing_accent_spots(text)
    if spots:
        print(f"  ⚠ Suspicious double-space gaps ({len(spots)}):")
        for s in spots[:5]:
            print(f"    '{s}'")

    # Show first 200 chars
    preview = text[:300].replace("\n", "\\n")
    print(f"  Preview: {preview}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python diagnose_accents.py <file.pdf>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    # ── Stage 1: Raw PyMuPDF extraction ──────────────────────────
    print("\n" + "─" * 60)
    print("  PYMUPDF RAW EXTRACTION")
    print("─" * 60)
    try:
        import fitz
        doc = fitz.open(str(path))
        raw_pages = []
        for page in doc:
            raw_pages.append(page.get_text("text", sort=True))
        doc.close()
        raw_pymupdf = "\n\n".join(raw_pages)
        report("PyMuPDF raw get_text('text', sort=True)", raw_pymupdf)
    except Exception as e:
        print(f"  PyMuPDF failed: {e}")
        raw_pymupdf = ""

    # ── Stage 1b: PyMuPDF with rawdict (character-level) ─────────
    try:
        import fitz
        doc = fitz.open(str(path))
        page0 = doc[0]
        blocks = page0.get_text("rawdict")["blocks"]
        chars_sample = []
        for b in blocks:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    for ch in s.get("chars", []):
                        c = ch["c"]
                        if c.strip():
                            chars_sample.append(
                                f"  U+{ord(c):04X} ({unicodedata.category(c)}) '{c}'"
                            )
        doc.close()
        print(f"\n  rawdict sample (first page, first 30 chars):")
        for cs in chars_sample[:30]:
            print(cs)
    except Exception as e:
        print(f"  rawdict failed: {e}")

    # ── Stage 2: Raw Docling extraction ──────────────────────────
    print("\n" + "─" * 60)
    print("  DOCLING RAW EXTRACTION")
    print("─" * 60)
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions(do_ocr=False)
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            },
        )
        result = converter.convert(path, raises_on_error=False)
        raw_docling = result.document.export_to_markdown(
            image_placeholder="",
            escape_underscores=False,
        )
        report("Docling export_to_markdown (do_ocr=False)", raw_docling)
    except Exception as e:
        print(f"  Docling failed: {e}")
        raw_docling = ""

    # ── Stage 3: After TextNormalizer ────────────────────────────
    if raw_docling or raw_pymupdf:
        text = raw_docling or raw_pymupdf
        source = "Docling" if raw_docling else "PyMuPDF"

        print("\n" + "─" * 60)
        print("  AFTER TEXTNORMALIZER")
        print("─" * 60)
        try:
            from cleaners import TextNormalizer
            normalizer = TextNormalizer()
            cleaned = normalizer(text)
            report(f"TextNormalizer({source})", cleaned)
        except Exception as e:
            print(f"  TextNormalizer failed: {e}")

    # ── Stage 4: After ftfy alone ────────────────────────────────
    if raw_docling or raw_pymupdf:
        text = raw_docling or raw_pymupdf
        print("\n" + "─" * 60)
        print("  FTFY ISOLATED TEST")
        print("─" * 60)
        try:
            import ftfy
            fixed = ftfy.fix_text(text, normalization="NFKC")
            before = count_accents(text)
            after = count_accents(fixed)
            print(f"  Before ftfy: {before}")
            print(f"  After ftfy:  {after}")
            if before != after:
                print("  ⚠ ftfy CHANGED accent counts!")
            else:
                print("  ✓ ftfy preserved all accents")
        except Exception as e:
            print(f"  ftfy test failed: {e}")

    print("\n" + "═" * 60)
    print("  DIAGNOSIS COMPLETE")
    print("═" * 60)


if __name__ == "__main__":
    main()
