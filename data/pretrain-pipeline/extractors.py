"""
Format-specific document extractors for local corpus building.

Each extractor is a generator that yields ``(source_path, raw_text)``
tuples.  A single file may yield multiple documents (JSON arrays,
JSONL lines, multi-article XML, etc.).

Supported formats
─────────────────
  • PDF   — Docling (default, GPU-accelerated) or pymupdf (legacy fallback)
  • HTML  — trafilatura content extraction
  • JSON  — recursive key search, hash-deduped within file
  • JSONL — line-by-line recursive key search
  • TXT   — direct read
  • XML   — placeholder (returns raw text content)
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Generator

import xxhash

from cleaners import CleanerConfig, MinimalNormalizer, PDFCleaner, TextNormalizer

logger = logging.getLogger(__name__)

# Type alias
DocStream = Generator[tuple[str, str], None, None]

# ──────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
    ".jsonl": "jsonl",
    ".txt": "txt",
    ".xml": "xml",
}


def discover_files(source: Path) -> dict[str, list[Path]]:
    """Walk *source* and group files by format.

    Returns:
        Dict mapping format name → sorted list of file paths.
    """
    result: dict[str, list[Path]] = {}

    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        fmt = SUPPORTED_EXTENSIONS.get(ext)
        if fmt:
            result.setdefault(fmt, []).append(path)

    return result


# ══════════════════════════════════════════════════════════════════════
#  PDF EXTRACTOR
# ══════════════════════════════════════════════════════════════════════


# --- Page-level garbage / table detection ────────────────────────────
# After extracting text from a PDF page, check if it's tabular garbage
# before including it.  This prevents word-salad from complex table
# layouts (fishing quotas, form specifications, e-AD/e-DAS templates)
# from contaminating the corpus.

# Data type specification codes from EU form definitions
_DTYPE_CODE = re.compile(
    r"\b(?:an?\d|n\d|an?\.\.\d+|[0-9]{1,3}[xX])\b"
)

# ELI page reference line
_ELI_LINE = re.compile(
    r"ELI:\s*http://data\.europa\.eu/eli/"
)

# Stock / quota codes (HER/1/2-, SOL/24-C.)
_STOCK_CODE = re.compile(
    r"\b[A-Z]{2,5}/[\dA-Z]{1,6}[-.]?"
)


def _is_page_garbage(page_text: str) -> bool:
    """Detect if a PDF page's extracted text is table/form garbage.

    Returns True if the page should be SKIPPED (garbage).

    Signals checked (any one is sufficient for rejection):
      1. Consecutive duplicate words ratio > 15%
      2. Unique word ratio < 45% (extreme repetition)
      3. Data type codes present (an..35, n1, etc.) with low prose
      4. Very high stock code density
    """
    words = page_text.split()
    n = len(words)
    if n < 20:
        return False  # too short to judge

    # ── Signal 1: Consecutive duplicate words ─────────────────────
    # "BEL BEL BEL BEL" or "zona zona zona zone zone"
    consec_dupes = sum(
        1 for i in range(1, n)
        if words[i].lower() == words[i - 1].lower()
    )
    if consec_dupes / n > 0.15:
        return True

    # ── Signal 2: Unique word ratio (vocabulary diversity) ────────
    lower_words = [w.lower().strip(".,;:()\"'") for w in words]
    unique = len(set(w for w in lower_words if len(w) > 1))
    if unique / n < 0.30:
        return True

    # ── Signal 3: Data type codes (form spec garbage) ─────────────
    # "an..35", "n1", "a2", "9x" — these appear in e-AD/e-DAS specs
    dtype_hits = len(_DTYPE_CODE.findall(page_text))
    if dtype_hits >= 3:
        return True

    # ── Signal 4: Stock code density ──────────────────────────────
    # "HER/1/2-", "SOL/24-C.", "WHG/7X7A-C" — fishing quota tables
    stock_hits = len(_STOCK_CODE.findall(page_text))
    if stock_hits >= 5:
        return True

    # ── Signal 5: Single R/O/C/D tokens (form field markers) ─────
    single_struct = sum(
        1 for w in words
        if re.fullmatch(r'"?[ROCD]"?', w)
    )
    if single_struct / n > 0.05 and n > 30:
        return True

    return False


def _is_document_garbage(text: str) -> bool:
    """Detect if an entire document's text is garbage (tables/forms).

    Same signals as ``_is_page_garbage()`` but applied to the full
    document text produced by Docling.

    Returns True if the document should be SKIPPED (garbage).
    """
    words = text.split()
    n = len(words)
    if n < 20:
        return False

    # ── Signal 1: Consecutive duplicate words ─────────────────────
    consec_dupes = sum(
        1 for i in range(1, n)
        if words[i].lower() == words[i - 1].lower()
    )
    if consec_dupes / n > 0.15:
        return True

    # ── Signal 2: Unique word ratio (vocabulary diversity) ────────
    lower_words = [w.lower().strip(".,;:()\"'") for w in words]
    unique = len(set(w for w in lower_words if len(w) > 1))
    if unique / n < 0.30:
        return True

    # ── Signal 3: Data type codes (form spec garbage) ─────────────
    dtype_hits = len(_DTYPE_CODE.findall(text))
    if dtype_hits >= 3:
        return True

    # ── Signal 4: Stock code density ──────────────────────────────
    stock_hits = len(_STOCK_CODE.findall(text))
    if stock_hits >= 5:
        return True

    # ── Signal 5: Single R/O/C/D tokens (form field markers) ─────
    single_struct = sum(
        1 for w in words
        if re.fullmatch(r'"?[ROCD]"?', w)
    )
    if single_struct / n > 0.05 and n > 30:
        return True

    return False


class PDFExtractor:
    """Extract text from PDF files using pymupdf.

    Workflow per file
    ─────────────────
      1. Open with fitz, extract text per page.
      2. **Skip table/garbage pages** (form specs, quota tables, etc.).
      3. Run PDFCleaner (header/footer removal + paragraph merge).
      4. Yield one document per PDF.
    """

    def __init__(self, cfg: CleanerConfig | None = None) -> None:
        self.cleaner = PDFCleaner(cfg)
        self.pages_skipped: int = 0
        self.pages_total: int = 0

    def extract(self, path: Path) -> DocStream:
        """Yield ``(source, text)`` for a single PDF."""
        try:
            import fitz  # pymupdf
        except ImportError:
            logger.error("pymupdf not installed — skipping %s", path)
            return

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            logger.warning("Failed to open PDF %s: %s", path, exc)
            return

        pages: list[str] = []
        try:
            for page in doc:
                self.pages_total += 1
                # "text" sort mode gives natural reading order
                page_text = page.get_text("text", sort=True)
                if not page_text or not page_text.strip():
                    continue

                # ── Table / garbage detection ─────────────────────
                if _is_page_garbage(page_text):
                    self.pages_skipped += 1
                    logger.debug(
                        "Skipped garbage page %d in %s",
                        page.number + 1, path,
                    )
                    continue

                # ── Strip ELI page references ─────────────────────
                page_text = _ELI_LINE.sub("", page_text)

                pages.append(page_text)
        except Exception as exc:
            logger.warning("Error reading pages from %s: %s", path, exc)
            return
        finally:
            doc.close()

        if not pages:
            return

        text = self.cleaner.clean_pages(pages)
        if text and text.strip():
            yield str(path), text


# ══════════════════════════════════════════════════════════════════════
#  DOCLING PDF EXTRACTOR (default)
# ══════════════════════════════════════════════════════════════════════


# Latin Extended A/B characters (U+0100–U+02FF) — these appear in EUR-Lex PDFs
# as corrupted Cyrillic/Greek/non-Latin text due to broken CMap font encoding.
# The codepoints are NOT consistently offset from their correct values, so
# remapping is not possible.  We detect and strip lines with high density of
# these characters since the corpus is Italian-only.
_LATIN_EXTENDED = re.compile(r"[\u0100-\u02FF]")
_LATIN_EXT_THRESHOLD = 0.25  # if >25% of chars in a line are Latin Extended → drop


def _strip_corrupted_lines(text: str) -> str:
    """Remove lines dominated by Latin Extended characters (corrupted non-Latin).

    EUR-Lex PDF CMap encoding errors produce characters like ǟǺǼȀǽDzǻ
    instead of proper Cyrillic (Н, и, etc.).  Since the Italian corpus has
    no legitimate use for dense Latin Extended text, we drop such lines.
    """
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        # Count Latin Extended characters vs total non-whitespace
        n_ext = len(_LATIN_EXTENDED.findall(stripped))
        n_total = len(stripped.replace(" ", ""))
        if n_total > 0 and n_ext / n_total > _LATIN_EXT_THRESHOLD:
            continue  # drop this line
        kept.append(line)
    return "\n".join(kept)


class DoclingPDFExtractor:
    """Extract text from PDF files using Docling with GPU acceleration.

    Uses ``DocumentConverter`` for layout-aware extraction that preserves
    reading order in multi-column PDFs (EUR-Lex, etc.).

    Workflow
    ────────
      1. Convert PDFs with Docling via ``convert_all()`` (batch GPU pipeline).
      2. Check conversion status per doc — fallback to pymupdf on FAILURE.
      3. ``export_to_markdown()`` for rich content extraction
         (headers, tables, lists — all with markdown formatting).
      4. Strip image placeholders and HTML comments.
      5. Strip lines with corrupted Latin Extended characters (CMap errors).
      6. Apply document-level garbage detection.
      7. Yield one document per PDF.

    The converter is initialized lazily on the first call to avoid
    importing heavy dependencies (torch, docling) until actually needed.

    If Docling fails on a specific PDF, falls back to the legacy
    pymupdf extractor for that file.
    """

    # Markdown artifacts to clean from export_to_markdown() output:
    # - image placeholders: <!-- image -->
    # - escaped underscores (Docling default): \_
    _MD_IMAGE_PLACEHOLDER = re.compile(r"<!--\s*image\s*-->")

    def __init__(self, cleaner_cfg: CleanerConfig | None = None) -> None:
        self._converter = None
        self._cleaner_cfg = cleaner_cfg
        self._fallback: PDFExtractor | None = None
        self.docs_processed: int = 0
        self.docs_garbage: int = 0
        self.docs_fallback: int = 0

    def _init_converter(self) -> None:
        """Lazy-initialize Docling converter with GPU acceleration."""
        if self._converter is not None:
            return

        from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
        )

        accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CUDA
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        pipeline_options.generate_picture_images = False
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=DoclingParseDocumentBackend
                )
            }
        )

        logger.info("Docling PDF converter initialized")

    def _get_fallback(self) -> PDFExtractor:
        """Get or create the legacy pymupdf fallback extractor."""
        if self._fallback is None:
            self._fallback = PDFExtractor(self._cleaner_cfg)
        return self._fallback

    def _postprocess(self, result, path: Path) -> DocStream:
        """Post-process a single ConversionResult into document text."""
        from docling.datamodel.base_models import ConversionStatus

        # ── Check conversion status ───────────────────────────────
        if result.status in (ConversionStatus.FAILURE, ConversionStatus.SKIPPED):
            errors = "; ".join(str(e) for e in (result.errors or []))
            logger.warning(
                "Docling failed on %s (status=%s%s) — falling back to pymupdf",
                path.name, result.status.name,
                f": {errors}" if errors else "",
            )
            self.docs_fallback += 1
            yield from self._get_fallback().extract(path)
            return

        self.docs_processed += 1

        # ── Markdown export (headers, tables, lists — rich formatting) ──
        text = result.document.export_to_markdown(
            image_placeholder="",
            escape_underscores=False,
        )
        if not text or not text.strip():
            return

        # ── Clean residual markdown artifacts ──────────────────────
        text = self._MD_IMAGE_PLACEHOLDER.sub("", text)

        # ── Strip ELI page references ─────────────────────────────
        text = _ELI_LINE.sub("", text)

        # ── Strip corrupted Latin Extended lines (CMap errors) ──
        text = _strip_corrupted_lines(text)
        if not text or not text.strip():
            return

        # ── Document-level garbage detection ──────────────────────
        if _is_document_garbage(text):
            self.docs_garbage += 1
            logger.debug("Skipped garbage document %s", path)
            return

        if text and text.strip():
            yield str(path), text

    def extract(self, path: Path) -> DocStream:
        """Yield ``(source, text)`` for a single PDF (non-batch fallback)."""
        try:
            self._init_converter()
        except ImportError as exc:
            logger.error("docling not installed — skipping %s: %s", path, exc)
            return
        except Exception as exc:
            logger.warning("Failed to init Docling converter: %s", exc)
            return

        try:
            result = self._converter.convert(path, raises_on_error=False)
        except Exception as exc:
            logger.warning(
                "Docling crashed on %s: %s — falling back to pymupdf",
                path.name, exc,
            )
            self.docs_fallback += 1
            yield from self._get_fallback().extract(path)
            return

        yield from self._postprocess(result, path)

    def extract_batch(self, paths: list[Path]) -> DocStream:
        """Convert multiple PDFs in one ``convert_all()`` call (batch GPU).

        Docling's ``convert_all()`` optimizes GPU utilisation by batching
        the layout analysis across multiple documents.  Individual failures
        are caught per-document and fall back to pymupdf.
        """
        if not paths:
            return

        try:
            self._init_converter()
        except ImportError as exc:
            logger.error("docling not installed — skipping %d PDFs: %s", len(paths), exc)
            return
        except Exception as exc:
            logger.warning("Failed to init Docling converter: %s", exc)
            return

        try:
            results_iter = self._converter.convert_all(
                paths,
                raises_on_error=False,
            )
        except Exception as exc:
            logger.warning(
                "Docling convert_all crashed: %s — falling back to pymupdf per-file",
                exc,
            )
            for p in paths:
                self.docs_fallback += 1
                yield from self._get_fallback().extract(p)
            return

        for result in results_iter:
            # Resolve the original path from the conversion result
            doc_path = Path(result.input.file)
            try:
                yield from self._postprocess(result, doc_path)
            except Exception as exc:
                logger.warning(
                    "Error post-processing %s: %s — falling back to pymupdf",
                    doc_path.name, exc,
                )
                self.docs_fallback += 1
                yield from self._get_fallback().extract(doc_path)


# ══════════════════════════════════════════════════════════════════════
#  HTML EXTRACTOR
# ══════════════════════════════════════════════════════════════════════


class HTMLExtractor:
    """Extract main content from HTML using trafilatura.

    trafilatura handles boilerplate removal (navigation, footers, ads,
    cookie banners) out of the box, returning clean prose.
    """

    def extract(self, path: Path) -> DocStream:
        try:
            import trafilatura
        except ImportError:
            logger.error("trafilatura not installed — skipping %s", path)
            return

        try:
            raw_html = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Failed to read HTML %s: %s", path, exc)
            return

        if not raw_html.strip():
            return

        try:
            text = trafilatura.extract(
                raw_html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=True,
                deduplicate=True,
            )
        except Exception as exc:
            logger.warning("trafilatura failed on %s: %s", path, exc)
            return

        if text and text.strip():
            yield str(path), text


# ══════════════════════════════════════════════════════════════════════
#  DOCLING HTML EXTRACTOR
# ══════════════════════════════════════════════════════════════════════


class DoclingHTMLExtractor:
    """Extract content from HTML files using Docling.

    Uses Docling's ``DocumentConverter`` with ``InputFormat.HTML`` for
    layout-aware content extraction.  Falls back to trafilatura
    (``HTMLExtractor``) if Docling is not installed or conversion fails.

    The converter is initialized lazily on first use.
    """

    _MD_IMAGE_PLACEHOLDER = re.compile(r"<!--\s*image\s*-->")

    def __init__(self) -> None:
        self._converter = None
        self._fallback: HTMLExtractor | None = None
        self.docs_processed: int = 0
        self.docs_fallback: int = 0

    def _init_converter(self) -> None:
        """Lazy-initialize Docling converter for HTML."""
        if self._converter is not None:
            return

        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat

        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.HTML],
        )
        logger.info("Docling HTML converter initialized")

    def _get_fallback(self) -> HTMLExtractor:
        """Get or create the trafilatura fallback extractor."""
        if self._fallback is None:
            self._fallback = HTMLExtractor()
        return self._fallback

    def extract(self, path: Path) -> DocStream:
        """Yield ``(source, text)`` for a single HTML file."""
        try:
            self._init_converter()
        except ImportError as exc:
            logger.warning(
                "docling not installed — falling back to trafilatura for %s: %s",
                path, exc,
            )
            yield from self._get_fallback().extract(path)
            return

        try:
            result = self._converter.convert(path, raises_on_error=False)
        except Exception as exc:
            logger.warning(
                "Docling failed on HTML %s: %s — falling back to trafilatura",
                path.name, exc,
            )
            self.docs_fallback += 1
            yield from self._get_fallback().extract(path)
            return

        from docling.datamodel.base_models import ConversionStatus

        if result.status in (ConversionStatus.FAILURE, ConversionStatus.SKIPPED):
            errors = "; ".join(str(e) for e in (result.errors or []))
            logger.warning(
                "Docling failed on %s (status=%s%s) — falling back to trafilatura",
                path.name, result.status.name,
                f": {errors}" if errors else "",
            )
            self.docs_fallback += 1
            yield from self._get_fallback().extract(path)
            return

        self.docs_processed += 1

        text = result.document.export_to_markdown(
            image_placeholder="",
            escape_underscores=False,
        )
        if not text or not text.strip():
            return

        # Clean residual markdown artifacts
        text = self._MD_IMAGE_PLACEHOLDER.sub("", text)

        if text and text.strip():
            yield str(path), text


# ══════════════════════════════════════════════════════════════════════
#  JSON / JSONL EXTRACTORS
# ══════════════════════════════════════════════════════════════════════


def _recursive_key_search(
        obj: object,
        keys: set[str],
) -> list[tuple[str, str]]:
    """Recursively find all string values at matching keys.

    Returns:
        List of ``(matched_key, value)`` tuples.
    """
    results: list[tuple[str, str]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                if isinstance(v, str) and v.strip():
                    results.append((k, v.strip()))
                elif isinstance(v, list):
                    # Array of strings under a matching key
                    for item in v:
                        if isinstance(item, str) and item.strip():
                            results.append((k, item.strip()))
                        elif isinstance(item, dict):
                            # Recurse into objects in the array
                            results.extend(_recursive_key_search(item, keys))
            else:
                # Recurse into non-matching keys
                results.extend(_recursive_key_search(v, keys))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_recursive_key_search(item, keys))

    return results


def _dedup_by_hash(pairs: list[tuple[str, str]]) -> list[str]:
    """Deduplicate extracted texts by xxhash within a single file.

    If multiple keys point to the same content (same hash),
    keep only one copy.
    """
    seen: set[int] = set()
    unique: list[str] = []

    for _key, text in pairs:
        h = xxhash.xxh64_intdigest(text.encode("utf-8"))
        if h not in seen:
            seen.add(h)
            unique.append(text)

    return unique


class JSONExtractor:
    """Extract documents from JSON files via recursive key search.

    Given search keys ``["text", "testo", "documento"]``, walks the
    entire JSON tree and extracts every string value found at a matching
    key.  Hash-deduplicates fragments within the file, then concatenates
    all unique fragments into **one document per file** (joined with
    ``"; "``).

    Rationale: a single JSON file with 10 "text" keys scattered across
    nested objects represents ONE logical document — those keys are
    different fields/sections of the same source, not separate articles.

    1 JSON file  → 1 document  (all unique text fragments joined)
    1000 JSON files → 1000 documents
    """

    def __init__(self, search_keys: list[str], separator: str = "\n") -> None:
        self._keys: set[str] = set(search_keys)
        self._sep = separator

    def extract(self, path: Path) -> DocStream:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in %s: %s", path, exc)
            return
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return

        pairs = _recursive_key_search(data, self._keys)
        if not pairs:
            return

        unique_texts = _dedup_by_hash(pairs)
        if not unique_texts:
            return

        merged = self._sep.join(unique_texts)
        yield str(path), merged


class JSONLExtractor:
    """Extract documents from JSONL (one JSON object per line).

    Same recursive key search as JSONExtractor, applied per line.
    All unique text fragments within a single line are concatenated
    into **one document per line** (joined with newline).

    1 JSONL line with 3 "text" keys → 1 document
    500-line JSONL file → up to 500 documents
    """

    def __init__(self, search_keys: list[str], separator: str = "\n") -> None:
        self._keys: set[str] = set(search_keys)
        self._sep = separator

    def extract(self, path: Path) -> DocStream:
        try:
            fh = path.open("r", encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Failed to open %s: %s", path, exc)
            return

        with fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Bad JSON at %s:%d — skipping", path, line_no)
                    continue

                pairs = _recursive_key_search(data, self._keys)
                if not pairs:
                    continue

                unique_texts = _dedup_by_hash(pairs)
                if not unique_texts:
                    continue

                merged = self._sep.join(unique_texts)
                yield str(path), merged


# ══════════════════════════════════════════════════════════════════════
#  TXT EXTRACTOR
# ══════════════════════════════════════════════════════════════════════


class TXTExtractor:
    """Read plain text files — one document per file."""

    def extract(self, path: Path) -> DocStream:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return

        if text and text.strip():
            yield str(path), text


# ══════════════════════════════════════════════════════════════════════
#  XML EXTRACTOR (basic)
# ══════════════════════════════════════════════════════════════════════


class XMLExtractor:
    """Basic XML text extraction via lxml.

    Extracts all text content from the XML tree, stripping tags.
    For Wikipedia dumps this gets article text; for other XML it
    concatenates all text nodes.
    """

    def extract(self, path: Path) -> DocStream:
        try:
            from lxml import etree
        except ImportError:
            logger.error("lxml not installed — skipping %s", path)
            return

        try:
            # Use iterparse for memory efficiency on large XML
            texts: list[str] = []
            for _event, elem in etree.iterparse(str(path), events=("end",)):
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())
                if elem.tail and elem.tail.strip():
                    texts.append(elem.tail.strip())
                elem.clear()

            if texts:
                full_text = "\n".join(texts)
                if full_text.strip():
                    yield str(path), full_text

        except Exception as exc:
            logger.warning("Failed to parse XML %s: %s", path, exc)


# ══════════════════════════════════════════════════════════════════════
#  EXTRACTOR REGISTRY
# ══════════════════════════════════════════════════════════════════════


class ExtractorRegistry:
    """Dispatches files to the appropriate extractor by format.

    Parameters
    ----------
    docling_formats : list[str] | None
        Formats to process with Docling (e.g. ``["pdf", "html"]``).
        Defaults to ``["pdf", "html"]``.  Pass an empty list to disable
        Docling entirely (legacy pymupdf + trafilatura).
    """

    def __init__(
            self,
            search_keys: list[str],
            cleaner_cfg: CleanerConfig | None = None,
            skip_xml: bool = True,
            skip_html: bool = False,
            json_separator: str = "\n",
            docling_formats: list[str] | None = None,
    ) -> None:
        self._normalizer = MinimalNormalizer()

        # ── Resolve Docling formats ────────────────────────────────
        if docling_formats is not None:
            self._docling_formats: set[str] = set(docling_formats)
        else:
            self._docling_formats = {"pdf", "html"}  # default

        # Backward-compat property used by phase1 batch processing
        self._use_docling = "pdf" in self._docling_formats

        # ── PDF extractor ──────────────────────────────────────────
        if "pdf" in self._docling_formats:
            pdf_extractor = DoclingPDFExtractor(cleaner_cfg)
        else:
            pdf_extractor = PDFExtractor(cleaner_cfg)

        self._extractors: dict[str, object] = {
            "pdf": pdf_extractor,
            "txt": TXTExtractor(),
            "json": JSONExtractor(search_keys, separator=json_separator),
            "jsonl": JSONLExtractor(search_keys, separator=json_separator),
        }

        # ── HTML extractor ─────────────────────────────────────────
        if not skip_html:
            if "html" in self._docling_formats:
                self._extractors["html"] = DoclingHTMLExtractor()
            else:
                self._extractors["html"] = HTMLExtractor()

        if not skip_xml:
            self._extractors["xml"] = XMLExtractor()

        self._skip_formats: set[str] = set()
        if skip_xml:
            self._skip_formats.add("xml")
        if skip_html:
            self._skip_formats.add("html")

    def extract_and_normalize(
            self,
            fmt: str,
            path: Path,
    ) -> Generator[tuple[str, str], None, None]:
        """Extract documents from *path*, normalize text, yield results.

        Yields:
            ``(source_path_str, normalized_text)`` tuples.
        """
        if fmt in self._skip_formats:
            return

        extractor = self._extractors.get(fmt)
        if extractor is None:
            logger.debug("No extractor for format %r — skipping %s", fmt, path)
            return

        for source, raw_text in extractor.extract(path):
            cleaned = self._normalizer(raw_text)
            if cleaned and len(cleaned.strip()) > 0:
                yield source, cleaned

    def extract_and_normalize_batch(
            self,
            fmt: str,
            paths: list[Path],
    ) -> Generator[tuple[str, str], None, None]:
        """Batch extract + normalize for formats that support it (PDF with Docling).

        For PDF with Docling, uses ``convert_all()`` to batch GPU processing.
        For other formats or legacy pymupdf, falls back to per-file extraction.

        Yields:
            ``(source_path_str, normalized_text)`` tuples.
        """
        if fmt in self._skip_formats:
            return

        extractor = self._extractors.get(fmt)
        if extractor is None:
            return

        # Batch path: Docling PDF
        if fmt == "pdf" and isinstance(extractor, DoclingPDFExtractor):
            for source, raw_text in extractor.extract_batch(paths):
                cleaned = self._normalizer(raw_text)
                if cleaned and len(cleaned.strip()) > 0:
                    yield source, cleaned
            return

        # Fallback: per-file extraction
        for path in paths:
            yield from self.extract_and_normalize(fmt, path)

    @property
    def pdf_stats(self) -> tuple[int, int, int]:
        """Return ``(total, skipped/garbage, fallback)`` from PDF extractor.

        For legacy PDFExtractor: ``(pages_total, pages_skipped, 0)``.
        For DoclingPDFExtractor: ``(docs_processed, docs_garbage, docs_fallback)``.
        """
        pdf_ext = self._extractors.get("pdf")
        if isinstance(pdf_ext, PDFExtractor):
            return pdf_ext.pages_total, pdf_ext.pages_skipped, 0
        if isinstance(pdf_ext, DoclingPDFExtractor):
            return pdf_ext.docs_processed, pdf_ext.docs_garbage, pdf_ext.docs_fallback
        return 0, 0, 0
