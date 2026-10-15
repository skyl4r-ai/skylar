#!/usr/bin/env python3
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
# filepath: ted_processor.py
"""TED Europa data processor for Italian pretrain corpus.

Recursively extracts nested archives (tar.gz → tar.gz → zip),
parses TED notices in all historical formats, filters Italian content,
and produces a <bos>doc<eos> delimited pretrain.txt file.

KEY DESIGN DECISIONS (based on actual TED data):
  - Legacy TXT files: we ONLY process IT_* files (Italian language).
    All notices in an IT_* file are already in Italian — no CY filter needed.
  - XML files: we extract the Italian language form (LG="IT").
  - Notice separator in legacy: "1.00/067192" (float/integer pattern)
  - Field format: "TI: value\\n    continuation line"
"""

from __future__ import annotations

import gzip
import logging
import re
import shutil
import tarfile
import textwrap
import unicodedata
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from xml.etree import ElementTree as ET

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

try:
    import typer
except ImportError:
    import sys

    print("Missing: typer. Install with: pip install typer rich")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOS = "<bos>"
EOS = "<eos>"

# Notice separator: lines like "1.00/067192" or "1.0/002725"
NOTICE_SEP_RE = re.compile(r"^\d+\.\d+/\d+\s*$", re.MULTILINE)

# Field tag: 2-3 uppercase letters followed by colon at start of line
# e.g. "TI: F-Paris: lighting supports"
FIELD_TAG_RE = re.compile(r"^([A-Z]{2,3}):\s?(.*)$")

# Italian file patterns (case-insensitive)
IT_FILE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^IT[_-]", re.IGNORECASE),          # IT_19930102_... or IT-...
    re.compile(r"[/\\]IT[_-]", re.IGNORECASE),       # path/.../IT_...
    re.compile(r"^it_.*utf8.*org$", re.IGNORECASE),   # it_20100102_001_utf8_org
    re.compile(r"^IT_.*ISO.*ORG$", re.IGNORECASE),    # IT_19980102_1998001_ISO_ORG
]

# Fields we extract for the corpus
WANTED_FIELDS = {"TI", "AB", "TX"}

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger("ted_processor")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class FormatEra(str, Enum):
    LEGACY_TXT = "legacy_txt"
    XML_TED = "xml_ted"
    UNKNOWN = "unknown"


@dataclass
class ExtractedDoc:
    """A single extracted document for the pretrain corpus."""

    doc_id: str
    title: str = ""
    abstract: str = ""
    body: str = ""
    source_file: str = ""

    def to_text(self) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(self.title.strip())
        if self.abstract:
            parts.append(self.abstract.strip())
        if self.body:
            parts.append(self.body.strip())
        return "\n\n".join(parts)

    def is_empty(self) -> bool:
        return not any((self.title, self.abstract, self.body))


@dataclass
class ProcessingStats:
    """Track processing statistics."""

    packages_processed: int = 0
    files_scanned: int = 0
    it_files_found: int = 0
    docs_extracted: int = 0
    docs_empty: int = 0
    errors: int = 0
    skipped_non_it: int = 0
    formats: dict[str, int] = field(default_factory=dict)

    def inc_format(self, fmt: str) -> None:
        self.formats[fmt] = self.formats.get(fmt, 0) + 1


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace, remove control chars, collapse blank lines."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Collapse runs of whitespace (but preserve single newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def elem_text(elem: ET.Element | None) -> str:
    """Recursively extract all text from an XML element."""
    if elem is None:
        return ""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(elem_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Italian file detection
# ---------------------------------------------------------------------------

def is_italian_file(path: Path) -> bool:
    """Check if a file is an Italian language version by its name."""
    name = path.name
    return any(pat.search(name) for pat in IT_FILE_PATTERNS)


# ---------------------------------------------------------------------------
# Archive extraction – recursive
# ---------------------------------------------------------------------------

def extract_recursive(
    path: Path, dest: Path, *, depth: int = 0, max_depth: int = 5
) -> None:
    """Recursively extract tar.gz, gz, and zip archives."""
    if depth > max_depth:
        log.warning("Max extraction depth reached at %s", path)
        return

    try:
        with path.open("rb") as f:
            magic = f.read(4)
    except OSError:
        return

    try:
        if magic[:2] == b"\x1f\x8b":
            # GZIP — could be tar.gz or plain .gz
            try:
                with tarfile.open(str(path), "r:gz") as tf:
                    for member in tf.getmembers():
                        resolved = (dest / member.name).resolve()
                        if not str(resolved).startswith(str(dest.resolve())):
                            continue
                    tf.extractall(dest, filter="data")
            except tarfile.TarError:
                # Plain .gz, not tar
                out_path = dest / path.stem
                with gzip.open(str(path), "rb") as gz_in:
                    with out_path.open("wb") as f_out:
                        shutil.copyfileobj(gz_in, f_out)

        elif magic[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(path, "r") as zf:
                for member in zf.infolist():
                    resolved = (dest / member.filename).resolve()
                    if not str(resolved).startswith(str(dest.resolve())):
                        continue
                zf.extractall(dest)

        elif len(magic) >= 4:
            # Check tar by trying to open
            try:
                with tarfile.open(str(path), "r:*") as tf:
                    tf.extractall(dest, filter="data")
            except tarfile.TarError:
                return
        else:
            return
    except (tarfile.TarError, zipfile.BadZipFile, gzip.BadGzipFile, OSError) as exc:
        log.debug("Extract failed %s: %s", path.name, exc)
        return

    # Recurse into newly extracted archives
    for child in sorted(dest.rglob("*")):
        if child.is_file() and _looks_like_archive(child):
            child_dest = child.parent / child.stem.replace(".tar", "")
            child_dest.mkdir(parents=True, exist_ok=True)
            extract_recursive(child, child_dest, depth=depth + 1, max_depth=max_depth)
            child.unlink(missing_ok=True)


def _looks_like_archive(path: Path) -> bool:
    """Quick check if a file might be an archive."""
    suffix = "".join(path.suffixes).lower()
    if any(suffix.endswith(ext) for ext in (".tar.gz", ".tgz", ".gz", ".zip", ".tar")):
        return True
    try:
        with path.open("rb") as f:
            h = f.read(4)
        if h[:2] == b"\x1f\x8b" or h[:4] == b"PK\x03\x04":
            return True
    except OSError:
        pass
    return False


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(file_path: Path) -> FormatEra:
    """Detect TED notice format from file content."""
    try:
        with file_path.open("rb") as f:
            header = f.read(2048)
    except OSError:
        return FormatEra.UNKNOWN

    if b"<?xml" in header or b"<TED_EXPORT" in header:
        return FormatEra.XML_TED

    # Legacy TXT: look for the notice separator pattern or field tags
    try:
        text = header.decode("utf-8", errors="replace")
    except Exception:
        text = header.decode("latin-1", errors="replace")

    # Check for notice separator "1.00/067192"
    if re.search(r"^\d+\.\d+/\d+", text, re.MULTILINE):
        return FormatEra.LEGACY_TXT

    # Check for field tags at start of lines
    if re.search(r"^(TI|ND|CY|AU|TX|AB):\s", text, re.MULTILINE):
        return FormatEra.LEGACY_TXT

    return FormatEra.UNKNOWN


# ---------------------------------------------------------------------------
# Parser: Legacy TXT
# ---------------------------------------------------------------------------

def _parse_fields_from_block(block: str) -> dict[str, str]:
    """Parse a single notice block into field → value dict.

    Field format:
        TI: Some title text here
            continuation of title on indented lines
        AB:  Abstract text
            more abstract
        TX:  1.  Full body text...
            continuation

    A new field starts when a line matches ^[A-Z]{2,3}: at column 0.
    Everything else is continuation of the previous field.
    """
    fields: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in block.split("\n"):
        match = FIELD_TAG_RE.match(line)
        if match:
            # Save previous field
            if current_key is not None:
                fields[current_key] = "\n".join(current_lines)
            current_key = match.group(1)
            current_lines = [match.group(2)]
        elif current_key is not None:
            # Continuation line — strip leading indent
            current_lines.append(line.strip())

    # Save last field
    if current_key is not None:
        fields[current_key] = "\n".join(current_lines)

    return fields


def parse_legacy_txt(file_path: Path) -> list[ExtractedDoc]:
    """Parse a legacy TED notice file (1993–2005).

    File structure:
        *** TED DAILY-DELIVERY ***
        *** ( ITALIAN  - VERSION) ***

        1.00/067192
        TI: I-Napoli: pasti
        ...
        TX: 1. Ente appaltante...

        1.00/067191
        TI: I-Roma: lavori stradali
        ...

    Each file is a specific language version (IT_*, EN_*, FR_*).
    We process ONLY IT_* files, so all notices are in Italian.
    """
    docs: list[ExtractedDoc] = []

    # Try multiple encodings
    content: str | None = None
    for encoding in ("utf-8", "latin-1", "cp1252", "iso-8859-15"):
        try:
            content = file_path.read_text(encoding=encoding)
            break
        except (UnicodeDecodeError, OSError):
            continue

    if content is None:
        return docs

    # Split into notice blocks by separator pattern
    # The separator is a line like "1.00/067192"
    parts = NOTICE_SEP_RE.split(content)

    # First part is the header (*** TED DAILY-DELIVERY ***), skip it
    notice_blocks = parts[1:] if len(parts) > 1 else parts

    for block in notice_blocks:
        if not block.strip():
            continue

        fields = _parse_fields_from_block(block)

        if not fields:
            continue

        title = clean_text(fields.get("TI", ""))
        abstract = clean_text(fields.get("AB", ""))
        body = clean_text(fields.get("TX", ""))

        doc_id = fields.get("ND", file_path.stem)

        doc = ExtractedDoc(
            doc_id=doc_id.strip(),
            title=title,
            abstract=abstract,
            body=body,
            source_file=str(file_path),
        )

        if not doc.is_empty():
            docs.append(doc)

    return docs


# ---------------------------------------------------------------------------
# Parser: XML (2006+ TED_EXPORT R2.0.x and eForms)
# ---------------------------------------------------------------------------

def parse_xml_ted(file_path: Path) -> list[ExtractedDoc]:
    """Parse TED XML notice (R2.0.x schema or eForms).

    Extracts Italian content from:
      1. FORM_SECTION with LG="IT" → TITLE, SHORT_DESCR, all <P> text
      2. TRANSLATION_SECTION → ML_TI_DOC LG="IT"
      3. Fallback: any element with xml:lang="it" or LG="IT"
    """
    docs: list[ExtractedDoc] = []

    try:
        tree = ET.parse(file_path)
    except ET.ParseError as exc:
        log.debug("XML parse error in %s: %s", file_path.name, exc)
        return docs

    root = tree.getroot()

    # Extract namespace
    ns = ""
    tag = root.tag
    if "}" in tag:
        ns = tag.split("}")[0] + "}"

    doc_id = root.get("DOC_ID", file_path.stem)

    # ── Strategy 1: Find Italian form in FORM_SECTION ─────────────────────
    form_section = root.find(f".//{ns}FORM_SECTION")
    if form_section is None:
        form_section = root.find(".//FORM_SECTION")

    it_form = None
    if form_section is not None:
        for form_elem in form_section:
            lg = form_elem.get("LG", "")
            if lg.upper() == "IT":
                it_form = form_elem
                break

    if it_form is not None:
        title = _extract_form_title(it_form, ns)
        short_descr = _extract_form_short_descr(it_form, ns)
        full_text = _extract_form_paragraphs(it_form, ns)

        doc = ExtractedDoc(
            doc_id=doc_id,
            title=clean_text(title),
            abstract=clean_text(short_descr),
            body=clean_text(full_text),
            source_file=str(file_path),
        )
        if not doc.is_empty():
            docs.append(doc)
            return docs

    # ── Strategy 2: TRANSLATION_SECTION for title + any Italian text ──────
    title_it = ""
    translation = root.find(f".//{ns}TRANSLATION_SECTION")
    if translation is None:
        translation = root.find(".//TRANSLATION_SECTION")

    if translation is not None:
        for ml_ti in translation.iter(f"{ns}ML_TI_DOC"):
            if ml_ti.get("LG", "").upper() == "IT":
                ti_text = ml_ti.find(f"{ns}TI_TEXT")
                if ti_text is None:
                    ti_text = ml_ti.find("TI_TEXT")
                if ti_text is not None:
                    title_it = elem_text(ti_text)
                break
        # Also try without namespace
        if not title_it:
            for ml_ti in translation.iter("ML_TI_DOC"):
                if ml_ti.get("LG", "").upper() == "IT":
                    ti_text = ml_ti.find("TI_TEXT")
                    if ti_text is not None:
                        title_it = elem_text(ti_text)
                    break

    if title_it:
        doc = ExtractedDoc(
            doc_id=doc_id,
            title=clean_text(title_it),
            source_file=str(file_path),
        )
        if not doc.is_empty():
            docs.append(doc)
            return docs

    # ── Strategy 3: xml:lang="it" on any elements ─────────────────────────
    italian_texts: list[str] = []
    for elem in root.iter():
        lang = (
            elem.get("{http://www.w3.org/XML/1998/namespace}lang", "")
            or elem.get("languageID", "")
            or elem.get("LG", "")
        )
        if lang.upper() in ("IT", "ITA"):
            text = elem_text(elem).strip()
            if text and len(text) > 15:
                italian_texts.append(text)

    if italian_texts:
        docs.append(
            ExtractedDoc(
                doc_id=doc_id,
                title=clean_text(italian_texts[0]),
                body=clean_text("\n\n".join(italian_texts[1:])),
                source_file=str(file_path),
            )
        )

    return docs


def _extract_form_title(form: ET.Element, ns: str) -> str:
    for tag in (f"{ns}TITLE", "TITLE"):
        for elem in form.iter(tag):
            return elem_text(elem)
    return ""


def _extract_form_short_descr(form: ET.Element, ns: str) -> str:
    parts: list[str] = []
    for tag in (f"{ns}SHORT_DESCR", "SHORT_DESCR"):
        for elem in form.iter(tag):
            parts.append(elem_text(elem))
    return "\n".join(parts)


def _extract_form_paragraphs(form: ET.Element, ns: str) -> str:
    """Extract all <P> text from a form — gives us the full notice body."""
    parts: list[str] = []
    seen: set[str] = set()
    for tag in (f"{ns}P", "P"):
        for p in form.iter(tag):
            text = elem_text(p).strip()
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_notice_files(root_dir: Path) -> tuple[list[Path], list[Path]]:
    """Walk extracted tree. Return (italian_legacy_files, xml_files).

    For legacy TXT: only return IT_* files.
    For XML: return all .xml files (Italian filtering at parse time).
    """
    it_legacy: list[Path] = []
    xml_files: list[Path] = []

    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue

        name_lower = path.name.lower()

        # Skip junk
        if name_lower.startswith((".", "__")):
            continue
        if name_lower.endswith((".xsd", ".dtd", ".pdf", ".jpg", ".png", ".css", ".html")):
            continue

        if name_lower.endswith(".xml"):
            xml_files.append(path)
            continue

        # Legacy TXT: only IT_* files
        if is_italian_file(path):
            fmt = detect_format(path)
            if fmt == FormatEra.LEGACY_TXT:
                it_legacy.append(path)
            elif fmt == FormatEra.XML_TED:
                xml_files.append(path)

    return it_legacy, xml_files


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_package(
    package_dir: Path,
    stats: ProcessingStats,
) -> list[ExtractedDoc]:
    """Process a single extracted monthly package directory."""
    all_docs: list[ExtractedDoc] = []

    # Extract any remaining nested archives
    for archive in list(package_dir.rglob("*")):
        if archive.is_file() and _looks_like_archive(archive):
            dest = archive.parent / archive.stem.replace(".tar", "")
            dest.mkdir(parents=True, exist_ok=True)
            extract_recursive(archive, dest)
            archive.unlink(missing_ok=True)

    it_legacy, xml_files = find_notice_files(package_dir)
    stats.files_scanned += len(it_legacy) + len(xml_files)
    stats.it_files_found += len(it_legacy)

    # Process Italian legacy TXT files
    for file_path in it_legacy:
        try:
            docs = parse_legacy_txt(file_path)
            stats.inc_format("legacy_txt")
            for doc in docs:
                if doc.is_empty():
                    stats.docs_empty += 1
                else:
                    all_docs.append(doc)
                    stats.docs_extracted += 1
        except Exception as exc:
            log.debug("Error processing %s: %s", file_path, exc)
            stats.errors += 1

    # Process XML files
    for file_path in xml_files:
        try:
            docs = parse_xml_ted(file_path)
            stats.inc_format("xml_ted")
            for doc in docs:
                if doc.is_empty():
                    stats.docs_empty += 1
                else:
                    all_docs.append(doc)
                    stats.docs_extracted += 1
            if not docs:
                stats.skipped_non_it += 1
        except Exception as exc:
            log.debug("Error processing %s: %s", file_path, exc)
            stats.errors += 1

    stats.packages_processed += 1
    return all_docs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_pretrain_batch(
    docs: list[ExtractedDoc],
    output_file: Path,
    *,
    append: bool = True,
) -> int:
    """Write documents to pretrain.txt in <bos>doc<eos> format."""
    mode = "a" if append else "w"
    written = 0
    with output_file.open(mode, encoding="utf-8") as f:
        for doc in docs:
            text = doc.to_text()
            if not text:
                continue
            f.write(f"{BOS}\n{text}\n{EOS}\n\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def _show_summary(stats: ProcessingStats, output: Path) -> None:
    """Display processing summary."""
    table = Table(
        title="Processing Summary",
        title_style="bold magenta",
        show_lines=True,
        padding=(0, 2),
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Packages processed", f"[green]{stats.packages_processed}")
    table.add_row("Files scanned", f"[cyan]{stats.files_scanned}")
    table.add_row("IT legacy files found", f"[bold cyan]{stats.it_files_found}")
    table.add_row("Italian docs extracted", f"[bold green]{stats.docs_extracted}")
    table.add_row("Empty docs skipped", f"[yellow]{stats.docs_empty}")
    table.add_row("Non-IT XML skipped", f"[dim]{stats.skipped_non_it}")
    table.add_row("Errors", f"[red]{stats.errors}" if stats.errors else "[green]0")
    console.print(table)

    if stats.formats:
        fmt_table = Table(title="Formats", title_style="bold cyan", padding=(0, 2))
        fmt_table.add_column("Format")
        fmt_table.add_column("Files", justify="right")
        for fmt, count in sorted(stats.formats.items()):
            fmt_table.add_row(fmt, str(count))
        console.print(fmt_table)

    if output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        doc_count = 0
        with output.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == BOS:
                    doc_count += 1
        console.print(
            f"\n[bold green]Output: {output} "
            f"({size_mb:.1f} MB, {doc_count} documents)"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ted-processor",
    help="Process downloaded TED packages into Italian pretrain corpus.",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command()
def process(
    source: Path = typer.Option(Path("./ted_data"), "--source", "-s"),
    output: Path = typer.Option(Path("./pretrain_ted_it.txt"), "--output", "-o"),
    cleanup: bool = typer.Option(True, "--cleanup/--no-cleanup"),
    packages: str = typer.Option("", "--packages", "-p"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Process TED packages into Italian pretrain corpus."""
    if verbose:
        log.setLevel(logging.DEBUG)

    if not source.exists():
        console.print(f"[red]Source not found: {source}")
        raise typer.Exit(1)

    pkg_filter: set[str] = set()
    if packages:
        pkg_filter = {p.strip() for p in packages.split(",")}

    pkg_dirs = sorted(
        p
        for p in source.iterdir()
        if p.is_dir() and not p.name.startswith(".") and "journal" not in p.name
    )
    if pkg_filter:
        pkg_dirs = [p for p in pkg_dirs if p.name in pkg_filter]

    if not pkg_dirs:
        console.print("[yellow]No package directories found.")
        raise typer.Exit(0)

    console.print(
        Panel(
            f"[cyan]Source: {source.resolve()}\n"
            f"Output: {output.resolve()}\n"
            f"Packages: {len(pkg_dirs)}",
            title="[bold magenta]TED Italian Corpus Processor",
            border_style="blue",
        )
    )

    stats = ProcessingStats()
    if output.exists() and not packages:
        output.unlink()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Processing", total=len(pkg_dirs))
        for pkg_dir in pkg_dirs:
            progress.update(task, description=f"[bold blue]{pkg_dir.name}")
            try:
                docs = process_package(pkg_dir, stats)
                if docs:
                    write_pretrain_batch(docs, output, append=True)
                    log.info("%s: %d Italian docs", pkg_dir.name, len(docs))

                if cleanup:
                    for child in pkg_dir.iterdir():
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
            except Exception as exc:
                log.error("Failed %s: %s", pkg_dir.name, exc)
                stats.errors += 1
            progress.advance(task)

    console.print()
    _show_summary(stats, output)


@app.command()
def sample(
    output: Path = typer.Option(Path("./pretrain_ted_it.txt"), "--output", "-o"),
    count: int = typer.Option(3, "--count", "-n"),
) -> None:
    """Preview sample documents from the pretrain file."""
    if not output.exists():
        console.print("[yellow]No output file. Run process first.")
        raise typer.Exit(1)

    docs: list[str] = []
    current: list[str] = []
    in_doc = False

    with output.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == BOS:
                in_doc = True
                current = []
            elif stripped == EOS and in_doc:
                docs.append("\n".join(current))
                in_doc = False
                if len(docs) >= count:
                    break
            elif in_doc:
                current.append(line.rstrip())

    for i, doc in enumerate(docs, 1):
        preview = textwrap.shorten(doc, width=500, placeholder="…")
        console.print(
            Panel(preview, title=f"[bold]Document {i}", border_style="cyan", padding=(1, 2))
        )


if __name__ == "__main__":
    app()