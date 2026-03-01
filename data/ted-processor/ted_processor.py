#!/usr/bin/env python3
# filepath: ted_processor.py
"""TED Europa data processor for Italian pretrain corpus.

Recursively extracts nested archives (tar.gz → tar.gz → zip),
parses TED notices in all historical formats (legacy TXT 1993-2005,
XML 2006-2022, eForms 2022+), filters Italian content, and produces
a <bos>doc<eos> delimited pretrain.txt file.
"""

from __future__ import annotations

import gzip
import io
import logging
import os
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

# XML namespaces found in TED exports across eras
NS_MAP: dict[str, str] = {
    "ted_r209": "http://publications.europa.eu/resource/schema/ted/R2.0.9/publication",
    "ted_export": "http://publications.europa.eu/TED_schema/Export",
    "n2016": "http://publications.europa.eu/resource/schema/ted/2016/nuts",
}

# Patterns for Italian ZIP files in early data (1998-2005)
IT_ZIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^IT[_-]", re.IGNORECASE),
    re.compile(r"[_-]IT[_-]", re.IGNORECASE),
    re.compile(r"^it_.*utf8.*org", re.IGNORECASE),
    re.compile(r"^IT_.*ISO.*ORG", re.IGNORECASE),
    re.compile(r"^it_.*_org", re.IGNORECASE),
]

# Legacy TXT field regex (multi-line fields ending at next TAG: or EOF)
LEGACY_FIELD_RE = re.compile(
    r"^([A-Z]{2})\s*[:=]\s*(.+?)(?=^[A-Z]{2}\s*[:=]|\Z)",
    re.MULTILINE | re.DOTALL,
)

# Fields we want from legacy notices
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
    LEGACY_TXT = "legacy_txt"     # 1993–2005
    XML_OLD = "xml_old"           # 2006–~2022
    XML_EFORMS = "xml_eforms"     # 2022+
    UNKNOWN = "unknown"


@dataclass
class ExtractedDoc:
    """A single extracted document ready for the pretrain corpus."""
    doc_id: str
    title: str = ""
    abstract: str = ""
    body: str = ""
    source_file: str = ""

    def to_text(self) -> str:
        """Combine fields into a single document string."""
        parts: list[str] = []
        if self.title:
            parts.append(self.title.strip())
        if self.abstract:
            parts.append(self.abstract.strip())
        if self.body:
            parts.append(self.body.strip())
        return "\n\n".join(parts)

    def is_empty(self) -> bool:
        return not (self.title or self.abstract or self.body)


@dataclass
class ProcessingStats:
    """Track processing statistics."""
    packages_processed: int = 0
    files_scanned: int = 0
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
    # Remove XML/HTML tags that might be leftover
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters except newlines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def elem_text(elem: ET.Element | None) -> str:
    """Recursively extract all text from an XML element and children."""
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
# Archive extraction – recursive, handles all nesting levels
# ---------------------------------------------------------------------------

def extract_recursive(path: Path, dest: Path, *, depth: int = 0, max_depth: int = 5) -> None:
    """Recursively extract tar.gz, gz, and zip archives.

    Handles the TED nesting: monthly.tar.gz → daily.tar.gz → notice.zip → files
    """
    if depth > max_depth:
        log.warning("Max extraction depth reached at %s", path)
        return

    suffix_lower = "".join(path.suffixes).lower()

    try:
        if suffix_lower.endswith(".tar.gz") or suffix_lower.endswith(".tgz"):
            _extract_tar(path, dest, depth, max_depth)
        elif suffix_lower.endswith(".gz") and not suffix_lower.endswith(".tar.gz"):
            _extract_gz(path, dest, depth, max_depth)
        elif suffix_lower.endswith(".zip"):
            _extract_zip(path, dest, depth, max_depth)
        elif tarfile.is_tarfile(str(path)):
            _extract_tar(path, dest, depth, max_depth)
        elif zipfile.is_zipfile(str(path)):
            _extract_zip(path, dest, depth, max_depth)
    except (tarfile.TarError, zipfile.BadZipFile, gzip.BadGzipFile, OSError) as exc:
        log.warning("Failed to extract %s: %s", path.name, exc)


def _extract_tar(path: Path, dest: Path, depth: int, max_depth: int) -> None:
    """Extract a tar/tar.gz and recurse into nested archives."""
    with tarfile.open(str(path), "r:*") as tf:
        # Security: prevent path traversal
        for member in tf.getmembers():
            resolved = (dest / member.name).resolve()
            if not str(resolved).startswith(str(dest.resolve())):
                raise tarfile.TarError(f"Path traversal: {member.name}")
        tf.extractall(dest, filter="data")

    # Recurse into any nested archives
    for child in dest.rglob("*"):
        if child.is_file() and _is_archive(child):
            child_dest = child.parent / child.stem.replace(".tar", "")
            child_dest.mkdir(parents=True, exist_ok=True)
            extract_recursive(child, child_dest, depth=depth + 1, max_depth=max_depth)
            child.unlink(missing_ok=True)


def _extract_gz(path: Path, dest: Path, depth: int, max_depth: int) -> None:
    """Extract a plain .gz file."""
    out_name = path.stem  # remove .gz
    out_path = dest / out_name
    with gzip.open(str(path), "rb") as gz_in:
        with out_path.open("wb") as f_out:
            shutil.copyfileobj(gz_in, f_out)
    if _is_archive(out_path):
        extract_recursive(out_path, dest, depth=depth + 1, max_depth=max_depth)
        out_path.unlink(missing_ok=True)


def _extract_zip(path: Path, dest: Path, depth: int, max_depth: int) -> None:
    """Extract a ZIP and recurse."""
    with zipfile.ZipFile(path, "r") as zf:
        for member in zf.infolist():
            resolved = (dest / member.filename).resolve()
            if not str(resolved).startswith(str(dest.resolve())):
                raise zipfile.BadZipFile(f"Path traversal: {member.filename}")
        zf.extractall(dest)

    for child in dest.rglob("*"):
        if child.is_file() and _is_archive(child):
            child_dest = child.parent / child.stem
            child_dest.mkdir(parents=True, exist_ok=True)
            extract_recursive(child, child_dest, depth=depth + 1, max_depth=max_depth)
            child.unlink(missing_ok=True)


def _is_archive(path: Path) -> bool:
    """Check if file is an extractable archive."""
    suffix = "".join(path.suffixes).lower()
    if any(suffix.endswith(ext) for ext in (".tar.gz", ".tgz", ".gz", ".zip")):
        return True
    # Check by magic bytes for extensionless files
    try:
        with path.open("rb") as f:
            header = f.read(4)
            # ZIP magic
            if header[:4] == b"PK\x03\x04":
                return True
            # GZIP magic
            if header[:2] == b"\x1f\x8b":
                return True
    except OSError:
        pass
    return False


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(file_path: Path) -> FormatEra:
    """Detect the TED notice format from file content."""
    try:
        with file_path.open("rb") as f:
            header = f.read(2048)
    except OSError:
        return FormatEra.UNKNOWN

    # XML detection
    if b"<?xml" in header or b"<TED_EXPORT" in header:
        if b"eForms" in header or b"eforms" in header:
            return FormatEra.XML_EFORMS
        if b"TED_EXPORT" in header:
            return FormatEra.XML_OLD
        return FormatEra.XML_OLD

    # Legacy TXT detection – look for field markers
    try:
        text_header = header.decode("utf-8", errors="replace")
    except Exception:
        text_header = header.decode("latin-1", errors="replace")

    if re.search(r"^(CY|ND|TI|AB|TX)\s*[:=]", text_header, re.MULTILINE):
        return FormatEra.LEGACY_TXT

    return FormatEra.UNKNOWN


# ---------------------------------------------------------------------------
# Parser: Legacy TXT (1993–2005)
# ---------------------------------------------------------------------------

def parse_legacy_txt(file_path: Path) -> list[ExtractedDoc]:
    """Parse a legacy TED notice in TXT format.

    Format: multi-line key-value pairs like:
        CY: IT
        TI: Titolo del bando...
        AB: Abstract...
        TX: Testo completo...

    Each file may contain one or more notices separated by ND: markers.
    """
    docs: list[ExtractedDoc] = []

    for encoding in ("utf-8", "latin-1", "cp1252", "iso-8859-15"):
        try:
            content = file_path.read_text(encoding=encoding)
            break
        except (UnicodeDecodeError, OSError):
            continue
    else:
        return docs

    # Split on ND: (notice delimiter) if multiple notices in one file
    notice_blocks = re.split(r"(?=^ND\s*[:=])", content, flags=re.MULTILINE)
    if not notice_blocks or (len(notice_blocks) == 1 and not notice_blocks[0].strip()):
        notice_blocks = [content]

    for block in notice_blocks:
        if not block.strip():
            continue

        fields: dict[str, str] = {}
        for match in LEGACY_FIELD_RE.finditer(block):
            key = match.group(1).strip().upper()
            value = match.group(2).strip()
            # Append if key already exists (multi-occurrence)
            if key in fields:
                fields[key] += "\n" + value
            else:
                fields[key] = value

        # Filter: only Italian notices
        cy = fields.get("CY", "").strip().upper()
        if "IT" not in cy and cy not in ("I", "ITALIA", "ITALY", "ITALIE"):
            continue

        doc = ExtractedDoc(
            doc_id=fields.get("ND", file_path.stem),
            title=clean_text(fields.get("TI", "")),
            abstract=clean_text(fields.get("AB", "")),
            body=clean_text(fields.get("TX", "")),
            source_file=str(file_path),
        )
        if not doc.is_empty():
            docs.append(doc)

    return docs


# ---------------------------------------------------------------------------
# Parser: XML old format (2006–~2022) – TED_EXPORT R2.0.x
# ---------------------------------------------------------------------------

def parse_xml_old(file_path: Path) -> list[ExtractedDoc]:
    """Parse old-format TED XML (R2.0.x schema).

    Structure:
        <TED_EXPORT>
            <CODED_DATA_SECTION>
                <NOTICE_DATA><ISO_COUNTRY VALUE="IT"/></NOTICE_DATA>
            </CODED_DATA_SECTION>
            <TRANSLATION_SECTION>
                <ML_TITLES><ML_TI_DOC LG="IT">...</ML_TI_DOC></ML_TITLES>
            </TRANSLATION_SECTION>
            <FORM_SECTION>
                <F02_2014 LG="IT">  <!-- Italian form -->
                    <OBJECT_CONTRACT><TITLE><P>...</P></TITLE></OBJECT_CONTRACT>
                </F02_2014>
            </FORM_SECTION>
        </TED_EXPORT>
    """
    docs: list[ExtractedDoc] = []

    try:
        tree = ET.parse(file_path)
    except ET.ParseError as exc:
        log.debug("XML parse error in %s: %s", file_path.name, exc)
        return docs

    root = tree.getroot()

    # Strip namespace from tag for easier matching
    ns = ""
    tag = root.tag
    if "}" in tag:
        ns = tag.split("}")[0] + "}"

    doc_id = root.get("DOC_ID", file_path.stem)

    # ── Strategy 1: Find Italian FORM_SECTION ────────────────────────────
    form_section = root.find(f".//{ns}FORM_SECTION")
    it_form = None

    if form_section is not None:
        for form_elem in form_section:
            lg = form_elem.get("LG", "")
            if lg.upper() == "IT":
                it_form = form_elem
                break

    if it_form is not None:
        title = _extract_xml_form_title(it_form, ns)
        short_descr = _extract_xml_form_short_descr(it_form, ns)
        full_text = _extract_xml_form_full_text(it_form, ns)

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

    # ── Strategy 2: Use TRANSLATION_SECTION for title ────────────────────
    translation = root.find(f".//{ns}TRANSLATION_SECTION")
    title_it = ""
    if translation is not None:
        for ml_ti in translation.iter(f"{ns}ML_TI_DOC"):
            if ml_ti.get("LG", "").upper() == "IT":
                ti_text = ml_ti.find(f"{ns}TI_TEXT")
                if ti_text is not None:
                    title_it = elem_text(ti_text)
                break

    # Check ISO_COUNTRY for IT relevance even without Italian form
    country_elem = root.find(f".//{ns}ISO_COUNTRY")
    is_italian_country = False
    if country_elem is not None:
        is_italian_country = country_elem.get("VALUE", "").upper() == "IT"

    if title_it or is_italian_country:
        # Try to get any text from the original form
        body_parts: list[str] = []
        if form_section is not None:
            for form_elem in form_section:
                cat = form_elem.get("CATEGORY", "")
                if cat == "ORIGINAL" or form_elem.get("LG", "").upper() == "IT":
                    body_parts.append(_extract_xml_form_full_text(form_elem, ns))
                    break

        doc = ExtractedDoc(
            doc_id=doc_id,
            title=clean_text(title_it),
            body=clean_text("\n".join(body_parts)),
            source_file=str(file_path),
        )
        if not doc.is_empty():
            docs.append(doc)

    return docs


def _extract_xml_form_title(form: ET.Element, ns: str) -> str:
    """Extract TITLE from a form element."""
    for title_elem in form.iter(f"{ns}TITLE"):
        return elem_text(title_elem)
    # Try without namespace
    for title_elem in form.iter("TITLE"):
        return elem_text(title_elem)
    return ""


def _extract_xml_form_short_descr(form: ET.Element, ns: str) -> str:
    """Extract SHORT_DESCR from a form element."""
    parts: list[str] = []
    for descr in form.iter(f"{ns}SHORT_DESCR"):
        parts.append(elem_text(descr))
    if not parts:
        for descr in form.iter("SHORT_DESCR"):
            parts.append(elem_text(descr))
    return "\n".join(parts)


def _extract_xml_form_full_text(form: ET.Element, ns: str) -> str:
    """Extract all textual <P> content from a form, which gives us
    TITLE + SHORT_DESCR + MAIN_SITE + INFO_ADD + everything else."""
    parts: list[str] = []
    seen: set[str] = set()

    for p in form.iter(f"{ns}P"):
        text = elem_text(p).strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)

    if not parts:
        for p in form.iter("P"):
            text = elem_text(p).strip()
            if text and text not in seen:
                seen.add(text)
                parts.append(text)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Parser: eForms XML (2022+)
# ---------------------------------------------------------------------------

def parse_xml_eforms(file_path: Path) -> list[ExtractedDoc]:
    """Parse eForms XML.

    eForms use a UBL-based structure. Italian content can be identified
    by xml:lang="it" or languageID="ITA" attributes. Falls back to
    the old-format parser since many 2022-2024 notices still use R2.0.9.
    """
    # First try old-format parser – many "eForms era" files still use R2.0.9
    docs = parse_xml_old(file_path)
    if docs:
        return docs

    # If that didn't work, try UBL-style eForms parsing
    try:
        tree = ET.parse(file_path)
    except ET.ParseError:
        return []

    root = tree.getroot()
    ns = ""
    if "}" in root.tag:
        ns = root.tag.split("}")[0] + "}"

    doc_id = root.get("DOC_ID", file_path.stem)

    # Look for Italian text blocks via xml:lang or LG attributes
    italian_texts: list[str] = []

    for elem in root.iter():
        lang = (
            elem.get("{http://www.w3.org/XML/1998/namespace}lang", "")
            or elem.get("languageID", "")
            or elem.get("LG", "")
        )
        if lang.upper() in ("IT", "ITA"):
            text = elem_text(elem).strip()
            if text and len(text) > 10:
                italian_texts.append(text)

    if italian_texts:
        docs.append(ExtractedDoc(
            doc_id=doc_id,
            title=clean_text(italian_texts[0]) if italian_texts else "",
            body=clean_text("\n\n".join(italian_texts[1:])) if len(italian_texts) > 1 else "",
            source_file=str(file_path),
        ))

    return docs


# ---------------------------------------------------------------------------
# File discovery – find notice files in extracted directory tree
# ---------------------------------------------------------------------------

def find_notice_files(root_dir: Path, *, it_only_zips: bool = True) -> list[Path]:
    """Walk extracted directory tree and find all notice files.

    For early data (IT_* zip pattern), filters only Italian content.
    For XML data, returns all .xml files (filtering happens at parse time).
    For extensionless files, checks if they look like notices.
    """
    notice_files: list[Path] = []

    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        # Skip metadata / non-content files
        name_lower = path.name.lower()
        if name_lower.startswith(".") or name_lower.startswith("__"):
            continue
        if name_lower.endswith((".xsd", ".dtd", ".pdf", ".jpg", ".png", ".css", ".html")):
            continue
        if "meta" in name_lower and not name_lower.endswith(".xml"):
            continue

        if name_lower.endswith(".xml"):
            notice_files.append(path)
        elif path.suffix == "" or name_lower.endswith((".txt", ".dat", ".notice")):
            # Could be legacy notice – check content
            fmt = detect_format(path)
            if fmt == FormatEra.LEGACY_TXT:
                notice_files.append(path)
            elif fmt in (FormatEra.XML_OLD, FormatEra.XML_EFORMS):
                notice_files.append(path)

    return notice_files


def is_italian_zip_name(name: str) -> bool:
    """Check if a ZIP file name matches Italian content patterns."""
    return any(pat.search(name) for pat in IT_ZIP_PATTERNS)


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_package(
    package_dir: Path,
    stats: ProcessingStats,
) -> list[ExtractedDoc]:
    """Process a single extracted monthly package directory.

    Returns list of Italian documents found.
    """
    all_docs: list[ExtractedDoc] = []

    # Extract any remaining nested archives
    for archive in list(package_dir.rglob("*")):
        if archive.is_file() and _is_archive(archive):
            dest = archive.parent / archive.stem.replace(".tar", "")
            dest.mkdir(parents=True, exist_ok=True)
            extract_recursive(archive, dest)
            archive.unlink(missing_ok=True)

    notice_files = find_notice_files(package_dir)
    stats.files_scanned += len(notice_files)

    for file_path in notice_files:
        try:
            fmt = detect_format(file_path)

            if fmt == FormatEra.LEGACY_TXT:
                docs = parse_legacy_txt(file_path)
                stats.inc_format("legacy_txt")
            elif fmt == FormatEra.XML_OLD:
                docs = parse_xml_old(file_path)
                stats.inc_format("xml_old")
            elif fmt == FormatEra.XML_EFORMS:
                docs = parse_xml_eforms(file_path)
                stats.inc_format("xml_eforms")
            else:
                stats.inc_format("unknown")
                continue

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


def write_pretrain_batch(
    docs: list[ExtractedDoc],
    output_file: Path,
    *,
    append: bool = True,
) -> int:
    """Write documents to pretrain.txt in <bos>doc<eos> format.

    Returns number of documents written.
    """
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
    source: Path = typer.Option(
        Path("./ted_data"),
        "--source", "-s",
        help="Directory containing downloaded TED packages.",
    ),
    output: Path = typer.Option(
        Path("./pretrain_ted_it.txt"),
        "--output", "-o",
        help="Output pretrain.txt file path.",
    ),
    cleanup: bool = typer.Option(
        True,
        "--cleanup/--no-cleanup",
        help="Remove extracted files after processing (keeps ZIPs if kept by downloader).",
    ),
    packages: str = typer.Option(
        "",
        "--packages", "-p",
        help="Comma-separated list of packages to process (e.g. '1993-01,2024-06'). Empty = all.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Process TED packages into a pretrain corpus for Italian language."""

    if verbose:
        log.setLevel(logging.DEBUG)

    if not source.exists():
        console.print(f"[red]Source directory not found: {source}")
        raise typer.Exit(1)

    # Find all package directories
    pkg_filter: set[str] = set()
    if packages:
        pkg_filter = {p.strip() for p in packages.split(",")}

    pkg_dirs: list[Path] = sorted(
        p for p in source.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and p.name != "__pycache__"
        and "journal" not in p.name
    )

    if pkg_filter:
        pkg_dirs = [p for p in pkg_dirs if p.name in pkg_filter]

    if not pkg_dirs:
        console.print("[yellow]No package directories found.")
        raise typer.Exit(0)

    # Banner
    banner = Text.assemble(
        ("TED Italian Corpus Processor\n", "bold magenta"),
        (f"Source   : {source.resolve()}\n", "cyan"),
        (f"Output   : {output.resolve()}\n", "cyan"),
        (f"Packages : {len(pkg_dirs)}\n", "cyan"),
        (f"Cleanup  : {'yes' if cleanup else 'no'}", "cyan"),
    )
    console.print(Panel(banner, border_style="blue", padding=(1, 2)))

    stats = ProcessingStats()

    # Fresh output file
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
                    log.info(
                        "%s: %d Italian docs extracted",
                        pkg_dir.name, len(docs),
                    )
                else:
                    log.debug("%s: no Italian docs found", pkg_dir.name)

                # Cleanup extracted subdirectories (not the package dir itself)
                if cleanup:
                    for child in pkg_dir.iterdir():
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)

            except Exception as exc:
                log.error("Failed processing %s: %s", pkg_dir.name, exc)
                stats.errors += 1

            progress.advance(task)

    # Summary
    console.print()
    _show_summary(stats, output)


def _show_summary(stats: ProcessingStats, output: Path) -> None:
    """Display processing summary."""
    from rich.table import Table

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
    table.add_row("Italian docs extracted", f"[bold green]{stats.docs_extracted}")
    table.add_row("Empty docs skipped", f"[yellow]{stats.docs_empty}")
    table.add_row("Non-IT notices skipped", f"[dim]{stats.skipped_non_it}")
    table.add_row("Errors", f"[red]{stats.errors}" if stats.errors else "[green]0")

    console.print(table)

    if stats.formats:
        fmt_table = Table(
            title="Format Distribution",
            title_style="bold cyan",
            padding=(0, 2),
        )
        fmt_table.add_column("Format")
        fmt_table.add_column("Files", justify="right")
        for fmt, count in sorted(stats.formats.items()):
            fmt_table.add_row(fmt, str(count))
        console.print(fmt_table)

    if output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        # Count docs in output
        doc_count = 0
        with output.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == BOS:
                    doc_count += 1
        console.print(
            f"\n[bold green]Output: {output} "
            f"({size_mb:.1f} MB, {doc_count} documents)"
        )


@app.command()
def sample(
    output: Path = typer.Option(
        Path("./pretrain_ted_it.txt"),
        "--output", "-o",
    ),
    count: int = typer.Option(3, "--count", "-n", help="Number of samples to show."),
) -> None:
    """Preview sample documents from the pretrain file."""
    if not output.exists():
        console.print("[yellow]Output file not found. Run [bold]process[/bold] first.")
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
        console.print(Panel(
            preview,
            title=f"[bold]Document {i}",
            border_style="cyan",
            padding=(1, 2),
        ))


if __name__ == "__main__":
    app()
