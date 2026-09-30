#!/usr/bin/env python3
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Local corpus builder for pre-training datasets.

Scans a source directory for PDF, HTML, XML, JSON, JSONL, and TXT files,
extracts text using format-specific extractors, runs a full cleaning +
dedup pipeline, wraps documents in <bos>/<eos> tokens, and produces
shuffled output chunks ready for training.

Usage
─────
  # Minimal — default keys ["text"], all filters on
  python corpus_builder.py ./raw_data

  # Custom search keys for JSON/JSONL + output path
  python corpus_builder.py ./raw_data -o ./corpus -k text -k testo -k documento

  # Skip HTML/XML, custom thresholds
  python corpus_builder.py ./raw_data --skip-html --skip-xml --min-chars 200

  # Full control
  python corpus_builder.py ./raw_data \\
      -o ./corpus \\
      -k text -k body -k content \\
      --max-gb 3 \\
      --nd-threshold 0.85 \\
      --config filters.yaml

Pipeline
────────
  Extraction  →  Normalization  →  PII  →  Quality  →  Spam
  →  ExactDedup  →  NearDedup  →  Temp chunks  →  Shuffle  →  Output
"""

from __future__ import annotations

import io
import shutil
import time
from pathlib import Path

import numpy as np
import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from cleaners import CleanerConfig, ColumnMergeConfig, ColumnMergeCleaner, RepetitionScorer
from extractors import ExtractorRegistry, discover_files
from journal import Journal, hash_file
from pipeline import Pipeline, PipelineConfig, PipelineStats, ExactDedup, NearDedup
from prose_filter import ProseConfig, ProseFilter, FilterStats as ProseStats, flatten_to_prose
from typo_detector import TypoDetector

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

BOS_TOKEN: str = "<bos>"
EOS_TOKEN: str = "<eos>"
NEWLINE: bytes = b"\n"

DOC_INDEX_DTYPE = np.dtype(
    [
        ("file_idx", np.uint16),
        ("byte_offset", np.uint64),
        ("byte_length", np.uint32),
    ]
)

console = Console()


class _PassthroughProseFilter:
    """Passthrough that only flattens text (no scoring/filtering)."""

    def __init__(self) -> None:
        self.stats = ProseStats()

    def process(self, text: str) -> str:
        self.stats.total_docs_in += 1
        result = flatten_to_prose(text)
        if result:
            self.stats.total_docs_out += 1
        return result

    def summary(self) -> list[tuple[str, int]]:
        return []


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _fmt_bytes(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def _pct(a: int, b: int) -> str:
    return f"{a / b * 100:.1f}%" if b else "—"


# Formats where one file == one document, so a giant file == a giant single doc.
# JSONL (one doc per line) and TXT (streamed/curated) must NOT be size-skipped.
SIZE_LIMITED_FORMATS: set[str] = {"json", "html", "htm", "xml", "pdf"}


def _apply_source_limits(
        file_map: dict[str, list[Path]],
        max_file_mb: float,
        max_source_gb: float,
        seed: int,
) -> tuple[dict[str, list[Path]], int, int]:
    """Drop oversize single-doc files and optionally cap total source bytes.

    ``max_file_mb`` skips files larger than the cap, but ONLY for formats where
    one file == one document (json/html/xml/pdf) — this is the guard against
    giant single docs (343MB leggi, 400MB EUR-Lex HTML) that OOM/stall the
    filters. JSONL/TXT stream per-line/per-file and are never size-skipped.

    ``max_source_gb`` randomly samples discovered files (seeded by ``seed``)
    until the cumulative source size reaches the cap — used for per-domain
    volume balancing (e.g. cap 92GB normattiva to 12GB).

    Returns ``(new_file_map, n_dropped_oversize, n_capped_out)``.
    """
    n_dropped = 0
    if max_file_mb > 0:
        cap = int(max_file_mb * 1024 ** 2)
        for fmt in list(file_map):
            if fmt not in SIZE_LIMITED_FORMATS:
                continue
            kept: list[Path] = []
            for f in file_map[fmt]:
                try:
                    too_big = f.stat().st_size > cap
                except OSError:
                    too_big = False
                if too_big:
                    n_dropped += 1
                else:
                    kept.append(f)
            file_map[fmt] = kept

    n_capped = 0
    if max_source_gb > 0:
        cap = int(max_source_gb * 1024 ** 3)
        flat: list[tuple[str, Path, int]] = []
        for fmt, files in file_map.items():
            for f in files:
                try:
                    flat.append((fmt, f, f.stat().st_size))
                except OSError:
                    continue
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(flat))
        selected: dict[str, list[Path]] = {fmt: [] for fmt in file_map}
        total = 0
        for idx in order:
            fmt, f, sz = flat[int(idx)]
            if total >= cap:
                n_capped += 1
                continue
            selected[fmt].append(f)
            total += sz
        file_map = {fmt: fs for fmt, fs in selected.items() if fs}

    return file_map, n_dropped, n_capped


# ──────────────────────────────────────────────────────────────────────
# Chunk writer
# ──────────────────────────────────────────────────────────────────────


class ChunkWriter:
    """Writes documents to size-limited binary chunk files."""

    def __init__(self, directory: Path, max_bytes: int, prefix: str = "") -> None:
        self.directory = directory
        self.max_bytes = max_bytes
        self.prefix = prefix
        self.directory.mkdir(parents=True, exist_ok=True)

        self._chunk_idx: int = 0
        self._bytes_written: int = 0
        self._total_bytes: int = 0
        self._fh: io.BufferedWriter | None = None
        self._files: list[Path] = []
        self._open_next()

    def _current_path(self) -> Path:
        return self.directory / f"{self.prefix}{self._chunk_idx:03d}.txt"

    def _open_next(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
        path = self._current_path()
        self._fh = open(path, "wb")  # noqa: SIM115
        self._files.append(path)
        self._bytes_written = 0

    def write_document(self, text: str) -> tuple[int, int, int]:
        """Write ``<bos>text<eos>\\n``, return *(file_idx, offset, length)*."""
        payload: bytes = f"{BOS_TOKEN}{text}{EOS_TOKEN}".encode("utf-8") + NEWLINE
        doc_len = len(payload)
        if self._bytes_written > 0 and self._bytes_written + doc_len > self.max_bytes:
            self._chunk_idx += 1
            self._open_next()
        assert self._fh is not None
        offset = self._bytes_written
        self._fh.write(payload)
        self._bytes_written += doc_len
        self._total_bytes += doc_len
        return self._chunk_idx, offset, doc_len

    def write_raw(self, raw: bytes) -> None:
        """Write pre-encoded bytes, rolling chunks as needed."""
        doc_len = len(raw)
        if self._bytes_written > 0 and self._bytes_written + doc_len > self.max_bytes:
            self._chunk_idx += 1
            self._open_next()
        assert self._fh is not None
        self._fh.write(raw)
        self._bytes_written += doc_len
        self._total_bytes += doc_len

    @property
    def files(self) -> list[Path]:
        return list(self._files)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def num_chunks(self) -> int:
        return self._chunk_idx + 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


# ──────────────────────────────────────────────────────────────────────
# Rich reporting
# ──────────────────────────────────────────────────────────────────────


def _print_source_scan(file_map: dict[str, list[Path]]) -> None:
    """Print a summary table of discovered source files."""
    table = Table(
        title="[bold]Source Scan",
        border_style="cyan",
        show_lines=False,
    )
    table.add_column("Format", style="bold cyan")
    table.add_column("Files", justify="right")
    table.add_column("Status", style="dim")

    total = 0
    for fmt in ("pdf", "html", "xml", "json", "jsonl", "txt"):
        files = file_map.get(fmt, [])
        count = len(files)
        total += count
        status = "[green]ready[/]" if count > 0 else "[dim]—[/]"
        table.add_row(fmt.upper(), str(count), status)

    table.add_section()
    table.add_row("[bold]Total", f"[bold]{total}", "")

    console.print(table)
    console.print()


def _print_extraction_stats(stats: dict[str, tuple[int, int]]) -> None:
    """Print per-format extraction stats: files processed, docs extracted."""
    table = Table(
        title="[bold]Extraction Results",
        border_style="cyan",
        show_lines=False,
    )
    table.add_column("Format", style="bold cyan")
    table.add_column("Files", justify="right")
    table.add_column("Documents", justify="right", style="green")

    total_files = 0
    total_docs = 0
    for fmt in ("pdf", "html", "xml", "json", "jsonl", "txt"):
        if fmt in stats:
            files, docs = stats[fmt]
            total_files += files
            total_docs += docs
            table.add_row(fmt.upper(), str(files), f"{docs:,}")

    table.add_section()
    table.add_row("[bold]Total", f"[bold]{total_files}", f"[bold green]{total_docs:,}")

    console.print(table)
    console.print()


def _print_pipeline_report(pipe_stats: PipelineStats, pipe: Pipeline) -> None:
    """Print pipeline rejection summary."""
    table = Table(
        title="[bold]Pipeline Results",
        border_style="yellow",
        show_lines=True,
    )
    table.add_column("Filter", style="bold cyan")
    table.add_column("Rejected", justify="right", style="red")
    table.add_column("% of total", justify="right")

    for name, count, pct in pipe.summary_table():
        table.add_row(name, f"{count:,}", f"{pct:.1f}%")

    table.add_section()
    table.add_row(
        "[bold]TOTAL REJECTED",
        f"[bold red]{pipe_stats.total_rejected:,}",
        f"[bold]{_pct(pipe_stats.total_rejected, pipe_stats.total_seen)}",
    )
    table.add_row(
        "[bold green]KEPT",
        f"[bold green]{pipe_stats.total_kept:,}",
        f"[bold green]{_pct(pipe_stats.total_kept, pipe_stats.total_seen)}",
    )

    reasons_table = Table(
        title="[bold]Top Rejection Reasons",
        border_style="yellow",
        show_lines=False,
    )
    reasons_table.add_column("#", justify="right", style="dim")
    reasons_table.add_column("Filter", style="cyan")
    reasons_table.add_column("Reason", style="white")
    reasons_table.add_column("Count", justify="right", style="red")

    for i, (filt, reason, count) in enumerate(pipe.top_reasons(15), 1):
        reasons_table.add_row(str(i), filt, reason, f"{count:,}")

    console.print(Columns([table, reasons_table], padding=(0, 4)))
    console.print()


# ──────────────────────────────────────────────────────────────────────
# Phase 1a — Extract, clean, filter (incremental — no dedup)
# ──────────────────────────────────────────────────────────────────────


def _process_single_doc(
        text: str,
        fmt: str,
        source: str,
        col_merge_cleaner: ColumnMergeCleaner | None,
        prose_filter: ProseFilter,
        repetition_scorer: RepetitionScorer,
        max_rep_ratio: float,
        pipe: Pipeline,
        typo_detector: TypoDetector | None,
        repetition_rejects: list[int],
) -> str | None:
    """Run one document through the full filter chain (no dedup).

    Returns cleaned text or None if rejected.
    """
    if fmt == "txt":
        text = flatten_to_prose(text)
        return text if text else None

    if col_merge_cleaner is not None:
        text = col_merge_cleaner.clean(text)
        if not text:
            return None

    text = prose_filter.process(text)
    if not text:
        return None

    if repetition_scorer.is_repetitive(text, max_rep_ratio):
        repetition_rejects[0] += 1
        return None

    keep, cleaned = pipe.process(text)
    if not keep:
        return None

    if typo_detector is not None:
        typo_detector.check(cleaned, source=source)

    return cleaned


def phase1a_extract_and_filter(
        file_map: dict[str, list[Path]],
        registry: ExtractorRegistry,
        col_merge_cleaner: ColumnMergeCleaner | None,
        prose_filter: ProseFilter,
        pipe: Pipeline,
        repetition_scorer: RepetitionScorer,
        max_rep_ratio: float,
        journal: Journal,
        typo_detector: TypoDetector | None = None,
) -> tuple[list[tuple[str, str]], dict[str, tuple[int, int]], int, int]:
    """Walk source files → extract → normalize → filter (incremental, no dedup).

    Uses the journal to skip already-processed files.  Dedup is NOT run
    here — it happens in Phase 1b on the full corpus.

    Returns:
        (all_docs, per_format_stats, cache_hits, cache_misses)
        where all_docs is list of (text, source) tuples.
    """
    console.rule("[bold cyan]Phase 1a[/]  Extract → Filter (incremental)")

    all_docs: list[tuple[str, str]] = []
    extraction_stats: dict[str, tuple[int, int]] = {}
    repetition_rejects: list[int] = [0]  # mutable counter for helper
    cache_hits = 0
    cache_misses = 0

    total_files = sum(len(files) for files in file_map.values())

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
    ) as progress:
        task = progress.add_task("[green]Processing files", total=total_files)

        for fmt, files in file_map.items():
            fmt_files = len(files)
            fmt_docs = 0

            # ── PDF with Docling: batch convert_all() for GPU efficiency ──
            if fmt == "pdf" and registry._use_docling:
                # Separate cached vs new files
                new_files: list[Path] = []
                file_hashes: dict[str, str] = {}  # abs_path → hash

                for fp in files:
                    fh = hash_file(fp)
                    abs_key = str(fp.resolve())
                    file_hashes[abs_key] = fh
                    cached = journal.lookup(fp, fh)
                    if cached is not None:
                        all_docs.extend(cached)
                        fmt_docs += len(cached)
                        cache_hits += 1
                    else:
                        new_files.append(fp)
                        cache_misses += 1

                if cache_hits > 0:
                    progress.update(
                        task,
                        description=(
                            f"[green]{fmt.upper()} (batch)  "
                            f"[dim]{cache_hits} cached, {len(new_files)} new"
                        ),
                    )

                # Batch-process only new files
                if new_files:
                    # Group results by source file for journal storage
                    batch_docs_by_file: dict[str, list[tuple[str, str]]] = {}
                    docs_seen = 0
                    try:
                        for _source, text in registry.extract_and_normalize_batch(fmt, new_files):
                            docs_seen += 1
                            cleaned = _process_single_doc(
                                text, fmt, _source,
                                col_merge_cleaner, prose_filter,
                                repetition_scorer, max_rep_ratio,
                                pipe, typo_detector, repetition_rejects,
                            )
                            if cleaned is None:
                                continue

                            all_docs.append((cleaned, _source))
                            fmt_docs += 1

                            batch_docs_by_file.setdefault(_source, []).append(
                                (cleaned, _source)
                            )

                            progress.update(
                                task,
                                description=(
                                    f"[green]{fmt.upper()} (batch)  "
                                    f"[dim]docs {len(all_docs):,}  "
                                    f"({cache_hits} cached)"
                                ),
                            )

                    except Exception as exc:
                        console.print(
                            f"  [yellow]⚠[/] Error in PDF batch processing: {exc}"
                        )

                    # Save to journal per source file
                    for fp in new_files:
                        abs_key = str(fp.resolve())
                        # Match against str(fp) since extractors yield str(path)
                        docs_for_file = batch_docs_by_file.get(str(fp), [])
                        journal.store(fp, file_hashes[abs_key], docs_for_file)

                progress.update(task, advance=fmt_files)

            else:
                # ── Per-file processing (TXT, HTML, JSON, JSONL, XML, legacy PDF) ──
                for file_path in files:
                    fh = hash_file(file_path)
                    cached = journal.lookup(file_path, fh)

                    if cached is not None:
                        all_docs.extend(cached)
                        fmt_docs += len(cached)
                        cache_hits += 1
                    else:
                        cache_misses += 1
                        file_docs: list[tuple[str, str]] = []
                        try:
                            for _source, text in registry.extract_and_normalize(fmt, file_path):
                                cleaned = _process_single_doc(
                                    text, fmt, _source,
                                    col_merge_cleaner, prose_filter,
                                    repetition_scorer, max_rep_ratio,
                                    pipe, typo_detector, repetition_rejects,
                                )
                                if cleaned is None:
                                    continue

                                file_docs.append((cleaned, _source))
                                fmt_docs += 1

                        except Exception as exc:
                            console.print(
                                f"  [yellow]⚠[/] Error processing {file_path.name}: {exc}"
                            )

                        all_docs.extend(file_docs)
                        journal.store(file_path, fh, file_docs)

                    progress.update(
                        task,
                        advance=1,
                        description=(
                            f"[green]{fmt.upper()}  "
                            f"[dim]docs {len(all_docs):,}  "
                            f"({cache_hits} cached)"
                        ),
                    )

            if fmt_files > 0:
                extraction_stats[fmt] = (fmt_files, fmt_docs)

    console.print(
        f"\n  [bold green]✓[/] {len(all_docs):,} documents after extraction + filtering"
    )
    console.print(
        f"  [dim]↳[/] {cache_hits} from cache, {cache_misses} newly processed"
    )
    if repetition_rejects[0] > 0:
        console.print(
            f"  [yellow]↳[/] {repetition_rejects[0]:,} rejected by repetition scorer"
        )

    # PDF garbage stats
    pdf_total, pdf_skipped, pdf_fallback = registry.pdf_stats
    if pdf_skipped > 0:
        unit = "documents" if registry._use_docling else "pages"
        console.print(
            f"  [yellow]↳[/] {pdf_skipped:,}/{pdf_total:,} PDF {unit} skipped "
            f"(table/form garbage)"
        )
    if pdf_fallback > 0:
        console.print(
            f"  [yellow]↳[/] {pdf_fallback:,} PDF fell back to pymupdf "
            f"(Docling conversion failed)"
        )

    # Column merge cleaner stats
    if col_merge_cleaner is not None:
        cms = col_merge_cleaner.stats
        if cms.docs_in > 0:
            cm_table = Table(
                title="[bold]Column Merge Cleaner",
                border_style="blue",
                show_lines=False,
            )
            cm_table.add_column("Metric", style="bold")
            cm_table.add_column("Value", justify="right")
            cm_table.add_row("Docs in", f"{cms.docs_in:,}")
            cm_table.add_row("[green]Docs out", f"[green]{cms.docs_out:,}")
            cm_table.add_row("[red]Docs rejected", f"[red]{cms.docs_rejected:,}")
            cm_table.add_row("Paragraphs total", f"{cms.paragraphs_total:,}")
            cm_table.add_row(
                "[green]Paragraphs clean",
                f"[green]{cms.paragraphs_clean:,}",
            )
            cm_table.add_row(
                "[red]Paragraphs corrupted",
                f"[red]{cms.paragraphs_corrupted:,}",
            )
            cm_table.add_row(
                "Gazette lines stripped",
                f"{cms.gazette_lines_stripped:,}",
            )
            console.print()
            console.print(cm_table)

    # Prose filter stats
    ps = prose_filter.stats
    if ps.total_chunks > 0:
        prose_table = Table(
            title="[bold]Prose Filter",
            border_style="magenta",
            show_lines=False,
        )
        prose_table.add_column("Metric", style="bold")
        prose_table.add_column("Value", justify="right")
        prose_table.add_row("Chunks evaluated", f"{ps.total_chunks:,}")
        prose_table.add_row("[green]Chunks kept", f"[green]{ps.kept_chunks:,}")
        prose_table.add_row("[red]Chunks rejected", f"[red]{ps.rejected_chunks:,}")
        prose_table.add_row(
            "Keep rate",
            f"{ps.kept_chunks / ps.total_chunks * 100:.1f}%"
            if ps.total_chunks else "—",
        )
        prose_table.add_row("Docs in", f"{ps.total_docs_in:,}")
        prose_table.add_row("[green]Docs out", f"[green]{ps.total_docs_out:,}")
        prose_table.add_row(
            "[red]Docs empty after filter",
            f"[red]{ps.empty_after_filter:,}",
        )

        reasons_table = Table(
            title="[bold]Prose Rejection Reasons",
            border_style="magenta",
            show_lines=False,
        )
        reasons_table.add_column("#", justify="right", style="dim")
        reasons_table.add_column("Reason", style="white")
        reasons_table.add_column("Count", justify="right", style="red")

        for i, (reason, count) in enumerate(prose_filter.summary(), 1):
            reasons_table.add_row(str(i), reason, f"{count:,}")

        console.print()
        console.print(Columns([prose_table, reasons_table], padding=(0, 4)))

    console.print()

    _print_extraction_stats(extraction_stats)
    _print_pipeline_report(pipe.stats, pipe)

    # Typo detector report
    if typo_detector is not None and typo_detector.total_checked > 0:
        typo_detector.close()
        console.print(
            f"  [dim]Typo detector:[/] {typo_detector.total_suspicious:,} "
            f"suspicious words / {typo_detector.total_checked:,} total "
            f"({typo_detector.suspicious_ratio:.2%})"
        )
        if typo_detector.total_suspicious > 0:
            console.print(
                f"  [dim]↳[/] Review: [cyan]{typo_detector._output_path}[/]"
            )
        console.print()

    return all_docs, extraction_stats, cache_hits, cache_misses


# ──────────────────────────────────────────────────────────────────────
# Phase 1b — Dedup & Chunk (always fresh)
# ──────────────────────────────────────────────────────────────────────


def phase1b_dedup_and_chunk(
        all_docs: list[tuple[str, str]],
        pipe_cfg: PipelineConfig,
        temp_dir: Path,
        max_chunk_bytes: int,
) -> tuple[list[Path], np.ndarray, PipelineStats]:
    """Run fresh ExactDedup + NearDedup on ALL documents, then write temp chunks.

    Returns:
        (temp_file_paths, index_array, dedup_stats)
    """
    console.rule("[bold cyan]Phase 1b[/]  Dedup → Chunk (fresh)")

    # Build fresh dedup filters
    dedup_filters: list = []
    if pipe_cfg.exact_dedup_enabled:
        dedup_filters.append(ExactDedup())
    if pipe_cfg.near_dedup_enabled:
        dedup_filters.append(NearDedup(pipe_cfg))

    writer = ChunkWriter(temp_dir, max_chunk_bytes, prefix="tmp_")
    indices: list[tuple[int, int, int]] = []
    dedup_stats = PipelineStats()
    dedup_rejects = 0

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
    ) as progress:
        task = progress.add_task("[green]Dedup + chunk", total=len(all_docs))

        for i, (text, _source) in enumerate(all_docs):
            dedup_stats.total_seen += 1
            rejected = False

            for filt in dedup_filters:
                result = filt(text)
                if not result.keep:
                    dedup_stats.record_reject(filt.name, result.reason)
                    dedup_rejects += 1
                    rejected = True
                    break

            if not rejected:
                dedup_stats.total_kept += 1
                file_idx, offset, length = writer.write_document(text)
                indices.append((file_idx, offset, length))

            if i % 5_000 == 0:
                progress.update(
                    task,
                    completed=i,
                    description=(
                        f"[green]Dedup + chunk  "
                        f"[dim]kept {dedup_stats.total_kept:,} / "
                        f"{dedup_stats.total_seen:,}"
                    ),
                )

        progress.update(task, completed=len(all_docs))

    writer.close()
    idx_array = np.array(indices, dtype=DOC_INDEX_DTYPE)

    console.print(
        f"\n  [bold green]✓[/] {len(idx_array):,} documents after dedup  •  "
        f"{_fmt_bytes(writer.total_bytes)}  •  "
        f"{writer.num_chunks} temp chunk(s)"
    )
    if dedup_rejects > 0:
        console.print(
            f"  [yellow]↳[/] {dedup_rejects:,} rejected by dedup"
        )
    console.print()

    return writer.files, idx_array, dedup_stats


# ──────────────────────────────────────────────────────────────────────
# Phase 2 — Shuffle
# ──────────────────────────────────────────────────────────────────────


def phase2_shuffle(indices: np.ndarray, seed: int) -> None:
    console.rule("[bold cyan]Phase 2[/]  Shuffle indices")
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    console.print(
        f"  [bold green]✓[/] Shuffled {len(indices):,} entries (seed={seed})\n"
    )


# ──────────────────────────────────────────────────────────────────────
# Phase 3 — Write shuffled output
# ──────────────────────────────────────────────────────────────────────


def phase3_write_shuffled(
        indices: np.ndarray,
        temp_files: list[Path],
        output_dir: Path,
        max_chunk_bytes: int,
) -> list[Path]:
    console.rule("[bold cyan]Phase 3[/]  Write shuffled output")
    output_dir.mkdir(parents=True, exist_ok=True)

    handles: dict[int, io.BufferedReader] = {}
    try:
        for idx, path in enumerate(temp_files):
            handles[idx] = open(path, "rb")  # noqa: SIM115

        writer = ChunkWriter(output_dir, max_chunk_bytes)
        total = len(indices)

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TextColumn("•"),
                TransferSpeedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                transient=False,
        ) as progress:
            task = progress.add_task("[magenta]Writing shuffled", total=total)

            for i, entry in enumerate(indices):
                fh = handles[int(entry["file_idx"])]
                fh.seek(int(entry["byte_offset"]))
                raw: bytes = fh.read(int(entry["byte_length"]))
                writer.write_raw(raw)

                if i % 10_000 == 0:
                    progress.update(task, completed=i)

            progress.update(task, completed=total)

        writer.close()
    finally:
        for fh in handles.values():
            fh.close()

    console.print(
        f"  [bold green]✓[/] {total:,} documents  •  "
        f"{_fmt_bytes(writer.total_bytes)}  •  "
        f"{writer.num_chunks} output chunk(s)\n"
    )
    return writer.files


# ──────────────────────────────────────────────────────────────────────
# Phase 4 — Cleanup
# ──────────────────────────────────────────────────────────────────────


def phase4_cleanup(temp_dir: Path) -> None:
    console.rule("[bold cyan]Phase 4[/]  Cleanup")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        console.print(f"  [bold green]✓[/] Removed {temp_dir}\n")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="corpus-builder",
    help=(
        "Build a filtered, deduplicated, shuffled pre-training corpus "
        "from local PDF, HTML, JSON, JSONL, and TXT files."
    ),
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command()
def build(
        source: Path = typer.Argument(
            Path("../../.datasets/pretrain"),
            help="Source directory containing raw files (PDF, JSON, JSONL, TXT, HTML, XML).",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
        output: Path = typer.Option(
            Path("../../.datasets/pretokenized"),
            "--output", "-o",
            help="Output directory for final chunk files.",
        ),
        search_keys: list[str] = typer.Option(
            ["text", "testo"],  # nel dataset per ora abbiamo questo
            "--key", "-k",
            help="Keys to search in JSON/JSONL (can repeat: -k text -k testo -k body).",
        ),
        seed: int = typer.Option(42, "--seed"),
        max_gb: float = typer.Option(1.0, "--max-gb", help="Max size per output chunk in GB."),
        max_file_mb: float = typer.Option(
            0.0, "--max-file-mb",
            help=(
                "Skip source files larger than this many MB (json/html/xml/pdf only). "
                "0 = no limit. Guards against giant single-document files "
                "(e.g. 343MB leggi, 400MB EUR-Lex HTML) that OOM/stall the filters."
            ),
        ),
        max_source_gb: float = typer.Option(
            0.0, "--max-source-gb",
            help=(
                "Cap total source size by randomly sampling discovered files "
                "(seeded by --seed) until this many GB. 0 = no limit. "
                "Use for per-domain volume balancing."
            ),
        ),
        config: Path | None = typer.Option(
            None, "--config", "-c",
            help="YAML config for pipeline thresholds.",
        ),
        # ── Format toggles ────────────────────────────────────────────
        skip_html: bool = typer.Option(False, "--skip-html", help="Skip HTML files."),
        skip_xml: bool = typer.Option(True, "--skip-xml", help="Skip XML files (default: skipped)."),
        no_skip_xml: bool = typer.Option(False, "--no-skip-xml", help="Process XML files."),
        # ── Filter toggles ────────────────────────────────────────────
        no_pii: bool = typer.Option(False, "--no-pii"),
        no_quality: bool = typer.Option(False, "--no-quality"),
        no_spam: bool = typer.Option(False, "--no-spam"),
        no_exact_dedup: bool = typer.Option(False, "--no-exact-dedup"),
        no_near_dedup: bool = typer.Option(False, "--no-near-dedup"),
        # ── Threshold overrides ───────────────────────────────────────
        near_dedup_threshold: float = typer.Option(0.80, "--nd-threshold"),
        near_dedup_num_perm: int = typer.Option(128, "--nd-perm"),
        min_doc_chars: int = typer.Option(100, "--min-chars"),
        min_word_count: int = typer.Option(20, "--min-words"),
        max_rep_ratio: float = typer.Option(
            0.20, "--max-rep",
            help="Max n-gram repetition ratio (0-1). Documents above this are dropped.",
        ),
        blacklist_domains: Path | None = typer.Option(None, "--blacklist-domains"),
        blacklist_keywords: Path | None = typer.Option(None, "--blacklist-keywords"),
        # ── Prose filter options ──────────────────────────────────────
        no_column_merge: bool = typer.Option(
            False, "--no-column-merge",
            help="Disable column merge detection (PDF two-column corruption).",
        ),
        no_prose_filter: bool = typer.Option(
            False, "--no-prose-filter",
            help="Disable semantic prose filtering (table/number detection).",
        ),
        max_digit_ratio: float = typer.Option(
            0.15, "--max-digit-ratio",
            help="Max digit/alpha char ratio per chunk (0-1).",
        ),
        max_number_word_ratio: float = typer.Option(
            0.25, "--max-num-ratio",
            help="Max fraction of number tokens per chunk (0-1).",
        ),
        gpu_perplexity: bool = typer.Option(
            True, "--gpu-perplexity/--no-gpu-perplexity",
            help="Enable GPU perplexity scoring (requires torch + transformers).",
        ),
        perplexity_model: str = typer.Option(
            "facebook/xglm-564M", "--ppl-model",
            help="HF model for perplexity scoring.",
        ),
        perplexity_max: float = typer.Option(
            1500.0, "--ppl-max",
            help="Max perplexity threshold (above = incoherent).",
        ),
        # ── Docling format toggles ─────────────────────────────────
        docling_formats: str = typer.Option(
            "pdf,html", "--docling-formats",
            help=(
                    "Comma-separated formats to process with Docling "
                    "(e.g., pdf,html). Empty string disables Docling."
            ),
        ),
        no_docling: bool = typer.Option(
            False, "--no-docling",
            help="Disable Docling for all formats (legacy pymupdf/trafilatura).",
        ),
        # ── Typo detection ────────────────────────────────────────
        typo_log: Path | None = typer.Option(
            None, "--typo-log",
            help="Log suspicious words (typos/OCR errors) to this CSV file for review.",
        ),
        # ── Cache / incremental ────────────────────────────────────
        no_cache: bool = typer.Option(
            False, "--no-cache",
            help="Ignore journal cache and reprocess all files from scratch.",
        ),
) -> None:
    """Build a pre-training corpus from local files."""

    # Resolve XML skip
    actual_skip_xml = skip_xml and not no_skip_xml

    max_chunk_bytes = int(max_gb * 1024 ** 3)
    temp_dir = output / ".tmp_shuffle"

    # ── Discover source files ─────────────────────────────────────
    console.print()
    file_map = discover_files(source)

    if not file_map:
        console.print(
            f"[bold red]Error:[/] No supported files found in {source}"
        )
        raise typer.Exit(code=1)

    # ── Source limits (giant-doc skip + per-domain volume cap) ────
    file_map, n_dropped_big, n_capped = _apply_source_limits(
        file_map, max_file_mb, max_source_gb, seed,
    )
    if n_dropped_big:
        console.print(
            f"  [yellow]↳[/] Skipped {n_dropped_big:,} oversize file(s) "
            f"(> {max_file_mb:.0f} MB)"
        )
    if n_capped:
        console.print(
            f"  [yellow]↳[/] Source cap {max_source_gb:.1f} GB: "
            f"dropped {n_capped:,} file(s) beyond budget"
        )
    if not any(file_map.values()):
        console.print("[bold red]Error:[/] No files left after applying source limits.")
        raise typer.Exit(code=1)

    # ── Pipeline config ───────────────────────────────────────────
    if config and config.exists():
        pipe_cfg = PipelineConfig.from_yaml(config)
    else:
        pipe_cfg = PipelineConfig()

    pipe_cfg.pii_enabled = not no_pii
    pipe_cfg.quality_enabled = not no_quality
    pipe_cfg.spam_enabled = not no_spam
    # Dedup disabled in Phase 1a pipeline — runs fresh in Phase 1b
    pipe_cfg.exact_dedup_enabled = False
    pipe_cfg.near_dedup_enabled = False
    pipe_cfg.near_dedup_threshold = near_dedup_threshold
    pipe_cfg.near_dedup_num_perm = near_dedup_num_perm
    pipe_cfg.min_doc_chars = min_doc_chars
    pipe_cfg.min_word_count = min_word_count
    if blacklist_domains:
        pipe_cfg.blacklist_domains_path = str(blacklist_domains)
    if blacklist_keywords:
        pipe_cfg.blacklist_keywords_path = str(blacklist_keywords)

    # Separate dedup config — used in Phase 1b with original user flags
    dedup_cfg = PipelineConfig()
    dedup_cfg.exact_dedup_enabled = not no_exact_dedup
    dedup_cfg.near_dedup_enabled = not no_near_dedup
    dedup_cfg.near_dedup_threshold = near_dedup_threshold
    dedup_cfg.near_dedup_num_perm = near_dedup_num_perm

    pipe = Pipeline(pipe_cfg)

    # ── Cleaner config ────────────────────────────────────────────
    cleaner_cfg = CleanerConfig()
    repetition_scorer = RepetitionScorer(ngram_size=cleaner_cfg.repetition_ngram_size)

    # ── Column merge cleaner ──────────────────────────────────────
    col_merge = ColumnMergeCleaner() if not no_column_merge else None

    # ── Prose filter config ───────────────────────────────────────
    prose_cfg = ProseConfig(
        max_digit_ratio=max_digit_ratio,
        max_number_word_ratio=max_number_word_ratio,
        perplexity_enabled=gpu_perplexity,
        perplexity_model=perplexity_model,
        perplexity_max=perplexity_max,
    )
    prose_filt = ProseFilter(prose_cfg) if not no_prose_filter else None

    # ── Docling formats ────────────────────────────────────────────
    if no_docling:
        docling_fmt_list: list[str] = []
    else:
        docling_fmt_list = [
            f.strip().lower()
            for f in docling_formats.split(",")
            if f.strip()
        ]

    # ── Extractor registry ────────────────────────────────────────
    registry = ExtractorRegistry(
        search_keys=search_keys,
        cleaner_cfg=cleaner_cfg,
        skip_xml=actual_skip_xml,
        skip_html=skip_html,
        docling_formats=docling_fmt_list,
    )

    # ── Journal ────────────────────────────────────────────────────
    journal = Journal(output, enabled=not no_cache)

    # ── Banner ────────────────────────────────────────────────────
    banner = Table.grid(padding=(0, 2))
    banner.add_row("[bold]Source", str(source.resolve()))
    banner.add_row("[bold]Output", str(output.resolve()))
    banner.add_row("[bold]Max chunk", _fmt_bytes(max_chunk_bytes))
    if max_file_mb > 0:
        banner.add_row("[bold]Max file", f"{max_file_mb:.0f} MB [dim](json/html/xml/pdf)[/]")
    if max_source_gb > 0:
        banner.add_row("[bold]Source cap", f"{max_source_gb:.1f} GB [dim](sampled, seed {seed})[/]")
    banner.add_row("[bold]Seed", str(seed))
    if journal.enabled:
        banner.add_row(
            "[bold]Cache",
            f"[green]enabled[/] [dim]({journal.n_entries} cached)[/]",
        )
    else:
        banner.add_row("[bold]Cache", "[yellow]disabled (--no-cache)[/]")
    if docling_fmt_list:
        docling_label = ", ".join(f.upper() for f in docling_fmt_list)
        banner.add_row("[bold]Docling formats", f"[green]{docling_label}[/]")
    else:
        banner.add_row("[bold]Docling", "[yellow]disabled (legacy)[/]")
    banner.add_row("[bold]Format", f"{BOS_TOKEN}text{EOS_TOKEN}")
    banner.add_row(
        "[bold]Search keys",
        ", ".join(f"[yellow]{k}[/]" for k in search_keys),
    )

    # Formats
    fmt_list: list[str] = []
    for fmt in ("pdf", "json", "jsonl", "txt"):
        if fmt in file_map:
            fmt_list.append(f"[green]{fmt.upper()}[/]")
    if not skip_html and "html" in file_map:
        fmt_list.append("[green]HTML[/]")
    if not actual_skip_xml and "xml" in file_map:
        fmt_list.append("[green]XML[/]")
    for fmt in ("html", "xml"):
        if fmt in file_map and (
                (fmt == "html" and skip_html) or (fmt == "xml" and actual_skip_xml)
        ):
            fmt_list.append(f"[dim]{fmt.upper()} (skipped)[/]")
    banner.add_row("[bold]Formats", "  ".join(fmt_list) or "[dim]none")

    # Filters
    active: list[str] = []
    if pipe_cfg.pii_enabled:
        active.append("[green]PII[/]")
    if pipe_cfg.quality_enabled:
        active.append("[green]Quality[/]")
    if pipe_cfg.spam_enabled:
        active.append("[green]Spam[/]")
    if dedup_cfg.exact_dedup_enabled:
        active.append("[green]ExactDedup[/]")
    if dedup_cfg.near_dedup_enabled:
        active.append(
            f"[green]NearDedup[/] [dim](J≥{dedup_cfg.near_dedup_threshold})"
        )
    active.append(f"[green]Repetition[/] [dim](≤{max_rep_ratio})")
    if not no_column_merge:
        active.append("[green]ColumnMerge[/]")
    if not no_prose_filter:
        pf_label = "[green]ProseFilter[/] [dim](dig≤{:.0%} num≤{:.0%})".format(
            max_digit_ratio, max_number_word_ratio,
        )
        if gpu_perplexity:
            pf_label += f" [dim]+GPU PPL[/]"
        active.append(pf_label)
    banner.add_row("[bold]Filters", "  →  ".join(active))

    console.print(
        Panel(
            banner,
            title="[bold white]📚  Local Corpus Builder[/]",
            border_style="bright_blue",
            padding=(1, 3),
        )
    )
    console.print()

    _print_source_scan(file_map)

    t0 = time.perf_counter()

    # ── Phase 1a — Extract & Filter (incremental) ─────────────────
    # If prose filter is disabled, create a passthrough that just flattens
    if prose_filt is None:
        prose_filt = _PassthroughProseFilter()

    # ── Typo detector (optional) ─────────────────────────────
    typo_det = TypoDetector(typo_log) if typo_log else None

    all_docs, extraction_stats, cache_hits, cache_misses = phase1a_extract_and_filter(
        file_map=file_map,
        registry=registry,
        col_merge_cleaner=col_merge,
        prose_filter=prose_filt,
        pipe=pipe,
        repetition_scorer=repetition_scorer,
        max_rep_ratio=max_rep_ratio,
        journal=journal,
        typo_detector=typo_det,
    )

    # Flush any batched journal entries not yet persisted (see Journal._save_every).
    journal.save()

    if len(all_docs) == 0:
        console.print("[bold red]No documents survived the pipeline. Exiting.[/]")
        raise typer.Exit(code=1)

    # ── Phase 1b — Dedup & Chunk (always fresh) ──────────────────
    temp_files, indices, dedup_stats = phase1b_dedup_and_chunk(
        all_docs=all_docs,
        pipe_cfg=dedup_cfg,
        temp_dir=temp_dir,
        max_chunk_bytes=max_chunk_bytes,
    )

    # Free docs from memory — they're now in temp chunks
    del all_docs

    if len(indices) == 0:
        console.print("[bold red]No documents survived dedup. Exiting.[/]")
        phase4_cleanup(temp_dir)
        raise typer.Exit(code=1)

    # ── Phase 2 ───────────────────────────────────────────────────
    phase2_shuffle(indices, seed)

    # ── Phase 3 — Clean old output files, write shuffled ─────────
    # Remove old corpus files (but not .cache or .tmp_shuffle)
    for old_file in output.glob("*.txt"):
        old_file.unlink()

    output_files = phase3_write_shuffled(
        indices, temp_files, output, max_chunk_bytes,
    )

    # ── Phase 4 ───────────────────────────────────────────────────
    phase4_cleanup(temp_dir)

    # ── Final summary ─────────────────────────────────────────────
    elapsed = time.perf_counter() - t0

    # Merge pipeline stats (Phase 1a filters + Phase 1b dedup)
    total_docs_extracted = sum(d for _, d in extraction_stats.values())
    total_seen_pipeline = pipe.stats.total_seen
    total_seen_dedup = dedup_stats.total_seen

    summary = Table(
        title="[bold]Final Summary",
        border_style="green",
        show_lines=True,
    )
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")

    summary.add_row("Files scanned", f"{sum(f for f, _ in extraction_stats.values()):,}")
    summary.add_row("Docs extracted", f"{total_docs_extracted:,}")
    summary.add_row(
        "Cache",
        f"[green]{cache_hits} hits[/] / {cache_misses} misses"
        if journal.enabled else "[dim]disabled[/]",
    )
    summary.add_row("Docs after filters", f"{total_seen_dedup:,}")
    summary.add_row("Docs rejected (dedup)", f"[red]{dedup_stats.total_rejected:,}[/]")
    summary.add_row("Docs kept", f"[green]{dedup_stats.total_kept:,}[/]")
    summary.add_row(
        "Keep rate (overall)",
        f"[green]{_pct(dedup_stats.total_kept, total_docs_extracted)}[/]",
    )
    summary.add_row("Output files", str(len(output_files)))

    total_out = sum(f.stat().st_size for f in output_files)
    summary.add_row("Total size", _fmt_bytes(total_out))
    summary.add_row("RAM (index)", _fmt_bytes(indices.nbytes))
    summary.add_row("Elapsed", f"{elapsed:.1f}s")
    summary.add_row(
        "Throughput",
        f"{total_docs_extracted / elapsed:,.0f} docs/s" if elapsed > 0 else "—",
    )

    console.print()
    console.print(summary)
    console.print()

    for f in output_files:
        console.print(f"  [dim]📄[/] {f.resolve()}")

    console.print(
        f"\n[bold green]Done![/] "
        f"{len(output_files)} file{'s' if len(output_files) > 1 else ''} "
        f"ready in [cyan]{output.resolve()}[/]\n"
    )


if __name__ == "__main__":
    app()
