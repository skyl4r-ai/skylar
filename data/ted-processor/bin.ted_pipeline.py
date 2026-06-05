#!/usr/bin/env python3
# filepath: ted_pipeline.py
"""TED Europa full pipeline: download → extract → parse → pretrain.txt

TED monthly packages come as tar.gz (most common) or zip.
This detects the format from magic bytes, NOT the extension.
"""

from __future__ import annotations

import gzip
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path

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
    TimeRemainingColumn,
)
from rich.text import Text

import logging

try:
    import httpx
    import typer
except ImportError:
    print("Missing dependencies. Install: pip install httpx typer rich")
    sys.exit(1)

from ted_downloader import (
    CHUNK_SIZE,
    DEFAULT_TIMEOUT,
    USER_AGENT,
    Journal,
    PackageStatus,
    build_url,
    generate_package_keys,
    show_summary,
)
from ted_processor import (
    ProcessingStats,
    process_package,
    write_pretrain_batch,
    _show_summary as show_proc_summary,
)


console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger("ted_pipeline")


# ---------------------------------------------------------------------------
# Archive format detection by magic bytes
# ---------------------------------------------------------------------------

def detect_archive_format(file_path: Path) -> str:
    """Detect actual archive format by reading magic bytes.

    Returns: 'tar.gz', 'tar', 'zip', or 'unknown'
    """
    try:
        with file_path.open("rb") as f:
            header = f.read(512)
    except OSError:
        return "unknown"

    # GZIP magic: 1f 8b
    if header[:2] == b"\x1f\x8b":
        return "tar.gz"

    # ZIP magic: PK\x03\x04
    if header[:4] == b"PK\x03\x04":
        return "zip"

    # TAR: 'ustar' at offset 257
    if len(header) >= 263 and header[257:262] == b"ustar":
        return "tar"

    return "unknown"


def extract_top_level(archive_path: Path, dest: Path) -> bool:
    """Extract the downloaded archive (tar.gz or zip) into dest."""
    fmt = detect_archive_format(archive_path)
    log.info("  Format: [bold cyan]%s[/] (%s)", fmt, archive_path.name, extra={"markup": True})

    dest.mkdir(parents=True, exist_ok=True)

    try:
        if fmt in ("tar.gz", "tar"):
            mode = "r:gz" if fmt == "tar.gz" else "r:"
            with tarfile.open(str(archive_path), mode) as tf:
                for member in tf.getmembers():
                    resolved = (dest / member.name).resolve()
                    if not str(resolved).startswith(str(dest.resolve())):
                        log.error("  Path traversal blocked: %s", member.name)
                        return False
                tf.extractall(dest, filter="data")
            return True

        elif fmt == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.infolist():
                    resolved = (dest / member.filename).resolve()
                    if not str(resolved).startswith(str(dest.resolve())):
                        log.error("  Path traversal blocked: %s", member.filename)
                        return False
                zf.extractall(dest)
            return True

        else:
            log.error("  Unknown archive format (magic bytes don't match gzip/zip/tar)")
            return False

    except (tarfile.TarError, zipfile.BadZipFile, gzip.BadGzipFile, OSError) as exc:
        log.error("  Extraction failed: %s", exc)
        return False


def count_files_recursive(directory: Path) -> dict[str, int]:
    """Count files by extension in a directory tree."""
    counts: dict[str, int] = {}
    for p in directory.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower() if p.suffix else "(no ext)"
            counts[ext] = counts.get(ext, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ted-pipeline",
    help="Download + process TED data into Italian pretrain corpus.",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command()
def run(
    dest: Path = typer.Option(Path("./ted_data"), "--dest", "-d", help="Working directory."),
    output: Path = typer.Option(Path("./pretrain_ted_it.txt"), "--output", "-o", help="Output pretrain file."),
    start_year: int = typer.Option(1993, "--start-year", "-sy"),
    start_month: int = typer.Option(1, "--start-month", "-sm"),
    end_year: int = typer.Option(2026, "--end-year", "-ey"),
    end_month: int = typer.Option(1, "--end-month", "-em"),
    delay: float = typer.Option(2.0, "--delay", help="Seconds between downloads."),
    max_retries: int = typer.Option(3, "--retries", "-r"),
    keep_extracted: bool = typer.Option(False, "--keep/--no-keep", help="Keep extracted data after processing."),
    timeout: float = typer.Option(DEFAULT_TIMEOUT, "--timeout", "-t"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download → extract → parse Italian → append pretrain.txt → cleanup.

    For each package:
      1. Download (auto-detect tar.gz or zip from magic bytes)
      2. Extract top-level archive
      3. Recursively extract nested archives (tar.gz → tar.gz → zip)
      4. Parse notice files, filter Italian content (TI, AB, TX)
      5. Append to pretrain.txt as <bos>doc<eos>
      6. Remove raw data (unless --keep)
    """
    if verbose:
        log.setLevel(logging.DEBUG)

    dest.mkdir(parents=True, exist_ok=True)
    journal = Journal(dest / "ted_journal.json")
    proc_stats = ProcessingStats()

    packages = generate_package_keys(start_year, start_month, end_year, end_month)

    to_process: list[tuple[int, int, str]] = []
    for year, month, key in packages:
        status = journal.get_status(key)
        if status in (PackageStatus.EXTRACTED, PackageStatus.SKIPPED):
            continue
        to_process.append((year, month, key))

    banner = Text.assemble(
        ("TED Full Pipeline\n", "bold magenta"),
        (f"Range   : {start_year}-{start_month:02d} → {end_year}-{end_month:02d}\n", "cyan"),
        (f"Packages: {len(packages)} total, {len(to_process)} to process\n", "cyan"),
        (f"Output  : {output.resolve()}\n", "cyan"),
        (f"Cleanup : {'yes' if not keep_extracted else 'no'}", "cyan"),
    )
    console.print(Panel(banner, border_style="blue", padding=(1, 2)))

    if not to_process:
        console.print("[green]Nothing to do – all packages already processed!")
        show_summary(journal)
        return

    overall = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    task = overall.add_task("Pipeline", total=len(to_process))
    total_docs = 0

    # identity encoding: don't let httpx decompress, we want the raw tar.gz
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    transport = httpx.HTTPTransport(retries=2)

    with httpx.Client(
        headers=headers,
        timeout=httpx.Timeout(timeout, connect=30.0),
        transport=transport,
    ) as client:
        with overall:
            for idx, (year, month, key) in enumerate(to_process):
                overall.update(task, description=f"[bold blue]{key}")

                # Generic temp name – we detect format AFTER download
                download_path = dest / f"_tmp_{year}-{month:02d}.bin"
                pkg_dir = dest / f"{year}-{month:02d}"
                status = PackageStatus.FAILED
                error_msg = ""
                attempts = 0

                while attempts < max_retries:
                    attempts += 1
                    try:
                        url = build_url(year, month)
                        log.info("[bold]▼ %s[/]", key, extra={"markup": True})

                        with client.stream("GET", url, follow_redirects=True) as resp:
                            if resp.status_code == 404:
                                log.warning("  404 – package does not exist")
                                status = PackageStatus.SKIPPED
                                break
                            resp.raise_for_status()

                            ct = resp.headers.get("content-type", "?")
                            cl = resp.headers.get("content-length", "?")
                            log.info("  HTTP 200 | content-type=%s size=%s", ct, cl)

                            with download_path.open("wb") as f:
                                for chunk in resp.iter_bytes(chunk_size=CHUNK_SIZE):
                                    f.write(chunk)

                        actual_mb = download_path.stat().st_size / (1024 * 1024)
                        log.info("  Downloaded: %.1f MB", actual_mb)

                        # ── Detect & extract top-level ────────────────────
                        ok = extract_top_level(download_path, pkg_dir)
                        download_path.unlink(missing_ok=True)

                        if not ok:
                            error_msg = f"Extraction failed for {key}"
                            shutil.rmtree(pkg_dir, ignore_errors=True)
                            if attempts < max_retries:
                                backoff = delay * (2 ** (attempts - 1))
                                log.warning("  Retry %d/%d in %.0fs", attempts, max_retries, backoff)
                                time.sleep(backoff)
                                continue
                            break

                        # Show what we got after top-level extraction
                        file_counts = count_files_recursive(pkg_dir)
                        total_files = sum(file_counts.values())
                        log.info("  Top-level: %d files %s", total_files, dict(sorted(file_counts.items())))

                        # ── Recursive extract + parse ─────────────────────
                        log.info("  Recursive extraction + Italian parsing...")
                        docs = process_package(pkg_dir, proc_stats)

                        if docs:
                            written = write_pretrain_batch(docs, output, append=True)
                            total_docs += written
                            log.info(
                                "  [bold green]✓ %d Italian docs[/] (corpus total: %d)",
                                written, total_docs,
                                extra={"markup": True},
                            )
                        else:
                            log.info("  [dim]0 Italian docs in this package[/]", extra={"markup": True})

                        # Cleanup extracted data
                        if not keep_extracted:
                            shutil.rmtree(pkg_dir, ignore_errors=True)

                        status = PackageStatus.EXTRACTED
                        error_msg = ""
                        break

                    except (httpx.HTTPStatusError, httpx.TransportError, OSError) as exc:
                        error_msg = f"{type(exc).__name__}: {exc}"
                        download_path.unlink(missing_ok=True)
                        shutil.rmtree(pkg_dir, ignore_errors=True)
                        if attempts < max_retries:
                            backoff = delay * (2 ** (attempts - 1))
                            log.warning(
                                "  Attempt %d/%d failed: %s – retry in %.0fs",
                                attempts, max_retries, error_msg, backoff,
                            )
                            time.sleep(backoff)

                journal.set_status(key, status, error=error_msg or None, attempts=attempts)
                journal.save()
                overall.advance(task)

                if idx < len(to_process) - 1 and status != PackageStatus.SKIPPED:
                    time.sleep(delay)

    # ── Final summary ─────────────────────────────────────────────────────
    console.print()
    show_summary(journal)
    show_proc_summary(proc_stats, output)

    if output.exists() and output.stat().st_size > 0:
        size_mb = output.stat().st_size / (1024 * 1024)
        console.print(
            f"\n[bold green]✓ Corpus: {output} "
            f"({size_mb:.1f} MB, {total_docs} Italian documents)"
        )
    else:
        console.print("\n[yellow]⚠ No output file created – 0 Italian docs found so far.")
        console.print("[yellow]  Tip: run with --verbose to see parsing details.")


if __name__ == "__main__":
    app()