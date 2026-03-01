#!/usr/bin/env python3
# filepath: ted_pipeline.py
"""TED Europa full pipeline: download → extract → parse → pretrain.txt

Combines ted_downloader.py and ted_processor.py into a single streamed
pipeline that processes each package as it downloads, avoiding disk bloat.
"""

from __future__ import annotations

import shutil
import sys
import time
import zipfile
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
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
    BASE_URL,
    CHUNK_SIZE,
    DEFAULT_TIMEOUT,
    USER_AGENT,
    Journal,
    PackageStatus,
    build_url,
    generate_package_keys,
    safe_extract_zip,
    show_summary,
)
from ted_processor import (
    ProcessingStats,
    extract_recursive,
    process_package,
    write_pretrain_batch,
)


console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger("ted_pipeline")


app = typer.Typer(
    name="ted-pipeline",
    help="Download + process TED data into Italian pretrain corpus in one pass.",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command()
def run(
    dest: Path = typer.Option(Path("./ted_data"), "--dest", "-d", help="Working directory."),
    output: Path = typer.Option(Path("./.datasets/pretrain/ted/pretrain_ted_it.txt"), "--output", "-o", help="Output pretrain file."),
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
    """Download each monthly package, process it for Italian text, then clean up.

    This is a streaming pipeline: each package is downloaded, extracted,
    parsed for Italian content, appended to pretrain.txt, then the raw
    data is removed to save disk space.
    """
    if verbose:
        log.setLevel(logging.DEBUG)

    dest.mkdir(parents=True, exist_ok=True)
    journal = Journal(dest / "ted_journal.json")
    proc_stats = ProcessingStats()

    packages = generate_package_keys(start_year, start_month, end_year, end_month)

    # Filter to pending packages
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
        console.print("[green]Nothing to do!")
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
    file_progress = Progress(
        TextColumn("  {task.description}"),
        BarColumn(bar_width=30),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    )

    task = overall.add_task("Pipeline", total=len(to_process))
    total_docs = 0

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    transport = httpx.HTTPTransport(retries=2)

    with httpx.Client(
        headers=headers,
        timeout=httpx.Timeout(timeout, connect=30.0),
        transport=transport,
    ) as client:
        with overall:
            for idx, (year, month, key) in enumerate(to_process):
                overall.update(task, description=f"[bold blue]{key}")
                pkg_dir = dest / f"{year}-{month:02d}"
                zip_path = dest / f"{year}-{month:02d}.zip"
                status = PackageStatus.FAILED
                error_msg = ""
                attempts = 0

                # ── Download ──────────────────────────────────────────
                while attempts < max_retries:
                    attempts += 1
                    try:
                        url = build_url(year, month)
                        with client.stream("GET", url, follow_redirects=True) as resp:
                            if resp.status_code == 404:
                                status = PackageStatus.SKIPPED
                                break
                            resp.raise_for_status()

                            total_size = int(resp.headers.get("content-length", 0)) or None
                            dl_task = file_progress.add_task(f"[cyan]↓ {key}", total=total_size)

                            with zip_path.open("wb") as f:
                                for chunk in resp.iter_bytes(chunk_size=CHUNK_SIZE):
                                    f.write(chunk)
                                    file_progress.update(dl_task, advance=len(chunk))
                            file_progress.remove_task(dl_task)

                        # ── Extract & Process ─────────────────────────
                        if zipfile.is_zipfile(zip_path):
                            pkg_dir.mkdir(parents=True, exist_ok=True)
                            safe_extract_zip(zip_path, pkg_dir)
                            zip_path.unlink(missing_ok=True)

                            docs = process_package(pkg_dir, proc_stats)
                            if docs:
                                write_pretrain_batch(docs, output, append=True)
                                total_docs += len(docs)
                                log.info("%s → %d Italian docs", key, len(docs))

                            # Cleanup extracted data
                            if not keep_extracted:
                                shutil.rmtree(pkg_dir, ignore_errors=True)

                            status = PackageStatus.EXTRACTED
                        else:
                            error_msg = "Invalid ZIP"
                            zip_path.unlink(missing_ok=True)

                        break

                    except (httpx.HTTPStatusError, httpx.TransportError, OSError) as exc:
                        error_msg = f"{type(exc).__name__}: {exc}"
                        zip_path.unlink(missing_ok=True)
                        if attempts < max_retries:
                            backoff = delay * (2 ** (attempts - 1))
                            log.warning("Retry %d/%d for %s in %.0fs", attempts, max_retries, key, backoff)
                            time.sleep(backoff)

                journal.set_status(key, status, error=error_msg or None, attempts=attempts)
                journal.save()
                overall.advance(task)

                if idx < len(to_process) - 1 and status != PackageStatus.SKIPPED:
                    time.sleep(delay)

    # ── Final summary ─────────────────────────────────────────────────────
    console.print()
    show_summary(journal)

    if output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        console.print(
            f"\n[bold green]Corpus: {output} "
            f"({size_mb:.1f} MB, {total_docs} new Italian documents)"
        )


if __name__ == "__main__":
    app()