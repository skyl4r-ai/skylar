#!/usr/bin/env python3
# filepath: ted_downloader.py
"""TED Europa bulk data downloader.

Downloads monthly data packages from ted.europa.eu spanning 1993 to 2026,
with journal-based resumption, rate limiting, and automatic extraction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import tarfile
import time
import zipfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.live import Live
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
from rich.table import Table
from rich.text import Text

try:
    import typer
except ImportError:
    print("Missing dependency: typer. Install with: pip install typer")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://ted.europa.eu/packages/monthly"
USER_AGENT = (
    "Mozilla/5.0 (compatible; TED-Downloader/1.0; "
    "+https://github.com/your-org/ted-downloader)"
)
JOURNAL_FILENAME = "ted_journal.json"
DEFAULT_TIMEOUT = 120.0
CHUNK_SIZE = 1024 * 64  # 64 KB

console = Console()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger("ted_downloader")


# ---------------------------------------------------------------------------
# Journal – tracks every package status across runs
# ---------------------------------------------------------------------------

class PackageStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    EXTRACTED = "extracted"
    FAILED = "failed"
    SKIPPED = "skipped"  # e.g. 404 – package doesn't exist


class Journal:
    """Persistent JSON journal for download state tracking."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Corrupted journal, starting fresh: %s", exc)
        return {"packages": {}, "last_run": None, "stats": {}}

    def save(self) -> None:
        self.data["last_run"] = datetime.now().isoformat()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def get_status(self, key: str) -> PackageStatus | None:
        pkg = self.data["packages"].get(key)
        if pkg is None:
            return None
        return PackageStatus(pkg["status"])

    def set_status(
        self,
        key: str,
        status: PackageStatus,
        *,
        error: str | None = None,
        attempts: int | None = None,
    ) -> None:
        entry = self.data["packages"].setdefault(key, {})
        entry["status"] = status.value
        entry["updated"] = datetime.now().isoformat()
        if error is not None:
            entry["last_error"] = error
        if attempts is not None:
            entry["attempts"] = attempts

    def count_by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pkg in self.data["packages"].values():
            s = pkg["status"]
            counts[s] = counts.get(s, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_package_keys(
    start_year: int, start_month: int, end_year: int, end_month: int
) -> list[tuple[int, int, str]]:
    """Generate (year, month, key) tuples for the requested range."""
    keys: list[tuple[int, int, str]] = []
    for year in range(start_year, end_year + 1):
        m_start = start_month if year == start_year else 1
        m_end = end_month if year == end_year else 12
        for month in range(m_start, m_end + 1):
            keys.append((year, month, f"{year}-{month}"))
    return keys


def build_url(year: int, month: int) -> str:
    return f"{BASE_URL}/{year}-{month}"


def safe_extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract ZIP avoiding path-traversal attacks (Zip Slip)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = dest / member.filename
            resolved = target.resolve()
            if not str(resolved).startswith(str(dest.resolve())):
                raise zipfile.BadZipFile(
                    f"Attempted path traversal in zip: {member.filename}"
                )
        zf.extractall(dest)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Downloader core
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

    if header[:2] == b"\x1f\x8b":
        return "tar.gz"
    if header[:4] == b"PK\x03\x04":
        return "zip"
    if len(header) >= 263 and header[257:262] == b"ustar":
        return "tar"
    return "unknown"


def extract_archive(archive_path: Path, dest: Path) -> None:
    """Extract any supported archive (tar.gz, tar, zip) with path traversal protection."""
    fmt = detect_archive_format(archive_path)
    log.info("Extracting %s (format: %s)", archive_path.name, fmt)

    if fmt in ("tar.gz", "tar"):
        mode = "r:gz" if fmt == "tar.gz" else "r:"
        with tarfile.open(str(archive_path), mode) as tf:
            for member in tf.getmembers():
                resolved = (dest / member.name).resolve()
                if not str(resolved).startswith(str(dest.resolve())):
                    raise tarfile.TarError(f"Path traversal: {member.name}")
            tf.extractall(dest, filter="data")

    elif fmt == "zip":
        safe_extract_zip(archive_path, dest)

    else:
        raise ValueError(f"Unknown archive format for {archive_path.name}")


def download_package(
    client: httpx.Client,
    year: int,
    month: int,
    dest_dir: Path,
    *,
    keep_archive: bool = False,
    file_progress: Progress,
) -> PackageStatus:
    """Download and extract a single monthly package.

    Auto-detects archive format (tar.gz or zip) from magic bytes.
    Returns the resulting PackageStatus.
    """
    url = build_url(year, month)
    package_dir = dest_dir / f"{year}-{month:02d}"
    # Generic temp name – format detected after download
    archive_path = dest_dir / f"_tmp_{year}-{month:02d}.bin"

    # If already fully extracted, skip
    if package_dir.exists() and any(package_dir.iterdir()):
        log.info("Already extracted: %s", package_dir.name)
        return PackageStatus.EXTRACTED

    package_dir.mkdir(parents=True, exist_ok=True)

    # -- Download --------------------------------------------------------
    try:
        with client.stream("GET", url, follow_redirects=True) as resp:
            if resp.status_code == 404:
                log.warning("Not found (404): %s", url)
                shutil.rmtree(package_dir, ignore_errors=True)
                return PackageStatus.SKIPPED

            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0)) or None
            task = file_progress.add_task(
                f"[cyan]{year}-{month:02d}",
                total=total,
            )

            with archive_path.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    file_progress.update(task, advance=len(chunk))

            file_progress.remove_task(task)

    except httpx.HTTPStatusError as exc:
        log.error("HTTP %s for %s", exc.response.status_code, url)
        archive_path.unlink(missing_ok=True)
        raise
    except httpx.TransportError as exc:
        log.error("Network error for %s: %s", url, exc)
        archive_path.unlink(missing_ok=True)
        raise

    # -- Detect format & extract -----------------------------------------
    fmt = detect_archive_format(archive_path)
    if fmt == "unknown":
        log.error("Unknown archive format for %s", archive_path.name)
        archive_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is not a valid archive: {archive_path}")

    log.info("Extracting %s (detected: %s) …", archive_path.name, fmt)
    extract_archive(archive_path, package_dir)

    sha = file_sha256(archive_path)
    log.debug("SHA-256: %s", sha)

    if not keep_archive:
        archive_path.unlink(missing_ok=True)

    return PackageStatus.EXTRACTED


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def show_summary(journal: Journal) -> None:
    counts = journal.count_by_status()
    table = Table(
        title="Download Summary",
        title_style="bold magenta",
        show_lines=True,
        padding=(0, 2),
    )
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")

    status_styles = {
        "extracted": "green",
        "downloaded": "blue",
        "skipped": "yellow",
        "failed": "red",
        "pending": "dim",
    }

    for status, count in sorted(counts.items()):
        style = status_styles.get(status, "white")
        table.add_row(f"[{style}]{status.upper()}", f"[{style}]{count}")

    total = sum(counts.values())
    table.add_row("[bold]TOTAL", f"[bold]{total}")
    console.print(table)

    # Show failed packages if any
    failed = [
        (k, v)
        for k, v in journal.data["packages"].items()
        if v["status"] == PackageStatus.FAILED.value
    ]
    if failed:
        console.print()
        err_table = Table(
            title="[red]Failed Packages",
            title_style="bold red",
            show_lines=True,
        )
        err_table.add_column("Package")
        err_table.add_column("Error")
        err_table.add_column("Attempts", justify="right")
        for key, info in failed:
            err_table.add_row(
                key,
                info.get("last_error", "unknown"),
                str(info.get("attempts", "?")),
            )
        console.print(err_table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ted-downloader",
    help="Download monthly data packages from ted.europa.eu",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command()
def download(
    dest: Path = typer.Option(
        Path("./ted_data"),
        "--dest", "-d",
        help="Destination directory for downloaded packages.",
    ),
    start_year: int = typer.Option(1993, "--start-year", "-sy", help="First year to download."),
    start_month: int = typer.Option(1, "--start-month", "-sm", help="First month to download."),
    end_year: int = typer.Option(2026, "--end-year", "-ey", help="Last year to download."),
    end_month: int = typer.Option(1, "--end-month", "-em", help="Last month to download."),
    delay: float = typer.Option(
        2.0, "--delay", help="Seconds to wait between downloads (rate-limiting)."
    ),
    max_retries: int = typer.Option(3, "--retries", "-r", help="Max retry attempts per package."),
    keep_archive: bool = typer.Option(False, "--keep-archive", help="Keep archive files after extraction."),
    retry_failed: bool = typer.Option(
        True, "--retry-failed/--skip-failed", help="Retry previously failed packages."
    ),
    timeout: float = typer.Option(
        DEFAULT_TIMEOUT, "--timeout", "-t", help="HTTP request timeout in seconds."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Download and extract TED Europa monthly data packages."""

    if verbose:
        log.setLevel(logging.DEBUG)

    # -- Validate --------------------------------------------------------
    now = datetime.now()
    if start_year < 1993 or end_year > now.year + 1:
        console.print("[red]Year range out of bounds (1993 – current+1).")
        raise typer.Exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    journal_path = dest / JOURNAL_FILENAME
    journal = Journal(journal_path)

    packages = generate_package_keys(start_year, start_month, end_year, end_month)

    # -- Figure out what needs to be done --------------------------------
    to_download: list[tuple[int, int, str]] = []
    for year, month, key in packages:
        status = journal.get_status(key)
        if status == PackageStatus.EXTRACTED:
            continue
        if status == PackageStatus.SKIPPED:
            continue
        if status == PackageStatus.FAILED and not retry_failed:
            continue
        to_download.append((year, month, key))

    # -- Banner ----------------------------------------------------------
    banner = Text.assemble(
        ("TED Europa Downloader\n", "bold magenta"),
        (f"Range    : {start_year}-{start_month:02d} → {end_year}-{end_month:02d}\n", "cyan"),
        (f"Packages : {len(packages)} total, ", "cyan"),
        (f"{len(to_download)} to download\n", "green" if to_download else "dim"),
        (f"Dest     : {dest.resolve()}\n", "cyan"),
        (f"Delay    : {delay}s between requests\n", "cyan"),
        (f"Retries  : {max_retries}\n", "cyan"),
        (f"Keep arc : {'yes' if keep_archive else 'no'}", "cyan"),
    )
    console.print(Panel(banner, border_style="blue", padding=(1, 2)))

    if not to_download:
        console.print("[green]Nothing to do – all packages already processed!")
        show_summary(journal)
        raise typer.Exit(0)

    # -- Progress bars ---------------------------------------------------
    overall_progress = Progress(
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

    overall_task = overall_progress.add_task("Downloading", total=len(to_download))

    # -- HTTP client -----------------------------------------------------
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    transport = httpx.HTTPTransport(retries=2)

    with httpx.Client(
        headers=headers,
        timeout=httpx.Timeout(timeout, connect=30.0),
        transport=transport,
    ) as client:
        with Live(
            Panel(
                overall_progress,
                title="[bold]Overall Progress",
                border_style="blue",
            ),
            console=console,
            refresh_per_second=4,
        ):
            for idx, (year, month, key) in enumerate(to_download):
                overall_progress.update(
                    overall_task,
                    description=f"[bold blue]Downloading [cyan]{key}[/cyan]",
                )

                attempts = 0
                status = PackageStatus.FAILED
                last_error = ""

                while attempts < max_retries:
                    attempts += 1
                    try:
                        status = download_package(
                            client,
                            year,
                            month,
                            dest,
                            keep_archive=keep_archive,
                            file_progress=file_progress,
                        )
                        last_error = ""
                        break
                    except (
                        httpx.HTTPStatusError,
                        httpx.TransportError,
                        tarfile.TarError,
                        zipfile.BadZipFile,
                        ValueError,
                        OSError,
                    ) as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        if attempts < max_retries:
                            backoff = delay * (2 ** (attempts - 1))
                            log.warning(
                                "Attempt %d/%d failed for %s, retrying in %.1fs…",
                                attempts,
                                max_retries,
                                key,
                                backoff,
                            )
                            time.sleep(backoff)
                        else:
                            log.error(
                                "Giving up on %s after %d attempts: %s",
                                key,
                                max_retries,
                                last_error,
                            )

                journal.set_status(
                    key,
                    status,
                    error=last_error or None,
                    attempts=attempts,
                )
                journal.save()

                overall_progress.advance(overall_task)

                # Rate limit – skip delay after last item
                if idx < len(to_download) - 1 and status != PackageStatus.SKIPPED:
                    time.sleep(delay)

    # -- Done ------------------------------------------------------------
    console.print()
    show_summary(journal)

    failed_count = sum(
        1
        for v in journal.data["packages"].values()
        if v["status"] == PackageStatus.FAILED.value
    )
    if failed_count:
        console.print(
            f"\n[yellow]Run again with [bold]--retry-failed[/bold] to retry "
            f"{failed_count} failed package(s)."
        )


@app.command()
def status(
    dest: Path = typer.Option(
        Path("./ted_data"),
        "--dest", "-d",
        help="Directory containing the journal.",
    ),
) -> None:
    """Show current journal status without downloading."""
    journal_path = dest / JOURNAL_FILENAME
    if not journal_path.exists():
        console.print("[yellow]No journal found. Run [bold]download[/bold] first.")
        raise typer.Exit(1)

    journal = Journal(journal_path)
    if journal.data.get("last_run"):
        console.print(f"[dim]Last run: {journal.data['last_run']}")
    show_summary(journal)


@app.command()
def reset(
    dest: Path = typer.Option(
        Path("./ted_data"),
        "--dest", "-d",
        help="Directory containing the journal.",
    ),
    only_failed: bool = typer.Option(
        False, "--only-failed", help="Reset only failed packages to pending."
    ),
) -> None:
    """Reset journal (all or only failed packages)."""
    journal_path = dest / JOURNAL_FILENAME
    if not journal_path.exists():
        console.print("[yellow]No journal found.")
        raise typer.Exit(1)

    journal = Journal(journal_path)

    if only_failed:
        count = 0
        for key, info in journal.data["packages"].items():
            if info["status"] == PackageStatus.FAILED.value:
                info["status"] = PackageStatus.PENDING.value
                info.pop("last_error", None)
                info.pop("attempts", None)
                count += 1
        journal.save()
        console.print(f"[green]Reset {count} failed package(s) to pending.")
    else:
        journal_path.unlink()
        console.print("[green]Journal deleted. Next run will start fresh.")


if __name__ == "__main__":
    app()