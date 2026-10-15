#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================

"""
Shard + clean di un grande TXT (es. Eur-Lex / DGT-Acquis / dump giurisprudenza).

Uso:
  python shard_clean.py --input big.txt --outdir shards --shard_mb 300 --with_bos_eos

Note:
- Lavora in streaming.
- "Document boundary" euristico: header tipici di atti/decisioni UE oppure 2+ blank lines.
- Se i tuoi dati hanno un separatore chiaro, puoi adattare `is_doc_boundary()`.
"""

import argparse

import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from rich import box

console = Console()

# --- Regex di pulizia e boundary ---
RE_LOWER_UPPER = re.compile(r'([a-zàèéìòù])([A-ZÀÈÉÌÒÙ])')
RE_DIGIT_LETTER = re.compile(r'(\d)([A-Za-zÀ-ÖØ-öø-ÿ])')
RE_LETTER_DIGIT = re.compile(r'([A-Za-zÀ-ÖØ-öø-ÿ])(\d)')
RE_MULTI_SPACES = re.compile(r'[ \t\f\v]+')
RE_MANY_NEWLINES = re.compile(r'\n{3,}')

# Headers ricorrenti (molto comuni in giurisprudenza/atti UE in IT)
RE_HEADER = re.compile(
    r'^(Ordinanza della Corte|Sentenza della Corte|Conclusioni dell\'avvocato generale|'
    r'Parere del Comitato|Regolamento \(|Direttiva \(|Decisione \(|Raccomandazione \(|'
    r'Comunicazione della Commissione|GU\s+[CL]\s+\d+|Gazzetta ufficiale|'
    r'nel procedimento\s+C-\d+/\d+|nella causa\s+C-\d+/\d+)',
    re.IGNORECASE
)


def normalize_unicode(s: str) -> str:
    """
    Normalizza unicode e rimuove spazi "strani".
    - NBSP (U+00A0) -> space
    - NNBSP (U+202F) -> space
    - thin space (U+2009) -> space
    - zero width / soft hyphen -> remove
    """
    s = unicodedata.normalize("NFKC", s)

    s = s.replace("\u00A0", " ")
    s = s.replace("\u202F", " ")
    s = s.replace("\u2009", " ")
    s = s.replace("\u2007", " ")

    s = s.replace("\u200B", "")
    s = s.replace("\u2060", "")
    s = s.replace("\u00AD", "")

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def clean_text(s: str) -> str:
    """
    Pulizia 'minima ma efficace' per legal corpora:
    - normalizza unicode e spazi
    - inserisce spazio tra minuscola->Maiuscola (proceduraDirettive -> procedura Direttive)
    - inserisce spazio tra cifra<->lettera (234CE -> 234 CE)
    - comprime spazi e newline eccessivi
    """
    s = normalize_unicode(s)

    s = RE_LOWER_UPPER.sub(r"\1 \2", s)

    s = RE_DIGIT_LETTER.sub(r"\1 \2", s)
    s = RE_LETTER_DIGIT.sub(r"\1 \2", s)

    s = RE_MULTI_SPACES.sub(" ", s)

    s = "\n".join(line.strip() for line in s.split("\n"))

    s = RE_MANY_NEWLINES.sub("\n\n", s)

    return s.strip()


def is_doc_boundary(line: str, blank_run: int) -> bool:
    """
    Euristica boundary:
    - se vedo un header tipico legale UE all'inizio di riga -> probabile inizio nuovo documento
    - oppure 2+ righe vuote consecutive -> possibile boundary
    """
    if RE_HEADER.match(line.strip()):
        return True
    if blank_run >= 2:
        return True
    return False


def write_shard_header(f, with_bos_eos: bool):
    pass


def write_doc(out_f, doc: str, with_bos_eos: bool):
    if not doc.strip():
        return
    if with_bos_eos:
        out_f.write("<bos>\n")
        out_f.write(doc.strip())
        out_f.write("\n<eos>\n\n")
    else:
        out_f.write(doc.strip())
        out_f.write("\n\n")


def _fmt_bytes(n: int) -> str:
    """Formatta bytes in modo leggibile."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def shard_clean(
    input_path: Path,
    outdir: Path,
    shard_mb: int,
    with_bos_eos: bool,
    min_doc_chars: int,
):
    outdir.mkdir(parents=True, exist_ok=True)

    input_size = input_path.stat().st_size

    # --- Banner di avvio ---
    config_table = Table(
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 2),
        title="[bold cyan]Configurazione[/bold cyan]",
        title_style="",
    )
    config_table.add_column("param", style="dim")
    config_table.add_column("value", style="bold white")
    config_table.add_row("Input", str(input_path))
    config_table.add_row("Dimensione", _fmt_bytes(input_size))
    config_table.add_row("Output dir", str(outdir))
    config_table.add_row("Shard size", f"{shard_mb} MB")
    config_table.add_row("BOS/EOS", "✓" if with_bos_eos else "✗")
    config_table.add_row("Min doc chars", str(min_doc_chars))
    console.print()
    console.print(
        Panel(
            config_table,
            title="[bold green]⚖  Shard & Clean — Legal Corpus[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()

    max_bytes = shard_mb * 1024 * 1024
    shard_id = 0
    cur_bytes = 0
    total_docs = 0
    total_docs_discarded = 0
    bytes_read = 0

    out_path = outdir / f"shard_{shard_id:05d}.txt"
    out_f = out_path.open("w", encoding="utf-8")
    write_shard_header(out_f, with_bos_eos)

    doc_lines: List[str] = []
    blank_run = 0
    started = False

    progress = Progress(
        SpinnerColumn("dots", style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("•"),
        DownloadColumn(),
        TextColumn("•"),
        TransferSpeedColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("→"),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )

    task_read = progress.add_task("Lettura & pulizia", total=input_size)

    def rotate_shard():
        nonlocal shard_id, cur_bytes, out_f, out_path
        out_f.close()
        console.log(
            f"[green]✓[/green] Shard [bold]{out_path.name}[/bold] completato "
            f"— [cyan]{_fmt_bytes(max_bytes)}[/cyan]"
        )
        shard_id += 1
        cur_bytes = 0
        out_path = outdir / f"shard_{shard_id:05d}.txt"
        out_f = out_path.open("w", encoding="utf-8")
        write_shard_header(out_f, with_bos_eos)

    def flush_doc(force: bool = False):
        nonlocal doc_lines, cur_bytes, total_docs, total_docs_discarded
        raw = "\n".join(doc_lines).strip()
        doc_lines = []
        if not raw:
            return
        doc = clean_text(raw)
        if len(doc) < min_doc_chars and not force:
            total_docs_discarded += 1
            return

        payload = (f"<bos>\n{doc}\n<eos>\n\n" if with_bos_eos else f"{doc}\n\n")
        b = len(payload.encode("utf-8"))
        if cur_bytes + b > max_bytes and cur_bytes > 0:
            rotate_shard()
        out_f.write(payload)
        cur_bytes += b
        total_docs += 1

    t_start = time.perf_counter()

    with progress:
        with input_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                bytes_read += len(line.encode("utf-8"))
                progress.update(task_read, completed=min(bytes_read, input_size))

                line = normalize_unicode(line)

                if line.strip() == "":
                    blank_run += 1
                else:
                    if started and is_doc_boundary(line, blank_run):
                        flush_doc()
                    blank_run = 0
                    started = True

                doc_lines.append(line.rstrip("\n"))

            flush_doc(force=True)

    out_f.close()

    elapsed = time.perf_counter() - t_start

    # --- Log ultimo shard ---
    if cur_bytes > 0:
        console.log(
            f"[green]✓[/green] Shard [bold]{out_path.name}[/bold] completato "
            f"— [cyan]{_fmt_bytes(cur_bytes)}[/cyan]"
        )

    # --- Riepilogo finale ---
    console.print()
    summary = Table(
        box=box.HEAVY_EDGE,
        show_header=False,
        padding=(0, 2),
        title="[bold yellow]Riepilogo[/bold yellow]",
        title_style="",
    )
    summary.add_column("metric", style="dim")
    summary.add_column("value", style="bold white", justify="right")
    summary.add_row("Documenti estratti", f"[green]{total_docs:,}[/green]")
    summary.add_row(
        "Documenti scartati",
        f"[red]{total_docs_discarded:,}[/red] [dim](< {min_doc_chars} chars)[/dim]",
    )
    summary.add_row("Shards generati", f"[cyan]{shard_id + 1}[/cyan]")
    summary.add_row("Bytes letti", _fmt_bytes(input_size))
    summary.add_row("Tempo", f"{elapsed:.1f}s")
    summary.add_row(
        "Velocità",
        f"{_fmt_bytes(int(input_size / elapsed))}/s" if elapsed > 0 else "—",
    )
    console.print(
        Panel(
            summary,
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print(
        f"\n[bold green]✔ Fatto![/bold green] Shards salvati in [underline]{outdir}[/underline]\n"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Shard & Clean di un grande TXT giuridico",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Path del file TXT enorme")
    ap.add_argument("--outdir", required=True, help="Cartella output shards")
    ap.add_argument("--shard_mb", type=int, default=300, help="Dimensione shard in MB (default 300)")
    ap.add_argument("--with_bos_eos", action="store_true", help="Aggiunge <bos>/<eos> per documento")
    ap.add_argument("--min_doc_chars", type=int, default=500, help="Scarta doc troppo corti (default 500 char)")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        console.print(f"[bold red]✗ Errore:[/bold red] file non trovato: {input_path}")
        sys.exit(1)

    shard_clean(
        input_path=input_path,
        outdir=Path(args.outdir),
        shard_mb=args.shard_mb,
        with_bos_eos=args.with_bos_eos,
        min_doc_chars=args.min_doc_chars,
    )


if __name__ == "__main__":
    main()