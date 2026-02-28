#!/usr/bin/env python3
"""
Pre-tokenize large text corpora into sharded binary token files (.bin) + metadata.

Workflow:
  1. Train (or load) a ByteLevel BPE tokenizer with special tokens
  2. Tokenize local .txt corpus file-by-file, chunk-by-chunk
  3. Inject BOS/EOS once per document (not per chunk)
  4. Write uint32-LE token IDs into sharded .bin files (streaming SHA256)
  5. Save tokenizer.json + pretokenized_meta.json
  6. Upload all artifacts to AWS S3

Output (local + S3):
  output/
    tokenizer.json
    pretokenized_meta.json
    shards/
      shard_000000.bin
      shard_000001.bin
      ...

Required env vars for S3:
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY

Examples
--------
# Train new tokenizer + upload to S3
python bin.tokenizer.py \
  --vocab_size 40960 \
  --s3_bucket my-bucket \
  --s3_prefix skylar/tokenized_corpus \
  --s3_region eu-west-1

# Reuse existing tokenizer
python pre_tokenize.py \\
  --data data/pretrain \\
  --output data/tokenized_corpus \\
  --tokenizer data/tokenized_corpus/tokenizer.json \\
  --s3_bucket my-bucket \\
  --s3_prefix skylar/pretrain_v1 \\
  --s3_region eu-west-1

# Local only (no S3)
python pre_tokenize.py \\
  --data data/pretrain \\
  --output data/tokenized_corpus \\
  --vocab_size 40960
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import struct
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from dotenv import load_dotenv
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
)

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn, TimeRemainingColumn,
)
from rich import box

# ─────────────────────────────────────────────────────────────
# LOGGING (Rich)
# ─────────────────────────────────────────────────────────────

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
        markup=True,
    )],
)
log = logging.getLogger("pre_tokenize")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SPECIAL_TOKENS: list[str] = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
]

DEFAULT_SAMPLE_MB: int = 500
DEFAULT_READ_CHUNK_MB: int = 50
DEFAULT_TARGET_SHARD_MB: int = 1024

S3_UPLOAD_MAX_RETRIES: int = 5
S3_UPLOAD_RETRY_BASE_SECONDS: float = 2.0


# ─────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────


def _ram_gb() -> float:
    """Return current process RSS in GB, or 0 if psutil is unavailable."""
    try:
        import psutil  # noqa: PLC0415

        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return 0.0


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_uint32_le(ids: list[int]) -> bytes:
    """Pack a list of ints into little-endian uint32 bytes — platform-independent."""
    return struct.pack(f"<{len(ids)}I", *ids)


def _np_to_uint32_le(arr: np.ndarray) -> bytes:
    """Convert numpy array to little-endian uint32 bytes."""
    out = arr.astype("<u4")  # force little-endian uint32
    return out.tobytes()


# ─────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────


def discover_text_files(data_path: str) -> list[Path]:
    """Return sorted list of .txt files from a file or directory."""
    p = Path(data_path)
    if p.is_file():
        if p.suffix != ".txt":
            raise ValueError(f"Expected .txt file, got: {p}")
        return [p]
    if p.is_dir():
        files = sorted(p.glob("**/*.txt"))
        total_bytes = sum(f.stat().st_size for f in files)
        log.info(
            "Found [bold]%d[/bold] .txt files in %s ([cyan]%.2f GB[/cyan])", len(files), p, total_bytes / 1e9
        )
        return files
    raise FileNotFoundError(f"Not a file or directory: {data_path}")


def _sample_iterator(files: list[Path], max_bytes: int) -> Iterator[str]:
    """Yield text chunks across files for tokenizer training, up to max_bytes total."""
    if not files:
        return

    bytes_per_file = max(1, max_bytes // len(files))
    total_read = 0
    chunk_size = 10 * 1024 * 1024  # 10 MB

    for fp in files:
        file_read = 0
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            while file_read < bytes_per_file and total_read < max_bytes:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
                n = len(chunk.encode("utf-8"))
                file_read += n
                total_read += n
        if total_read >= max_bytes:
            break

    log.info("Sampled ~[bold]%.0f MB[/bold] for tokenizer training", total_read / 1e6)


# ─────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────


def _attach_bos_eos(tokenizer: Tokenizer) -> None:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer is missing <bos> and/or <eos> special tokens")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos>:0 $A:0 <eos>:0",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )


def _ensure_tokenizer_ready(tokenizer: Tokenizer) -> None:
    """Attach decoder and BOS/EOS postprocessor if missing (for loaded tokenizers)."""
    if tokenizer.decoder is None:
        tokenizer.decoder = decoders.ByteLevel()
    if tokenizer.post_processor is None:
        _attach_bos_eos(tokenizer)


def train_tokenizer(
        files: list[Path],
        vocab_size: int,
        sample_mb: int,
        save_path: Path,
) -> Tokenizer:
    """Train a ByteLevel BPE tokenizer from corpus samples and save it."""
    log.info(
        "Training ByteLevel BPE tokenizer (vocab_size=[bold]%d[/bold], sample_mb=%d)",
        vocab_size,
        sample_mb,
    )
    log.info("[dim]RAM before training: %.1f GB[/dim]", _ram_gb())

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        _sample_iterator(files, max_bytes=sample_mb * 1_000_000),
        trainer=trainer,
    )

    tokenizer.decoder = decoders.ByteLevel()
    _attach_bos_eos(tokenizer)

    _ensure_dir(save_path.parent)
    tokenizer.save(str(save_path))

    log.info("[green]✓[/green] Tokenizer saved: %s (vocab=[bold]%d[/bold])", save_path, tokenizer.get_vocab_size())
    log.info("[dim]RAM after training: %.1f GB[/dim]", _ram_gb())
    return tokenizer


def load_or_train_tokenizer(
        files: list[Path],
        tokenizer_arg: str | None,
        vocab_size: int,
        sample_mb: int,
        output_dir: Path,
) -> Tokenizer:
    """Load an existing tokenizer.json or train a new one. Always copies to output_dir."""
    dest = output_dir / "tokenizer.json"

    if tokenizer_arg:
        tp = Path(tokenizer_arg)
        if tp.is_dir():
            tp = tp / "tokenizer.json"
        if not tp.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tp}")

        log.info("Loading existing tokenizer from [bold]%s[/bold]", tp)
        tokenizer = Tokenizer.from_file(str(tp))
        _ensure_tokenizer_ready(tokenizer)

        _ensure_dir(output_dir)
        tokenizer.save(str(dest))
        log.info("[green]✓[/green] Tokenizer copied to %s (vocab=[bold]%d[/bold])", dest, tokenizer.get_vocab_size())
        return tokenizer

    return train_tokenizer(files, vocab_size, sample_mb, dest)


# ─────────────────────────────────────────────────────────────
# S3 UPLOADER
# ─────────────────────────────────────────────────────────────


class S3Uploader:
    """Upload files to AWS S3 with exponential backoff retry.

    Requires env vars:
        AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY
    """

    def __init__(self, bucket: str, prefix: str, region: str) -> None:
        try:
            import boto3  # noqa: PLC0415
            from botocore.config import Config  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3 upload. Install: pip install boto3"
            ) from exc

        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if not access_key or not secret_key:
            raise EnvironmentError(
                "Missing AWS_ACCESS_KEY_ID and/or AWS_SECRET_ACCESS_KEY env vars"
            )

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region

        boto_config = Config(
            retries={"max_attempts": 0, "mode": "standard"},  # we handle retries
            max_pool_connections=10,
        )
        self._client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=boto_config,
        )
        log.info("[green]✓[/green] S3 uploader ready: [cyan]s3://%s/%s[/cyan] (region=%s)", bucket, self.prefix, region)

    def _full_key(self, name: str) -> str:
        if not self.prefix:
            return name
        return f"{self.prefix}/{name}"

    def upload(
            self,
            local_path: Path,
            remote_name: str,
            max_retries: int = S3_UPLOAD_MAX_RETRIES,
    ) -> str:
        """Upload a file to S3 with retries. Returns the s3:// URI."""
        key = self._full_key(remote_name)
        uri = f"s3://{self.bucket}/{key}"

        for attempt in range(1, max_retries + 1):
            try:
                file_size = local_path.stat().st_size

                # Use multipart for files > 100 MB
                from boto3.s3.transfer import TransferConfig  # noqa: PLC0415

                transfer_cfg = TransferConfig(
                    multipart_threshold=100 * 1024 * 1024,
                    multipart_chunksize=64 * 1024 * 1024,
                    max_concurrency=4,
                    use_threads=True,
                )
                self._client.upload_file(
                    str(local_path),
                    self.bucket,
                    key,
                    Config=transfer_cfg,
                )
                log.info(
                    "[green]✓[/green] Uploaded %s ([cyan]%.2f MB[/cyan]) → %s",
                    local_path.name,
                    file_size / 1e6,
                    uri,
                )
                return uri

            except Exception:
                if attempt == max_retries:
                    log.exception(
                        "[bold red]✗[/bold red] S3 upload FAILED after %d attempts: %s", max_retries, local_path
                    )
                    raise
                wait = S3_UPLOAD_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                log.warning(
                    "[yellow]⚠[/yellow] S3 upload attempt %d/%d failed for %s, retrying in %.1fs...",
                    attempt,
                    max_retries,
                    local_path.name,
                    wait,
                )
                time.sleep(wait)

        # unreachable, but satisfies type checker
        raise RuntimeError("S3 upload exhausted retries")  # pragma: no cover


# ─────────────────────────────────────────────────────────────
# SHARD WRITER (fixed)
# ─────────────────────────────────────────────────────────────


@dataclass
class ShardRecord:
    """Metadata for one completed shard file."""

    index: int
    filename: str
    num_tokens: int
    num_bytes: int
    sha256: str
    s3_uri: str | None = None


class ShardedBinWriter:
    """Write uint32-LE token IDs into sharded .bin files.

    Fixes vs original:
      - Forces little-endian via struct.pack / numpy '<u4' (no native-endian bug)
      - Single shard counter (no index/records desync)
      - SHA256 computed in streaming during writes (no re-read)
    """

    def __init__(
            self,
            shards_dir: Path,
            target_shard_bytes: int,
    ) -> None:
        self.shards_dir = shards_dir
        _ensure_dir(self.shards_dir)

        self.target_shard_bytes = target_shard_bytes

        self._fp = None
        self._path: Path | None = None
        self._hasher: hashlib._Hash | None = None
        self._cur_tokens: int = 0
        self._cur_bytes: int = 0
        self._next_index: int = 0

        self.records: list[ShardRecord] = []
        self.total_tokens: int = 0

        self._open_new_shard()

    # ── internal ──

    def _shard_filename(self, idx: int) -> str:
        return f"shard_{idx:06d}.bin"

    def _open_new_shard(self) -> None:
        if self._fp is not None:
            self._finalize_current()

        idx = self._next_index
        self._next_index += 1

        filename = self._shard_filename(idx)
        self._path = self.shards_dir / filename
        self._fp = self._path.open("wb")
        self._hasher = hashlib.sha256()
        self._cur_tokens = 0
        self._cur_bytes = 0

    def _finalize_current(self) -> None:
        if self._fp is None or self._path is None:
            return

        self._fp.flush()
        self._fp.close()

        size = self._path.stat().st_size if self._path.exists() else 0
        if size == 0:
            self._path.unlink(missing_ok=True)
            self._fp = None
            self._path = None
            return

        rec = ShardRecord(
            index=len(self.records),
            filename=self._path.name,
            num_tokens=self._cur_tokens,
            num_bytes=size,
            sha256=self._hasher.hexdigest() if self._hasher else "",
        )
        self.records.append(rec)

        self._fp = None
        self._path = None
        self._hasher = None

    # ── public ──

    def write_ids(self, ids: list[int]) -> None:
        """Write token IDs as little-endian uint32. Rotates shard when target size is reached."""
        if not ids:
            return

        # Convert to LE bytes
        if len(ids) <= 64:
            raw = _to_uint32_le(ids)
        else:
            raw = _np_to_uint32_le(np.asarray(ids, dtype=np.uint32))

        # Rotate if current shard has data and would exceed target
        if self._cur_bytes > 0 and (self._cur_bytes + len(raw)) > self.target_shard_bytes:
            self._open_new_shard()

        self._fp.write(raw)
        self._hasher.update(raw)
        self._cur_tokens += len(ids)
        self._cur_bytes += len(raw)
        self.total_tokens += len(ids)

    def close(self) -> None:
        self._finalize_current()

    def __enter__(self) -> ShardedBinWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.close()
        except Exception:
            log.exception("Error finalizing shard writer")
        return False


# ─────────────────────────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────────────────────────


def tokenize_file(
        filepath: Path,
        tokenizer: Tokenizer,
        writer: ShardedBinWriter,
        read_chunk_bytes: int,
        inject_bos_eos_per_file: bool = True,
) -> int:
    """Tokenize one file into the shard writer.

    Se inject_bos_eos_per_file=True:
      - inietta <bos> all'inizio file e <eos> a fine file
    Se False:
      - NON inietta nulla (utile se il corpus contiene già <bos>/<eos> interni)
    """
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer missing <bos>/<eos> IDs")

    # Disable postprocessor to avoid BOS/EOS per chunk
    saved_pp = tokenizer.post_processor
    tokenizer.post_processor = None

    written = 0
    try:
        if inject_bos_eos_per_file:
            writer.write_ids([bos_id])
            written += 1

        with filepath.open("r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(read_chunk_bytes)
                if not chunk:
                    break
                # Backtrack to last newline to avoid splitting words/special tokens
                if len(chunk) == read_chunk_bytes:
                    last_nl = chunk.rfind('\n')
                    if last_nl > 0:
                        excess = len(chunk) - last_nl - 1
                        f.seek(f.tell() - excess)
                        chunk = chunk[:last_nl + 1]
                ids = tokenizer.encode(chunk, add_special_tokens=False).ids
                if ids:
                    writer.write_ids(ids)
                    written += len(ids)

        if inject_bos_eos_per_file:
            writer.write_ids([eos_id])
            written += 1
    finally:
        tokenizer.post_processor = saved_pp

    return written


# ─────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────


def build_metadata(
        tokenizer: Tokenizer,
        files: list[Path],
        writer: ShardedBinWriter,
        args: argparse.Namespace,
) -> dict:
    """Build the pretokenized_meta.json content."""
    special_ids = {
        tok: tokenizer.token_to_id(tok)
        for tok in SPECIAL_TOKENS
        if tokenizer.token_to_id(tok) is not None
    }

    return {
        "format": "pretokenized_sharded_bin_v1",
        "created_at_utc": _utc_iso(),
        "dtype": "uint32",
        "endianness": "little",
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_file": "tokenizer.json",
        "special_ids": special_ids,
        "num_input_files": len(files),
        "total_tokens": writer.total_tokens,
        "num_shards": len(writer.records),
        "shards": [asdict(r) for r in writer.records],
        "config": {
            "trained_new_tokenizer": not bool(args.tokenizer),
            "sample_mb": args.sample_mb,
            "vocab_size_requested": args.vocab_size,
            "read_chunk_mb": args.read_chunk_mb,
            "target_shard_mb": args.target_shard_mb,
            "bos_eos_per_document": not args.input_has_bos_eos,
            "input_contains_bos_eos_markers": bool(args.input_has_bos_eos),
        },
        "s3": {
            "enabled": bool(args.s3_bucket),
            "bucket": args.s3_bucket or None,
            "prefix": args.s3_prefix or None,
            "region": args.s3_region or None,
        },
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    # Load .env from project root (searches parent dirs automatically)
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

    p = argparse.ArgumentParser(
        description="Pre-tokenize corpus to sharded .bin + upload to S3"
    )
    # I/O
    p.add_argument(
        "--data",
        default=Path("../.datasets/pretokenized"),
        required=False, help="Text file or folder with .txt files")
    p.add_argument(
        "--output",
        default=Path("../.datasets/tokenized"),
        required=False,
        help="Local output directory")
    p.add_argument(
        "--tokenizer",
        default=None,
        help="Path to existing tokenizer.json (file or directory)",
    )
    # Tokenizer training
    p.add_argument("--vocab_size", type=int, default=40960)
    p.add_argument("--sample_mb", type=int, default=DEFAULT_SAMPLE_MB)
    # Tokenization
    p.add_argument("--read_chunk_mb", type=int, default=DEFAULT_READ_CHUNK_MB)
    p.add_argument("--target_shard_mb", type=int, default=DEFAULT_TARGET_SHARD_MB)
    # S3 (CLI overrides env vars; if neither is set, S3 upload is skipped)
    p.add_argument("--s3_bucket", default=None, help="AWS S3 bucket name (fallback: S3_BUCKET env)")
    p.add_argument("--s3_prefix", default=None, help="Key prefix inside bucket (fallback: S3_PREFIX env)")
    p.add_argument("--s3_region", default=None, help="AWS region (fallback: S3_REGION env, default: eu-south-1)")
    p.add_argument(
        "--delete_local_after_upload",
        action="store_true",
        help="Delete local shard .bin after successful S3 upload",
    )
    p.add_argument(
        "--input_has_bos_eos",
        action=argparse.BooleanOptionalAction,
        default=True,  # se il tuo scenario standard è già con <bos>/<eos> nel testo
        help="Se True, il testo contiene già marker <bos>/<eos> nel corpus e NON vengono iniettati per file",
    )
    args = p.parse_args()

    # Resolve S3 params: CLI > env > None
    args.s3_bucket = args.s3_bucket or os.environ.get("S3_BUCKET") or None
    args.s3_prefix = args.s3_prefix or os.environ.get("S3_PREFIX") or None
    args.s3_region = args.s3_region or os.environ.get("S3_REGION") or "eu-south-1"

    return args


def main() -> None:
    args = parse_args()
    t0 = time.time()

    console.print()
    console.rule("[bold cyan]Pre-Tokenization Pipeline[/bold cyan]", style="cyan")
    console.print()

    out_dir = Path(args.output)
    shards_dir = out_dir / "shards"
    _ensure_dir(shards_dir)

    log.info("[dim]RAM at start: %.1f GB[/dim]", _ram_gb())

    # ── 1. Discover corpus files ──
    console.rule("[bold]1. Corpus Discovery[/bold]", style="dim")
    files = discover_text_files(args.data)
    if not files:
        log.error("[bold red]✗[/bold red] No .txt files found in %s", args.data)
        sys.exit(1)

    # ── 2. Tokenizer ──
    console.print()
    console.rule("[bold]2. Tokenizer[/bold]", style="dim")
    tokenizer = load_or_train_tokenizer(
        files=files,
        tokenizer_arg=args.tokenizer,
        vocab_size=args.vocab_size,
        sample_mb=args.sample_mb,
        output_dir=out_dir,
    )

    # ── 3. Tokenize corpus → shards ──
    console.print()
    console.rule("[bold]3. Tokenization[/bold]", style="dim")
    read_chunk_bytes = args.read_chunk_mb * 1_000_000
    target_shard_bytes = args.target_shard_mb * 1_000_000

    inject_bos_eos_per_file = not args.input_has_bos_eos
    log.info(
        "Tokenizing corpus (read_chunk=[bold]%d[/bold]MB, shard_target=[bold]%d[/bold]MB, inject_bos_eos_per_file=%s)",
        args.read_chunk_mb,
        args.target_shard_mb,
        inject_bos_eos_per_file,
    )

    with ShardedBinWriter(
            shards_dir=shards_dir,
            target_shard_bytes=target_shard_bytes,
    ) as writer:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=35),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
        ) as progress:
            task = progress.add_task("Tokenizing files", total=len(files))
            for i, fp in enumerate(files, 1):
                size_mb = fp.stat().st_size / 1e6

                n_tokens = tokenize_file(
                    fp,
                    tokenizer,
                    writer,
                    read_chunk_bytes,
                    inject_bos_eos_per_file=inject_bos_eos_per_file,
                )

                pending = 1 if writer._fp else 0
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"[bold]{fp.name}[/bold] ({size_mb:.1f}MB) → "
                        f"{n_tokens:,} tok │ total={writer.total_tokens:,} │ "
                        f"shards={len(writer.records) + pending}"
                    ),
                )

        log.info("[dim]RAM after tokenization: %.1f GB[/dim]", _ram_gb())

    # ── 4. Upload to S3 ──
    s3_uploader: S3Uploader | None = None
    if args.s3_bucket:
        console.print()
        console.rule("[bold]4. S3 Upload[/bold]", style="dim")
        s3_uploader = S3Uploader(
            bucket=args.s3_bucket,
            prefix=args.s3_prefix or "",
            region=args.s3_region or "us-east-1",
        )

        # Upload shards
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=35),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=console,
        ) as progress:
            upload_task = progress.add_task("Uploading shards", total=len(writer.records))
            for rec in writer.records:
                shard_path = shards_dir / rec.filename
                if not shard_path.exists():
                    log.warning("[yellow]⚠[/yellow] Shard file missing, skipping: %s", shard_path)
                    progress.update(upload_task, advance=1)
                    continue

                remote_name = f"shards/{rec.filename}"
                progress.update(upload_task, description=f"[bold]{rec.filename}[/bold] ({rec.num_bytes / 1e6:.1f} MB)")
                rec.s3_uri = s3_uploader.upload(shard_path, remote_name)

                if args.delete_local_after_upload:
                    shard_path.unlink(missing_ok=True)
                    log.info("[dim]Deleted local shard: %s[/dim]", shard_path)

                progress.update(upload_task, advance=1)

        # Upload tokenizer
        tokenizer_path = out_dir / "tokenizer.json"
        s3_uploader.upload(tokenizer_path, "tokenizer.json")

    # ── 5. Save + upload metadata ──
    console.print()
    console.rule("[bold]5. Metadata[/bold]", style="dim")
    meta = build_metadata(tokenizer, files, writer, args)
    meta_path = out_dir / "pretokenized_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info("[green]✓[/green] Metadata saved: %s", meta_path)

    if s3_uploader:
        s3_uploader.upload(meta_path, "pretokenized_meta.json")

    # ── Summary ──
    elapsed = time.time() - t0
    local_bytes = sum(
        p.stat().st_size for p in shards_dir.glob("*.bin") if p.exists()
    )

    console.print()
    summary_table = Table(
        box=box.DOUBLE_EDGE,
        border_style="green",
        title="✅ Pre-Tokenization Complete",
        title_style="bold green",
    )
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="cyan")
    summary_table.add_row("Total tokens", f"{meta['total_tokens']:,}")
    summary_table.add_row("Num shards", str(meta["num_shards"]))
    summary_table.add_row("Local shards", f"{local_bytes / 1e9:.2f} GB")
    summary_table.add_row("Time", f"{elapsed / 60:.1f} min")
    summary_table.add_row("RAM final", f"{_ram_gb():.1f} GB")
    if s3_uploader:
        summary_table.add_row("S3 location", f"s3://{args.s3_bucket}/{args.s3_prefix or ''}")
    console.print(summary_table)
    console.print()


if __name__ == "__main__":
    main()
