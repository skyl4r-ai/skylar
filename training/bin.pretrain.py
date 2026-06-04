"""
Training script — loads pre-tokenized shards from S3 or local disk.

Requires output from pre_tokenize.py:
  tokenizer.json
  pretokenized_meta.json
  shards/shard_000000.bin ...

Supports:
  - Load shards from AWS S3 or local directory
  - Cosine or WSD (Warmup-Stable-Decay) LR schedule
  - µP for HP transfer across model widths
  - Mixed precision (bf16/fp16)
  - Gradient accumulation
  - Wandb logging (optional)
  - HuggingFace checkpointing (save_pretrained)
  - Resume from checkpoint
  - FlexAttention document masking (automatic when packing)
  - Async S3 checkpoint upload (best-effort, non-blocking)

Usage:
  # From S3
  python train.py \\
    --s3_bucket my-bucket \\
    --s3_prefix skylar/pretrain_v1 \\
    --s3_region eu-west-1 \\
    --preset medium --bf16

  # From local pre_tokenize.py output
  python train.py --data data/tokenized_corpus --preset medium --bf16

  # Resume from checkpoint
  python train.py --data data/tokenized_corpus --resume checkpoints/step_5000

  # WSD schedule
  python train.py --data data/tokenized_corpus --lr_schedule wsd --lr_decay_ratio 0.1

  # µP: tune on proxy, transfer to target
  python train.py --data data/tokenized_corpus --preset xl --mup_base_d_model 256 --bf16

Env for S3:
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
"""

import os
import sys
import json
import math
import time
import random
import shutil
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from models.config import get_config, PRESETS
from models.decoder import NanoTransformer, HAS_FLEX_ATTENTION

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn, TimeRemainingColumn,
)
from rich.text import Text
from rich import box

logger = logging.getLogger(__name__)
console = Console()

# ─────────────────────────────────────────────────────────────
# S3 CLIENT
# ─────────────────────────────────────────────────────────────

S3_MAX_RETRIES: int = 5
S3_RETRY_BASE_SEC: float = 2.0


class S3Client:
    """Download/upload files from/to AWS S3 with retry logic.

    Env vars required:
        AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY
    """

    def __init__(self, bucket: str, prefix: str, region: str) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ImportError("boto3 required: pip install boto3") from exc

        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if not access_key or not secret_key:
            raise EnvironmentError(
                "Missing AWS_ACCESS_KEY_ID and/or AWS_SECRET_ACCESS_KEY"
            )

        self.bucket = bucket
        self.prefix = prefix.strip("/") if prefix else ""
        self.region = region

        boto_config = Config(
            retries={"max_attempts": 0, "mode": "standard"},
            max_pool_connections=10,
        )
        self._client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=boto_config,
        )
        console.print(f"  [green]✓[/green] S3 client ready: [cyan]s3://{bucket}/{self.prefix}[/cyan] (region={region})")

    def _full_key(self, name: str) -> str:
        if not self.prefix:
            return name
        return f"{self.prefix}/{name}"

    def download(self, remote_name: str, local_path: Path) -> Path:
        """Download a file from S3 with retries."""
        key = self._full_key(remote_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, S3_MAX_RETRIES + 1):
            try:
                self._client.download_file(self.bucket, key, str(local_path))
                return local_path
            except Exception:
                if attempt == S3_MAX_RETRIES:
                    console.print(f"  [bold red]✗[/bold red] S3 download FAILED after {S3_MAX_RETRIES} attempts: {key}")
                    raise
                wait = S3_RETRY_BASE_SEC * (2 ** (attempt - 1))
                console.print(
                    f"  [yellow]⚠[/yellow] S3 download attempt {attempt}/{S3_MAX_RETRIES} failed: {remote_name}, retry in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def upload(self, local_path: Path, remote_name: str) -> str:
        """Upload a file to S3 with retries. Returns s3:// URI."""
        from boto3.s3.transfer import TransferConfig

        key = self._full_key(remote_name)
        uri = f"s3://{self.bucket}/{key}"

        transfer_cfg = TransferConfig(
            multipart_threshold=100 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=4,
            use_threads=True,
        )

        for attempt in range(1, S3_MAX_RETRIES + 1):
            try:
                self._client.upload_file(
                    str(local_path), self.bucket, key, Config=transfer_cfg
                )
                return uri
            except Exception:
                if attempt == S3_MAX_RETRIES:
                    console.print(
                        f"  [bold red]✗[/bold red] S3 upload FAILED after {S3_MAX_RETRIES} attempts: {local_path}")
                    raise
                wait = S3_RETRY_BASE_SEC * (2 ** (attempt - 1))
                console.print(
                    f"  [yellow]⚠[/yellow] S3 upload attempt {attempt}/{S3_MAX_RETRIES} failed: {local_path.name}, retry in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def upload_directory(self, local_dir: Path, remote_prefix: str) -> list[str]:
        """Upload all files in a directory (non-recursive). Returns list of URIs."""
        uris = []
        for f in sorted(local_dir.iterdir()):
            if f.is_file():
                remote_name = f"{remote_prefix}/{f.name}"
                uri = self.upload(f, remote_name)
                uris.append(uri)
        return uris


# ─────────────────────────────────────────────────────────────
# ASYNC S3 CHECKPOINT UPLOAD
# ─────────────────────────────────────────────────────────────

_upload_pool: Optional[ThreadPoolExecutor] = None


def _get_upload_pool() -> ThreadPoolExecutor:
    global _upload_pool
    if _upload_pool is None:
        _upload_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-upload")
    return _upload_pool


def async_upload_checkpoint(
        s3_client: S3Client,
        local_dir: str,
        remote_prefix: str,
) -> None:
    """Fire-and-forget S3 upload of a checkpoint directory.

    Copies the checkpoint dir first so training can continue even if upload is slow.
    Best-effort: logs success or failure, never crashes training.
    """
    src = Path(local_dir)
    staging = src.parent / f".upload_staging_{src.name}"

    try:
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(src, staging)
    except Exception as e:
        console.print(f"  [yellow]⚠[/yellow] S3 upload staging failed for {src.name}: {e}")
        return

    def _do_upload():
        try:
            uris = s3_client.upload_directory(staging, remote_prefix)
            console.print(f"  [cyan]☁[/cyan] S3 checkpoint uploaded: [bold]{remote_prefix}[/bold] ({len(uris)} files)")
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] S3 checkpoint upload failed for {remote_prefix}: {e}")
        finally:
            try:
                shutil.rmtree(staging)
            except Exception:
                pass

    _get_upload_pool().submit(_do_upload)


# ─────────────────────────────────────────────────────────────
# SHARD LOADING
# ─────────────────────────────────────────────────────────────

def load_pretokenized_data(
        data_dir: Optional[str],
        s3_client: Optional[S3Client],
        cache_dir: Path,
) -> tuple[torch.Tensor, Tokenizer, dict]:
    """Load pre-tokenized shards + tokenizer from local dir or S3.

    Returns (tokens_tensor, tokenizer, metadata_dict).
    """
    if s3_client is not None:
        local_dir = cache_dir / "pretokenized"
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download metadata (always re-download — tiny file, may have changed)
        meta_path = local_dir / "pretokenized_meta.json"
        console.print("  Downloading [bold]pretokenized_meta.json[/bold] from S3...")
        s3_client.download("pretokenized_meta.json", meta_path)

        with meta_path.open("r") as f:
            meta = json.load(f)

        # Download tokenizer (cache if already present)
        tok_path = local_dir / "tokenizer.json"
        if tok_path.exists():
            console.print("  [dim]tokenizer.json (cached)[/dim]")
        else:
            console.print("  Downloading [bold]tokenizer.json[/bold] from S3...")
            s3_client.download("tokenizer.json", tok_path)

        # Download shards
        shards_dir = local_dir / "shards"
        shards_dir.mkdir(exist_ok=True)
        shard_records = meta["shards"]

        console.print(f"  Downloading [bold]{len(shard_records)}[/bold] shards from S3...")
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,
        ) as progress:
            task = progress.add_task("Shards", total=len(shard_records))
            for rec in shard_records:
                filename = rec["filename"]
                shard_path = shards_dir / filename
                if shard_path.exists() and shard_path.stat().st_size == rec["num_bytes"]:
                    progress.update(task, advance=1, description=f"[dim]{filename} (cached)[/dim]")
                    continue
                remote_name = f"shards/{filename}"
                progress.update(task, description=f"{filename} ({rec['num_bytes'] / 1e6:.0f} MB)")
                s3_client.download(remote_name, shard_path)
                progress.update(task, advance=1)

    elif data_dir is not None:
        local_dir = Path(data_dir)
        meta_path = local_dir / "pretokenized_meta.json"
        if not meta_path.exists():
            console.print(f"  [bold red]✗[/bold red] pretokenized_meta.json not found in {local_dir}")
            sys.exit(1)

        with meta_path.open("r") as f:
            meta = json.load(f)

        tok_path = local_dir / "tokenizer.json"
        shards_dir = local_dir / "shards"
        shard_records = meta["shards"]
    else:
        console.print("  [bold red]✗[/bold red] Provide --data (local) or --s3_bucket (S3) for pre-tokenized data")
        sys.exit(1)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    console.print(f"  [green]✓[/green] Tokenizer loaded: vocab_size=[bold]{tokenizer.get_vocab_size()}[/bold]")

    # Load shards → single uint32 LE tensor
    all_arrays = []
    total_tokens = 0

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
    ) as progress:
        task = progress.add_task(f"Loading {len(shard_records)} shards", total=len(shard_records))
        for rec in shard_records:
            filename = rec["filename"]
            shard_path = shards_dir / filename
            if not shard_path.exists():
                console.print(f"  [bold red]✗[/bold red] Shard missing: {shard_path}")
                sys.exit(1)

            arr = np.fromfile(str(shard_path), dtype="<u4")  # little-endian uint32
            all_arrays.append(arr)
            total_tokens += len(arr)
            progress.update(task, advance=1, description=f"Loading shards ({total_tokens:,} tokens)")

    tokens = torch.from_numpy(np.concatenate(all_arrays)).long()
    console.print(
        f"  [green]✓[/green] Loaded [bold]{total_tokens:,}[/bold] tokens ([cyan]{tokens.nbytes / 1e9:.2f} GB[/cyan])")

    # Sanity check
    expected = meta.get("total_tokens", 0)
    if expected > 0 and total_tokens != expected:
        console.print(f"  [yellow]⚠[/yellow] Token count mismatch: loaded {total_tokens:,} vs meta {expected:,}")

    try:
        import psutil
        console.print(f"  [dim]RAM after loading: {psutil.Process().memory_info().rss / 1e9:.1f} GB[/dim]")
    except ImportError:
        pass

    return tokens, tokenizer, meta


# ─────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Tokenized text dataset for causal language modeling.

    Uses deterministic slicing: chunk[idx] starts at idx * seq_len.
    This ensures validation loss is reproducible across evaluations.
    """

    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_samples = max(1, (len(tokens) - 1) // seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start: start + self.seq_len]
        y = self.tokens[start + 1: start + self.seq_len + 1]
        min_len = min(len(x), len(y))
        return x[:min_len], y[:min_len]


class PackedDataset(Dataset):
    """
    Packed dataset: concatenate all tokens and slice into exact chunks.

    Eliminates padding waste — every token in every batch is a real token.
    Multiple documents are packed into each chunk, separated by BOS/EOS.

    When document boundaries are provided, also returns document_ids per chunk
    for FlexAttention document masking (prevents cross-doc attention leakage).
    """

    def __init__(self, tokens, seq_len, doc_boundaries=None):
        n_chunks = len(tokens) // (seq_len + 1)
        if n_chunks == 0:
            n_chunks = 1
            tokens = torch.cat([tokens, torch.zeros(seq_len + 1 - len(tokens), dtype=tokens.dtype)])
        self.total_tokens = n_chunks * (seq_len + 1)
        self.chunks = tokens[:self.total_tokens].reshape(n_chunks, seq_len + 1)

        # Build document_ids if boundaries are provided
        self.doc_ids = None
        if doc_boundaries is not None and len(doc_boundaries) > 0:
            doc_id_flat = self._build_doc_ids(doc_boundaries, len(tokens))
            self.doc_ids = doc_id_flat[:self.total_tokens].reshape(n_chunks, seq_len + 1)

    @staticmethod
    def _build_doc_ids(boundaries, total_len):
        """Convert document boundary positions to a flat doc_id tensor."""
        doc_ids = torch.zeros(total_len, dtype=torch.long)
        for doc_idx in range(len(boundaries)):
            start = boundaries[doc_idx]
            end = boundaries[doc_idx + 1] if doc_idx + 1 < len(boundaries) else total_len
            doc_ids[start:end] = doc_idx
        return doc_ids

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x, y = chunk[:-1], chunk[1:]
        if self.doc_ids is not None:
            return x, y, self.doc_ids[idx][:-1]
        return x, y


# ─────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────

def _find_doc_boundaries(tokens, bos_id):
    """Find document boundary positions (BOS token locations)."""
    if bos_id is None:
        return []
    return (tokens == bos_id).nonzero(as_tuple=True)[0].tolist()


def build_datasets(
        tokens: torch.Tensor,
        tokenizer: Tokenizer,
        seq_len: int,
        val_split: float = 0.05,
        packing: bool = True,
) -> tuple:
    """Split tokens into train/val and build Dataset objects."""
    split = int(len(tokens) * (1 - val_split))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    # Document boundaries for packing mask
    train_boundaries = None
    val_boundaries = None

    if packing:
        bos_id = tokenizer.token_to_id("<bos>")
        all_boundaries = _find_doc_boundaries(tokens, bos_id)
        if all_boundaries:
            train_boundaries = [b for b in all_boundaries if b < split]
            val_boundaries = [b - split for b in all_boundaries if b >= split]
            if val_boundaries and val_boundaries[0] != 0:
                val_boundaries = [0] + val_boundaries
            console.print(
                f"  [green]✓[/green] Found [bold]{len(train_boundaries)}[/bold] train / [bold]{len(val_boundaries)}[/bold] val document boundaries")
        else:
            console.print("  [yellow]⚠[/yellow] No BOS tokens found — packing without document masking")

    if packing:
        train_ds = PackedDataset(train_tokens, seq_len, doc_boundaries=train_boundaries)
        val_ds = PackedDataset(val_tokens, seq_len, doc_boundaries=val_boundaries)
    else:
        train_ds = TextDataset(train_tokens, seq_len)
        val_ds = TextDataset(val_tokens, seq_len)

    mode = "packed" if packing else "strided"
    has_mask = " + doc masking" if packing and train_boundaries else ""

    data_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    data_table.add_column(style="bold")
    data_table.add_column()
    data_table.add_row("Train", f"{len(train_tokens):,} tokens → {len(train_ds)} {mode} chunks{has_mask}")
    data_table.add_row("Val", f"{len(val_tokens):,} tokens → {len(val_ds)} {mode} chunks")
    console.print(data_table)

    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULES
# ─────────────────────────────────────────────────────────────

def get_lr_cosine(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine decay with linear warmup — standard LLM schedule."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_lr_wsd(step, warmup_steps, max_steps, max_lr, min_lr, decay_ratio=0.1):
    """
    Warmup-Stable-Decay schedule (DeepSeek-V3, MiniCPM, OLMo 2, GLM-4.5).

    Three phases:
      1. Warmup:  linear ramp from 0 to max_lr
      2. Stable:  constant max_lr (bulk of training — model explores loss landscape)
      3. Decay:   linear decay to min_lr (final convergence into local minimum)

    Advantages over cosine:
      - No need to know max_steps in advance during stable phase
      - Can pause/resume training freely during stable phase
      - Theoretically optimal for hard-task regime (arxiv:2602.06797)
      - Better SFT downstream: models stay in flatter minima (OpenReview 2025)

    Args:
        decay_ratio: fraction of total steps for decay phase (default 0.1 = last 10%)
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr

    decay_steps = max(1, int(max_steps * decay_ratio))
    stable_end = max_steps - decay_steps

    if step < stable_end:
        return max_lr

    progress = (step - stable_end) / decay_steps
    return min_lr + (max_lr - min_lr) * (1.0 - progress)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr, schedule="cosine", decay_ratio=0.1):
    """Dispatch to the appropriate LR schedule."""
    if schedule == "wsd":
        return get_lr_wsd(step, warmup_steps, max_steps, max_lr, min_lr, decay_ratio)
    return get_lr_cosine(step, warmup_steps, max_steps, max_lr, min_lr)


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50, dtype=torch.bfloat16):
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        if len(batch) == 3:
            x, y, doc_ids = batch
            x, y, doc_ids = x.to(device), y.to(device), doc_ids.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            doc_ids = None

        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            out = model(x, labels=y, document_ids=doc_ids)
        total_loss += out["loss"].item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def _make_train_progress(console_obj: Console) -> Progress:
    """Build the rich Progress bar for the training loop."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console_obj,
        refresh_per_second=2,
    )


def train(args):
    console.print()
    console.rule("[bold cyan]NanoTransformer Pre-Training[/bold cyan]", style="cyan")
    console.print()

    # ── Seed ──
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Accelerate (multi-GPU) ──
    try:
        from accelerate import Accelerator
        HAS_ACCELERATE = True
    except ImportError:
        HAS_ACCELERATE = False

    use_accelerate = args.multi_gpu and HAS_ACCELERATE
    accelerator = None

    if args.multi_gpu and not HAS_ACCELERATE:
        console.print("  [yellow]⚠[/yellow] --multi_gpu requires accelerate: [bold]pip install accelerate[/bold]")
        console.print("  [dim]Falling back to single device[/dim]")

    if use_accelerate:
        mixed = "bf16" if args.bf16 else ("fp16" if args.fp16 else "no")
        accelerator = Accelerator(mixed_precision=mixed)
        device = accelerator.device
        is_main = accelerator.is_main_process
        if is_main:
            console.print(f"  🚀 [bold green]Multi-GPU[/bold green]: {accelerator.num_processes} processes (accelerate)")
            console.print(f"     Device: [cyan]{device}[/cyan]")
    else:
        is_main = True

    # ── Device (single GPU/CPU/MPS) ──
    if not use_accelerate:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name()
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(f"  🔥 [bold green]GPU[/bold green]: {gpu_name}")
            console.print(f"     VRAM: [cyan]{vram_total:.1f} GB[/cyan]")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            console.print("  🍎 [bold green]Using Apple MPS[/bold green]")
        else:
            device = torch.device("cpu")
            console.print("  💻 [bold yellow]Using CPU[/bold yellow]")

    dtype = torch.float32
    if not use_accelerate:
        if args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            console.print("  [dim]Using bfloat16 mixed precision[/dim]")
        elif args.fp16 and device.type == "cuda":
            dtype = torch.float16
            console.print("  [dim]Using float16 mixed precision[/dim]")

    # ── S3 client ──
    s3_client: Optional[S3Client] = None
    if args.s3_bucket:
        s3_client = S3Client(
            bucket=args.s3_bucket,
            prefix=args.s3_prefix or "",
            region=args.s3_region or "us-east-1",
        )

    # ── Load pre-tokenized data ──
    console.print()
    console.rule("[bold]Data Loading[/bold]", style="dim")
    cache_dir = Path(args.out_dir) / ".cache"
    all_tokens, tokenizer, meta = load_pretokenized_data(
        data_dir=args.data,
        s3_client=s3_client,
        cache_dir=cache_dir,
    )
    vocab_size = tokenizer.get_vocab_size()

    # ── Model ──
    console.print()
    console.rule("[bold]Model[/bold]", style="dim")
    if args.resume:
        console.print(f"  [cyan]↻[/cyan] Resuming from [bold]{args.resume}[/bold]")
        model = NanoTransformer.from_pretrained(args.resume)
        config = model.config
    else:
        config_overrides = dict(
            max_seq_len=args.seq_len,
            dropout=args.dropout,
        )
        if args.mup_base_d_model is not None:
            config_overrides["mup_base_d_model"] = args.mup_base_d_model
        config = get_config(
            preset=args.preset,
            vocab_size=vocab_size,
            **config_overrides,
        )
        model = NanoTransformer(config)

    model = model.to(device)
    n_params = model.count_params()
    use_mup = config.mup_base_d_model is not None

    if is_main:
        model_table = Table(
            title=f"🧠 {args.preset.upper()} — {n_params:,} parameters",
            box=box.ROUNDED,
            title_style="bold magenta",
            border_style="dim",
            padding=(0, 1),
        )
        model_table.add_column("Parameter", style="bold")
        model_table.add_column("Value", style="cyan")
        model_table.add_row("d_model", str(config.d_model))
        model_table.add_row("n_heads", str(config.n_heads))
        model_table.add_row("n_layers", str(config.n_layers))
        model_table.add_row("d_ff", str(config.d_ff))
        model_table.add_row("seq_len", str(config.max_seq_len))
        model_table.add_row("vocab_size", f"{vocab_size:,}")
        console.print(model_table)

        if use_mup:
            console.print(f"  📐 [bold]µP enabled[/bold] — base_d={config.mup_base_d_model}, "
                          f"width_mult={config.mup_width_mult:.1f}x, "
                          f"hidden_lr_mult={1.0 / config.mup_width_mult:.4f}")
        if HAS_FLEX_ATTENTION:
            console.print("  🔒 [green]FlexAttention available[/green] — document masking at zero VRAM cost")
        else:
            console.print("  [yellow]⚠[/yellow] FlexAttention not available — packing without document masking")

    # ── Auto batch size ──
    if args.batch_size == 0 and device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        param_mem_gb = n_params * 2 / 1e9
        available = vram_gb - param_mem_gb * 4
        bytes_per_sample = config.max_seq_len * config.d_model * config.n_layers * 4
        estimated_batch = max(1, int(available * 1e9 / bytes_per_sample))
        args.batch_size = min(estimated_batch, 64)
        if is_main:
            console.print(
                f"  [green]✓[/green] Auto batch_size: [bold]{args.batch_size}[/bold] (VRAM: {vram_gb:.0f}GB, available: {available:.1f}GB)")
    elif args.batch_size == 0:
        args.batch_size = 4
        if is_main:
            console.print(f"  [green]✓[/green] Auto batch_size: [bold]{args.batch_size}[/bold] (non-CUDA device)")

    # ── Gradient checkpointing ──
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        console.print("  ♻ [green]Gradient checkpointing attivo[/green]")

    # ── torch.compile ──
    if args.compile:
        try:
            model = torch.compile(model)
            console.print("  ⚡ [green]torch.compile() attivo[/green]")
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] torch.compile() non disponibile: {e}")

    # ── Build datasets ──
    console.print()
    console.rule("[bold]Datasets[/bold]", style="dim")
    train_ds, val_ds = build_datasets(
        tokens=all_tokens,
        tokenizer=tokenizer,
        seq_len=config.max_seq_len,
        packing=args.packing,
    )

    # Free raw tokens — datasets hold their own slices
    del all_tokens

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── Optimizer ──
    use_fused = device.type == "cuda" and not use_accelerate

    if use_mup:
        param_groups = model.mup_param_groups(args.lr, args.weight_decay)
        optimizer = torch.optim.AdamW(
            param_groups, betas=(0.9, 0.95), eps=1e-8, fused=use_fused,
        )
        if is_main:
            opt_table = Table(title="µP Param Groups", box=box.SIMPLE, border_style="dim")
            opt_table.add_column("Group", style="bold")
            opt_table.add_column("Params", justify="right")
            opt_table.add_column("LR", justify="right", style="cyan")
            opt_table.add_column("WD", justify="right")
            for g in param_groups:
                n_p = sum(p.numel() for p in g["params"])
                opt_table.add_row(g["name"], f"{n_p:,}", f"{g['lr']:.2e}", f"{g['weight_decay']}")
            console.print(opt_table)
    else:
        decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
        nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        n_decay = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        if is_main:
            console.print(f"  [dim]Decay params: {n_decay:,} | No-decay params: {n_nodecay:,}[/dim]")

    # ── Accelerate: prepare ──
    if use_accelerate:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        val_loader = accelerator.prepare(val_loader)

    # ── Wandb ──
    if args.wandb and is_main:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        wandb.watch(model, log_freq=100)

    # ── Training ──
    os.makedirs(args.out_dir, exist_ok=True)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16 and not use_accelerate))

    model.train()
    step = 0
    best_val_loss = float("inf")
    t0 = time.time()
    tokens_processed = 0

    # log grad_norm
    grad_norm = torch.tensor(0.0)  # ← aggiungi qui

    # Resume training state
    if args.resume:
        state_path = os.path.join(args.resume, "training_state.pt")
        if os.path.exists(state_path):
            if is_main:
                console.print(f"  [cyan]↻[/cyan] Restoring optimizer & training state...")
            training_state = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(training_state["optimizer"])
            if not use_accelerate:
                scaler.load_state_dict(training_state["scaler"])
            step = training_state["step"] + 1
            best_val_loss = training_state.get("best_val_loss", float("inf"))
            tokens_processed = training_state.get("tokens_processed", 0)
            if is_main:
                console.print(
                    f"  [green]✓[/green] Resumed from step [bold]{step}[/bold] (best_val={best_val_loss:.4f})")

    min_lr = args.lr * 0.1

    if is_main:
        schedule_info = args.lr_schedule
        if args.lr_schedule == "wsd":
            schedule_info += f", decay_ratio={args.lr_decay_ratio} (last {args.lr_decay_ratio * 100:.0f}%)"

        train_cfg_table = Table(box=box.ROUNDED, border_style="cyan", title="🚀 Training Configuration",
                                title_style="bold cyan")
        train_cfg_table.add_column("Setting", style="bold")
        train_cfg_table.add_column("Value", style="white")
        train_cfg_table.add_row("Max steps", f"{args.max_steps:,}")
        train_cfg_table.add_row("Batch size", str(args.batch_size))
        train_cfg_table.add_row("Grad accumulation", str(args.grad_accum))
        train_cfg_table.add_row("Effective batch", str(args.batch_size * args.grad_accum))
        train_cfg_table.add_row("Learning rate", f"{args.lr:.2e}")
        train_cfg_table.add_row("Warmup steps", str(args.warmup_steps))
        train_cfg_table.add_row("Schedule", schedule_info)
        if use_mup:
            train_cfg_table.add_row("µP base_d", str(config.mup_base_d_model))
        console.print()
        console.print(train_cfg_table)
        console.print()

    def unwrap_model():
        if use_accelerate:
            return accelerator.unwrap_model(model)
        return model

    def save_checkpoint(name: str, is_best: bool = False):
        """Save checkpoint locally + async upload to S3."""
        save_path = os.path.join(args.out_dir, name)
        unwrap_model().save_pretrained(save_path)
        tokenizer.save(os.path.join(save_path, "tokenizer.json"))
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "tokens_processed": tokens_processed,
        }, os.path.join(save_path, "training_state.pt"))

        label = "Best model" if is_best else "Checkpoint"
        extra = f" (val_loss={best_val_loss:.4f})" if is_best else ""
        style = "bold green" if is_best else "bold"
        console.print(f"  💾 [{style}]{label} saved[/{style}]: {save_path}{extra}")

        # Async S3 upload (best-effort)
        if s3_client is not None:
            run_name = os.path.basename(args.out_dir.rstrip("/"))
            remote_prefix = f"{run_name}/{name}"
            async_upload_checkpoint(s3_client, save_path, remote_prefix)

    # ── Training progress bar ──
    train_progress = _make_train_progress(console)
    train_progress.start()
    train_task = train_progress.add_task("Training", total=args.max_steps, completed=step)

    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break

            # Unpack batch (with or without doc_ids)
            if len(batch) == 3:
                x, y, doc_ids = batch
            else:
                x, y = batch
                doc_ids = None

            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, min_lr,
                        schedule=args.lr_schedule, decay_ratio=args.lr_decay_ratio)

            # Apply LR schedule — scale all groups proportionally.
            # Capture each group's base LR ONCE. Must use dict-membership ("not in"),
            # NOT hasattr: param_groups are dicts, so hasattr(pg, "_base_lr_set") is
            # always False, which re-captured the already-scheduled LR every step and
            # compounded it toward 0 (silent LR collapse). Mirrors bin.sft.py.
            for pg in optimizer.param_groups:
                if use_mup:
                    if "_base_lr_set" not in pg:
                        pg["_base_lr"] = pg["lr"]
                        pg["_base_lr_set"] = True
                    pg["lr"] = pg["_base_lr"] * (lr / args.lr)
                else:
                    pg["lr"] = lr

            if not use_accelerate:
                x, y = x.to(device), y.to(device)
                if doc_ids is not None:
                    doc_ids = doc_ids.to(device)

            # Forward + backward
            if use_accelerate:
                out = model(x, labels=y, document_ids=doc_ids)
                loss = out["loss"] / args.grad_accum
                accelerator.backward(loss)
            else:
                with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                    out = model(x, labels=y, document_ids=doc_ids)
                    loss = out["loss"] / args.grad_accum
                scaler.scale(loss).backward()

            tokens_processed += x.numel()

            if (step + 1) % args.grad_accum == 0 or step == args.max_steps - 1:
                if use_accelerate:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # ── Logging ──
            if step % args.log_every == 0 and is_main:
                dt = time.time() - t0
                tok_per_sec = tokens_processed / max(dt, 1e-6)
                raw_loss = loss.item() * args.grad_accum
                display_lr = lr
                if use_mup:
                    for pg in optimizer.param_groups:
                        if pg.get("name") == "hidden":
                            display_lr = pg["lr"]
                            break

                loss_color = "green" if raw_loss < 4.0 else ("yellow" if raw_loss < 6.0 else "red")
                train_progress.update(
                    train_task,
                    completed=step,
                    description=(
                        f"[bold]step {step:>6d}[/bold] │ "
                        f"loss [{loss_color}]{raw_loss:.4f}[/{loss_color}] │ "
                        f"lr [cyan]{display_lr:.2e}[/cyan] │ "
                        f"∇ {grad_norm:.3f} │ "
                        f"[dim]{tok_per_sec:,.0f} tok/s[/dim]"
                    ),
                )

                if args.wandb:
                    import wandb
                    wandb.log({"train/loss": raw_loss, "train/lr": lr,
                               "train/tok_per_sec": tok_per_sec,
                               "train/grad_norm": grad_norm.item()}, step=step)

            # ── Evaluation ──
            if step > 0 and step % args.eval_every == 0 and is_main:
                train_progress.stop()
                val_loss = evaluate(unwrap_model(), val_loader, device, dtype=dtype)
                console.print(f"  ✦ val_loss: [bold magenta]{val_loss:.4f}[/bold magenta]")

                if args.wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss}, step=step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint("best", is_best=True)
                train_progress.start()

            # ── Checkpoint ──
            if step > 0 and step % args.save_every == 0 and is_main:
                train_progress.stop()
                save_checkpoint(f"step_{step}")
                train_progress.start()

            # ── Sample generation ──
            if step > 0 and step % args.sample_every == 0 and is_main:
                train_progress.stop()
                raw_model = unwrap_model()
                raw_model.eval()
                # Seed with a clean single <bos>: encoding the literal string
                # "<bos>" with the post-processor active yields [bos, bos, eos].
                bos_id = tokenizer.token_to_id("<bos>")
                if bos_id is None:
                    bos_id = getattr(raw_model.config, "bos_token_id", 0) or 0
                input_ids = torch.tensor([[bos_id]], device=device)
                gen = raw_model.generate(input_ids, max_new_tokens=100, temperature=0.8, repetition_penalty=1.2)
                text = tokenizer.decode(gen[0].tolist())
                console.print(Panel(
                    Text(text[:200], style="italic"),
                    title="📝 Sample Generation",
                    border_style="blue",
                    width=min(console.width, 100),
                    padding=(0, 1),
                ))
                raw_model.train()
                train_progress.start()

            step += 1

    train_progress.update(train_task, completed=args.max_steps)
    train_progress.stop()

    # ── Final save ──
    if is_main:
        total_time = time.time() - t0
        save_checkpoint("final")

        final_table = Table(box=box.DOUBLE_EDGE, border_style="green", title="✅ Training Complete",
                            title_style="bold green")
        final_table.add_column("Metric", style="bold")
        final_table.add_column("Value", style="cyan")
        final_table.add_row("Total time", f"{total_time / 60:.1f} min")
        final_table.add_row("Final train loss", f"{loss.item() * args.grad_accum:.4f}")
        final_table.add_row("Best val loss", f"{best_val_loss:.4f}")
        final_table.add_row("Model saved to", os.path.join(args.out_dir, "final"))
        if s3_client:
            final_table.add_row("S3 checkpoints", f"s3://{args.s3_bucket}/{args.s3_prefix or ''}/checkpoints/")
        console.print()
        console.print(final_table)
        console.print()

    # Shutdown upload pool gracefully
    if _upload_pool is not None:
        console.print("  [dim]Waiting for pending S3 uploads to finish...[/dim]")
        _upload_pool.shutdown(wait=True)
        console.print("  [green]✓[/green] All S3 uploads completed.")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train NanoTransformer from pre-tokenized data")

    # Data source (local OR S3 — one is required)
    parser.add_argument("--data", type=str, default=None,
                        help="Local directory from pre_tokenize.py output")

    # S3
    parser.add_argument("--s3_bucket", type=str, default=None,
                        help="AWS S3 bucket name (for data download + checkpoint upload)")
    parser.add_argument("--s3_prefix", type=str, default=None,
                        help="Key prefix inside bucket")
    parser.add_argument("--s3_region", type=str, default=None,
                        help="AWS region (e.g. eu-west-1)")

    # Model
    parser.add_argument("--preset", type=str, default="test", choices=list(PRESETS.keys()),
                        help="Model size preset")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Override max sequence length from preset")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout (default 0.0: standard for LLM pretraining; "
                             "the FlexAttention packed path cannot apply attn dropout)")

    # Training
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (0 = auto-estimate from VRAM)")
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # LR Schedule
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "wsd"])
    parser.add_argument("--lr_decay_ratio", type=float, default=0.1,
                        help="WSD only: fraction of training for decay phase")

    # µP
    parser.add_argument("--mup_base_d_model", type=int, default=None,
                        help="µP proxy model width for HP transfer")

    # Precision
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--sample_every", type=int, default=500)

    # Output
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a save_pretrained checkpoint")

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="nano-transformer")
    parser.add_argument("--wandb_run", type=str, default=None)

    # Speed
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=True,
                        help="Pack sequences to eliminate padding (default: True)")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use HuggingFace Accelerate for multi-GPU")

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)

    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = min(4, os.cpu_count() or 1)

    if args.seq_len is None:
        args.seq_len = PRESETS[args.preset]["max_seq_len"]

    if not args.data and not args.s3_bucket:
        parser.error("Provide --data (local path) or --s3_bucket (AWS S3)")

    train(args)


if __name__ == "__main__":
    main()
