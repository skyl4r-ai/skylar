"""
Supervised Fine-Tuning (SFT) — turn a base model into a chat model.

This is Stage 2 of the pipeline:
  Stage 1: Pre-train on raw text     → python train.py (base model)
  Stage 2: SFT on conversations      → python train_sft.py (chat model) ← YOU ARE HERE
  Stage 3: RLHF/DPO (optional)       → alignment

Supports:
  - Cosine or WSD LR schedule (aligned with train.py)
  - µP optimizer groups (when base model was trained with µP)
  - Multi-epoch training (arxiv:2602.11149 — multi-epoch on small dataset
    beats single-epoch on large dataset for SFT reasoning)
  - Mixed precision (bf16/fp16)
  - Gradient accumulation + gradient norm logging
  - Resume from checkpoint
  - Async S3 checkpoint upload (best-effort, non-blocking)

Usage:
  # Step 1: Generate demo dataset
  python chat_format.py

  # Step 2: Fine-tune a pre-trained base model
  python train_sft.py --data data/sft_train.jsonl --base_model checkpoints/final --max_steps 5000

  # Step 2b: Multi-epoch on small dataset (recommended for <5K examples)
  python train_sft.py --data data/sft_train.jsonl --base_model checkpoints/final --epochs 5

  # Step 2c: WSD schedule (better for continual SFT)
  python train_sft.py --data data/sft_train.jsonl --base_model checkpoints/final --lr_schedule wsd

  # Step 2d: Resume from SFT checkpoint
  python train_sft.py --data data/sft_train.jsonl --resume checkpoints_sft/step_2000 --max_steps 5000

  # Step 2e: With S3 checkpoint backup
  python train_sft.py --data data/sft_train.jsonl --base_model checkpoints/final \\
    --s3_bucket my-bucket --s3_prefix skylar/sft_v1 --s3_region eu-west-1

  # Step 3: Chat with it
  python chat.py --model checkpoints_sft/best

Env for S3 (optional):
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
"""

import os
import sys
import time
import math
import random
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import torch
import torch.nn.functional as F

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)
from rich.text import Text
from rich import box

console = Console()

# B200 Blackwell: cuDNN SDPA crashes on variable-length sequences
if torch.cuda.is_available() and "B200" in torch.cuda.get_device_name(0):
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    console.print(
        f"  [yellow]⚠[/yellow] SDP backends disabled: cudnn={torch.backends.cuda.cudnn_sdp_enabled()}, flash={torch.backends.cuda.flash_sdp_enabled()}, mem={torch.backends.cuda.mem_efficient_sdp_enabled()}")

from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, decoders

# make the repo root importable when run as a script without `pip install -e .` (like bin.pretrain.py)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from models.decoder import Skylar2ForCausalLM
from training.optim import OptimizerSet, build_optimizer

try:
    from accelerate import Accelerator

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

from utils.chatML import (
    create_loss_mask,
    load_dataset_jsonl,
    encode_chatml,
)

# ─────────────────────────────────────────────────────────────
# S3 CLIENT + ASYNC UPLOAD
# ─────────────────────────────────────────────────────────────

S3_MAX_RETRIES: int = 5
S3_RETRY_BASE_SEC: float = 2.0


class S3Client:
    """Upload files to AWS S3 with retry logic.

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
# SFT DATASET
# ─────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning on chat conversations.

    Key difference from pre-training:
      - Pre-training: loss on ALL tokens
      - SFT: loss ONLY on assistant responses (everything else is masked with -100)
    """

    def __init__(self, examples, tokenizer, max_seq_len, overlong="tail_truncate"):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.overlong = overlong
        self.samples = []

        skipped = 0
        truncated = 0
        dropped_long = 0
        for ex in examples:
            # Token-based loss mask: tokenize each segment separately
            # for exact boundaries (no decode/re-encode approximation)
            token_ids, labels = create_loss_mask(ex["messages"], tokenizer)

            # Shift per causal LM: logits[t] predice token_ids[t+1]
            token_ids = token_ids[:-1]
            labels = labels[1:]

            if len(token_ids) < 4:
                skipped += 1
                continue

            # Over-length policy. Default 'tail_truncate' — keep the TAIL, not the head:
            # long examples must retain the final <|im_end|> label (the stop token); head
            # truncation silently dropped it and trained the model to never stop. 'drop'
            # (audit H2) discards the example entirely instead — for datasets where a
            # tail-truncated user turn would corrupt the instruction.
            if len(token_ids) > max_seq_len:
                if overlong == "drop":
                    dropped_long += 1
                    continue
                truncated += 1
                token_ids = token_ids[-max_seq_len:]
                labels = labels[-max_seq_len:]

            # Only keep examples where we have at least some assistant tokens
            if any(l != -100 for l in labels):
                self.samples.append({
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                })
            else:
                skipped += 1

        if skipped > 0:
            console.print(f"  [yellow]⚠[/yellow] Skipped {skipped} examples (too short or no assistant content)")
        if truncated > 0:
            console.print(f"  [yellow]⚠[/yellow] Tail-truncated {truncated} examples longer than seq_len={max_seq_len} (kept the ending with <|im_end|>)")
        if dropped_long > 0:
            console.print(f"  [yellow]⚠[/yellow] Dropped {dropped_long} examples longer than seq_len={max_seq_len} (--overlong drop)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad sequences to same length within a batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)  # pad with 0
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)  # -100 = ignore

    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]

    return input_ids, labels


# ─────────────────────────────────────────────────────────────
# LR SCHEDULE — aligned with train.py
# ─────────────────────────────────────────────────────────────

def get_lr_cosine(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_lr_wsd(step, warmup_steps, max_steps, max_lr, min_lr, decay_ratio=0.2):
    """
    Warmup-Stable-Decay schedule.

    For SFT, default decay_ratio is 0.2 (last 20%) — slightly more aggressive
    decay than pre-training (10%) since SFT is shorter and needs to converge.
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


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr, schedule="cosine", decay_ratio=0.2):
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
    for i, (input_ids, labels) in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids, labels = input_ids.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            out = model(input_ids, labels=labels)
        if out["loss"] is not None:
            total_loss += out["loss"].item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


# ─────────────────────────────────────────────────────────────
# TRAINING
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


def train_sft(args):
    console.print()
    console.rule("[bold cyan]Skylar2ForCausalLM SFT[/bold cyan]", style="cyan")
    console.print()

    # ── Seed ──
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Accelerate (multi-GPU) ──
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
    else:
        is_main = True

    # ── Device (single GPU/CPU/MPS) ──
    if not use_accelerate:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            console.print(f"  🔥 [bold green]GPU[/bold green]: {torch.cuda.get_device_name()}")
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

    # ── S3 client (optional) ──
    s3_client: Optional[S3Client] = None
    if args.s3_bucket:
        s3_client = S3Client(
            bucket=args.s3_bucket,
            prefix=args.s3_prefix or "",
            region=args.s3_region or "us-east-1",
        )

    # ── Data ──
    console.print()
    console.rule("[bold]Data[/bold]", style="dim")
    if args.data and os.path.exists(args.data):
        train_examples = load_dataset_jsonl(args.data)
    else:
        exit("  No data file found, generating demo dataset...")

    if args.val_data and os.path.exists(args.val_data):
        val_examples = load_dataset_jsonl(args.val_data)
    else:
        # Split 10% for validation
        split = max(1, len(train_examples) // 10)
        val_examples = train_examples[-split:]
        train_examples = train_examples[:-split]

    # ── Tokenizer (always from base_model or resume) ──
    tokenizer_path = None
    if args.resume and os.path.exists(os.path.join(args.resume, "tokenizer.json")):
        tokenizer_path = os.path.join(args.resume, "tokenizer.json")
        console.print(f"  [green]✓[/green] Loading tokenizer from checkpoint: [bold]{args.resume}[/bold]")
    elif args.base_model and os.path.exists(os.path.join(args.base_model, "tokenizer.json")):
        tokenizer_path = os.path.join(args.base_model, "tokenizer.json")
        console.print(f"  [green]✓[/green] Loading tokenizer from base model: [bold]{args.base_model}[/bold]")
    else:
        console.print(
            "  [bold red]✗[/bold red] No tokenizer found. Provide --base_model or --resume with a tokenizer.json")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    if tokenizer.decoder is None:
        tokenizer.decoder = decoders.ByteLevel()

    vocab_size = tokenizer.get_vocab_size()

    # ── Model ──
    console.print()
    console.rule("[bold]Model[/bold]", style="dim")
    if args.resume and os.path.exists(args.resume):
        console.print(f"  [cyan]↻[/cyan] Resuming from checkpoint: [bold]{args.resume}[/bold]")
        model = Skylar2ForCausalLM.from_pretrained(args.resume)
        config = model.config
    elif args.base_model and os.path.exists(args.base_model):
        console.print(f"  [green]✓[/green] Loading pre-trained base model: [bold]{args.base_model}[/bold]")
        model = Skylar2ForCausalLM.from_pretrained(args.base_model)
        config = model.config
    else:
        console.print("  [bold red]✗[/bold red] Provide --base_model (pre-trained) or --resume (SFT checkpoint)")
        sys.exit(1)

    model = model.to(device)
    n_params = model.count_params()
    use_mup = config.mup_base_d_model is not None

    if is_main:
        model_table = Table(
            title=f"🧠 SFT Model — {n_params:,} parameters",
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
                          f"width_mult={config.mup_width_mult:.1f}x")

    # ── Auto batch size ──
    if args.batch_size == 0 and device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        param_mem_gb = n_params * 2 / 1e9
        available = vram_gb - param_mem_gb * 4
        bytes_per_sample = config.max_seq_len * config.d_model * config.n_layers * 4
        estimated_batch = max(1, int(available * 1e9 / bytes_per_sample))
        args.batch_size = min(estimated_batch, 64)
        if is_main:
            console.print(f"  [green]✓[/green] Auto batch_size: [bold]{args.batch_size}[/bold] (VRAM: {vram_gb:.0f}GB)")
    elif args.batch_size == 0:
        args.batch_size = 4
        if is_main:
            console.print(f"  [green]✓[/green] Auto batch_size: [bold]{args.batch_size}[/bold] (non-CUDA device)")

    # ── Gradient checkpointing ──
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        console.print("  ♻ [green]Gradient checkpointing attivo[/green] (VRAM ridotta, training leggermente più lento)")

    # ── torch.compile ──
    if args.compile:
        try:
            model = torch.compile(model)
            console.print("  ⚡ [green]torch.compile() attivo[/green]")
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] torch.compile() non disponibile: {e}")

    # ── Datasets ──
    console.print()
    console.rule("[bold]Datasets[/bold]", style="dim")
    sft_seq_len = min(args.seq_len, config.max_seq_len)
    console.print(f"  Preparing SFT datasets... (seq_len=[bold]{sft_seq_len}[/bold], model_ctx={config.max_seq_len})")
    train_ds = SFTDataset(train_examples, tokenizer, sft_seq_len, overlong=args.overlong)
    val_ds = SFTDataset(val_examples, tokenizer, sft_seq_len)   # val: always tail-truncate (don't drop val examples)

    data_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    data_table.add_column(style="bold")
    data_table.add_column()
    data_table.add_row("Train", f"{len(train_ds)} examples")
    data_table.add_row("Val", f"{len(val_ds)} examples")
    console.print(data_table)

    # Dopo aver creato train_ds, conta una volta sola
    _ime_id = tokenizer.token_to_id("<|im_end|>")
    _n_ime = sum((s["labels"] == _ime_id).sum().item() for s in train_ds.samples)
    _n_train = sum((s["labels"] != -100).sum().item() for s in train_ds.samples)
    _ime_ratio = _n_ime / max(_n_train, 1)

    # Target: im_end si comporta come se fosse il 5% dei token
    _TARGET_RATIO = 0.05
    _SPECIAL_BOOST = min(_TARGET_RATIO / max(_ime_ratio, 1e-6), 10.0)  # cap a 10x

    console.print(f"  ⚖️  im_end: {_n_ime}/{_n_train} ({_ime_ratio:.2%}) → boost=[bold]{_SPECIAL_BOOST:.1f}x[/bold]")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── Compute max_steps from epochs if specified ──
    # arxiv:2602.11149: multi-epoch on small dataset > single-epoch on large dataset
    if args.epochs is not None:
        steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
        args.max_steps = steps_per_epoch * args.epochs
        if is_main:
            console.print(
                f"  📊 Multi-epoch: [bold]{args.epochs}[/bold] epochs × {steps_per_epoch} steps/epoch = [bold]{args.max_steps}[/bold] steps")

    # warmup as a fraction of the finalized max_steps (overrides --warmup_steps when set)
    if args.warmup_frac is not None:
        args.warmup_steps = max(1, int(args.max_steps * args.warmup_frac))
        if is_main:
            console.print(f"  Warmup: [bold]{args.warmup_frac:.1%}[/bold] of {args.max_steps} = [bold]{args.warmup_steps}[/bold] steps")

    # ── Optimizer ──
    # µP: per-parameter LR scaling when base model was trained with µP
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
        # exclude norm/bias (1D) AND embeddings/lm_head from weight decay (consistent with mup_param_groups).
        # --optimizer muon: Muon on the hidden linear maps, for a base pretrained with Muon (training/optim.py).
        optimizer, opt_info = build_optimizer(
            model, args.lr, args.weight_decay, args.optimizer, eps=1e-8, fused=use_fused,
            decay=lambda n, p: p.dim() >= 2 and "emb" not in n.lower() and "lm_head" not in n.lower())
        if is_main:
            console.print(f"  [dim]{args.optimizer}: {opt_info}[/dim]")

    # ── Accelerate: prepare model, optimizer, dataloader ──
    if use_accelerate:
        if isinstance(optimizer, OptimizerSet):        # Muon + AdamW: accelerate prepares each one
            model, train_loader, *prepared = accelerator.prepare(model, train_loader, *optimizer.opts)
            optimizer = OptimizerSet(prepared)
        else:
            model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        val_loader = accelerator.prepare(val_loader)

    # ── Wandb ──
    if args.wandb and is_main:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run or "sft", config=vars(args))

    # ── Train ──
    os.makedirs(args.out_dir, exist_ok=True)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16 and not use_accelerate))
    if scaler.is_enabled() and isinstance(optimizer, OptimizerSet):
        raise SystemExit("--optimizer muon needs bf16 (or accelerate): the fp16 GradScaler drives one optimizer")

    model.train()
    step = 0  # optimizer step (quello "vero")
    micro_step = 0  # singolo batch / micro-batch
    best_val_loss = float("inf")
    t0 = time.time()

    min_lr = args.lr * 0.1

    # Resume training state (optimizer momentum, scaler, step counter)
    if args.resume:
        state_path = os.path.join(args.resume, "training_state.pt")
        if os.path.exists(state_path):
            if is_main:
                console.print(f"  [cyan]↻[/cyan] Restoring optimizer & training state...")
            training_state = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(training_state["optimizer"])

            if not use_accelerate:
                scaler.load_state_dict(training_state["scaler"])

            step = training_state.get("step", 0)  # optimizer step già completati
            micro_step = training_state.get("micro_step", step * args.grad_accum)
            best_val_loss = training_state.get("best_val_loss", float("inf"))
            if is_main:
                console.print(
                    f"  [green]✓[/green] Resumed from optimizer step [bold]{step}[/bold] (micro_step={micro_step}, best_val={best_val_loss:.4f})")

    if is_main:
        schedule_info = args.lr_schedule
        if args.lr_schedule == "wsd":
            schedule_info += f", decay_ratio={args.lr_decay_ratio} (last {args.lr_decay_ratio * 100:.0f}%)"

        train_cfg_table = Table(box=box.ROUNDED, border_style="cyan", title="🚀 SFT Training Configuration",
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
        if s3_client:
            train_cfg_table.add_row("S3 backup", f"s3://{args.s3_bucket}/{args.s3_prefix or ''}/checkpoints_sft/")
        console.print()
        console.print(train_cfg_table)
        console.print()

    # Helper to get the unwrapped model (for save_pretrained / generate)
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
            "micro_step": micro_step,
            "best_val_loss": best_val_loss,
        }, os.path.join(save_path, "training_state.pt"))

        label = "Best model" if is_best else "Checkpoint"
        extra = f" (val_loss={best_val_loss:.4f})" if is_best else ""
        style = "bold green" if is_best else "bold"
        console.print(f"  💾 [{style}]{label} saved[/{style}]: {save_path}{extra}")

        # Async S3 upload (best-effort)
        if s3_client is not None:
            remote_prefix = f"checkpoints_sft/{name}"
            async_upload_checkpoint(s3_client, save_path, remote_prefix)

    # ── DEBUG ──
    debug_table = Table(title="Debug Info", box=box.SIMPLE, border_style="dim", title_style="dim")
    debug_table.add_column("Check", style="dim")
    debug_table.add_column("Value", style="dim")
    debug_table.add_row("tied", str(model.lm_head.weight is model.token_emb.weight))
    debug_table.add_row("params NaN", str(any(p.isnan().any().item() for p in model.parameters())))
    debug_table.add_row("model dtype", str(next(model.parameters()).dtype))
    debug_table.add_row("scaler enabled", str(scaler.is_enabled()))
    debug_table.add_row("SDP backends",
                        f"cudnn={torch.backends.cuda.cudnn_sdp_enabled()} flash={torch.backends.cuda.flash_sdp_enabled()} mem={torch.backends.cuda.mem_efficient_sdp_enabled()}")

    _ti, _tl = next(iter(train_loader))
    _ti, _tl = _ti.to(device), _tl.to(device)
    with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
        _to = model(_ti, labels=_tl)
    debug_table.add_row("first batch",
                        f"shape={_ti.shape}, loss={_to['loss'].item():.4f}, nan={_to['loss'].isnan().item()}")
    del _ti, _tl, _to
    console.print(debug_table)

    # ── Training progress bar ──
    train_progress = _make_train_progress(console)
    train_progress.start()
    train_task = train_progress.add_task("Training", total=args.max_steps, completed=step)

    while step < args.max_steps:
        for input_ids, labels in train_loader:
            if step >= args.max_steps:
                break

            # LR schedulata su optimizer-step reali
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, min_lr,
                        schedule=args.lr_schedule, decay_ratio=args.lr_decay_ratio)

            for pg in optimizer.param_groups:
                if use_mup:
                    if "_base_lr_set" not in pg:
                        pg["_base_lr"] = pg["lr"]
                        pg["_base_lr_set"] = True
                    pg["lr"] = pg["_base_lr"] * (lr / args.lr)
                else:
                    pg["lr"] = lr

            if not use_accelerate:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

            # Forward + backward (micro-step)
            if use_accelerate:
                out = model(input_ids)  # no labels → no loss
                logits = out["logits"]
                flat_logits = logits.view(-1, logits.size(-1))
                flat_labels = labels.view(-1)
                ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='none')
                w = torch.ones_like(ce)
                w[flat_labels == _ime_id] = _SPECIAL_BOOST
                valid = (flat_labels != -100).float()
                loss = (ce * w * valid).sum() / (w * valid).sum()
                loss = loss / args.grad_accum
                accelerator.backward(loss)
            else:
                with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                    out = model(input_ids)  # no labels → no loss
                    logits = out["logits"]
                    flat_logits = logits.view(-1, logits.size(-1))
                    flat_labels = labels.view(-1)
                    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='none')
                    w = torch.ones_like(ce)
                    w[flat_labels == _ime_id] = _SPECIAL_BOOST
                    valid = (flat_labels != -100).float()
                    loss = (ce * w * valid).sum() / (w * valid).sum()
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()

                # ── DEBUG NaN ──
                if micro_step <= 4:
                    raw = loss.item() * args.grad_accum  # usa la loss calcolata manualmente
                    has_nan_grad = any(
                        p.grad is not None and p.grad.isnan().any().item()
                        for p in model.parameters()
                    )
                    has_nan_param = any(p.isnan().any().item() for p in model.parameters())
                    console.print(
                        f"  [dim]TRACE micro={micro_step}: loss={raw:.4f}, nan_grad={has_nan_grad}, nan_param={has_nan_param}[/dim]")

            micro_step += 1
            do_update = (micro_step % args.grad_accum == 0)

            last_grad_norm = None
            if do_update:
                if use_accelerate:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                optimizer.zero_grad(set_to_none=True)

                # Ora incrementiamo lo step "vero"
                step += 1

                # ── Logging (su optimizer-step) ──
                if step % args.log_every == 0 and is_main:
                    raw_loss = loss.item() * args.grad_accum
                    elapsed = time.time() - t0
                    display_lr = lr
                    if use_mup:
                        for pg in optimizer.param_groups:
                            if pg.get("name") == "hidden":
                                display_lr = pg["lr"]
                                break

                    grad_norm_str = ""
                    if last_grad_norm is not None:
                        gn = last_grad_norm.item() if torch.is_tensor(last_grad_norm) else last_grad_norm
                        grad_norm_str = f" │ ∇ {gn:.2f}"

                    loss_color = "green" if raw_loss < 2.0 else ("yellow" if raw_loss < 4.0 else "red")
                    train_progress.update(
                        train_task,
                        completed=step,
                        description=(
                            f"[bold]step {step:>5d}[/bold] │ "
                            f"loss [{loss_color}]{raw_loss:.4f}[/{loss_color}] │ "
                            f"lr [cyan]{display_lr:.2e}[/cyan]"
                            f"{grad_norm_str} │ "
                            f"[dim]{elapsed:.0f}s[/dim]"
                        ),
                    )

                    if args.wandb:
                        import wandb
                        log_dict = {"sft/loss": raw_loss, "sft/lr": lr}
                        if last_grad_norm is not None:
                            log_dict["sft/grad_norm"] = gn
                        wandb.log(log_dict, step=step)

                # ── Eval ──
                if step > 0 and step % args.eval_every == 0 and is_main:
                    train_progress.stop()
                    val_loss = evaluate(unwrap_model(), val_loader, device, dtype=dtype)
                    console.print(f"  ✦ val_loss: [bold magenta]{val_loss:.4f}[/bold magenta]")

                    if args.wandb:
                        import wandb
                        wandb.log({"sft/val_loss": val_loss}, step=step)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint("best", is_best=True)
                    train_progress.start()

                # ── Sample generation ──
                if step > 0 and step % args.sample_every == 0 and is_main:
                    train_progress.stop()
                    raw_model = unwrap_model()
                    raw_model.eval()

                    token_ids = encode_chatml([
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": args.sample_prompt},
                    ], tokenizer, add_generation_prompt=True)
                    input_ids_gen = torch.tensor([token_ids], device=device)

                    output_ids = raw_model.generate(input_ids_gen, max_new_tokens=80, temperature=0.7,
                                                    repetition_penalty=1.2)

                    generated_ids = output_ids[0].tolist()
                    prompt_len = len(token_ids)
                    new_ids = generated_ids[prompt_len:]

                    ime_id = tokenizer.token_to_id("<|im_end|>")
                    if ime_id is not None and ime_id in new_ids:
                        new_ids = new_ids[:new_ids.index(ime_id)]

                    response = tokenizer.decode(new_ids)
                    if "<|im_end|>" in response:
                        response = response.split("<|im_end|>")[0]
                    console.print(Panel(
                        Text(response[:150], style="italic"),
                        title="📝 Sample Response",
                        border_style="blue",
                        width=min(console.width, 100),
                        padding=(0, 1),
                    ))

                    raw_model.train()
                    train_progress.start()

                # ── Checkpoint ──
                if step > 0 and step % args.save_every == 0 and is_main:
                    train_progress.stop()
                    save_checkpoint(f"step_{step}")
                    train_progress.start()

    train_progress.update(train_task, completed=args.max_steps)
    train_progress.stop()

    # ── Final save ──
    if is_main:
        total_time = time.time() - t0
        save_checkpoint("final")

        final_table = Table(box=box.DOUBLE_EDGE, border_style="green", title="✅ SFT Complete", title_style="bold green")
        final_table.add_column("Metric", style="bold")
        final_table.add_column("Value", style="cyan")
        final_table.add_row("Total time", f"{total_time / 60:.1f} min")
        final_table.add_row("Best val loss", f"{best_val_loss:.4f}")
        final_table.add_row("Model saved to", os.path.join(args.out_dir, "final"))
        if s3_client:
            final_table.add_row("S3 checkpoints", f"s3://{args.s3_bucket}/{args.s3_prefix or ''}/checkpoints_sft/")
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
    parser = argparse.ArgumentParser(description="SFT — Fine-tune for chat")

    # Data
    parser.add_argument("--data", type=str, default=None, help="JSONL chat dataset")
    parser.add_argument("--val_data", type=str, default=None, help="Validation JSONL")

    # Model
    parser.add_argument("--base_model", type=str, default=None,
                        help="Pre-trained base model path (from train.py)")
    parser.add_argument("--seq_len", type=int, default=512)

    # Training
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (0 = auto-estimate from VRAM)")
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Train for N epochs instead of max_steps. "
                             "Overrides max_steps. Recommended for small datasets (<5K examples). "
                             "arxiv:2602.11149: multi-epoch on small dataset beats single-epoch on large.")
    parser.add_argument("--lr", type=float, default=2e-5)  # Lower LR for fine-tuning
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--warmup_frac", type=float, default=None,
                        help="warmup as a fraction of max_steps (overrides --warmup_steps when set)")
    parser.add_argument("--overlong", default="tail_truncate", choices=["tail_truncate", "drop"],
                        help="over-length examples: tail_truncate (keep the ending w/ <|im_end|>) or drop them")
    parser.add_argument("--sample_prompt", type=str, default="Cosa è la normativa bancaria italiana?",
                        help="user prompt used for the periodic in-training sample generation")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"],
                        help="muon for a base pretrained with Muon (Skylar 2): Muon in pretraining + Muon in "
                             "SFT is the best pairing (Moonlight 2502.16982, Table 6)")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # LR Schedule
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "wsd"],
                        help="Learning rate schedule: cosine (default) or wsd (warmup-stable-decay)")
    parser.add_argument("--lr_decay_ratio", type=float, default=0.2,
                        help="WSD only: fraction of training for decay phase (default: 0.2 = last 20%%)")

    # Precision
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--sample_every", type=int, default=200)

    # Output
    parser.add_argument("--out_dir", type=str, default="checkpoints_sft")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a checkpoint (loads model, tokenizer, optimizer state)")

    # S3 (optional — async checkpoint backup)
    parser.add_argument("--s3_bucket", type=str, default=None,
                        help="AWS S3 bucket for checkpoint backup")
    parser.add_argument("--s3_prefix", type=str, default=None,
                        help="Key prefix inside bucket")
    parser.add_argument("--s3_region", type=str, default=None,
                        help="AWS region (e.g. eu-west-1)")

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="nano-transformer")
    parser.add_argument("--wandb_run", type=str, default=None)

    # Speed
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for faster training (CUDA only)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Trade compute for VRAM — reduces activation memory from O(n_layers) to O(sqrt(n_layers))")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use HuggingFace Accelerate for multi-GPU training (requires: pip install accelerate)")

    # System
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_sft(args)


if __name__ == "__main__":
    main()
