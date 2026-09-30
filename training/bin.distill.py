"""
Knowledge Distillation (KD) — compress a large teacher into a small student.

This is an optional Stage-2 variant of the pipeline:
  Stage 1: Pre-train on raw text            → python training/bin.pretrain.py (base model)
  Stage 2: SFT on conversations             → python training/bin.sft.py     (chat model)
  Stage 2*: Distill a teacher into a student → python training/bin.distill.py ← YOU ARE HERE
  Stage 3: RLVR / preference alignment       → alignment

Idea (Hinton et al. 2015; LFM2 tech report, arxiv:2511.23404): instead of training the
student ONLY against the hard next-token label, also make it imitate the full output
DISTRIBUTION of a stronger, frozen teacher. The soft targets carry the teacher's
"dark knowledge" (how plausible every *other* token was), which a one-hot label cannot.

  loss = alpha * CE(hard label) + (1 - alpha) * KD(student || teacher)

This is ONLINE distillation: the teacher lives in memory and is queried every step (no
pre-computed logits on disk). The teacher is FROZEN — gradients only ever touch the
student. Saved weights are the student's, born from random/base init: nothing of the
teacher's parameters is ever copied. KD ≠ warm-start.

SOVEREIGNTY RULE (Skylar): the teacher MUST be one of OUR OWN from-scratch Skylar models
(e.g. a 4B distilled into a 1B). NEVER distill from a third-party model — that would
inject someone else's knowledge and break the "100% from-scratch / no foreign weights"
rule. The student stays from-scratch; only the supervision signal is OUR larger model.

Teacher and student MUST share the SAME tokenizer / vocab — logit index k must mean the
same token for both. The script asserts vocab_size matches and loads the tokenizer from
the student's base model.

Two KD objectives (--kd_objective):
  full  — forward KL over the FULL vocab, temperature-scaled (Hinton). Default; the right
          choice when the teacher is in memory (you have every logit, no truncation needed).
  topk  — tempered, DECOUPLED Top-K KD (LFM2 §3.3). Splits the KL into (a) a Bernoulli
          "membership" term matching the probability mass on the teacher's Top-K set and
          (b) a temperature-scaled "shape" term inside the Top-K. Avoids the support
          mismatch that naive Top-K truncation + temperature would cause. This is the form
          you precompute offline to save logit storage on huge corpora; here it runs online.

Usage:
  # Distill a chat teacher (e.g. our 4B SFT) into a small student (base or SFT checkpoint)
  python training/bin.distill.py --data data/sft_train.jsonl \
    --base_model checkpoints/skylar-1b-base --teacher_model checkpoints/skylar-4b-chat \
    --epochs 3 --lr 2e-5 --bf16 --alpha 0.5 --kd_temperature 2.0

  # LFM2-style decoupled Top-K objective, teacher kept in bf16 to save VRAM
  python training/bin.distill.py --data data/sft_train.jsonl \
    --base_model checkpoints/skylar-1b-base --teacher_model checkpoints/skylar-4b-chat \
    --kd_objective topk --kd_top_k 32 --teacher_dtype bf16 --bf16 --epochs 3

  # Resume + S3 checkpoint backup
  python training/bin.distill.py --data data/sft_train.jsonl --resume checkpoints_distill/step_2000 \
    --teacher_model checkpoints/skylar-4b-chat --max_steps 5000 \
    --s3_bucket my-bucket --s3_prefix skylar/distill_v1 --s3_region eu-west-1

Note on checkpoint selection (memory skylar-checkpoint-eval-not-loss): the trainer tracks
val CE loss only to pick a "best" snapshot, but for the COBOL line the FINAL checkpoint must
be chosen on the executable benchmark (COBOLEval pass@1 / CSR), not on loss. Keep several
epoch snapshots and pick on the benchmark.

Env for S3 (optional):
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
"""

import argparse
import math
import os
import shutil
import sys
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# make the repo root importable when run as a script without `pip install -e .`
# (relative to THIS file — generic, no machine-specific path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)
from rich import box

console = Console()

# B200 Blackwell: cuDNN SDPA crashes on variable-length sequences
if torch.cuda.is_available() and "B200" in torch.cuda.get_device_name(0):
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    console.print(
        f"  [yellow]⚠[/yellow] SDP backends disabled: cudnn={torch.backends.cuda.cudnn_sdp_enabled()}, "
        f"flash={torch.backends.cuda.flash_sdp_enabled()}, mem={torch.backends.cuda.mem_efficient_sdp_enabled()}")

from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, decoders

from models.decoder import Skylar2ForCausalLM
from utils.chatML import create_loss_mask, load_dataset_jsonl


# ─────────────────────────────────────────────────────────────
# S3 CLIENT + ASYNC UPLOAD (best-effort, non-blocking)
# ─────────────────────────────────────────────────────────────

_upload_pool: Optional[ThreadPoolExecutor] = None


def _get_pool() -> ThreadPoolExecutor:
    global _upload_pool
    if _upload_pool is None:
        _upload_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-upload")
    return _upload_pool


class S3Client:
    """Minimal S3 upload with retries. Needs AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."""

    def __init__(self, bucket, prefix, region):
        import boto3
        from botocore.config import Config
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            raise EnvironmentError("Missing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
        self.bucket, self.prefix, self.region = bucket, (prefix or "").strip("/"), region
        self._c = boto3.client("s3", region_name=region, config=Config(max_pool_connections=10))
        console.print(f"  [green]✓[/green] S3 client ready: [cyan]s3://{bucket}/{self.prefix}[/cyan] (region={region})")

    def _key(self, name):
        return f"{self.prefix}/{name}" if self.prefix else name

    def upload(self, local_path, remote_name, retries=5):
        from boto3.s3.transfer import TransferConfig
        cfg = TransferConfig(multipart_threshold=100 * 1024 * 1024, multipart_chunksize=64 * 1024 * 1024,
                             max_concurrency=4, use_threads=True)
        key = self._key(remote_name)
        for attempt in range(1, retries + 1):
            try:
                self._c.upload_file(str(local_path), self.bucket, key, Config=cfg)
                return f"s3://{self.bucket}/{key}"
            except Exception:
                if attempt == retries:
                    console.print(f"  [bold red]✗[/bold red] S3 upload FAILED after {retries} attempts: {local_path}")
                    raise
                time.sleep(2.0 * (2 ** (attempt - 1)))

    def upload_directory(self, local_dir, remote_prefix):
        uris = []
        for f in sorted(Path(local_dir).iterdir()):
            if f.is_file():
                uris.append(self.upload(f, f"{remote_prefix}/{f.name}"))
        return uris


def async_upload_checkpoint(s3_client: S3Client, local_dir: str, remote_prefix: str) -> None:
    """Fire-and-forget S3 upload of a checkpoint dir. Copies first so training continues."""
    src = Path(local_dir)
    staging = src.parent / f".upload_staging_{src.name}"
    try:
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(src, staging)
    except Exception as e:
        console.print(f"  [yellow]⚠[/yellow] S3 staging failed for {src.name}: {e}")
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

    _get_pool().submit(_do_upload)


# ─────────────────────────────────────────────────────────────
# DATASET (ChatML, assistant-only supervision — same masking as SFT)
# ─────────────────────────────────────────────────────────────

class DistillDataset(Dataset):
    """ChatML conversations with assistant-only loss mask.

    Identical preprocessing to SFT: tokenize each segment, mask everything except the
    assistant content + closing <|im_end|> to -100, then shift for causal LM. Both the
    hard CE and the KD term are computed only on the unmasked (assistant) positions.
    """

    def __init__(self, examples, tokenizer, max_seq_len):
        self.samples = []
        skipped = truncated = 0
        for ex in examples:
            token_ids, labels = create_loss_mask(ex["messages"], tokenizer)

            # Shift for causal LM: logits[t] predicts token_ids[t+1]
            token_ids = token_ids[:-1]
            labels = labels[1:]

            if len(token_ids) < 4:
                skipped += 1
                continue

            # Tail-truncate: keep the ending so the final <|im_end|> stop label survives.
            if len(token_ids) > max_seq_len:
                truncated += 1
            token_ids = token_ids[-max_seq_len:]
            labels = labels[-max_seq_len:]

            if any(l != -100 for l in labels):
                self.samples.append({
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                })
            else:
                skipped += 1

        if skipped:
            console.print(f"  [yellow]⚠[/yellow] Skipped {skipped} examples (too short or no assistant content)")
        if truncated:
            console.print(f"  [yellow]⚠[/yellow] Tail-truncated {truncated} examples > seq_len={max_seq_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad sequences to the same length within a batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)   # pad with 0
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)   # -100 = ignore
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
    return input_ids, labels


# ─────────────────────────────────────────────────────────────
# KD OBJECTIVES
# ─────────────────────────────────────────────────────────────

def kd_loss_full(student_logits, teacher_logits, tau):
    """Forward-KL distillation over the full vocab, temperature-scaled (Hinton 2015).

    student_logits / teacher_logits: (N, V) — already gathered on the supervised positions.
    KL(teacher || student) on softened distributions, scaled by tau^2 so the gradient
    magnitude is preserved across temperatures.
    """
    log_p_s = F.log_softmax(student_logits / tau, dim=-1)
    p_t = F.softmax(teacher_logits / tau, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau ** 2)


def kd_loss_topk_decoupled(student_logits, teacher_logits, tau, k):
    """Tempered, DECOUPLED Top-K KD (LFM2 §3.3), computed online from full teacher logits.

    Splits the teacher→student KL via the chain rule into:
      L_B — a Bernoulli "membership" KL matching the probability MASS the student puts on
            the teacher's Top-K set to the mass the teacher puts there (un-tempered).
      L_T — a temperature-scaled KL on the SHAPE inside the Top-K, weighted by the teacher
            mass. Temperature is applied ONLY here, where teacher and student share support,
            which is what avoids the support-mismatch divergence of naive Top-K + temperature.

    student_logits / teacher_logits: (N, V) on supervised positions.
    """
    k = min(k, teacher_logits.size(-1))
    topk_val, topk_idx = teacher_logits.topk(k, dim=-1)                  # (N, k)

    log_p_s = F.log_softmax(student_logits, dim=-1)                      # (N, V)
    log_p_s_k = torch.gather(log_p_s, 1, topk_idx)                       # (N, k)

    # masses on the Top-K set
    p_t_full = F.softmax(teacher_logits, dim=-1)
    t_mass = p_t_full.gather(1, topk_idx).sum(-1).clamp(1e-6, 1 - 1e-6)  # (N,)
    s_mass = log_p_s_k.exp().sum(-1).clamp(1e-6, 1 - 1e-6)               # (N,)

    # (a) membership — Bernoulli KL on the Top-K mass (un-tempered)
    L_B = (t_mass * (t_mass / s_mass).log()
           + (1 - t_mass) * ((1 - t_mass) / (1 - s_mass)).log())

    # (b) shape — tempered KL inside the Top-K, renormalized on the same support
    p_t_cond = F.softmax(topk_val / tau, dim=-1)                         # (N, k)
    log_p_s_cond = F.log_softmax(log_p_s_k / tau, dim=-1)                # (N, k)
    L_T = (p_t_cond * (p_t_cond.log() - log_p_s_cond)).sum(-1) * (tau ** 2)

    return (L_B + t_mass * L_T).mean()


def distill_step_loss(student_logits, teacher_logits, labels, alpha, tau,
                      kd_objective="full", kd_top_k=32):
    """Combined per-batch loss: alpha * CE(hard) + (1 - alpha) * KD(soft).

    The KD term is computed ONLY on supervised positions (labels != -100), matching where
    the hard CE applies. Teacher logits must be detached (the teacher is frozen).
    """
    V = student_logits.size(-1)
    flat_s = student_logits.view(-1, V)
    flat_t = teacher_logits.view(-1, V)
    flat_labels = labels.view(-1)

    ce = F.cross_entropy(flat_s, flat_labels, ignore_index=-100)

    mask = flat_labels != -100
    if mask.any():
        s = flat_s[mask]
        t = flat_t[mask]
        if kd_objective == "topk":
            kd = kd_loss_topk_decoupled(s, t, tau, kd_top_k)
        else:
            kd = kd_loss_full(s, t, tau)
    else:
        kd = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    loss = alpha * ce + (1.0 - alpha) * kd
    return loss, ce.detach(), kd.detach()


# ─────────────────────────────────────────────────────────────
# LR SCHEDULE (aligned with bin.sft.py / bin.pretrain.py)
# ─────────────────────────────────────────────────────────────

def get_lr_cosine(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_lr_wsd(step, warmup_steps, max_steps, max_lr, min_lr, decay_ratio=0.2):
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    decay_steps = max(1, int(max_steps * decay_ratio))
    stable_end = max_steps - decay_steps
    if step < stable_end:
        return max_lr
    return min_lr + (max_lr - min_lr) * (1.0 - (step - stable_end) / decay_steps)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr, schedule="cosine", decay_ratio=0.2):
    if schedule == "wsd":
        return get_lr_wsd(step, warmup_steps, max_steps, max_lr, min_lr, decay_ratio)
    return get_lr_cosine(step, warmup_steps, max_steps, max_lr, min_lr)


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(student, val_loader, device, max_batches=50, dtype=torch.bfloat16):
    """Average hard-CE loss on the validation set (used to pick the 'best' snapshot)."""
    student.eval()
    total, n = 0.0, 0
    for i, (input_ids, labels) in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids, labels = input_ids.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            out = student(input_ids, labels=labels)
        if out["loss"] is not None:
            total += out["loss"].item()
            n += 1
    student.train()
    return total / max(n, 1)


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def _make_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"), TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn(),
        console=console, refresh_per_second=2,
    )


def load_frozen_teacher(path, device, teacher_dtype):
    """Load the teacher, freeze it, move to device. Returns (model, vocab_size)."""
    teacher = Skylar2ForCausalLM.from_pretrained(path)
    teacher.eval()
    teacher.requires_grad_(False)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    td = dtype_map.get(teacher_dtype, torch.float32)
    if td != torch.float32 and device.type != "cuda":
        td = torch.float32  # half precision only meaningful on CUDA
    teacher = teacher.to(device=device, dtype=td)
    return teacher, teacher.config.vocab_size


def train_distill(args):
    console.print()
    console.rule("[bold cyan]Skylar2ForCausalLM Distillation[/bold cyan]", style="cyan")
    console.print()

    # ── Seed ──
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Device ──
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
    if args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        console.print("  [dim]Using bfloat16 mixed precision[/dim]")
    elif args.fp16 and device.type == "cuda":
        dtype = torch.float16
        console.print("  [dim]Using float16 mixed precision[/dim]")

    # ── S3 (optional) ──
    s3_client: Optional[S3Client] = None
    if args.s3_bucket:
        s3_client = S3Client(args.s3_bucket, args.s3_prefix or "", args.s3_region or "us-east-1")

    # ── Data ──
    console.print()
    console.rule("[bold]Data[/bold]", style="dim")
    if not (args.data and os.path.exists(args.data)):
        console.print("  [bold red]✗[/bold red] No --data JSONL found")
        sys.exit(1)
    train_examples = load_dataset_jsonl(args.data)

    if args.val_data and os.path.exists(args.val_data):
        val_examples = load_dataset_jsonl(args.val_data)
    else:
        split = max(1, len(train_examples) // 10)
        val_examples = train_examples[-split:]
        train_examples = train_examples[:-split]

    # ── Tokenizer (from resume or base model) ──
    if args.resume and os.path.exists(os.path.join(args.resume, "tokenizer.json")):
        tokenizer_path = os.path.join(args.resume, "tokenizer.json")
        console.print(f"  [green]✓[/green] Tokenizer from checkpoint: [bold]{args.resume}[/bold]")
    elif args.base_model and os.path.exists(os.path.join(args.base_model, "tokenizer.json")):
        tokenizer_path = os.path.join(args.base_model, "tokenizer.json")
        console.print(f"  [green]✓[/green] Tokenizer from base model: [bold]{args.base_model}[/bold]")
    else:
        console.print("  [bold red]✗[/bold red] No tokenizer.json — provide --base_model or --resume")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    if tokenizer.decoder is None:
        tokenizer.decoder = decoders.ByteLevel()
    vocab_size = tokenizer.get_vocab_size()

    # ── Student ──
    console.print()
    console.rule("[bold]Student[/bold]", style="dim")
    if args.resume and os.path.exists(args.resume):
        console.print(f"  [cyan]↻[/cyan] Resuming student from: [bold]{args.resume}[/bold]")
        student = Skylar2ForCausalLM.from_pretrained(args.resume)
    elif args.base_model and os.path.exists(args.base_model):
        console.print(f"  [green]✓[/green] Student init from base model: [bold]{args.base_model}[/bold]")
        student = Skylar2ForCausalLM.from_pretrained(args.base_model)
    else:
        console.print("  [bold red]✗[/bold red] Provide --base_model (student init) or --resume")
        sys.exit(1)

    config = student.config
    student = student.to(device)
    use_mup = config.mup_base_d_model is not None

    # ── Teacher (frozen) ──
    console.print()
    console.rule("[bold]Teacher[/bold]", style="dim")
    if not (args.teacher_model and os.path.exists(args.teacher_model)):
        console.print("  [bold red]✗[/bold red] --teacher_model is required and must exist")
        sys.exit(1)
    console.print(f"  [green]✓[/green] Frozen teacher: [bold]{args.teacher_model}[/bold] (dtype={args.teacher_dtype})")
    teacher, teacher_vocab = load_frozen_teacher(args.teacher_model, device, args.teacher_dtype)

    # ── Vocab compatibility (hard requirement) ──
    if teacher_vocab != vocab_size:
        console.print(
            f"  [bold red]✗[/bold red] Vocab mismatch: teacher={teacher_vocab} vs student/tokenizer={vocab_size}. "
            "Distillation requires the SAME tokenizer — logit index k must mean the same token for both.")
        sys.exit(1)
    console.print(f"  [green]✓[/green] Vocab match: [bold]{vocab_size:,}[/bold] — teacher and student aligned")

    # ── Model table ──
    info = Table(title="🧪 Distillation Setup", box=box.ROUNDED, title_style="bold magenta", border_style="dim",
                 padding=(0, 1))
    info.add_column("Field", style="bold")
    info.add_column("Value", style="cyan")
    info.add_row("Student params", f"{student.count_params():,}")
    info.add_row("Teacher params", f"{teacher.count_params():,}")
    info.add_row("d_model (student)", str(config.d_model))
    info.add_row("n_layers (student)", str(config.n_layers))
    info.add_row("vocab_size", f"{vocab_size:,}")
    info.add_row("KD objective", args.kd_objective + (f" (k={args.kd_top_k})" if args.kd_objective == "topk" else ""))
    info.add_row("alpha (CE weight)", f"{args.alpha}")
    info.add_row("temperature", f"{args.kd_temperature}")
    console.print(info)
    if use_mup:
        console.print(f"  📐 [bold]µP[/bold] — base_d={config.mup_base_d_model}, width_mult={config.mup_width_mult:.1f}x")

    # ── Gradient checkpointing / compile (student only) ──
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        console.print("  ♻ [green]Gradient checkpointing[/green] (student)")
    if args.compile:
        try:
            student = torch.compile(student)
            console.print("  ⚡ [green]torch.compile()[/green] (student)")
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] torch.compile() unavailable: {e}")

    # ── Datasets ──
    console.print()
    console.rule("[bold]Datasets[/bold]", style="dim")
    seq_len = min(args.seq_len, config.max_seq_len)
    console.print(f"  Preparing datasets... (seq_len=[bold]{seq_len}[/bold], model_ctx={config.max_seq_len})")
    train_ds = DistillDataset(train_examples, tokenizer, seq_len)
    val_ds = DistillDataset(val_examples, tokenizer, seq_len)
    console.print(f"  Train: [bold]{len(train_ds)}[/bold] · Val: [bold]{len(val_ds)}[/bold]")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ── Steps from epochs ──
    if args.epochs is not None:
        steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
        args.max_steps = steps_per_epoch * args.epochs
        console.print(f"  📊 {args.epochs} epochs × {steps_per_epoch} steps/epoch = [bold]{args.max_steps}[/bold] steps")

    # ── Optimizer (µP-aware, no WD on norms/embeddings) ──
    use_fused = device.type == "cuda"
    if use_mup:
        param_groups = student.mup_param_groups(args.lr, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    else:
        decay, nodecay = [], []
        for n, p in student.named_parameters():
            (nodecay if (p.dim() < 2 or "emb" in n.lower() or "lm_head" in n.lower()) else decay).append(p)
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    # ── Wandb ──
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run or "distill", config=vars(args))

    # ── Train state ──
    os.makedirs(args.out_dir, exist_ok=True)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
    student.train()
    step = micro_step = 0
    best_val_loss = float("inf")
    min_lr = args.lr * 0.1
    t0 = time.time()

    if args.resume:
        state_path = os.path.join(args.resume, "training_state.pt")
        if os.path.exists(state_path):
            console.print("  [cyan]↻[/cyan] Restoring optimizer & training state...")
            st = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(st["optimizer"])
            if dtype == torch.float16 and "scaler" in st:
                scaler.load_state_dict(st["scaler"])
            step = st.get("step", 0)
            micro_step = st.get("micro_step", step * args.grad_accum)
            best_val_loss = st.get("best_val_loss", float("inf"))
            console.print(f"  [green]✓[/green] Resumed at step [bold]{step}[/bold] (best_val={best_val_loss:.4f})")

    def save_checkpoint(name, is_best=False):
        save_path = os.path.join(args.out_dir, name)
        # unwrap torch.compile to get a clean save_pretrained
        to_save = getattr(student, "_orig_mod", student)
        to_save.save_pretrained(save_path)
        tokenizer.save(os.path.join(save_path, "tokenizer.json"))
        torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "micro_step": micro_step, "best_val_loss": best_val_loss},
                   os.path.join(save_path, "training_state.pt"))
        label = "Best model" if is_best else "Checkpoint"
        extra = f" (val_loss={best_val_loss:.4f})" if is_best else ""
        console.print(f"  💾 [bold]{label} saved[/bold]: {save_path}{extra}")
        if s3_client is not None:
            async_upload_checkpoint(s3_client, save_path, f"checkpoints_distill/{name}")

    # ── Config table ──
    cfg = Table(box=box.ROUNDED, border_style="cyan", title="🚀 Distill Configuration", title_style="bold cyan")
    cfg.add_column("Setting", style="bold")
    cfg.add_column("Value", style="white")
    cfg.add_row("Max steps", f"{args.max_steps:,}")
    cfg.add_row("Batch size", str(args.batch_size))
    cfg.add_row("Grad accumulation", str(args.grad_accum))
    cfg.add_row("Effective batch", str(args.batch_size * args.grad_accum))
    cfg.add_row("Learning rate", f"{args.lr:.2e}")
    cfg.add_row("Warmup steps", str(args.warmup_steps))
    cfg.add_row("Schedule", args.lr_schedule)
    cfg.add_row("Precision", str(dtype).replace("torch.", ""))
    console.print()
    console.print(cfg)
    console.print()

    progress = _make_progress()
    progress.start()
    task = progress.add_task("Distilling", total=args.max_steps, completed=step)

    while step < args.max_steps:
        for input_ids, labels in train_loader:
            if step >= args.max_steps:
                break

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

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                # teacher: frozen, no grad — only an oracle producing target logits
                with torch.no_grad():
                    teacher_logits = teacher(input_ids)["logits"]
                # student: trainable
                student_logits = student(input_ids)["logits"]
                loss, ce, kd = distill_step_loss(
                    student_logits, teacher_logits.to(student_logits.dtype).detach(), labels,
                    alpha=args.alpha, tau=args.kd_temperature,
                    kd_objective=args.kd_objective, kd_top_k=args.kd_top_k)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            micro_step += 1
            if micro_step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    raw = loss.item() * args.grad_accum
                    gn = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                    progress.update(task, completed=step, description=(
                        f"[bold]step {step:>5d}[/bold] │ "
                        f"loss [green]{raw:.4f}[/green] │ ce {ce.item():.3f} │ kd {kd.item():.3f} │ "
                        f"lr [cyan]{lr:.2e}[/cyan] │ ∇ {gn:.2f}"))
                    if args.wandb:
                        import wandb
                        wandb.log({"distill/loss": raw, "distill/ce": ce.item(), "distill/kd": kd.item(),
                                   "distill/lr": lr, "distill/grad_norm": gn}, step=step)

                if step > 0 and step % args.eval_every == 0:
                    progress.stop()
                    val_loss = evaluate(getattr(student, "_orig_mod", student), val_loader, device, dtype=dtype)
                    console.print(f"  ✦ val_loss (CE): [bold magenta]{val_loss:.4f}[/bold magenta]")
                    if args.wandb:
                        import wandb
                        wandb.log({"distill/val_loss": val_loss}, step=step)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint("best", is_best=True)
                    progress.start()

                if step > 0 and step % args.save_every == 0:
                    progress.stop()
                    save_checkpoint(f"step_{step}")
                    progress.start()

    progress.update(task, completed=args.max_steps)
    progress.stop()

    save_checkpoint("final")
    total_time = time.time() - t0
    done = Table(box=box.DOUBLE_EDGE, border_style="green", title="✅ Distillation Complete", title_style="bold green")
    done.add_column("Metric", style="bold")
    done.add_column("Value", style="cyan")
    done.add_row("Total time", f"{total_time / 60:.1f} min")
    done.add_row("Best val loss (CE)", f"{best_val_loss:.4f}")
    done.add_row("Saved to", os.path.join(args.out_dir, "final"))
    console.print()
    console.print(done)
    console.print()

    if _upload_pool is not None:
        console.print("  [dim]Waiting for pending S3 uploads...[/dim]")
        _upload_pool.shutdown(wait=True)
        console.print("  [green]✓[/green] All S3 uploads completed.")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation — compress a teacher into a student")

    # Data
    parser.add_argument("--data", type=str, default=None, help="JSONL ChatML dataset")
    parser.add_argument("--val_data", type=str, default=None, help="Validation JSONL (else 10%% split)")

    # Models
    parser.add_argument("--base_model", type=str, default=None,
                        help="Student init: a pre-trained Skylar base (or SFT) checkpoint")
    parser.add_argument("--teacher_model", type=str, default=None,
                        help="Frozen teacher: a LARGER from-scratch Skylar checkpoint (required). "
                             "Must share the student's tokenizer/vocab. NEVER a third-party model.")
    parser.add_argument("--teacher_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Teacher inference dtype (bf16 saves VRAM; CUDA only for half precision)")
    parser.add_argument("--seq_len", type=int, default=2048)

    # Distillation
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight of the hard CE term: loss = alpha*CE + (1-alpha)*KD")
    parser.add_argument("--kd_temperature", type=float, default=2.0,
                        help="Softmax temperature for the KD term (>1 softens, surfaces dark knowledge)")
    parser.add_argument("--kd_objective", type=str, default="full", choices=["full", "topk"],
                        help="full = forward-KL over full vocab (default); topk = decoupled tempered Top-K (LFM2)")
    parser.add_argument("--kd_top_k", type=int, default=32, help="Top-K size for --kd_objective topk")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Train for N epochs instead of max_steps (overrides max_steps)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # LR schedule
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "wsd"])
    parser.add_argument("--lr_decay_ratio", type=float, default=0.2, help="WSD only: fraction for the decay phase")

    # Precision
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # Logging / eval / save
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)

    # Output / resume
    parser.add_argument("--out_dir", type=str, default="checkpoints_distill")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a distill checkpoint")

    # S3 (optional)
    parser.add_argument("--s3_bucket", type=str, default=None)
    parser.add_argument("--s3_prefix", type=str, default=None)
    parser.add_argument("--s3_region", type=str, default=None)

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="nano-transformer")
    parser.add_argument("--wandb_run", type=str, default=None)

    # Speed / system
    parser.add_argument("--compile", action="store_true", help="torch.compile() the student (CUDA)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Trade compute for VRAM (student)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")

    train_distill(args)


if __name__ == "__main__":
    main()
