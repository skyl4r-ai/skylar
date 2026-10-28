# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
From-scratch decoder-only pretrain — the framework's real long-run trainer.

Memmap uint16/uint32 streaming loader (O(1) RAM) + HuggingFace Accelerate (DDP) multi-GPU +
torch.compile, with full telemetry to metrics.jsonl, milestone snapshots, and crash-safe resume
(model + optimizer + RNG + data-sampler/prefetch state). Cosine or WSD LR schedule. Optional
document-masked attention and optional async S3 checkpoint upload.

  single-GPU:  python training/bin.pretrain.py --preset medium --data data/tokenized ...
  multi-GPU :  accelerate launch --num_processes 4 --mixed_precision no \
                 training/bin.pretrain.py --preset 1b --data data/tokenized --seq_len 8192 \
                 --batch_size 4 --grad_accum 8 --epochs 1 --out checkpoints/run
               # per-rank seed + tok/step x num_processes -> 1 epoch = corpus, split across ranks

LAUNCH WITH `--mixed_precision no`: bf16 is done via a MANUAL autocast around the forward.
accelerate's bf16 path wraps the output in convert_to_fp32 -> .float() on the FULL logits tensor
(GBs at long seq -> wasteful + OOM). Only the scalar loss matters; logits stay bf16.

History: this unifies the old framework trainer with an optimized long-run trainer. Kept from
the old bin.pretrain.py: WSD schedule, S3 checkpoint upload, document masking, dtype-flex (now in
data/memmap_dataset.py). NOT ported (recover from git history if needed): auto-batch (--batch_size 0),
µP param-group rich table, sample-during-train (--sample_every), fp16/GradScaler (obsolete — bf16
wins), and S3 DATA download (now a pre-step: fetch shards to disk first; the loader reads local).
"""
import argparse, hashlib, json, math, os, shutil, signal, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

# make the repo root importable when run as a script without `pip install -e .`
# (relative to THIS file — generic, no machine-specific path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import (DDPCommunicationHookType, DistributedDataParallelKwargs, InitProcessGroupKwargs,
                              gather_object, set_seed)
from tokenizers import Tokenizer

from models.config import Skylar2Config, get_config
from models.decoder import Skylar2ForCausalLM
from data.memmap_dataset import MemmapTokenDataset, MixtureTokenDataset, Prefetcher
from training.optim import OptimizerSet, build_optimizer
from training.telemetry import NodeTelemetry
from eval.bpb import bpb_sums

ROOT = Path(__file__).resolve().parent.parent


# ── LR schedules ──
def lr_cosine(step, warmup, total, base, min_ratio):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def lr_wsd(step, warmup, total, base, min_ratio, decay_ratio=0.1):
    """Warmup-Stable-Decay (DeepSeek-V3/OLMo-2/MiniCPM): warmup -> constant -> linear decay.
    Resumable mid-stable without knowing total in advance; flatter minima -> better SFT."""
    min_lr = base * min_ratio
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    decay_steps = max(1, int(total * decay_ratio))
    stable_end = total - decay_steps
    if step < stable_end:
        return base
    return min_lr + (base - min_lr) * (1.0 - (step - stable_end) / decay_steps)


def lr_constant(step, warmup, base):
    """WSM (2507.17634): warmup lineare 0->peak, poi LR COSTANTE per sempre. NESSUN decay online.
    Il 'decay' e' emulato POST-HOC dal merge degli ultimi N checkpoint (bin.merge_wsm.py) — il che
    e' algebricamente un LR-decay sui gradienti di quella finestra (Teorema 3.1 del paper)."""
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    return base


def lr_at(step, warmup, total, base, min_ratio=0.1, schedule="cosine", decay_ratio=0.1):
    if schedule == "constant":
        return lr_constant(step, warmup, base)
    if schedule == "wsd":
        return lr_wsd(step, warmup, total, base, min_ratio, decay_ratio)
    return lr_cosine(step, warmup, total, base, min_ratio)


# ── optional async S3 checkpoint upload ──
_upload_pool = None


def _get_pool():
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

    def _key(self, name):
        return f"{self.prefix}/{name}" if self.prefix else name

    def upload(self, local_path, remote_name, retries=5):
        from boto3.s3.transfer import TransferConfig
        cfg = TransferConfig(multipart_threshold=100 << 20, multipart_chunksize=64 << 20,
                             max_concurrency=4, use_threads=True)
        for a in range(1, retries + 1):
            try:
                self._c.upload_file(str(local_path), self.bucket, self._key(remote_name), Config=cfg)
                return
            except Exception:
                if a == retries:
                    raise
                time.sleep(2.0 * (2 ** (a - 1)))

    def upload_directory(self, local_dir, remote_prefix):
        for f in sorted(Path(local_dir).iterdir()):
            if f.is_file():
                self.upload(f, f"{remote_prefix}/{f.name}")


def async_upload_checkpoint(s3, local_dir, remote_prefix):
    """Fire-and-forget: stage a copy, upload in a background thread, never crash training."""
    src = Path(local_dir)
    staging = src.parent / f".upload_staging_{src.name}"
    try:
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(src, staging)
    except Exception as e:
        print(f"[s3] staging failed for {src.name}: {e}", flush=True)
        return

    def _do():
        try:
            s3.upload_directory(staging, remote_prefix)
            print(f"[s3] uploaded {remote_prefix}", flush=True)
        except Exception as e:
            print(f"[s3] upload failed {remote_prefix}: {e}", flush=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    _get_pool().submit(_do)


# ── telemetry ──
def gpu_telemetry():
    try:
        q = "power.draw,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        gpus = []
        for i, line in enumerate(out.splitlines()):
            v = [c.strip() for c in line.split(",")]
            try:
                gpus.append({"idx": i, "power_w": float(v[0]), "util_pct": float(v[1]),
                             "mem_used_mb": float(v[2]), "mem_total_mb": float(v[3]),
                             "temp_c": float(v[4]), "sm_clock_mhz": float(v[5])})
            except (ValueError, IndexError):
                continue
        return gpus
    except Exception:
        return []


@torch.no_grad()
def global_weight_norm(m):
    s = 0.0
    for p in m.parameters():
        s += float(p.detach().float().pow(2).sum().item())
    return math.sqrt(s)


@torch.no_grad()
def matrix_norms(m, k=8):
    mats = [(n, p) for n, p in m.named_parameters() if p.ndim >= 2]
    if not mats:
        return {}
    n = len(mats)
    idx = sorted(set([0, n - 1] + [round(i * (n - 1) / max(1, k - 1)) for i in range(k)]))
    return {mats[i][0]: float(mats[i][1].detach().float().norm().item()) for i in idx}


@torch.no_grad()
def state_finite(m, opt=None):
    """True if every parameter and every floating tensor of the optimizer state (AdamW moments, Muon momentum) is
    finite: one device sync, read before anything is saved. A NaN in a moment with finite gradients turns into NaN
    weights only at the next update, after the save."""
    tensors = list(m.parameters())
    if opt is not None:
        for o in getattr(opt, "opts", [opt]):
            for st in o.state.values():
                tensors += [v for v in st.values() if torch.is_tensor(v) and v.is_floating_point() and v.numel() > 1]
    bad = None
    for t in tensors:
        b = (~torch.isfinite(t)).sum()
        bad = b if bad is None else bad + b
    return bad is None or int(bad.item()) == 0


@torch.no_grad()
def out_proj_norms(m):
    """Frobenius norm of every output projection (W_o of attention and KDA): under Muon with weak decay their
    growth late in a constant-LR run is the first sign that the effective step is shrinking."""
    return {n.removesuffix(".weight"): round(float(p.detach().float().norm().item()), 4)
            for n, p in m.named_parameters() if n.endswith("W_o.weight")}


def _prev_name(path, k):
    return path.with_name(path.name + (".prev" if k == 1 else f".prev{k}"))


def _ring_index(p):
    """1 for <name>.prev, k for <name>.prevK, None for anything else."""
    head, sep, k = Path(p).name.rpartition(".prev")
    return int(k or 1) if sep and head and (k == "" or k.isdigit()) else None


def _ring_prevs(path):
    """The existing previous saves of `path` (path.prev, path.prev2, ...), newest first."""
    path = Path(path)
    found = []
    for p in path.parent.glob(path.name + ".prev*"):
        k = p.name[len(path.name) + len(".prev"):]
        if k == "" or k.isdigit():
            found.append((int(k or 1), p))
    return [p for _, p in sorted(found)]


def save_ckpt(path, raw_model, opt, step, tokens_seen, args, best_val, ds=None, train_pf=None, s3=None,
              data_state=None, keep=1):
    """Atomic save (tmp -> os.replace, .prev backup) of model + opt + RNG + sampler/prefetch RNG, with the
    tokenizer next to the weights (every checkpoint can be fine-tuned or evaluated as it is).
    keep > 1: the previous saves stay as <path>.prev, <path>.prev2 ... <path>.prev{keep-1} (a ring to go back to
    after a problem noticed late); keep = 1 deletes the previous one once the new one is in place.
    Optional fire-and-forget S3 upload of the finished checkpoint dir."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    raw_model.save_pretrained(str(tmp))
    if args.tokenizer and Path(args.tokenizer).is_file():
        shutil.copyfile(args.tokenizer, tmp / "tokenizer.json")
    state = {
        "opt": opt.state_dict(), "step": step, "tokens_seen": tokens_seen,
        "best_val": best_val, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "args": vars(args),
    }
    state.update(data_state or {})
    if ds is not None:
        try:
            state["sampler_rng"] = ds.rng.bit_generator.state
        except Exception:
            pass
    if train_pf is not None:
        try:
            state["train_pf_rng"] = train_pf.rng_state()
        except Exception:
            pass
    torch.save(state, tmp / "training_state.pt")
    if data_state:   # small: what a later run needs to continue this one's data (--sampler_state), without the optimizer
        (tmp / "sampler_state.json").write_text(json.dumps(
            {**{k: data_state[k] for k in SAMPLER_KEYS if k in data_state}, "seed": args.seed}, default=int))
    # what a launcher needs without loading the optimizer state (GBs): where this checkpoint is
    (tmp / "progress.json").write_text(json.dumps({"step": step, "tokens": tokens_seen, "t": round(time.time(), 1)}))
    bak = _prev_name(path, 1)
    if path.exists():
        if keep > 1:
            # shift the ring, compacting it: the existing previous saves, newest first, become prev2, prev3, ... and
            # the ones beyond keep-1 drop out. (A gap, left by a resume that set the newest ones aside, used to make
            # the oldest slot drop out first: the checkpoint the run had just come back to.)
            olds = _ring_prevs(path)
            for p in olds[keep - 2:]:
                shutil.rmtree(p, ignore_errors=True)
            moves = [(olds[i], _prev_name(path, i + 2)) for i in range(min(len(olds), keep - 2))]
            # A folder cannot be renamed onto one that still exists. Below a gap (a rollback to last.prev3 sets prev and
            # prev2 aside) the entries go down by one or more slots, the rest go up by one: the ones going up move
            # oldest first, the ones going down newest first, so every target is free when its turn comes.
            up = [(a, b) for a, b in moves if _ring_index(b) > _ring_index(a)]
            down = [(a, b) for a, b in moves if _ring_index(b) < _ring_index(a)]
            for a, b in reversed(up):
                os.replace(a, b)
            for a, b in down:
                os.replace(a, b)
        else:
            shutil.rmtree(bak, ignore_errors=True)
        os.replace(path, bak)
    os.replace(tmp, path)
    if keep <= 1:
        shutil.rmtree(bak, ignore_errors=True)
    if s3 is not None:
        async_upload_checkpoint(s3, str(path), path.name)


def resume_ring(out):
    """`last` and its ring, newest first: last, last.prev, last.prev2, ... A kill between the two renames of save_ckpt
    leaves no `last` and a complete last.prev; a checkpoint found broken later is skipped for the next older one."""
    return [Path(out) / "last"] + _ring_prevs(Path(out) / "last")


def run_history(out):
    """What says this run has already trained, or None at a first launch: a status or heartbeat past step 0, a
    milestone, a WSM snapshot, a `last` folder. A run with a history and no usable checkpoint must not start over from
    random weights without anybody noticing."""
    out = Path(out)
    for name in ("status.json", "heartbeat.json"):
        try:
            st = json.loads((out / name).read_text())
            if int(st.get("step") or 0) > 0:
                return f"{name} says step {st['step']}"
        except (OSError, ValueError, TypeError):
            pass
    found = sorted(out.glob("step_tok*B")) + sorted((out / "wsm").glob("snap_*")) + sorted(out.glob("last*"))
    return f"{found[-1]} exists" if found else None


def load_resume_point(path):
    """(state dict of the weights, training state) of a checkpoint, both read and checked: raises if a file is missing,
    unreadable or holds non-finite weights."""
    path = Path(path)
    if not (path / "training_state.pt").is_file():
        raise FileNotFoundError(f"{path}/training_state.pt not found")
    sd = Skylar2ForCausalLM.from_pretrained(str(path)).state_dict()
    bad = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise ValueError(f"non-finite weights in {len(bad)} tensors ({bad[0]}, ...)")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    return sd, state


def _tokens_of(path):
    """Token count a milestone (progress.json) or a WSM snapshot (wsm_snapshot.json) was saved at; None if unknown."""
    for name in ("progress.json", "wsm_snapshot.json"):
        try:
            return int(json.loads((Path(path) / name).read_text())["tokens"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return None


def set_aside_future(out, tokens, newer_ring=(), stamp=None):
    """After a resume from `tokens`: the milestones and WSM snapshots saved beyond it belong to a trajectory that no
    longer exists (a rollback, or a kill after a milestone and before `last`). Left in place, the milestone would not be
    saved again on the new trajectory and the merge would average two branches. They move to <out>/abandoned/<stamp>/,
    with the ring entries beyond it: the unusable ones skipped by the resume, and after a rollback by hand (last.prev3
    copied to last) the newer last.prevN, which the next save would shift into the ring of the new trajectory. Returns
    the moved paths."""
    out = Path(out)
    moved = []
    future = [p for p in sorted(out.glob("step_tok*B")) + sorted((out / "wsm").glob("snap_*")) + _ring_prevs(out / "last")
              if p.is_dir() and p not in newer_ring and (_tokens_of(p) or 0) > tokens]   # undated folders stay put
    for p in list(newer_ring) + future:
        if not p.exists():
            continue
        dest = out / "abandoned" / (stamp or time.strftime("%Y%m%d_%H%M%S")) / p.relative_to(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, dest)
        moved.append(dest)
    return moved


def save_weights_only(path, raw_model, meta=None, tokenizer_path=None, s3=None):
    """WSM snapshot: SOLO pesi (save_pretrained + config), NIENTE training_state.pt.
    Per un 4B: ~8GB bf16 vs ~30GB con l'opt state -> permette 8-10 snapshot nella merge-window
    senza esplodere il disco. Scrive wsm_snapshot.json {step, tokens} cosi' bin.merge_wsm.py
    puo' selezionare la finestra per token. Atomico (tmp -> os.replace). NON e' un resume-point."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    raw_model.save_pretrained(str(tmp))
    if meta is not None:
        (tmp / "wsm_snapshot.json").write_text(json.dumps(meta))
    if tokenizer_path:
        try:
            Tokenizer.from_file(tokenizer_path).save(str(tmp / "tokenizer.json"))
        except Exception:
            pass
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)
    if s3 is not None:
        async_upload_checkpoint(s3, str(path), f"wsm/{path.name}")


VAL_SEED = 1_000_003


def val_plan(ds, n_windows, batch_size, rank, world):
    """This rank's share of the fixed validation set, in batches: [(folder, [window, ...]), ...]. The set is the first
    n_windows windows of every folder's val split in a fixed shuffled order (window_at with VAL_SEED): the same windows
    for any number of GPUs, any mixture weights and any seed, split across the ranks by index. Until 10/2026 every
    rank drew 40 batches with its own generator and the mixture weights: the curve moved with the world size and at
    the burst."""
    plan = []
    for s, part in enumerate(getattr(ds, "parts", [ds])):
        ks = list(range(rank, min(n_windows, part.n_windows("val")), world))
        plan += [(s, ks[i:i + batch_size]) for i in range(0, len(ks), batch_size)]
    return plan


def val_batch(ds, s, ks, device, return_doc_ids=False, bos_id=None):
    """The (x, y[, doc]) batch of validation windows `ks` of folder `s`."""
    part = getattr(ds, "parts", [ds])[s]
    return part.batch_of("val", ks, VAL_SEED, device, return_doc_ids, bos_id)


def _needs_doc_ids(args):
    """Il modello che sta per essere costruito ha layer ricorrenti?"""
    return bool(getattr(args, "kda_ratio", None))


# ── long runs on a scheduler: deadline, stop signals, heartbeat, status ──
def _slurm_seconds(s):
    """Slurm time ([days-]hours:minutes:seconds, minutes:seconds, ...) -> seconds; None if not a time."""
    s = s.strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + sec


def parse_data_mix(spec):
    """'path=weight,path=weight' -> [(path, weight)]. Every folder must carry the same tokenizer.json: token ids of
    two tokenizers mixed in one model are noise that no metric shows until the end."""
    sources = []
    for item in spec.split(","):
        if item.strip():
            path, _, w = item.strip().rpartition("=")
            if not path:
                raise SystemExit(f"--data_mix: '{item}' is not path=weight")
            sources.append((path, float(w)))
    digests = {}
    for path, _ in sources:
        tok = Path(path) / "tokenizer.json"
        if not tok.is_file():
            raise SystemExit(f"--data_mix: {path} has no tokenizer.json")
        digests[path] = hashlib.sha256(tok.read_bytes()).hexdigest()
    if len(set(digests.values())) > 1:
        raise SystemExit("--data_mix: the folders have different tokenizers:\n"
                         + "\n".join(f"  {d[:12]}  {p}" for p, d in digests.items()))
    return sources


def folder_id(path):
    """What identifies a tokenized folder's content: its pretokenized_meta.json (tokens, shards) and its tokenizer."""
    h = hashlib.sha256()
    for name in ("pretokenized_meta.json", "tokenizer.json"):
        f = Path(path) / name
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


# the data state a later run needs to continue this one's (--sampler_state), saved with every checkpoint as sampler_state.json
SAMPLER_KEYS = ("data_fp", "samples_done", "seq_len", "pack", "mix_ids", "mix_state")


def load_sampler_state(path):
    """The data state of a checkpoint (samples read, mix state, folder ids, seq_len, packing, seed) from its
    sampler_state.json, or from its training_state.pt memory-mapped (the optimizer state is not read). None if
    it has neither."""
    p = Path(path)
    if p.is_dir():
        p = next((p / n for n in ("sampler_state.json", "training_state.pt") if (p / n).is_file()), None)
    if p is None or not p.is_file():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text())
    st = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    old_args = st.get("args") or {}
    out = {k: st[k] for k in SAMPLER_KEYS if k in st}
    out.setdefault("seq_len", old_args.get("seq_len"))
    out.setdefault("pack", old_args.get("pack", "off"))
    out["seed"] = old_args.get("seed")
    return out


def carry_sampler_state(args, ds, sources, data_fp, is_main):
    """Where a run from --init_from starts its data: (samples_done, old mix state, folders to start again).

    Every run used to start each folder at the first window of the same permutation (seed + folder name), so the
    next stage of a chain, with the same folders, read the windows of the stage before again (28/10/2026). With
    --sampler_state the earlier run's position is carried: every folder continues its shuffled passes, a folder with
    another weight too, a new folder starts at 0. Without it, a sampler state in the --init_from dir is refused, as
    is a state that cannot be continued (another seq_len, packing or seed, a folder whose content changed): the
    windows would be read again. --sampler_fresh_ok accepts that explicitly."""
    src = args.sampler_state or args.init_from
    prev = load_sampler_state(src) if src else None
    if prev is None:
        if args.sampler_state:
            raise SystemExit(f"--sampler_state {args.sampler_state}: no sampler_state.json or training_state.pt there")
        return 0, None, []
    done = prev.get("samples_done")
    if done is None:
        if args.sampler_state:
            raise SystemExit(f"--sampler_state {src}: the checkpoint does not record the samples it read")
        if is_main:
            print(f"[sampler] {src} does not record the samples it read: the data starts at the first window",
                  flush=True)
        return 0, None, []
    done = int(done)
    if args.data_mix:
        names = ds.names
        windows = {n: p.n_windows("train") for n, p in zip(names, ds.parts)}
        read = MixtureTokenDataset.given_by(prev["mix_state"], done) if prev.get("mix_state") else {}
        ids_now = {Path(p).name: folder_id(p) for p, _ in sources}
        ids_old = prev.get("mix_ids") or {}
        changed = [n for n in names if read.get(n, 0) and n in ids_old and ids_old[n] != ids_now[n]]
    else:
        names = [Path(args.data).name]
        windows = {names[0]: ds.n_windows("train")}
        same = not prev.get("mix_state") and prev.get("data_fp") == data_fp
        read, changed = ({names[0]: done} if same else {}), []
    other = [f"{k} {prev.get(k)} -> {v}" for k, v in (("seed", args.seed), ("seq_len", args.seq_len),
                                                     ("pack", args.pack)) if prev.get(k) != v]
    lost = [n for n in names if read.get(n, 0)] if (other or not args.sampler_state) else changed
    if lost and not args.sampler_fresh_ok:
        why = (f"{', '.join(other)}: other windows" if other else
               f"their content changed since then ({', '.join(lost)})" if args.sampler_state else
               f"pass --sampler_state {src} to continue them")
        raise SystemExit(f"[sampler] {src} read {', '.join(f'{n} {read[n]:,}' for n in lost)} windows: this run would "
                         f"start them again from the first window ({why}). --sampler_fresh_ok to accept it")
    if other or not args.sampler_state or not read:      # nothing carried: every folder from its first window
        if is_main and lost:
            print(f"[sampler] --sampler_fresh_ok: {', '.join(lost)} start again from the first window although "
                  f"{src} read them", flush=True)
        elif is_main and args.sampler_state:
            print(f"[sampler] {src} read none of these folders: every folder starts at its first window", flush=True)
        return 0, None, []
    if is_main:
        for n in names:
            g, w = read.get(n, 0), windows[n]
            if n in changed:
                print(f"[sampler] {n}: content changed since {src} (--sampler_fresh_ok): starts again at its first "
                      f"window ({w:,} windows)", flush=True)
            elif g:
                print(f"[sampler] {n}: {g:,} windows already read / {w:,} (pass {g // w + 1}, {g % w / w:.1%} of it): "
                      f"continues at window {g + 1:,} of its shuffled passes"
                      + (" (%s #%d)" % ds.window_id(n, g, args.seed) if args.data_mix else ""), flush=True)
            else:
                print(f"[sampler] {n}: not read before, starts at its first window ({w:,} windows"
                      + (", %s #%d)" % ds.window_id(n, 0, args.seed) if args.data_mix else ")"), flush=True)
        for n in [n for n in read if n not in names]:
            print(f"[sampler] {n}: {read[n]:,} windows read by {src}, not in this run", flush=True)
    return done, prev.get("mix_state"), changed


# config keys that may change at a resume: the rest must match the checkpoint, or the same weights compute another
# function without any error (a shape-compatible flag: --nope, --hidden_act, the RoPE base, ...)
_CFG_MAY_CHANGE = {"dropout", "max_seq_len", "max_position_embeddings", "transformers_version", "_name_or_path",
                   "torch_dtype", "dtype", "architectures", "use_cache"}


def check_resume_identity(args, cfg, resume_dir, resume_state, mix_sources):
    """Refuses a resume whose model, tokenizer, sampler seed or mixture folders are not the checkpoint's. Returns the
    {folder name: id} of the mixture, saved with the data state."""
    problems = []
    try:
        old_cfg = Skylar2Config.from_pretrained(str(resume_dir)).to_dict()    # config.json holds only non-defaults
        new_cfg = cfg.to_dict()
        diff = sorted(k for k in set(old_cfg) | set(new_cfg)
                      if k not in _CFG_MAY_CHANGE and old_cfg.get(k) != new_cfg.get(k))
        if diff:
            problems.append("model config differs from the checkpoint on "
                            + ", ".join(f"{k} ({old_cfg.get(k)!r} -> {new_cfg.get(k)!r})" for k in diff))
    except (OSError, ValueError) as e:
        problems.append(f"config.json of the checkpoint not readable ({type(e).__name__}: {e})")
    old_tok = Path(resume_dir) / "tokenizer.json"
    if args.tokenizer and old_tok.is_file() and \
            hashlib.sha256(old_tok.read_bytes()).digest() != hashlib.sha256(Path(args.tokenizer).read_bytes()).digest():
        problems.append(f"tokenizer {args.tokenizer} is not the checkpoint's")
    old_args = resume_state.get("args") or {}
    if old_args.get("seed") is not None and old_args.get("seed") != args.seed and args.sampler == "permutation":
        problems.append(f"--seed {args.seed}, the checkpoint was trained with {old_args['seed']}: the sampler's "
                        f"permutation would change and windows would be read twice or never")
    ids = {Path(p).name: folder_id(p) for p, _ in mix_sources} if mix_sources else {}
    old_ids = resume_state.get("mix_ids") or {}
    changed = sorted(n for n in ids if n in old_ids and ids[n] != old_ids[n])
    if changed:
        problems.append(f"--data_mix folders {changed} have the name of the checkpoint's but other content: a new "
                        f"folder needs a new name (it then starts from its first window)")
    # the position in each folder is kept by its name: the same corpus moved under another name (from $WORK to $FAST
    # as skylar_v6) would start again from its first window, re-reading what was trained and never reaching the rest
    renamed = sorted(f"{n} (was {o})" for n in ids if n not in old_ids
                     for o, oid in old_ids.items() if oid == ids[n] and o not in ids)
    if renamed:
        problems.append(f"--data_mix folders {renamed} hold the content of a checkpoint folder under another name: the "
                        f"position in a folder goes by its name, so it would start again from its first window. Keep the "
                        f"old name (move the folder with it, or point a symlink with the old name at it)")
    if problems:
        raise SystemExit(f"[resume] {resume_dir} does not match this run:\n  " + "\n  ".join(problems))
    return ids


def parse_wsm_windows(args):
    """[(start, end, every)] in tokens, from --wsm_window START:END:EVERY (repeatable) and the older
    --wsm_every_tok/--wsm_start_tok pair (one window with no end). Empty = no snapshots."""
    windows = []
    if args.wsm_every_tok and args.wsm_every_tok > 0:
        windows.append((int(args.wsm_start_tok), None, int(args.wsm_every_tok)))
    for spec in args.wsm_window or []:
        parts = spec.split(":")
        try:
            start, end, every = (int(float(x)) for x in parts)
        except ValueError:
            raise SystemExit(f"--wsm_window: '{spec}' is not START:END:EVERY in tokens (e.g. 16e9:20e9:0.5e9)")
        if every <= 0 or end < start:
            raise SystemExit(f"--wsm_window: '{spec}' needs EVERY > 0 and END >= START")
        windows.append((start, end, every))
    return windows


def parse_skip_tokens(spec):
    """'41000:0.6e9' -> (41000, 600000000): at the checkpoint of step 41000, skip 0.6B tokens of data."""
    try:
        step, tokens = spec.split(":")
        step, tokens = int(step), int(float(tokens))
    except ValueError:
        raise SystemExit(f"--skip_tokens_at: '{spec}' is not STEP:TOKENS (e.g. 41000:0.6e9)")
    if step < 0 or tokens <= 0:
        raise SystemExit(f"--skip_tokens_at: '{spec}' needs STEP >= 0 and TOKENS > 0")
    return step, tokens


def next_wsm_mark(tokens, windows):
    """The first snapshot mark after `tokens`: in each window the multiples of EVERY from START to END (inclusive).
    Computed from the token count alone, so a resume, or a resume with other windows, continues from the right mark.
    0 = no mark left."""
    marks = []
    for start, end, every in windows:
        m = max(tokens // every + 1, -(-start // every)) * every
        if end is None or m <= end:
            marks.append(m)
    return min(marks) if marks else 0


def parse_deadline(spec):
    """When the job ends, as a unix time. `spec`: a unix time, `+90m` / `+2h` / `+600s` from now, or
    `slurm` = the time left to this Slurm job (squeue %L). None = no deadline."""
    if not spec:
        return None
    spec = str(spec).strip()
    if spec == "slurm":
        jid = os.environ.get("SLURM_JOB_ID")
        if not jid:
            return None
        try:
            left = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%L"], capture_output=True, text=True,
                                  timeout=60).stdout.strip()
            return time.time() + _slurm_seconds(left)
        except Exception as e:
            print(f"[deadline] squeue failed ({e}): no deadline", flush=True)
            return None
    if spec.startswith("+"):
        unit = spec[-1] if spec[-1] in "smh" else "s"
        val = float(spec[1:-1] if spec[-1] in "smh" else spec[1:])
        return time.time() + val * {"s": 1, "m": 60, "h": 3600}[unit]
    return float(spec)


def write_json_atomic(path, obj):
    """A small JSON file that a reader never sees half-written (heartbeat, status)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


_STOP_SIGNAL = {"sig": None}


# Steps at which the loss is made NaN, to test the non-finite guard: SKYLAR_TEST_NONFINITE_AT="120,121,122". Empty in
# any real run.
_TEST_NONFINITE_AT = {int(x) for x in os.environ.get("SKYLAR_TEST_NONFINITE_AT", "").split(",") if x.strip()}


def _on_stop_signal(signum, frame):
    """SIGTERM (Slurm at the time limit, scancel, preemption; torchrun forwards it to the workers) and
    SIGUSR1 (`#SBATCH --signal=USR1@600`) ask for a clean stop: the loop saves `last` at the end of the
    current step and exits. The handler only sets a flag: saving inside a signal handler, in the middle
    of a collective, would hang the other ranks."""
    _STOP_SIGNAL["sig"] = signum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="test")
    ap.add_argument("--data", default=str(ROOT / "data/tokenized"))
    ap.add_argument("--data_mix", default=None,
                    help="several tokenized folders read as one corpus, 'path=weight,path=weight' (weights = shares "
                         "of samples); replaces --data. Every folder carries the same tokenizer.json. The weights "
                         "can change at a resume: every folder continues where it was")
    ap.add_argument("--tokenizer", default=None, help="default: <data>/tokenizer.json")
    ap.add_argument("--max_tokens", type=float, default=None, help="default = epochs * corpus")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_epochs", type=float, default=4.0, help="hard clamp (data-constrained)")
    ap.add_argument("--steps", type=int, default=None, help="override; else derived from tokens")
    ap.add_argument("--batch_size", type=int, default=8, help="PER-GPU micro-batch")
    ap.add_argument("--grad_accum", type=int, default=6)
    ap.add_argument("--seq_len", type=int, default=2048)
    # Il default della classe e' 0.1 e NESSUN preset lo tocca: finora ogni training e' girato con
    # dropout attivo su FFN e residual (il 980M pubblicato ha dropout=0.1 nel suo config.json),
    # semplicemente perche' non era raggiungibile da riga di comando. Nel pretrain moderno si fa
    # circa UNA passata su un corpus enorme, quindi non c'e' overfitting da regolarizzare: PaLM
    # (arXiv 2204.02311) non usa dropout in pretrain, Llama (2302.13971) e Qwen2.5 (2412.15115)
    # nemmeno; GPT-3 (2005.14165) lo usava, ma era il 2020 e con molti meno token per parametro.
    # Qui NON si cambia il default — i checkpoint esistenti restano riproducibili — ma ora si puo'
    # passare `--dropout 0` sul pretrain. Da confermare con una misura nostra prima del 4B.
    ap.add_argument("--dropout", type=float, default=None,
                    help="sovrascrive il dropout del preset (0 = spento, consigliato in pretrain)")
    # ── architettura Skylar 2 (the Skylar 2 report). Ogni default = comportamento v1. ──
    ap.add_argument("--kda_ratio", type=str, default=None,
                    help="ibrido ricorrente/attention, es. '3:1'. Richiede --doc_masking")
    ap.add_argument("--attn_res", action="store_true", help="AttnRes: attenzione sulla profondita'")
    ap.add_argument("--attn_res_mode", default="block", choices=["block", "full", "window"],
                    help="sorgenti di AttnRes: block (Kimi 2603.15031, consigliato) | full | "
                         "window (valutata e scartata: the Skylar 2 report §6.2-6.3)")
    ap.add_argument("--attn_res_block_size", type=int, default=8,
                    help="block: sotto-layer per blocco (8 → 9 blocchi su 36 layer)")
    ap.add_argument("--attn_res_block", type=int, default=6,
                    help="solo --attn_res_mode window: finestra di 2S+1 sorgenti")
    ap.add_argument("--gated_norm", type=int, default=0,
                    help="GatedNorm (2601.22966) su ln1/ln2/ln_f: rango del gate, 0 = spento, 16 consigliato")
    ap.add_argument("--attn_out_gate", nargs="?", const="fullrank", default=None,
                    choices=["fullrank", "perhead"],
                    help="output gate sui layer full-attention: per canale (fullrank) o per testa")
    ap.add_argument("--hidden_act", type=str, default=None, choices=["swiglu", "situ_glu"])
    ap.add_argument("--mtp_layers", type=int, default=None,
                    help="teste di multi-token prediction (0 = spente). Si scartano a fine "
                         "pretrain: costano il 3%% in training, zero in inferenza")
    ap.add_argument("--mtp_loss_weight", type=float, default=None)
    ap.add_argument("--nope", action="store_true",
                    help="NoPE sui layer full-attention. IRREVERSIBILE e non verificabile "
                         "dalla loss: leggi the Skylar 2 report §3.7 prima di usarlo")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"],
                    help="muon: Muon (torch.optim.Muon, update RMS allineato ad AdamW) sulle matrici "
                         "delle mappe lineari nascoste, AdamW su tutto il resto")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1,
                    help="weight decay of the matrices (AdamW and Muon). At a resume the value given here is applied, "
                         "not the one saved with the optimizer")
    ap.add_argument("--wd_embed", type=float, default=None,
                    help="weight decay of the token embedding (with tied weights, also the output head) in its own "
                         "AdamW group: 0 as in OLMo 2 (arXiv 2501.00656 §3.4.2) and SmolLM3. Default: like --wd. Set "
                         "at launch: a resume across a change of the groups is refused")
    ap.add_argument("--z_loss", type=float, default=0.0,
                    help="z-loss on the output softmax: coef * log Z^2 per labelled token (PaLM 2204.02311: 1e-4; "
                         "OLMo 2 2501.00656: 1e-5). 0 = off, the loss path without it")
    ap.add_argument("--z_loss_floor", type=float, default=None,
                    help="with --z_loss: coef * max(0, log Z - floor)^2, no pull below the floor (an assumption, "
                         "without a source: to hold log Z where it is instead of pulling it to 0)")
    ap.add_argument("--lr_schedule", default="cosine", choices=["cosine", "wsd", "constant"])
    ap.add_argument("--lr_decay_ratio", type=float, default=0.1, help="WSD: last-fraction decay")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--val_windows", type=int, default=512,
                    help="validation windows per data folder, fixed for the whole run and split across the GPUs: the "
                         "same set whatever the number of GPUs and the --data_mix weights")
    ap.add_argument("--log_every", type=int, default=0,
                    help="scrive train loss/grad norm/lr ogni N step in <out>/train_steps.jsonl "
                         "(senza validazione; serve agli stress test). 0 = spento")
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--milestones", default="", help="comma token marks -> step_tok<N>B snapshots")
    ap.add_argument("--wsm_every_tok", type=float, default=0.0,
                    help="WSM: every N tokens save a WEIGHTS-ONLY snapshot to <out>/wsm/ (merge candidates). "
                         "0=off. Space them so that the merge window holds enough snapshots to average.")
    ap.add_argument("--wsm_start_tok", type=float, default=0.0,
                    help="WSM: first snapshot at this many tokens (e.g. the last ~30%% of the run). A weights-only "
                         "snapshot of a 1B model is ~4 GB")
    ap.add_argument("--wsm_window", action="append", default=None, metavar="START:END:EVERY",
                    help="WSM: snapshots every EVERY tokens from START to END inclusive; repeatable, combines with "
                         "--wsm_every_tok. E.g. --wsm_window 16e9:20e9:0.5e9 --wsm_window 90e9:100e9:1e9 (two separate merge "
                         "windows)")
    ap.add_argument("--no_checkpoints", action="store_true",
                    help="esperimenti: niente best/last/milestone e niente stato dell'ottimizzatore; "
                         "a fine run salva solo i pesi in <out>/final (un decimo dello spazio)")
    ap.add_argument("--keep_last", type=int, default=1,
                    help="keep the N most recent saves of <out>/last (last, last.prev, last.prev2, ...): a ring to go "
                         "back to after a loss spike or a NaN noticed hours later. 1 = only the current one")
    ap.add_argument("--no_best", action="store_true",
                    help="do not write <out>/best (a full checkpoint every time the validation improves): at a "
                         "constant LR it is ~last, and the final model is a merge of snapshots")
    ap.add_argument("--max_nonfinite", type=int, default=3,
                    help="a step whose gradient norm is not finite is skipped (no update, logged); after this many "
                         "in a row the run stops with status 'nonfinite' and exit code 3, without overwriting last")
    ap.add_argument("--max_nonfinite_per_1k", type=int, default=10,
                    help="the run also stops ('nonfinite', exit 3) when more than this many updates were skipped in "
                         "the last 1,000 steps, in a row or not: a NaN every few steps, from a bad stretch of data or "
                         "a kernel, would otherwise be skipped for ever while the tokens count on. 0 = off")
    ap.add_argument("--param_check_every", type=int, default=1000,
                    help="every N steps compare a parameter checksum across the ranks (logged as param_spread): "
                         "they must stay identical under DDP. 0 = off")
    ap.add_argument("--ckpt_every_min", type=float, default=0.0,
                    help="also save <out>/last every N minutes of wall time (0 = off): on a cluster the "
                         "lost work after a fault is bounded in time, not in steps")
    ap.add_argument("--deadline", default=None,
                    help="when the job ends: unix time, '+90m', or 'slurm' (time left to this job). The run "
                         "saves <out>/last --stop_margin_min before it and exits cleanly (status 'deadline')")
    ap.add_argument("--stop_margin_min", type=float, default=10.0,
                    help="how long before --deadline to stop: longer than one step + one checkpoint save")
    ap.add_argument("--heartbeat_s", type=float, default=30.0,
                    help="rank 0 rewrites <out>/heartbeat.json every N seconds (0 = off): a watchdog that "
                         "sees it go stale kills a hung job instead of letting it burn GPU hours")
    ap.add_argument("--telemetry_s", type=float, default=30.0,
                    help="per-node GPU power/energy sampling into <out>/telemetry/<host>.jsonl (0 = off)")
    ap.add_argument("--ddp_comm", default="fp32", choices=["fp32", "bf16"],
                    help="gradient all-reduce precision: bf16 halves the traffic between nodes")
    ap.add_argument("--ddp_bucket_mb", type=int, default=25, help="DDP gradient bucket size")
    ap.add_argument("--dist_timeout_min", type=float, default=30.0,
                    help="collective timeout: a rank that waits longer for the others fails instead of hanging "
                         "(the first step includes torch.compile, minutes on an A100)")
    ap.add_argument("--dist_backend", default=None, choices=["nccl", "gloo"],
                    help="default: nccl on GPU. gloo lets several ranks share one GPU (tests on one card)")
    ap.add_argument("--bpb_set", default=None,
                    help="directory of frozen *.txt slices: their bits per byte are measured "
                         "every --bpb_every_tok tokens, with the protocol of the stopping rule (eval/bpb.py)")
    ap.add_argument("--bpb_every_tok", type=float, default=1e9)
    ap.add_argument("--bpb_seq_len", type=int, default=2048)
    ap.add_argument("--bpb_stride", type=int, default=1024)
    ap.add_argument("--samples_prompts", default=None,
                    help="JSONL of {name, prompt}: every --samples_every_tok tokens the "
                         "model continues each prompt, greedy, into <out>/samples.jsonl, to read what it is learning. "
                         "The prompts are split across the ranks, so the pause is one generation long")
    ap.add_argument("--samples_every_tok", type=float, default=10e9)
    ap.add_argument("--samples_max_new", type=int, default=128)
    ap.add_argument("--ckpt_per_node", action="store_true",
                    help="multi-node without a shared filesystem: every node's first process writes its own "
                         "<out>/last, so each node can resume. Never with an --out shared between nodes")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--no_prefetch", action="store_true")
    ap.add_argument("--sampler", default="permutation", choices=["permutation", "random"],
                    help="permutation: one shuffled pass over the corpus without repeats, resumed exactly "
                         "from the step (300B tokens of a 502B corpus = 60%% of it, each window once); "
                         "random: random windows with replacement (45%% of the corpus seen, the rest repeated)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--doc_masking", action="store_true",
                    help="document-masked attention (no cross-document attention in a window)")
    ap.add_argument("--bos_id", type=int, default=None, help="bos id for doc masking (else from tokenizer)")
    ap.add_argument("--pack", default="off", choices=["off", "keep", "drop"],
                    help="best-fit packed windows (data/packing.py, arXiv 2404.10830): every document that fits a window "
                         "is read whole instead of being cut by consecutive windows (v6 at 8,192: 13.1%% of documents "
                         "cut, 32.9%% of positions without their document's start). keep/drop: train on the chunks "
                         "after the first of a document longer than a window, or leave them out. Needs --doc_masking and "
                         "the index of every folder (python data/bin.pack_index.py --data <folder> --seq_len <T> "
                         "--tails keep|drop)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", default=None, help="checkpoint dir, or 'auto': <out>/last, then last.prev, last.prev2, ... (the newest that "
                         "loads; a run with a history and no usable checkpoint stops instead of starting over)")
    ap.add_argument("--skip_tokens_at", default=None, metavar="STEP:TOKENS",
                    help="after a loss spike: when resuming from the checkpoint of step STEP (e.g. a milestone before "
                         "the spike), skip TOKENS of data (e.g. 41000:0.6e9). Only from that checkpoint, so it is "
                         "safe to leave in a launch script")
    ap.add_argument("--sampler_state", default=None, metavar="CHECKPOINT",
                    help="a run from --init_from that continues the data of an earlier run (the next stage of a chain): "
                         "that run's checkpoint dir, or its sampler_state.json / training_state.pt. With the same --seed, "
                         "--seq_len and --pack every folder goes on with its shuffled passes where that run left it, so "
                         "no window is read again before its pass ends; a folder with another weight goes on too, a new "
                         "folder starts at its first window. Ignored on a resume (the checkpoint's own state is used)")
    ap.add_argument("--sampler_fresh_ok", action="store_true",
                    help="start folders from their first window although a sampler state says they were read: one in the "
                         "--init_from dir, a --sampler_state with another --seq_len/--pack/--seed, or a folder whose "
                         "content changed. Refused by default: the run would read the same windows again")
    ap.add_argument("--init_from", default=None,
                    help="start from these weights, e.g. a model grown with training/bin.upscale.py (beta): its "
                         "config.json sets the architecture (--preset and the architecture flags are ignored), the "
                         "optimizer, the step and the token count start from zero. Once the run has a resume point, "
                         "--resume wins, so the same command chains")
    ap.add_argument("--out", default=str(ROOT / "checkpoints/pretrain"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="skylar-pretrain")
    ap.add_argument("--wandb_run", default=None)
    ap.add_argument("--s3_bucket", default=None, help="optional: async checkpoint upload bucket")
    ap.add_argument("--s3_prefix", default="")
    ap.add_argument("--s3_region", default=None)
    for a in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff"):
        ap.add_argument(f"--{a}", type=int, default=None)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_stop_signal)
    signal.signal(signal.SIGUSR1, _on_stop_signal)
    pg_kwargs = {"timeout": timedelta(minutes=args.dist_timeout_min)}
    # only in a distributed launch: with a backend set, a single process would ask for a process group nobody created
    if args.dist_backend and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        pg_kwargs["backend"] = args.dist_backend
    ddp_kwargs = DistributedDataParallelKwargs(
        bucket_cap_mb=args.ddp_bucket_mb,
        comm_hook=DDPCommunicationHookType.BF16 if args.ddp_comm == "bf16" else DDPCommunicationHookType.NO)
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              kwargs_handlers=[InitProcessGroupKwargs(**pg_kwargs), ddp_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process
    # Who writes the resume point <out>/last: the main process, or with --ckpt_per_node the first
    # process of every node (on a cluster each node has its own local disk).
    saves_last = accelerator.is_local_main_process if args.ckpt_per_node else is_main
    world = accelerator.num_processes
    rank = accelerator.process_index
    amp_dtype = torch.bfloat16

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed + rank)   # per-rank: each rank draws DIFFERENT windows (true data parallelism)
    sources = None
    pack = None if args.pack == "off" else args.pack
    if pack and not args.doc_masking:
        raise SystemExit("--pack needs --doc_masking: a chunk of a long document would attend to the documents packed "
                         "before it")
    if args.data_mix:
        sources = parse_data_mix(args.data_mix)
        ds = MixtureTokenDataset(sources, seq_len=args.seq_len, seed=args.seed + rank, pack=pack)
        args.data = sources[0][0]                # the tokenizer default and the messages
    else:
        ds = MemmapTokenDataset(args.data, seq_len=args.seq_len, seed=args.seed + rank, pack=pack)
    corpus_tok = ds.total_tokens
    if args.tokenizer is None:                  # the data dir carries the tokenizer it was encoded with
        args.tokenizer = str(Path(args.data) / "tokenizer.json")
    if not Path(args.tokenizer).is_file():
        if is_main:
            print(f"[warn] tokenizer not found ({args.tokenizer}): checkpoints will not carry tokenizer.json, "
                  f"and --doc_masking needs --bos_id", flush=True)
        args.tokenizer = None

    bos_id = args.bos_id
    if args.doc_masking and bos_id is None:
        try:
            bos_id = Tokenizer.from_file(args.tokenizer).token_to_id("<bos>")
        except Exception:
            bos_id = None
    use_doc = args.doc_masking and (bos_id is not None or pack is not None)   # packed: the ids come from the index
    batch_kwargs = {"return_doc_ids": True, "bos_id": bos_id} if use_doc else {}

    # Con i layer ricorrenti il doc-masking smette di essere un'opzione. Per
    # l'attention e' una raffinatezza: senza, un token guarda anche il documento
    # precedente. Per KDA e' un'altra cosa — non c'e' nessuna maschera, c'e' uno
    # STATO, e senza confini quello stato porta il contesto di un programma COBOL
    # dentro il successivo per tutta la sua lunghezza. Non crasha e non compare in
    # nessuna metrica: si scopre solo a modello finito. Quindi qui si ferma prima.
    if _needs_doc_ids(args) and not use_doc:
        raise SystemExit(
            "Questo modello ha layer ricorrenti (kda_ratio impostato), quindi i confini "
            "fra documenti DEVONO arrivare al trainer.\n"
            "  Aggiungi --doc_masking (e --tokenizer, o --bos_id, per trovare <bos>).\n"
            "Senza, lo stato ricorrente attraversa i documenti impacchettati: il training "
            "non fallisce, impara peggio — ed e' invisibile fino alla valutazione finale."
        )

    tok_per_step = args.batch_size * args.grad_accum * args.seq_len * world
    if args.steps:
        total_steps = args.steps
    else:
        # --max_tokens counts the whole run, also what came before a resume. The --max_epochs cap is applied
        # after the resume (below), on the tokens drawn from THIS dataset.
        target = args.max_tokens if args.max_tokens else corpus_tok * args.epochs
        total_steps = max(1, int(target / tok_per_step))
    milestones = sorted(int(float(x)) for x in args.milestones.split(",") if x.strip())

    overrides = {k: v for k, v in (("d_model", args.d_model), ("n_layers", args.n_layers),
                                   ("n_heads", args.n_heads), ("n_kv_heads", args.n_kv_heads),
                                   ("d_ff", args.d_ff), ("dropout", args.dropout),
                                   ("kda_ratio", args.kda_ratio),
                                   ("hidden_act", args.hidden_act),
                                   ("attn_res", args.attn_res or None),
                                   ("attn_res_block", args.attn_res_block),
                                   ("attn_res_mode", args.attn_res_mode if args.attn_res else None),
                                   ("attn_res_block_size", args.attn_res_block_size if args.attn_res else None),
                                   ("gated_norm", args.gated_norm or None),
                                   ("attn_out_gate", args.attn_out_gate),
                                   ("nope_on_attention", args.nope or None),
                                   ("mtp_layers", args.mtp_layers),
                                   ("mtp_loss_weight", args.mtp_loss_weight)) if v is not None}
    if args.init_from:
        # The architecture is the checkpoint's own (a grown model has no preset). Only dropout follows the command.
        cfg = Skylar2Config.from_pretrained(args.init_from)
        if cfg.vocab_size != ds.vocab_size:
            raise SystemExit(f"--init_from {args.init_from}: vocab {cfg.vocab_size}, the data has {ds.vocab_size}")
        if args.dropout is not None:
            cfg.dropout = args.dropout
        ignored = sorted(k for k in overrides               # attn_res_block has a default: only with --attn_res
                         if k != "dropout" and not (k == "attn_res_block" and not args.attn_res))
        if is_main and ignored:
            print(f"[init] the architecture comes from {args.init_from}: ignored {', '.join(ignored)}", flush=True)
    else:
        cfg = get_config(args.preset, vocab_size=ds.vocab_size, **overrides)
    for attr in ("max_seq_len", "max_position_embeddings"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, max(args.seq_len, getattr(cfg, attr, 0) or 0))

    model = Skylar2ForCausalLM(cfg)
    model.z_loss, model.z_loss_floor = args.z_loss, args.z_loss_floor
    if is_main and args.z_loss:
        print(f"[loss] z-loss {args.z_loss:g}" + (f" above log Z {args.z_loss_floor:g}"
                                                  if args.z_loss_floor is not None else ""), flush=True)
    if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())

    if getattr(cfg, "mup_base_d_model", None):
        opt = torch.optim.AdamW(model.mup_param_groups(args.lr, args.wd), betas=(0.9, 0.95))
    else:
        # Weight decay SOLO sulle matrici: norm, bias e A_log/dt_bias di KDA esclusi (training/optim.py).
        # Con --optimizer muon le mappe lineari nascoste passano a Muon, il resto resta su AdamW.
        opt, opt_info = build_optimizer(model, args.lr, args.wd, args.optimizer, embed_weight_decay=args.wd_embed)
        if is_main:
            print(f"[opt] {opt_info}")

    # resume: load weights BEFORE prepare (so DDP broadcasts identical weights)
    start_step, tokens_seen, best_val = 0, 0, float("inf")
    resume_dir, resume_state, newer_ring = None, None, []
    if args.resume == "auto":
        # The newest usable entry of the ring. save_ckpt swaps `last` for the new one in two renames (last -> last.prev,
        # last.tmp -> last): a kill between them leaves only last.prev, complete. An entry that does not load (missing
        # or truncated file, non-finite weights) is skipped for the next older one, loudly.
        for cand in resume_ring(args.out):
            if not cand.exists():
                continue
            try:
                sd, resume_state = load_resume_point(cand)
            except Exception as e:
                print(f"[resume] rank {rank}: {cand} is not usable ({type(e).__name__}: {str(e)[:200]}), trying an "
                      f"older one", flush=True)
                newer_ring.append(cand)
                continue
            resume_dir = cand
            break
        if resume_dir is None:
            # Only a first launch starts from scratch. Until 10/2026 a run with progress and no readable `last` started
            # again from random weights here, without a word.
            history = run_history(args.out)
            if history:
                raise SystemExit(f"--resume auto: no usable checkpoint in {args.out} ({', '.join(map(str, newer_ring)) or 'no last*'}), "
                                 f"but the run has trained ({history}). Not starting from random weights: resume from an "
                                 f"explicit checkpoint (--resume <dir>, e.g. a milestone) or move {args.out} aside to start over")
    elif args.resume:
        try:
            sd, resume_state = load_resume_point(args.resume)       # an explicit checkpoint that is not there is an error
        except Exception as e:
            raise SystemExit(f"--resume {args.resume}: {type(e).__name__}: {e}")
        resume_dir = Path(args.resume)
    if resume_state is not None:
        check_resume_identity(args, cfg, resume_dir, resume_state, sources)
        model.load_state_dict(sd)
        del sd
        start_step = resume_state["step"]; tokens_seen = resume_state["tokens_seen"]
        best_val = resume_state.get("best_val", float("inf"))
    elif args.init_from:
        model.load_state_dict(Skylar2ForCausalLM.from_pretrained(args.init_from).state_dict())
        if is_main:
            print(f"[init] weights from {args.init_from}: step 0, optimizer and token count from zero", flush=True)

    # compile BEFORE accelerate wraps it (compiling the DDP-wrapped model -> per-step recompiles,
    # profiled 3x slower). dynamic=False: the loader feeds a fresh tensor each step; without it
    # dynamo's dynamic-shape inference recompiles on step 2.
    if args.compile:
        # Without activation checkpointing every block is a compiled frame of its own, and with
        # AttnRes dynamo specialises it on the depth state (the source count changes at every
        # layer): one variant per layer. Past recompile_limit (8) dynamo stops compiling the frame
        # and everything it calls, flex_attention included, which then materialises the full T x T
        # scores (990M at seq 8192: out of memory on a 180 GB B200). Room for a variant per layer.
        n_layers = model.config.n_layers
        dc = torch._dynamo.config
        dc.recompile_limit = max(dc.recompile_limit, 2 * n_layers)
        dc.accumulated_recompile_limit = max(dc.accumulated_recompile_limit, 16 * n_layers)
        model = torch.compile(model, dynamic=False)
    if isinstance(opt, OptimizerSet):
        model, *prepared = accelerator.prepare(model, *opt.opts)
        opt = OptimizerSet(prepared)
    else:
        model, opt = accelerator.prepare(model, opt)
    # The plain model for save/val, under DDP and compile. Not accelerator.unwrap_model: with the
    # compiled model inside DDP (the order above) accelerate 1.12 looks for _orig_mod in __dict__
    # and fails with KeyError on every multi-GPU run with --compile (cluster smoke, 30/09/2026).
    raw_model = model
    while isinstance(raw_model, torch.nn.parallel.DistributedDataParallel) or hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model.module if hasattr(raw_model, "module") else raw_model._orig_mod

    if resume_state is not None:
        # load_state_dict restores the saved group settings, weight decay included: the decay asked for on this command
        # line is put back afterwards (the learning rate is set at every step anyway). Until 10/2026 a --wd changed at a
        # resume did nothing.
        wd_asked = [g["weight_decay"] for g in opt.param_groups]
        try:
            opt.load_state_dict(resume_state["opt"])
        except ValueError as e:
            raise SystemExit(f"[resume] the optimizer state of {resume_dir} does not fit this run: {e}")
        for g, wd in zip(opt.param_groups, wd_asked):
            if g["weight_decay"] != wd:
                if is_main:
                    print(f"[resume] weight decay of a group of {len(g['params'])} tensors: {g['weight_decay']} in the "
                          f"checkpoint -> {wd}", flush=True)
                g["weight_decay"] = wd
        try:
            torch.set_rng_state(resume_state["torch_rng"].cpu())
            if resume_state.get("cuda_rng"):
                torch.cuda.set_rng_state_all([r.cpu() for r in resume_state["cuda_rng"]])
        except Exception:
            pass
        if resume_state.get("sampler_rng") is not None:
            try:
                ds.rng.bit_generator.state = resume_state["sampler_rng"]
            except Exception:
                pass
        if accelerator.is_local_main_process:
            print(f"[resume] rank {rank}: from {resume_dir} @ step {start_step} tokens {tokens_seen/1e9:.2f}B",
                  flush=True)
    # The burst resumes the main run's checkpoint on ANOTHER dataset. From that step its sampler starts a
    # fresh pass (from the main run's position it would start mid-pass: about a quarter of the windows read
    # twice and a quarter never), and the epoch cap counts this dataset only. Without the cap moving, a burst
    # resumed with --max_tokens computed its end below the current step and did not train at all.
    # A mixture is the same data across resumes, whatever its weights and folders: MixtureTokenDataset keeps
    # every folder's position itself (below).
    data_fp = (["mixture"] if args.data_mix
               else [int(corpus_tok), len(ds.splits["train"]), len(ds.splits["val"])])
    data_start_step = 0
    if resume_state is not None:
        if resume_state.get("data_fp") not in (None, data_fp):
            data_start_step = start_step
            if is_main:
                print(f"[resume] new dataset {args.data_mix or args.data}: its sampler starts at its first window",
                      flush=True)
        else:
            data_start_step = resume_state.get("data_start_step", 0)

    # Where the sampler restarts, counted in SAMPLES of this dataset, not in steps: the window of global
    # sample k does not depend on the number of GPUs, so a resume on a different number of nodes reads the
    # next windows and none twice. With a different global batch the steps are re-counted in the new size
    # (the schedule, warmup and --max_tokens work in tokens).
    samples_per_step = args.batch_size * args.grad_accum * world
    samples_done = 0
    if resume_state is not None and data_start_step != start_step:
        old_args = resume_state.get("args") or {}
        old_seq = resume_state.get("seq_len", old_args.get("seq_len", args.seq_len))
        old_tps = resume_state.get("tok_per_step")
        if old_tps is None:   # checkpoints written before 10/2026 do not carry it
            per_rank = (old_args.get("batch_size", args.batch_size) * old_args.get("grad_accum", args.grad_accum)
                        * old_seq)
            old_world = max(1, round(tokens_seen / max(1, per_rank * max(1, start_step))))
            old_tps = per_rank * old_world
        old_pack = resume_state.get("pack", "off")
        if old_seq != args.seq_len or old_pack != args.pack:
            # other window length or packing, other windows: a new pass, like a new dataset
            data_start_step, samples_done = start_step, 0
            if is_main:
                print(f"[resume] seq_len {old_seq} -> {args.seq_len}, packing {old_pack} -> {args.pack}: the sampler "
                      f"starts a new pass", flush=True)
        else:
            samples_done = int(resume_state.get("samples_done",
                                                (start_step - data_start_step) * old_tps // args.seq_len))
            if old_tps != tok_per_step:
                new_start = int(round(tokens_seen / tok_per_step))
                data_start_step = new_start - int(round(samples_done * args.seq_len / tok_per_step))
                if is_main:
                    print(f"[resume] global batch {old_tps:,} -> {tok_per_step:,} tokens/step: step {start_step} "
                          f"-> {new_start}, the sampler continues from sample {samples_done:,}", flush=True)
                start_step = new_start
    # After a loss spike: resume from a checkpoint before it and step over the data that followed (PaLM, arXiv
    # 2204.02311 §5.1: ~100 steps back, 200-500 batches skipped). Applied only when the checkpoint was saved at STEP,
    # so the flag left in a launch script does not skip again at the next resume. The skipped tokens are not
    # counted as trained.
    if args.skip_tokens_at and resume_state is not None:
        at_step, n_tok = parse_skip_tokens(args.skip_tokens_at)
        if int(resume_state.get("step", -1)) == at_step:
            n_skip = n_tok // args.seq_len
            samples_done += n_skip
            if is_main:
                print(f"[resume] --skip_tokens_at: checkpoint of step {at_step}, the sampler steps over {n_skip:,} "
                      f"samples ({n_skip * args.seq_len / 1e9:.3f}B tokens) and continues from sample "
                      f"{samples_done:,}", flush=True)
        elif is_main:
            print(f"[resume] --skip_tokens_at {args.skip_tokens_at}: this checkpoint is of step "
                  f"{resume_state.get('step')}, nothing skipped", flush=True)
    carried_mix, carried_reset = None, []
    if resume_state is None and args.sampler == "permutation" and (args.sampler_state or args.init_from):
        samples_done, carried_mix, carried_reset = carry_sampler_state(args, ds, sources, data_fp, is_main)
    elif args.sampler_state and is_main:
        print(f"[sampler] --sampler_state {args.sampler_state}: "
              + ("a resume, the checkpoint's own sampler state is used" if resume_state is not None
                 else "the random sampler has no passes to continue"), flush=True)
    data_state = {"data_fp": data_fp, "data_start_step": data_start_step, "samples_done": samples_done,
                  "tok_per_step": tok_per_step, "seq_len": args.seq_len, "pack": args.pack}
    if args.data_mix:
        data_state["mix_ids"] = {Path(p).name: folder_id(p) for p, _ in sources}
        # weights changed since the checkpoint (or the run --sampler_state continues) -> every folder's base moves to
        # what it has given
        old_mix = resume_state.get("mix_state") if resume_state is not None and samples_done else carried_mix
        data_state["mix_state"] = ds.restate(old_mix, samples_done, reset=carried_reset)
    if not args.steps:
        total_steps = min(total_steps, data_start_step + max(1, int(corpus_tok * args.max_epochs / tok_per_step)))
    implied_epochs = (total_steps - data_start_step) * tok_per_step / max(1, corpus_tok)

    # Every rank has to restart from the same step. A node without the checkpoint would start from
    # zero, and DDP would hang on the first all-reduce the others never make.
    if world > 1:
        steps_all = accelerator.gather(torch.tensor([start_step], device=device))
        if int(steps_all.min()) != int(steps_all.max()):
            raise RuntimeError(f"ranks resume from different steps {steps_all.tolist()}: without a shared "
                               f"--out, use --ckpt_per_node so every node has its own <out>/last")
    if resume_state is not None and saves_last and not args.no_checkpoints:
        moved = set_aside_future(args.out, tokens_seen, newer_ring)
        if moved:
            where = Path(args.out) / "abandoned" / moved[0].relative_to(Path(args.out) / "abandoned").parts[0]
            print(f"[resume] {len(moved)} folders from beyond {tokens_seen/1e9:.3f}B tokens (another trajectory, or "
                  f"unusable) moved to {where}: " + ", ".join(p.name for p in moved), flush=True)

    s3 = None
    if args.s3_bucket and args.s3_region and is_main:
        try:
            s3 = S3Client(args.s3_bucket, args.s3_prefix, args.s3_region)
            print(f"[s3] ckpt upload -> s3://{args.s3_bucket}/{args.s3_prefix}", flush=True)
        except Exception as e:
            print(f"[s3] disabled ({e})", flush=True)

    metrics_fp = None
    steps_fp = None
    use_wandb = False
    if is_main:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        metrics_fp = open(Path(args.out) / "metrics.jsonl", "a")
        steps_fp = open(Path(args.out) / "train_steps.jsonl", "a") if args.log_every else None
        if args.wandb:
            try:
                import wandb
                wandb.init(project=args.wandb_project, name=args.wandb_run,
                           config={**vars(args), "n_params": n_params, "world": world,
                                   "tok_per_step": tok_per_step, "total_steps": total_steps})
                use_wandb = True
            except Exception as e:
                print(f"[wandb] disabled ({e})", flush=True)
        print(f"device={device} preset={args.preset} params={n_params/1e6:.1f}M vocab={ds.vocab_size} "
              f"gpus={world} corpus={corpus_tok/1e9:.2f}B seq={args.seq_len} lr_sched={args.lr_schedule} "
              f"doc_mask={use_doc} tok/step={tok_per_step:,} total_steps={total_steps:,} "
              f"implied_epochs={implied_epochs:.2f} sampler={args.sampler} pack={args.pack}"
              + (f" windows={ds.n_windows('train'):,}" if args.sampler == "permutation" else ""), flush=True)
        for part in getattr(ds, "parts", [ds]):
            if part.pack:
                i = part.pack.info
                print(f"[pack] {part.dir.name}: {i['windows']:,} windows ({i['consecutive_windows']:,} consecutive), "
                      f"padding {i['padding_share']:.3%}, documents longer than a window {i['docs_split_share']:.2%}, "
                      f"positions without their document's start {i['without_doc_start_share']:.2%}, tokens left out "
                      f"{i['dropped_tokens_share']:.2%}", flush=True)
        if args.data_mix:
            k_end = samples_done + max(0, total_steps - start_step) * samples_per_step
            for (name, w), q, (_, p_now), (_, p_end) in zip(zip(ds.names, ds.weights), ds.quota,
                                                           ds.passes(samples_done).items(),
                                                           ds.passes(k_end).items()):
                print(f"[mix] {name}: weight {w:.4f} ({q}/{ds.block} samples), passes {p_now:.3f} now, "
                      f"{p_end:.3f} at the end", flush=True)

    def log_metrics(rec):
        if not is_main:
            return
        metrics_fp.write(json.dumps(rec) + "\n"); metrics_fp.flush()
        if use_wandb:
            import wandb
            flat = {k: v for k, v in rec.items() if isinstance(v, (int, float))}
            for g in rec.get("gpus", []):
                for kk, vv in g.items():
                    if kk != "idx":
                        flat[f"gpu{g['idx']}/{kk}"] = vv
            wandb.log(flat, step=rec["step"])

    vplan = val_plan(ds, args.val_windows, args.batch_size, rank, world)
    vnames = list(getattr(ds, "names", [Path(args.data).name]))
    if not any(p.n_windows("val") for p in getattr(ds, "parts", [ds])):
        raise SystemExit(f"no validation windows: every data folder ({', '.join(vnames)}) says \"validation\": "
                         f"\"none\" in its meta; add the folder that holds the validation shards")

    logit_stats = {}       # level and spread of the output logits at the last evaluation (val_loss)

    @torch.no_grad()
    def val_loss():
        """
        Ritorna (loss_totale, loss_CE, max logit, log Z medio, {cartella: CE}).

        Con MTP acceso la loss totale include il termine ausiliario: confrontarla
        con quella di un run senza MTP e' mele-contro-pere, e la curva smette di
        essere confrontabile con lo storico. La CE della testa principale e' quella
        che va guardata e su cui si seleziona il checkpoint; la totale resta per
        vedere se l'ausiliaria sta convergendo.
        The windows are a fixed set per data folder (val_plan): the curve of a folder moves only with the model. The
        headline is the mean over the folders, each with the same weight: it changes only when a folder is added. A
        folder without validation ("validation": "none" in its meta, the COBOL burst) is not part of it.
        """
        raw_model.eval(); max_logit, log_z, lg_mean, lg_std = 0.0, float("nan"), float("nan"), float("nan")
        sums = torch.zeros(3 * len(vnames), dtype=torch.float64, device=device)   # per folder: loss, CE, windows
        for i, (s_idx, ks) in enumerate(vplan):
            batch = val_batch(ds, s_idx, ks, device, **batch_kwargs)
            doc = batch[2] if len(batch) == 3 else None
            x, y = batch[0], batch[1]
            with torch.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = raw_model(input_ids=x, labels=y, document_ids=doc, loss_only=True)
                if i == 0:
                    # the largest output logit on the first 1,024 positions: output-logit growth is the
                    # instability that QK-Norm does not cover (tied embedding, constant LR)
                    try:
                        lg = raw_model(input_ids=x[:, :1024],
                                       document_ids=doc[:, :1024] if doc is not None else None)["logits"]
                        max_logit = float(lg.float().abs().amax().item())
                        # the mean log-normaliser over the positions with a target: the quantity z-loss would hold
                        # near 0 (PaLM 2204.02311); its drift, not the largest logit, is the output-logit divergence
                        # of Wortsman et al. 2309.14322 §3.1.2 (without z-loss this is the alarm for it)
                        lse = torch.logsumexp(lg.float(), dim=-1)
                        valid = y[:, :1024] != -100
                        if bool(valid.any()):
                            log_z = float(lse[valid].mean().item())
                            # log Z = level + spread: the mean logit over the vocabulary (a shift of every logit,
                            # which no probability sees) and the standard deviation around it (the confidence)
                            lg_mean = float(lg.float().mean(-1)[valid].mean().item())
                            lg_std = float(lg.float().std(-1)[valid].mean().item())
                        del lg, lse
                    except Exception as e:
                        if is_main:
                            print(f"[val] max logit not measured: {type(e).__name__}: {str(e)[:120]}", flush=True)
            parts = out.get("loss_parts") or {}
            sums[3 * s_idx] += out["loss"].item() * len(ks)
            sums[3 * s_idx + 1] += (float(parts["ce"]) if "ce" in parts else out["loss"].item()) * len(ks)
            sums[3 * s_idx + 2] += len(ks)
        raw_model.train()
        g = accelerator.gather(torch.cat([sums, torch.tensor([lg_mean, lg_std, log_z, max_logit], dtype=torch.float64,
                                                             device=device)]).unsqueeze(0))
        def finite_mean(col):
            v = col[torch.isfinite(col)]
            return float(v.mean().item()) if v.numel() else None
        log_z_mean = finite_mean(g[:, -2])
        logit_stats.update(logit_mean=finite_mean(g[:, -4]), logit_std=finite_mean(g[:, -3]))
        tot = g[:, :-4].sum(0).view(-1, 3)
        n = tot[:, 2].clamp(min=1)
        per_loss, per_ce = (tot[:, 0] / n).tolist(), (tot[:, 1] / n).tolist()
        keep = [i for i in range(len(vnames)) if tot[i, 2] > 0]      # folders with validation windows
        return (sum(per_loss[i] for i in keep) / len(keep), sum(per_ce[i] for i in keep) / len(keep),
                g[:, -1].max().item(), log_z_mean, {vnames[i]: per_ce[i] for i in keep})

    # Bits per byte on a folder of frozen text slices, live: written to metrics.jsonl while the run goes, not by a
    # separate job. The protocol of eval/bpb.py, the windows split across the ranks.
    bpb_slices = []
    if args.bpb_set:
        if not args.tokenizer:
            raise SystemExit("--bpb_set needs the tokenizer (--tokenizer or <data>/tokenizer.json)")
        _tok = Tokenizer.from_file(args.tokenizer)
        for f in sorted(Path(args.bpb_set).glob("*.txt")):
            text = f.read_text()
            bpb_slices.append((f.stem, _tok.encode(text).ids, len(text.encode("utf-8"))))
        if not bpb_slices:
            raise SystemExit(f"--bpb_set {args.bpb_set}: no *.txt slice there (the post-cutoff texts are not in the "
                             f"public repository: point it at your own frozen slices, or drop the flag)")
        if is_main:
            print(f"[bpb] {len(bpb_slices)} slices from {args.bpb_set} every {args.bpb_every_tok/1e9:g}B tokens: "
                  + ", ".join(f"{n} ({b/1e3:.0f} KB)" for n, _, b in bpb_slices), flush=True)
    bpb_every = int(args.bpb_every_tok) if bpb_slices and args.bpb_every_tok > 0 else 0
    next_bpb = ((tokens_seen // bpb_every) + 1) * bpb_every if bpb_every else 0

    def measure_bpb():
        raw_model.eval()
        sums = torch.zeros(2 * len(bpb_slices), dtype=torch.float64, device=device)
        for i, (_, ids, _) in enumerate(bpb_slices):
            sums[2 * i], sums[2 * i + 1] = bpb_sums(raw_model, ids, device, args.bpb_seq_len, args.bpb_stride,
                                                    rank, world)
        if world > 1:
            sums = sums.to(ctl_device)
            torch.distributed.all_reduce(sums)
        raw_model.train()
        return {n: float(sums[2 * i]) / math.log(2) / nb for i, (n, _, nb) in enumerate(bpb_slices)}

    # Samples during the run: what the model writes, not only its loss. Greedy, so two points of the run compare.
    sample_prompts = []
    if args.samples_prompts:
        if not args.tokenizer:
            raise SystemExit("--samples_prompts needs the tokenizer (--tokenizer or <data>/tokenizer.json)")
        _stok = Tokenizer.from_file(args.tokenizer)
        _sbos, _seos = _stok.token_to_id("<bos>"), _stok.token_to_id("<eos>")
        for line in open(args.samples_prompts, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                # without add_special_tokens=False the tokenizer's post-processor wraps the prompt in <bos>...<eos>:
                # the model then starts a new document instead of continuing the prompt (every sample until 10/2026)
                ids = ([_sbos] if _sbos is not None else []) + _stok.encode(r["prompt"], add_special_tokens=False).ids
                sample_prompts.append((r["name"], r["prompt"], ids))
    samples_every = int(args.samples_every_tok) if sample_prompts and args.samples_every_tok > 0 else 0
    next_samples = ((tokens_seen // samples_every) + 1) * samples_every if samples_every else 0

    def take_samples():
        raw_model.eval()
        mine = []
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            for i in range(rank, len(sample_prompts), world):
                name, text, ids = sample_prompts[i]
                out = raw_model.generate(torch.tensor([ids], device=device), max_new_tokens=args.samples_max_new,
                                         temperature=0, eos_token_id=_seos)
                mine.append({"name": name, "prompt": text, "completion": _stok.decode(out[0, len(ids):].tolist())})
        raw_model.train()
        return gather_object(mine) if world > 1 else mine

    # The milestones behind the resume point are done, from the token count alone: the same set on every rank, whatever
    # each one sees on its disk (with --ckpt_per_node only node 0 has them, and ranks disagreeing here would call
    # different barriers below). The ones beyond it were set aside above.
    done_ms = {m for m in milestones if m <= tokens_seen}
    if is_main and not args.no_checkpoints:
        missing = [m for m in done_ms if not (Path(args.out) / f"step_tok{int(m/1e9)}B").exists()]
        if missing and resume_state is not None:
            print(f"[resume] milestones behind the resume point but not on disk (not saved again): "
                  + ", ".join(f"{m/1e9:g}B" for m in sorted(missing)), flush=True)
    wsm_windows = parse_wsm_windows(args)
    next_wsm = next_wsm_mark(tokens_seen, wsm_windows)                         # resume-safe: next mark ahead
    last_wsm_tok = None
    step = start_step - 1
    model.train(); t0 = time.time(); seen0 = tokens_seen; train_secs = 0.0
    # permutation: ONE permutation shared by all ranks (same seed), each rank reads its own slots; the
    # position is the number of samples already trained, so a resume continues exactly.
    indexed = args.sampler == "permutation"
    samples_at_start = samples_done
    train_pf = None if args.no_prefetch else Prefetcher(
        ds, "train", args.batch_size, device, seed=args.seed if indexed else args.seed + rank + 100000,
        indexed=indexed, start=0, offset=samples_at_start, rank=rank, world=world, **batch_kwargs)
    if train_pf is not None and resume_state is not None and resume_state.get("train_pf_rng") is not None:
        train_pf.set_rng_state(resume_state["train_pf_rng"])

    # Long runs on a scheduler: a deadline, the stop signals, a heartbeat, the status of the run, telemetry.
    deadline = parse_deadline(args.deadline) if is_main else None
    if is_main and deadline:
        print(f"[deadline] stop at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline))} minus "
              f"{args.stop_margin_min:g} min", flush=True)
    ctl_device = device
    if world > 1 and torch.distributed.is_initialized() and torch.distributed.get_backend() != "nccl":
        ctl_device = torch.device("cpu")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    telemetry = None
    if args.telemetry_s > 0 and accelerator.is_local_main_process and torch.cuda.is_available():
        telemetry = NodeTelemetry(args.out, every_s=args.telemetry_s, tag={"job": job_id}).start()

    def heartbeat(state, step_done, loss=None):
        if not (is_main and args.heartbeat_s > 0):
            return
        tps_now = (tokens_seen - seen0) / train_secs if train_secs > 0 else None
        write_json_atomic(Path(args.out) / "heartbeat.json",
                          {"state": state, "step": step_done, "total_steps": total_steps, "tokens": tokens_seen,
                           "samples": samples_done, "loss": loss, "tok_s": tps_now, "t": round(time.time(), 1),
                           "job": job_id, "world": world})

    def write_status(state, step_done, **extra):
        if is_main:
            write_json_atomic(Path(args.out) / "status.json",
                              {"state": state, "step": step_done, "total_steps": total_steps, "tokens": tokens_seen,
                               "t": round(time.time(), 1), "job": job_id, "world": world, **extra})

    heartbeat("starting", start_step)
    if _TEST_NONFINITE_AT and is_main:
        print(f"[TEST] SKYLAR_TEST_NONFINITE_AT={sorted(_TEST_NONFINITE_AT)}: the loss is made NaN at these steps, a test "
              f"of the non-finite guard. Unset it for a real run (submit.sh exports the shell's environment)", flush=True)
    last_save_t, last_hb = time.time(), 0.0
    steps_done = start_step          # steps whose optimizer update is complete: the resume point
    stop_reason = None
    nonfinite_run, nonfinite_total = 0, 0     # skipped steps (non-finite gradient norm): in a row, and in this job
    nonfinite_recent = []                     # the steps skipped in the last 1,000 (--max_nonfinite_per_1k)
    clipped, clip_window = 0, 0               # steps whose pre-clip gradient norm exceeded 1.0, since the last eval
    param_spread = None
    interrupted = False
    update_open = False

    def commit_ok():
        """Weights and optimizer state finite, the same answer on every rank (one all-reduce): asked before a save."""
        ok = torch.tensor([1.0 if state_finite(raw_model, opt) else 0.0], device=ctl_device)
        if world > 1:
            torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
        return bool(ok.item())

    try:
        for step in range(start_step, total_steps):
            lr = lr_at(step, args.warmup, total_steps, args.lr, args.min_lr_ratio,
                       args.lr_schedule, args.lr_decay_ratio)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_scale", 1.0) if "lr_scale" in g else lr

            _t_step = time.time()
            loss_accum, grad_norm = torch.zeros((), device=device), 0.0
            z_accum = torch.zeros((), device=device)
            skip_update = False
            for micro in range(args.grad_accum):
                if train_pf is not None:
                    batch = train_pf.next()
                elif indexed:
                    batch = ds.get_batch_at("train", args.batch_size,
                                            (step - start_step) * args.grad_accum + micro, device,
                                            rank=rank, world=world, seed=args.seed, offset=samples_at_start,
                                            **batch_kwargs)
                else:
                    batch = ds.get_batch("train", args.batch_size, device, **batch_kwargs)
                doc = batch[2] if len(batch) == 3 else None
                x, y = batch[0], batch[1]
                with accelerator.accumulate(model):
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                        out_ = model(input_ids=x, labels=y, document_ids=doc, loss_only=True)
                        loss = out_["loss"]
                    if args.z_loss:
                        z_accum += out_["loss_parts"]["z"]
                    if step in _TEST_NONFINITE_AT:             # test hook (env SKYLAR_TEST_NONFINITE_AT), off by default
                        loss = loss * float("nan")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = float(accelerator.clip_grad_norm_(model.parameters(), 1.0))
                        # A non-finite gradient would write NaN into every parameter (both optimizers): skip this
                        # update. The norm is computed on the all-reduced gradient, so every rank decides the same.
                        skip_update = not math.isfinite(grad_norm)
                    if not skip_update:
                        # AdamW and then Muon: an exception in between leaves the weights half updated, with steps_done
                        # still at the previous step. The exception handler must not save that as a resume point.
                        update_open = True
                        opt.step()
                        update_open = False
                    opt.zero_grad(set_to_none=True)
                loss_accum += loss.detach()        # stay on GPU — one GPU->CPU sync per step
            # The optimizer update of this step is done: count it before any collective that can still fail,
            # so a crash from here on saves `last` at the right step (and a resume does not apply it twice).
            tokens_seen += tok_per_step
            samples_done += samples_per_step
            data_state["samples_done"] = samples_done
            steps_done = step + 1
            if skip_update:
                nonfinite_run += 1; nonfinite_total += 1
                nonfinite_recent = [s_ for s_ in nonfinite_recent if s_ > step - 1000] + [step]
                if is_main:
                    print(f"  [nonfinite] step {step}: gradient norm {grad_norm}, update skipped "
                          f"({nonfinite_run} in a row, {nonfinite_total} in this job)", flush=True)
            else:
                nonfinite_run = 0
                clip_window += 1
                clipped += int(grad_norm > 1.0)
            if world > 1 and args.param_check_every and step % args.param_check_every == 0:
                # Ranks update their own copy of the weights from the same gradient: they must stay identical.
                with torch.no_grad():
                    cs = sum(float(p.detach().double().abs().sum()) for p in list(raw_model.parameters())[::16])
                mm = torch.tensor([cs, -cs], dtype=torch.float64, device=ctl_device)
                torch.distributed.all_reduce(mm, op=torch.distributed.ReduceOp.MAX)
                param_spread = (mm[0].item() + mm[1].item()) / max(abs(mm[0].item()), 1e-30)
                if is_main and param_spread > 1e-6:
                    print(f"  [WARN] step {step}: parameters differ across ranks (relative spread {param_spread:.2e})",
                          flush=True)
            # Control, decided the same way on every rank, in ONE all-reduce with the loss: stop (a signal on
            # any rank, or rank 0 past the deadline) and the time-based checkpoint (rank 0's clock). The logged
            # train loss is the mean over ALL ranks: rank 0 alone is 1/world of the batch.
            now = time.time()
            sig_local = 1.0 if _STOP_SIGNAL["sig"] is not None else 0.0
            dl_local = 1.0 if (deadline and now >= deadline - args.stop_margin_min * 60) else 0.0
            save_local = float(is_main and args.ckpt_every_min > 0 and now - last_save_t >= args.ckpt_every_min * 60)
            if world > 1:
                ctl = torch.tensor([0.0, sig_local, dl_local, save_local], device=ctl_device)
                ctl[0] = (loss_accum / args.grad_accum).float().to(ctl_device)
                torch.distributed.all_reduce(ctl, op=torch.distributed.ReduceOp.SUM)
                ctl = ctl.tolist()
                train_loss, sig_any, dl_any, save_due = ctl[0] / world, ctl[1] > 0, ctl[2] > 0, ctl[3] > 0
            else:
                train_loss = (loss_accum / args.grad_accum).item()
                sig_any, dl_any, save_due = sig_local > 0, dl_local > 0, save_local > 0
            stop_code = 2 if dl_any else (1 if sig_any else 0)
            if nonfinite_run >= args.max_nonfinite or (
                    args.max_nonfinite_per_1k and len(nonfinite_recent) > args.max_nonfinite_per_1k
                    and nonfinite_recent[0] > step - 1000):
                stop_reason = "nonfinite"
                break
            if args.log_every and is_main and step % args.log_every == 0:
                steps_fp.write(json.dumps({"step": step, "tokens": tokens_seen, "samples": samples_done, "lr": lr,
                                           "train_loss": train_loss, "grad_norm": grad_norm,
                                           **({"z_term": (z_accum / args.grad_accum).item()} if args.z_loss else {}),
                                           "t": round(time.time(), 3)}) + "\n")
                steps_fp.flush()
            if step == start_step:
                seen0, train_secs = tokens_seen, 0.0   # drop compile/warmup step from throughput
            else:
                train_secs += time.time() - _t_step

            if step % args.eval_every == 0 or step == total_steps - 1:
                vl, vl_ce, vl_max_logit, vl_log_z, vl_sources = val_loss()
                el = time.time() - t0
                tps = (tokens_seen - seen0) / max(1e-9, train_secs)
                eta_h = (total_steps - step) * tok_per_step / max(1, tps) / 3600
                if is_main:
                    wnorm = global_weight_norm(raw_model)
                    upd_ratio = (lr * grad_norm / wnorm) if wnorm > 0 else 0.0
                    rec = {"step": step, "tokens": tokens_seen, "lr": lr, "train_loss": train_loss,
                           "val_loss": vl, "val_ce": vl_ce, "ppl": math.exp(min(vl_ce, 20)),
                           "tok_s": tps, "eta_h": eta_h,
                           "grad_norm": grad_norm, "weight_norm": wnorm, "update_ratio": upd_ratio,
                           "clip_frac": clipped / max(1, clip_window), "max_logit": vl_max_logit, "log_z": vl_log_z,
                           **logit_stats,
                           "nonfinite_skipped": nonfinite_total, "param_spread": param_spread,
                           **({"val_ce_sources": vl_sources} if len(vl_sources) > 1 else {}),
                           "matrix_norms": matrix_norms(raw_model), "out_proj_norms": out_proj_norms(raw_model),
                           "gpus": gpu_telemetry(), "wall_s": el}
                    clipped, clip_window = 0, 0
                    log_metrics(rec)
                    watts = sum(g["power_w"] for g in rec["gpus"]) if rec["gpus"] else 0.0
                    print(f"  step {step:6d}/{total_steps}  tok {tokens_seen/1e9:.2f}B  lr {lr:.2e}  "
                          f"train {train_loss:.3f}  val {vl_ce:.3f}  ppl {math.exp(min(vl_ce,20)):.1f}  "
                          f"gnorm {grad_norm:.2f}  {tps/1e3:.1f}k tok/s  {watts:.0f}W  ETA {eta_h:.1f}h", flush=True)
                    # Il "best" si seleziona sulla CE della testa principale, non
                    # sulla loss totale: con MTP la totale include l'ausiliaria, e un
                    # checkpoint scelto su quella ottimizza in parte un obiettivo che
                    # all'inferenza viene buttato via insieme alla testa.
                    if vl_ce < best_val:
                        best_val = vl_ce
                        if not args.no_checkpoints and not args.no_best and state_finite(raw_model, opt):
                            save_ckpt(Path(args.out) / "best", raw_model, opt, steps_done, tokens_seen, args, best_val, ds, train_pf, s3, data_state=data_state)

            for m in ([] if args.no_checkpoints else milestones):
                if m not in done_ms and tokens_seen >= m:
                    done_ms.add(m)
                    accelerator.wait_for_everyone()
                    if not commit_ok():
                        stop_reason = "nonfinite"
                        break
                    if is_main:
                        # steps_done, like `last`: a resume from a milestone (after a loss spike) starts at the next step
                        save_ckpt(Path(args.out) / f"step_tok{int(m/1e9)}B", raw_model, opt, steps_done, tokens_seen, args, best_val, ds, train_pf, s3, data_state=data_state)
                        print(f"  [milestone] snapshot @ {m/1e9:.0f}B tokens (step {step})", flush=True)

            if stop_reason:
                break

            # WSM: weights-only snapshots (merge candidates) — every rank steps next_wsm/barrier symmetrically
            if next_wsm and tokens_seen >= next_wsm:
                next_wsm = next_wsm_mark(tokens_seen, wsm_windows)
                accelerator.wait_for_everyone()
                if not commit_ok():
                    stop_reason = "nonfinite"
                    break
                if is_main:
                    tb = tokens_seen / 1e9
                    save_weights_only(Path(args.out) / "wsm" / f"snap_step{step:08d}_tok{tb:.2f}B",
                                      raw_model, meta={"step": step, "tokens": tokens_seen},
                                      tokenizer_path=args.tokenizer, s3=s3)
                    print(f"  [wsm] weights-only snapshot @ {tb:.2f}B tokens (step {step})", flush=True)
                last_wsm_tok = tokens_seen

            if bpb_every and tokens_seen >= next_bpb:
                while next_bpb <= tokens_seen:
                    next_bpb += bpb_every
                _tb = time.time()
                bpb = measure_bpb()
                if is_main:
                    with open(Path(args.out) / "bpb.jsonl", "a") as _f:       # its own file: metrics.jsonl readers stay
                        _f.write(json.dumps({"step": step, "tokens": tokens_seen, "t": round(time.time(), 1),
                                             **{f"bpb_{k}": v for k, v in bpb.items()}}) + "\n")
                    if use_wandb:
                        import wandb
                        wandb.log({f"bpb/{k}": v for k, v in bpb.items()}, step=step)
                    print(f"  [bpb] {tokens_seen/1e9:.2f}B tokens: "
                          + "  ".join(f"{k} {v:.4f}" for k, v in bpb.items()) + f"  ({time.time() - _tb:.0f}s)",
                          flush=True)

            if samples_every and tokens_seen >= next_samples:
                while next_samples <= tokens_seen:
                    next_samples += samples_every
                _ts = time.time()
                recs = take_samples()
                if is_main:
                    with open(Path(args.out) / "samples.jsonl", "a", encoding="utf-8") as _f:
                        for r in recs:
                            _f.write(json.dumps({"step": step, "tokens": tokens_seen, "t": round(time.time(), 1), **r},
                                                ensure_ascii=False) + "\n")
                    print(f"  [samples] {len(recs)} prompts at {tokens_seen/1e9:.2f}B tokens ({time.time() - _ts:.0f}s)",
                          flush=True)

            # `last` holds steps_done = step + 1: the resume starts at the next step. (Until 10/2026 the periodic
            # save wrote `step`, and a resume trained that step a second time.)
            periodic = (step - start_step) > 0 and step % args.ckpt_every == 0
            if not args.no_checkpoints and (periodic or save_due or stop_code):
                accelerator.wait_for_everyone()
                if not commit_ok():
                    stop_reason = "nonfinite"
                    break
                if saves_last:
                    _ts = time.time()
                    save_ckpt(Path(args.out) / "last", raw_model, opt, steps_done, tokens_seen, args, best_val, ds,
                              train_pf, s3 if is_main else None, data_state=data_state, keep=args.keep_last)
                    if is_main:
                        why = "stop" if stop_code else ("time" if save_due else "steps")
                        print(f"  [ckpt] last @ step {steps_done} ({why}, {time.time() - _ts:.1f}s)", flush=True)
                last_save_t = time.time()
            if stop_code or now - last_hb >= args.heartbeat_s:
                heartbeat("train", steps_done, train_loss)
                last_hb = now
            if stop_code:
                stop_reason = "deadline" if stop_code == 2 else "signal"
                break
    except BaseException as e:        # OOM / OS-kill / Ctrl-C -> save before dying
        if saves_last and not args.no_checkpoints and update_open:
            print(f"\n[interrupt] {type(e).__name__} inside the optimizer update of step {steps_done}: the weights are "
                  f"half updated, last NOT overwritten (resume from the previous save)", flush=True)
        elif saves_last and not args.no_checkpoints:
            print(f"\n[interrupt] {type(e).__name__}: saving last checkpoint ...", flush=True)
            try:
                if not state_finite(raw_model, opt):     # local: the other ranks may be gone, no collective here
                    raise RuntimeError("non-finite weights or optimizer state: last not overwritten")
                save_ckpt(Path(args.out) / "last", raw_model, opt, steps_done, tokens_seen, args, best_val, ds, train_pf, s3,
                          data_state=data_state, keep=args.keep_last)
            except Exception as e2:
                print(f"  (last-save failed: {e2})", flush=True)
        if not isinstance(e, KeyboardInterrupt):
            write_status("crashed", steps_done, error=f"{type(e).__name__}: {str(e)[:300]}")
            if telemetry is not None:
                telemetry.stop()
            raise
        interrupted = True
    finally:
        if train_pf is not None:
            train_pf.close()       # stop prefetch thread on EVERY exit path -> no NCCL-shutdown hang

    if stop_reason is None and not interrupted and not args.no_checkpoints:
        accelerator.wait_for_everyone()
        if not commit_ok():                    # the last updates since the last check: no `final` from NaN weights
            stop_reason = "nonfinite"
    if stop_reason == "nonfinite":
        # Non-finite gradients several steps in a row or too often, or non-finite weights: `last` is NOT overwritten
        # (it is the previous good save, and --keep_last keeps older ones). Exit 3; the chain's ledger reads status
        # 'nonfinite' and stops at once.
        if is_main:
            print(f"\nNONFINITE at step {steps_done}: {nonfinite_run} non-finite gradient norms in a row, "
                  f"{len(nonfinite_recent)} in the last 1,000 steps ({nonfinite_total} in this job), or non-finite weights "
                  f"or optimizer state; last not overwritten. Resume from "
                  f"{args.out}/last (or an older last.prevN) with --skip_tokens_at to step over the data", flush=True)
        write_status("nonfinite", steps_done, skipped=nonfinite_total)
        heartbeat("nonfinite", steps_done)
        if telemetry is not None:
            telemetry.stop()
        # every rank waits for rank 0's status: a rank that exits first makes torchrun and srun tear the job down,
        # and status.json would keep the previous state
        accelerator.wait_for_everyone()
        accelerator.end_training()
        sys.exit(3)

    def terminal_snapshot(why):
        """The last weights of a job that ends inside a WSM window are a snapshot too, marked `terminal`. At the end of
        the run the step count is the token target floored to whole steps, so the mark at the end of the window is
        never crossed; and a run stopped by its deadline stops between two marks. Without it the newest weights
        of the merge would be missing. A terminal snapshot followed by newer ones (the chain went on) is a job boundary,
        and bin.merge_wsm.py leaves it out."""
        if (is_main and last_wsm_tok != tokens_seen
                and any(start <= tokens_seen and (end is None or tokens_seen <= end) for start, end, _ in wsm_windows)):
            tb = tokens_seen / 1e9
            save_weights_only(Path(args.out) / "wsm" / f"snap_step{step:08d}_tok{tb:.2f}B", raw_model,
                              meta={"step": step, "tokens": tokens_seen, "terminal": True},
                              tokenizer_path=args.tokenizer, s3=s3)
            print(f"  [wsm] weights-only snapshot @ {tb:.3f}B tokens (step {step}, {why})", flush=True)

    if stop_reason:
        # A planned stop (the job's deadline, or SIGTERM/SIGUSR1 from the scheduler): `last` is saved, the run
        # is NOT finished. Exit 0, so a launcher does not count it as a failure; status.json says why.
        terminal_snapshot(f"{stop_reason} stop")
        accelerator.wait_for_everyone()
        if is_main:
            if metrics_fp:
                metrics_fp.close()
            print(f"\nSTOPPED ({stop_reason}) at step {steps_done}/{total_steps}, tokens {tokens_seen/1e9:.3f}B: "
                  f"resume with --resume auto (from {args.out}/last)", flush=True)
        write_status(stop_reason, steps_done)
        heartbeat(stop_reason, steps_done)
        if telemetry is not None:
            telemetry.stop()
        accelerator.end_training()
        sys.exit(0)

    if interrupted:
        write_status("interrupted", steps_done)
        if telemetry is not None:
            telemetry.stop()
        # A Ctrl-C is not the end of the run: no `final`, no DONE (a supervisor would take the run as finished).
        # `last`, saved above, is where --resume auto restarts.
        if is_main:
            if metrics_fp:
                metrics_fp.close()
            print(f"\nINTERRUPTED at step {steps_done}/{total_steps}: resume with --resume auto "
                  f"(from {args.out}/last)", flush=True)
        accelerator.end_training()
        sys.exit(130)

    accelerator.wait_for_everyone()
    if saves_last and not is_main and not args.no_checkpoints:
        save_ckpt(Path(args.out) / "last", raw_model, opt, steps_done, tokens_seen, args, best_val, ds, train_pf,
                  data_state=data_state, keep=args.keep_last)
    terminal_snapshot("end of the run")
    if is_main:
        if args.no_checkpoints:
            raw_model.save_pretrained(str(Path(args.out) / "final"))
        else:
            save_ckpt(Path(args.out) / "last", raw_model, opt, steps_done, tokens_seen, args, best_val, ds, train_pf, s3,
                      data_state=data_state, keep=args.keep_last)
            save_ckpt(Path(args.out) / "final", raw_model, opt, steps_done, tokens_seen, args, best_val, ds, train_pf, s3, data_state=data_state)
        if args.tokenizer:                      # save_ckpt already copies it; --no_checkpoints does not
            shutil.copyfile(args.tokenizer, Path(args.out) / "final" / "tokenizer.json")
        if metrics_fp:
            metrics_fp.close()
        print(f"\nDONE step={step} tokens={tokens_seen/1e9:.2f}B best_val={best_val:.3f} "
              f"-> {args.out}/final ({(time.time()-t0)/3600:.1f}h)", flush=True)
    write_status("done", steps_done, best_val=best_val)
    heartbeat("done", steps_done)
    if telemetry is not None:
        telemetry.stop()

    accelerator.end_training()   # clean NCCL shutdown


if __name__ == "__main__":
    main()
