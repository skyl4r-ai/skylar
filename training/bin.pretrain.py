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
import argparse, json, math, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# make the repo root importable when run as a script without `pip install -e .`
# (relative to THIS file — generic, no machine-specific path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tokenizers import Tokenizer

from models.config import get_config
from models.decoder import Skylar2ForCausalLM
from data.memmap_dataset import MemmapTokenDataset, Prefetcher
from training.optim import OptimizerSet, build_optimizer

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
    e' algebricamente un LR-decay sui gradienti di quella finestra (Teorema 3.1 del paper). Sul burst
    COBOL a LR costante i token di nicchia entrano con PESO PIENO (nel cosine il LR e' gia' schiacciato
    a fine run e il COBOL 'non si imprime'). Vedi projects/skylar-cobol/SCHEDULER_LR_4B.md."""
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


def save_ckpt(path, raw_model, opt, step, tokens_seen, args, best_val, ds=None, train_pf=None, s3=None,
              data_state=None):
    """Atomic save (tmp -> os.replace, .prev backup) of model + opt + RNG + sampler/prefetch RNG, with the
    tokenizer next to the weights (every checkpoint can be fine-tuned or evaluated as it is).
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
    bak = path.with_name(path.name + ".prev")
    if path.exists():
        shutil.rmtree(bak, ignore_errors=True)
        os.replace(path, bak)
    os.replace(tmp, path)
    shutil.rmtree(bak, ignore_errors=True)
    if s3 is not None:
        async_upload_checkpoint(s3, str(path), path.name)


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


def _needs_doc_ids(args):
    """Il modello che sta per essere costruito ha layer ricorrenti?"""
    return bool(getattr(args, "kda_ratio", None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="test")
    ap.add_argument("--data", default=str(ROOT / "data/tokenized"))
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
    # ── architettura Skylar 2 (docs/PAPER_V2.md). Ogni default = comportamento v1. ──
    ap.add_argument("--kda_ratio", type=str, default=None,
                    help="ibrido ricorrente/attention, es. '3:1'. Richiede --doc_masking")
    ap.add_argument("--attn_res", action="store_true", help="AttnRes: attenzione sulla profondita'")
    ap.add_argument("--attn_res_mode", default="block", choices=["block", "full", "window"],
                    help="sorgenti di AttnRes: block (Kimi 2603.15031, consigliato) | full | "
                         "window (valutata e scartata: docs/PAPER_V2.md §6.2-6.3)")
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
                         "dalla loss: leggi docs/PAPER_V2.md §3.7 prima di usarlo")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"],
                    help="muon: Muon (torch.optim.Muon, update RMS allineato ad AdamW) sulle matrici "
                         "delle mappe lineari nascoste, AdamW su tutto il resto")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--lr_schedule", default="cosine", choices=["cosine", "wsd", "constant"])
    ap.add_argument("--lr_decay_ratio", type=float, default=0.1, help="WSD: last-fraction decay")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=0,
                    help="scrive train loss/grad norm/lr ogni N step in <out>/train_steps.jsonl "
                         "(senza validazione; serve agli stress test). 0 = spento")
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--milestones", default="", help="comma token marks -> step_tok<N>B snapshots")
    ap.add_argument("--wsm_every_tok", type=float, default=0.0,
                    help="WSM: every N tokens save a WEIGHTS-ONLY snapshot to <out>/wsm/ (merge candidates). "
                         "0=off. Use ~2e9 during the constant-LR burst -> >=8-10 ckpt in the merge-window.")
    ap.add_argument("--no_checkpoints", action="store_true",
                    help="esperimenti: niente best/last/milestone e niente stato dell'ottimizzatore; "
                         "a fine run salva solo i pesi in <out>/final (un decimo dello spazio)")
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
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", default=None, help="checkpoint dir or 'auto' = <out>/last")
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

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
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
    ds = MemmapTokenDataset(args.data, seq_len=args.seq_len, seed=args.seed + rank)
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
    use_doc = args.doc_masking and bos_id is not None
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
    cfg = get_config(args.preset, vocab_size=ds.vocab_size, **overrides)
    for attr in ("max_seq_len", "max_position_embeddings"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, max(args.seq_len, getattr(cfg, attr, 0) or 0))

    model = Skylar2ForCausalLM(cfg)
    if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())

    if getattr(cfg, "mup_base_d_model", None):
        opt = torch.optim.AdamW(model.mup_param_groups(args.lr, args.wd), betas=(0.9, 0.95))
    else:
        # Weight decay SOLO sulle matrici: norm, bias e A_log/dt_bias di KDA esclusi (training/optim.py).
        # Con --optimizer muon le mappe lineari nascoste passano a Muon, il resto resta su AdamW.
        opt, opt_info = build_optimizer(model, args.lr, args.wd, args.optimizer)
        if is_main:
            print(f"[opt] {opt_info}")

    # resume: load weights BEFORE prepare (so DDP broadcasts identical weights)
    start_step, tokens_seen, best_val = 0, 0, float("inf")
    resume_dir = (Path(args.out) / "last") if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    resume_state = None
    if resume_dir and (resume_dir / "training_state.pt").exists():
        model.load_state_dict(Skylar2ForCausalLM.from_pretrained(str(resume_dir)).state_dict())
        resume_state = torch.load(resume_dir / "training_state.pt", map_location="cpu", weights_only=False)
        start_step = resume_state["step"]; tokens_seen = resume_state["tokens_seen"]
        best_val = resume_state.get("best_val", float("inf"))

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
        opt.load_state_dict(resume_state["opt"])
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
    data_fp = [int(corpus_tok), len(ds.splits["train"]), len(ds.splits["val"])]
    data_start_step = 0
    if resume_state is not None:
        if resume_state.get("data_fp") not in (None, data_fp):
            data_start_step = start_step
            if is_main:
                print(f"[resume] new dataset {args.data}: its sampler starts at its first window", flush=True)
        else:
            data_start_step = resume_state.get("data_start_step", 0)
    data_state = {"data_fp": data_fp, "data_start_step": data_start_step}
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
              f"implied_epochs={implied_epochs:.2f} sampler={args.sampler}"
              + (f" windows={ds.n_windows('train'):,}" if args.sampler == "permutation" else ""), flush=True)

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

    @torch.no_grad()
    def val_loss(n=40):
        """
        Ritorna (loss_totale, loss_CE).

        Con MTP acceso la loss totale include il termine ausiliario: confrontarla
        con quella di un run senza MTP e' mele-contro-pere, e la curva smette di
        essere confrontabile con lo storico. La CE della testa principale e' quella
        che va guardata e su cui si seleziona il checkpoint; la totale resta per
        vedere se l'ausiliaria sta convergendo.
        """
        raw_model.eval(); tot = 0.0; tot_ce = 0.0
        for _ in range(n):
            batch = ds.get_batch("val", args.batch_size, device, **batch_kwargs)
            doc = batch[2] if len(batch) == 3 else None
            x, y = batch[0], batch[1]
            with torch.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = raw_model(input_ids=x, labels=y, document_ids=doc, loss_only=True)
            tot += out["loss"].item()
            parts = out.get("loss_parts") or {}
            tot_ce += float(parts["ce"]) if "ce" in parts else out["loss"].item()
        raw_model.train()
        t = torch.tensor([tot / max(1, n), tot_ce / max(1, n)], device=device)
        g = accelerator.gather(t.unsqueeze(0)).mean(0)
        return g[0].item(), g[1].item()

    done_ms = {m for m in milestones if (Path(args.out) / f"step_tok{int(m/1e9)}B").exists()}
    wsm_step = int(args.wsm_every_tok) if args.wsm_every_tok and args.wsm_every_tok > 0 else 0
    next_wsm = ((tokens_seen // wsm_step) + 1) * wsm_step if wsm_step else 0   # resume-safe: next multiple ahead
    step = start_step - 1
    model.train(); t0 = time.time(); seen0 = tokens_seen; train_secs = 0.0
    # permutation: ONE permutation shared by all ranks (same seed), each rank reads its own slots; the
    # position is the number of micro-batches already trained, so a resume continues exactly.
    indexed = args.sampler == "permutation"
    train_pf = None if args.no_prefetch else Prefetcher(
        ds, "train", args.batch_size, device, seed=args.seed if indexed else args.seed + rank + 100000,
        indexed=indexed, start=(start_step - data_start_step) * args.grad_accum, rank=rank, world=world,
        **batch_kwargs)
    if train_pf is not None and resume_state is not None and resume_state.get("train_pf_rng") is not None:
        train_pf.set_rng_state(resume_state["train_pf_rng"])
    interrupted = False
    try:
        for step in range(start_step, total_steps):
            lr = lr_at(step, args.warmup, total_steps, args.lr, args.min_lr_ratio,
                       args.lr_schedule, args.lr_decay_ratio)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_scale", 1.0) if "lr_scale" in g else lr

            _t_step = time.time()
            loss_accum, grad_norm = torch.zeros((), device=device), 0.0
            for micro in range(args.grad_accum):
                if train_pf is not None:
                    batch = train_pf.next()
                elif indexed:
                    batch = ds.get_batch_at("train", args.batch_size,
                                            (step - data_start_step) * args.grad_accum + micro, device,
                                            rank=rank, world=world, seed=args.seed, **batch_kwargs)
                else:
                    batch = ds.get_batch("train", args.batch_size, device, **batch_kwargs)
                doc = batch[2] if len(batch) == 3 else None
                x, y = batch[0], batch[1]
                with accelerator.accumulate(model):
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                        loss = model(input_ids=x, labels=y, document_ids=doc, loss_only=True)["loss"]
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = float(accelerator.clip_grad_norm_(model.parameters(), 1.0))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                loss_accum += loss.detach()        # stay on GPU — one GPU->CPU sync per step
            train_loss = (loss_accum / args.grad_accum).item()
            tokens_seen += tok_per_step
            if args.log_every and is_main and step % args.log_every == 0:
                steps_fp.write(json.dumps({"step": step, "tokens": tokens_seen, "lr": lr,
                                           "train_loss": train_loss, "grad_norm": grad_norm,
                                           "t": round(time.time(), 3)}) + "\n")
                steps_fp.flush()
            if step == start_step:
                seen0, train_secs = tokens_seen, 0.0   # drop compile/warmup step from throughput
            else:
                train_secs += time.time() - _t_step

            if step % args.eval_every == 0 or step == total_steps - 1:
                vl, vl_ce = val_loss()
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
                           "matrix_norms": matrix_norms(raw_model), "gpus": gpu_telemetry(), "wall_s": el}
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
                        if not args.no_checkpoints:
                            save_ckpt(Path(args.out) / "best", raw_model, opt, step, tokens_seen, args, best_val, ds, train_pf, s3, data_state=data_state)

            for m in ([] if args.no_checkpoints else milestones):
                if m not in done_ms and tokens_seen >= m:
                    done_ms.add(m)
                    accelerator.wait_for_everyone()
                    if is_main:
                        save_ckpt(Path(args.out) / f"step_tok{int(m/1e9)}B", raw_model, opt, step, tokens_seen, args, best_val, ds, train_pf, s3, data_state=data_state)
                        print(f"  [milestone] snapshot @ {m/1e9:.0f}B tokens (step {step})", flush=True)

            # WSM: weights-only snapshots (merge candidates) — every rank steps next_wsm/barrier symmetrically
            if wsm_step and tokens_seen >= next_wsm:
                while next_wsm <= tokens_seen:
                    next_wsm += wsm_step
                accelerator.wait_for_everyone()
                if is_main:
                    tb = tokens_seen / 1e9
                    save_weights_only(Path(args.out) / "wsm" / f"snap_step{step:08d}_tok{tb:.2f}B",
                                      raw_model, meta={"step": step, "tokens": tokens_seen},
                                      tokenizer_path=args.tokenizer, s3=s3)
                    print(f"  [wsm] weights-only snapshot @ {tb:.2f}B tokens (step {step})", flush=True)

            if not args.no_checkpoints and (step - start_step) > 0 and step % args.ckpt_every == 0:
                accelerator.wait_for_everyone()
                if saves_last:
                    save_ckpt(Path(args.out) / "last", raw_model, opt, step, tokens_seen, args, best_val, ds, train_pf,
                              s3 if is_main else None, data_state=data_state)
    except BaseException as e:        # OOM / OS-kill / Ctrl-C -> save before dying
        if saves_last and not args.no_checkpoints:
            print(f"\n[interrupt] {type(e).__name__}: saving last checkpoint ...", flush=True)
            try:
                save_ckpt(Path(args.out) / "last", raw_model, opt, step + 1, tokens_seen, args, best_val, ds, train_pf, s3, data_state=data_state)
            except Exception as e2:
                print(f"  (last-save failed: {e2})", flush=True)
        if not isinstance(e, KeyboardInterrupt):
            raise
        interrupted = True
    finally:
        if train_pf is not None:
            train_pf.close()       # stop prefetch thread on EVERY exit path -> no NCCL-shutdown hang

    if interrupted:
        # A Ctrl-C is not the end of the run: no `final`, no DONE (a supervisor would take the run as finished).
        # `last`, saved above, is where --resume auto restarts.
        if is_main:
            if metrics_fp:
                metrics_fp.close()
            print(f"\nINTERRUPTED at step {step + 1}/{total_steps}: resume with --resume auto "
                  f"(from {args.out}/last)", flush=True)
        accelerator.end_training()
        sys.exit(130)

    accelerator.wait_for_everyone()
    if saves_last and not is_main and not args.no_checkpoints:
        save_ckpt(Path(args.out) / "last", raw_model, opt, min(step + 1, total_steps), tokens_seen, args, best_val, ds, train_pf, data_state=data_state)
    if is_main:
        if args.no_checkpoints:
            raw_model.save_pretrained(str(Path(args.out) / "final"))
        else:
            save_ckpt(Path(args.out) / "last", raw_model, opt, min(step + 1, total_steps), tokens_seen, args, best_val, ds, train_pf, s3, data_state=data_state)
            save_ckpt(Path(args.out) / "final", raw_model, opt, total_steps, tokens_seen, args, best_val, ds, train_pf, s3, data_state=data_state)
        if args.tokenizer:                      # save_ckpt already copies it; --no_checkpoints does not
            shutil.copyfile(args.tokenizer, Path(args.out) / "final" / "tokenizer.json")
        if metrics_fp:
            metrics_fp.close()
        print(f"\nDONE step={step} tokens={tokens_seen/1e9:.2f}B best_val={best_val:.3f} "
              f"-> {args.out}/final ({(time.time()-t0)/3600:.1f}h)", flush=True)

    accelerator.end_training()   # clean NCCL shutdown


if __name__ == "__main__":
    main()
