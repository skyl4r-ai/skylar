# Training on a Slurm cluster

Long pretraining runs on an HPC cluster (written for Leonardo Booster, CINECA; nothing in it is specific to
one machine): 24-hour jobs chained into one run, a checkpoint every 30 minutes, resume on any number of
nodes, a watchdog for hung jobs, a GPU-hour budget the chain cannot exceed, and per-node energy accounting.

| file | what it does |
|---|---|
| `run.env.example` | one file per run: account, nodes, paths, training arguments, budget. Copy and fill it |
| `submit.sh` | `bash submit.sh run.env` starts the chain; `bash submit.sh run.env preflight` runs the day-1 checks |
| `train.sbatch` | one link of the chain (below) |
| `one.sbatch` | one command in the run's environment, one node, no chain: SFT, preference training, GRPO, the WSM merge, COBOLEval (`bash submit.sh run.env one <time> <gpus> -- <command...>`); its hours are not in the chain's ledger, `sacct` has them |
| `preflight.sbatch` | ~25 min on 2 debug nodes: environment, kernel gates, all-reduce (NET/IB), disk, train + resume on 1 node |
| `watchdog.sh` | kills a training step whose heartbeat stopped (a dead node or a hung collective) |
| `ledger.py` | GPU hours, kWh, steps and tokens per job; the budget and crash-loop checks of the chain |
| `check_env.py` | versions, driver vs wheel, Triton, the KDA kernels, NCCL: the first thing to run on a new machine |
| `allreduce_bench.py` | all-reduce bandwidth (nccl-tests convention), and the time of one gradient all-reduce |
| `build_env.sh` | the pinned environment: online, or a wheelhouse built elsewhere and installed without network |
| `stage_data.sh` | the corpus from a Hugging Face bucket (or after an rsync): resumable, every shard checked against `checksums.sha256` |
| `build_gnucobol.sh` | GnuCOBOL 3.2 without root (conda-forge, with its own C compiler), online or from packages downloaded elsewhere; checks that it compiles and runs |
| `fsdp_bench.py` | memory and tokens/s of the 4B and 8B with FSDP2 (per-block checkpointing that keeps the AttnRes state, Muon on sharded matrices): a benchmark, not yet a trainer |

## A link of the chain

1. **May it run?** `ledger.py --can-chain`: not if the run is done, if this job could cross the GPU-hour budget,
   or if the last jobs made no progress (a crash loop would otherwise burn the allocation one job at a time).
   Progress is counted in tokens: the step number changes scale when a job resumes on a different number of GPUs.
   Jobs killed without writing their ledger line are recovered from `sacct`.
2. **Queue the next link now**, `--dependency=afterany`: its wait in the queue overlaps this job.
3. **Train** with `--resume auto` and `--deadline` = the job's end: the trainer saves `<out>/last` and exits
   `STOP_MARGIN_MIN` before the time limit, and every `CKPT_EVERY_MIN` in between. SIGTERM/SIGUSR1 (scancel,
   preemption) also save and exit; the job script waits for that save and still writes its ledger line.
4. **Watchdog** on the side: rank 0 rewrites `<out>/heartbeat.json` every few seconds; older than
   `WATCHDOG_STALL_MIN` (after a `WATCHDOG_GRACE_MIN` startup) means hung, and the step is killed.
5. **Ledger line**: nodes, wall time, GPU hours, steps and tokens before and after, final state.

## What the trainer does for this (`training/bin.pretrain.py`)

- `--deadline`, `--stop_margin_min`, `--ckpt_every_min`: planned stops and time-based checkpoints; one
  all-reduce per step makes every rank take the same decision (and carries the global mean loss).
- **Resume on a different number of GPUs**: the sampler position is counted in samples, and the window of
  global sample k does not depend on the number of ranks. Tested 2 -> 1 -> 2 ranks: same samples, none twice,
  the same loss curve as an uninterrupted run to 1e-4.
- `<out>/status.json` (done / deadline / signal / interrupted / crashed), `<out>/heartbeat.json`,
  `<out>/last/progress.json` (step and tokens without loading the optimizer state).
- `--ddp_comm bf16` halves the gradient traffic between nodes; `--dist_timeout_min` turns a hang into an error.
- `--telemetry_s`: per-node power, utilisation, memory and the NVML energy counter in `<out>/telemetry/`.
- `--bpb_set <dir>`: bits per byte on frozen `*.txt` slices every `--bpb_every_tok`, in `<out>/bpb.jsonl`, with the
  protocol of the stopping rule (`eval/bpb.py`, the windows split across the ranks; identical to
  `eval/bin.bits_per_byte.py` on the same checkpoint to 1e-4). Our post-cutoff texts are not redistributed:
  `eval/postcutoff_bytes/manifest.json` lists what to fetch to rebuild them (`eval/README.md`).

## What has been tested, and what has not

- **RTX 4090:** the trainer features above, one at a time: deadline, signals, time-based checkpoints, resume
  2 -> 1 -> 2 ranks, live bits per byte. The chain on a mock Slurm (sbatch, squeue, scancel, sacct): resubmission,
  the budget stop, the crash-loop stop, a job killed without its ledger line, the watchdog killing a hung step
  and the next job resuming from `last`.
- **2x A100 SXM 80 GB** (sm_80, as on Leonardo): `build_env.sh online` from scratch, `check_env.py`, the 18 kernel
  gates, `allreduce_bench.py`, the 990M under DDP (92% per GPU, peak 54.8 GiB with a 62 GiB cap), bf16 gradients,
  the energy counter, an `srun` step with GPU gres under Slurm 23.11; `fsdp_bench.py` on the 4B and 8B.
- **An older driver** (560, CUDA 12.6) with the cu128 wheels: `check_env.py` passes, Triton and the KDA kernels
  included.
- **Post-training on Skylar 2 (RTX 4090, 10/10/2026):** the WSM merge (also on four real-size 990M snapshots: 40 s,
  17 GB of RAM, constant in the number of snapshots), SFT with Muon, ORPO, SimPO, GRPO with the COBOL reward and
  COBOLEval, all on the hybrid model; `one.sbatch` on a mock Slurm; `build_gnucobol.sh` online and offline (network
  blocked), with the same outcome as the previous GnuCOBOL on all 146 COBOLEval problems of a reference sample set.
- **A Slurm cluster of 3 nodes x 4 A100 SXM 80 GB with InfiniBand (10/10/2026):** `preflight.sbatch` all PASS
  (NCCL on `NET/IB` with GPUDirect RDMA); the chain of 15-minute jobs with stops at the deadline; a rank frozen with
  SIGSTOP (the watchdog killed the step after 3 minutes, the next job resumed); `scancel`; resumes 3 -> 1 -> 2 nodes
  with no sample repeated or skipped; WSM snapshots, bits per byte and samples during the run; `one.sbatch` with SFT
  and COBOLEval (the GnuCOBOL of `build_gnucobol.sh` scored a reference sample set exactly as at home); the run
  watched from outside with `training/monitor/`. The 990M: 11,720 tokens/s per GPU on 3 nodes against 12,050 on one
  (97%), peak 53.5 GiB per GPU. Six bugs of this kit found there are fixed in this version.
- **Not yet:** Leonardo itself, and more than 3 nodes. `preflight.sbatch` is the first check there.

## Day 1 on a new cluster

```bash
bash training/slurm/build_env.sh online $WORK/venv/skylar cu128        # or download + install, see the script
$WORK/venv/skylar/bin/python training/slurm/check_env.py                 # on a GPU node
cp training/slurm/run.env.example my_run.env && $EDITOR my_run.env        # account, paths, NCCL_ENV
bash training/slurm/submit.sh my_run.env preflight                       # read <out>/preflight/<job>/SUMMARY.txt
# not on Leonardo: PREFLIGHT_QOS="" (no debug QOS), PREFLIGHT_NODES=<n>; one phase: PREFLIGHT_PHASES=env
bash training/slurm/submit.sh my_run.env                                 # the run
python training/slurm/ledger.py <out>                                    # where it is, GPU hours, kWh
```
