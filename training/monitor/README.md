# Watching a run

The trainer (`training/bin.pretrain.py`) writes a few small files next to its checkpoints. This folder reads them:
it never touches a checkpoint and never talks to the training processes, so it can run anywhere, on a copy.

| file in `--out` | written | what it holds |
|:--|:--|:--|
| `heartbeat.json`, `status.json` | every few seconds; at every stop | state, step, tokens, the reason of a stop |
| `train_steps.jsonl` | every `--log_every` steps | loss, learning rate, grad norm |
| `metrics.jsonl` | every `--eval_every` steps | validation loss, tokens/s, ETA, weight and update norms |
| `bpb.jsonl` | every `--bpb_every_tok` tokens | bits per byte on frozen text published after the corpus cutoff (`--bpb_set`) |
| `samples.jsonl` | every `--samples_every_tok` tokens | the model's greedy continuation of fixed prompts (`--samples_prompts eval/samples_prompts.jsonl`) |
| `telemetry/<host>.jsonl` | every `--telemetry_s` seconds | power, utilisation, memory, temperature and energy of every GPU of the node |
| `ledger.jsonl` | at the end of every Slurm job | nodes, wall time, GPU hours, steps (training/slurm) |

```bash
python training/bin.monitor.py top   <out>                    # in the terminal, like btop
python training/bin.monitor.py serve <out> [<out2> ...]       # in the browser, http://127.0.0.1:8020
python training/bin.monitor.py check <out> --budget 4500 --stop_rule code_new:0.512@20e9    # exit 0 / 1 / 2
python training/bin.monitor.py watch <out> --every 600 --notify 'curl -d "$MONITOR_MESSAGE" ntfy.sh/<topic>'
```

**A run on a cluster** is watched through a local copy of its small files, refreshed by `pull.sh` (rsync over ssh,
a few MB; the JSONL files only grow and are appended in place):

```bash
bash training/monitor/pull.sh leonardo:/leonardo_work/<project>/runs/skylar-990m mirror/skylar-990m 300 &
python training/bin.monitor.py serve mirror/skylar-990m --budget 4500 --stop_rule code_new:0.512@20e9
```

`pull.sh` writes `<copy>/.pulled` after every good pull. When the copy stops updating (an expired ssh certificate,
the network) the checks say so, instead of calling the run stalled.

**What is checked** (`core.py`, `check`): no heartbeat while training (a hung collective or a dead node), a NaN loss,
a loss well above its recent median, grad-norm spikes, tokens/s below 80% of the run's, a silent node, an idle or
hot GPU, the GPU-hour budget and its projection, the stopping rule on bits per byte, a crash.
