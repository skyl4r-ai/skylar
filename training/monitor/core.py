# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Reads a run directory (the --out of bin.pretrain.py, or a mirror of it pulled from a cluster) and turns it into what a
person or a watchdog needs: where the run is, its curves, every GPU of every node, the GPU hours, the text samples,
and the problems. It only reads the small files the trainer writes, never a checkpoint:

  status.json, heartbeat.json      state, step, tokens (heartbeat: every few seconds while training)
  train_steps.jsonl                loss, lr, grad norm per logged step
  metrics.jsonl                    validation, tokens/s, ETA at every evaluation
  bpb.jsonl                        bits per byte on the frozen post-cutoff slices (--bpb_set)
  samples.jsonl                    the model's continuation of fixed prompts (--samples_prompts)
  telemetry/<host>.jsonl           power, utilisation, memory, temperature and energy of each GPU (one file per node)
  ledger.jsonl                     one line per Slurm job of the chain (training/slurm)
  ALERT                            what the chain refused or failed to do (over budget, a crash loop, sbatch failed)

`read_run(out)` returns a dict ready for JSON; `check(run, ...)` returns the problems, worst first.
"""
import json
import math
import os
import time
from pathlib import Path

ALARM, WARN, INFO = "ALARM", "WARN", "INFO"
RUNNING = ("starting", "train")                 # heartbeat states of a live trainer
PAUSED = ("deadline", "signal", "interrupted")  # stopped cleanly with `last` saved: a chain resumes it


def _json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class _Follow:
    """An append-only JSONL file read incrementally: every call parses only the lines added since the last one, so
    a dashboard refreshing a run of days on 32 nodes does not reread hundreds of MB. A file replaced or truncated
    (a new mirror) is read again from the start."""

    def __init__(self, path):
        self.path, self.inode, self.offset, self.buf = Path(path), None, 0, b""
        self.reset()

    def reset(self):
        self.rows = []

    def add(self, rec):
        self.rows.append(rec)

    def update(self):
        try:
            st = self.path.stat()
        except OSError:
            return self
        if st.st_ino != self.inode or st.st_size < self.offset:
            self.inode, self.offset, self.buf = st.st_ino, 0, b""
            self.reset()
        if st.st_size > self.offset:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = self.buf + f.read()
                self.offset = f.tell()
            lines = data.split(b"\n")
            self.buf = lines.pop()                  # a line still being written
            for line in lines:
                try:
                    self.add(json.loads(line))
                except ValueError:
                    pass
        return self


class _Telemetry(_Follow):
    """One node's telemetry reduced as it is read: the last sample, a power/utilisation series, and the GPU hours
    and energy of finished jobs (their summary lines) plus the job running now (from its first sample)."""

    GAP_S = 900                                     # samples further apart belong to different jobs

    def reset(self):
        self.last, self.series = None, []
        self.done_gpu_h = self.done_kwh = 0.0
        self.seg0 = self.prev = None

    def _close(self):
        if self.seg0 and self.prev and self.prev is not self.seg0:
            self.done_gpu_h += len(self.prev["gpus"]) * (self.prev["t"] - self.seg0["t"]) / 3600
            self.done_kwh += self._kwh(self.seg0, self.prev)
        self.seg0 = self.prev = None

    @staticmethod
    def _kwh(a, b):
        return sum(max(0.0, (gb.get("energy_j") or 0) - (ga.get("energy_j") or 0))
                   for ga, gb in zip(a["gpus"], b["gpus"])) / 3.6e6

    def add(self, rec):
        if rec.get("summary"):                      # a job ended cleanly: its own totals replace the estimate
            self.done_gpu_h += float(rec.get("gpu_hours") or 0)
            self.done_kwh += float(rec.get("energy_kwh") or 0)
            self.seg0 = self.prev = None
            return
        if "gpus" not in rec:
            return
        if self.prev and rec["t"] - self.prev["t"] > self.GAP_S:
            self._close()                           # a job killed hard wrote no summary
        if self.seg0 is None:
            self.seg0 = rec
        self.prev = self.last = rec
        self.series.append({"t": rec["t"], "power_w": sum(g.get("power_w") or 0 for g in rec["gpus"]),
                            "util": sum(g.get("util_pct") or 0 for g in rec["gpus"]) / max(1, len(rec["gpus"]))})

    def totals(self):
        gpu_h, kwh = self.done_gpu_h, self.done_kwh
        if self.seg0 and self.prev:
            gpu_h += len(self.prev["gpus"]) * (self.prev["t"] - self.seg0["t"]) / 3600
            kwh += self._kwh(self.seg0, self.prev)
        return gpu_h, kwh


_CACHE = {}


def _follow(path, kind=_Follow):
    key = (str(path), kind)
    if key not in _CACHE:
        _CACHE[key] = kind(path)
    return _CACHE[key].update()


def _thin(rows, n):
    """At most n rows, evenly spaced, always keeping the last one."""
    if len(rows) <= n:
        return rows
    k = len(rows) / n
    out = [rows[int(i * k)] for i in range(n - 1)]
    return out + [rows[-1]]


def _tok(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _median(xs):
    xs = sorted(x for x in xs if _finite(x))
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def read_run(out, max_points=1500):
    out = Path(out)
    now = time.time()
    status = _json(out / "status.json") or {}
    hb = _json(out / "heartbeat.json") or {}
    hb_path = out / "heartbeat.json"
    hb_age = now - float(hb["t"]) if _finite(hb.get("t")) else (
        now - hb_path.stat().st_mtime if hb_path.exists() else None)
    # a mirror pulled from a cluster (training/monitor/pull.sh): how old the copy is
    try:
        mirror_age = now - float((out / ".pulled").read_text().strip())
    except (OSError, ValueError):
        mirror_age = None

    steps = _follow(out / "train_steps.jsonl").rows
    evals = _follow(out / "metrics.jsonl").rows
    bpb = _follow(out / "bpb.jsonl").rows
    ledger = _follow(out / "ledger.jsonl").rows

    # tokens/s from the logged steps, over windows of at least a minute (an evaluation or a save between two log
    # lines would otherwise show as a drop): the last ~10 minutes give the speed now
    speed, a = [], None
    for b in steps:
        if a is None or not b.get("t") or not b.get("tokens"):
            a = b if b.get("t") and b.get("tokens") else a
            continue
        dt, dtok = b["t"] - a["t"], b["tokens"] - a["tokens"]
        if dt > 900 or dtok < 0:                        # a gap (restart, queue) is not a speed
            a = b
        elif dt >= 60:
            speed.append({"t": b["t"], "tokens": b["tokens"], "tok_s": dtok / dt})
            a = b
    recent = [s["tok_s"] for s in speed if s["t"] >= (speed[-1]["t"] - 600 if speed else 0)]
    tok_s_now = _median(recent)
    tok_s_run = _median([s["tok_s"] for s in speed[len(speed) // 20:]])

    step = hb.get("step", status.get("step"))
    total = hb.get("total_steps", status.get("total_steps"))
    tokens = hb.get("tokens", status.get("tokens"))
    tok_per_step = (tokens / step) if step and tokens else None
    target_tokens = total * tok_per_step if total and tok_per_step else None
    eta_s = ((target_tokens - tokens) / tok_s_now) if target_tokens and tok_s_now and tokens is not None else None

    # GPUs: the last sample of every node, and per node a thinned series of total power and mean utilisation
    nodes, energy = [], {"kwh": 0.0, "gpu_hours": 0.0}
    for f in sorted((out / "telemetry").glob("*.jsonl")):
        tel = _follow(f, _Telemetry)
        gpu_h, kwh = tel.totals()
        energy["gpu_hours"] += gpu_h
        energy["kwh"] += kwh
        if tel.last is None:
            continue
        nodes.append({"host": tel.last.get("host", f.stem), "job": tel.last.get("job"), "age_s": now - tel.last["t"],
                      "gpus": tel.last["gpus"], "series": _thin(tel.series, max_points // 3)})

    samples = {}
    for r in _follow(out / "samples.jsonl").rows:
        samples.setdefault(r.get("name", "?"), []).append(
            {"tokens": r.get("tokens"), "step": r.get("step"), "prompt": r.get("prompt"),
             "completion": r.get("completion")})

    gpu_h_ledger = sum(float(j.get("gpu_h") or 0) for j in ledger)
    try:
        alerts = [x for x in (out / "ALERT").read_text().splitlines() if x.strip()][-20:]
    except OSError:
        alerts = []
    return {
        "out": str(out.resolve()), "name": out.resolve().name, "now": now,
        "state": hb.get("state") or status.get("state") or ("empty" if not steps else "?"),
        "status": status, "heartbeat": hb, "heartbeat_age_s": hb_age, "mirror_age_s": mirror_age,
        "step": step, "total_steps": total, "tokens": tokens, "target_tokens": target_tokens,
        "world": hb.get("world", status.get("world")), "job": hb.get("job", status.get("job")),
        "tok_s_now": tok_s_now, "tok_s_run": tok_s_run, "eta_s": eta_s,
        "steps": _thin([{"tokens": r.get("tokens"), "loss": r.get("train_loss"), "lr": r.get("lr"),
                         "grad_norm": r.get("grad_norm"), "t": r.get("t")} for r in steps], max_points),
        "recent_losses": [r.get("train_loss") for r in steps[-300:]],
        "recent_grad_norms": [r.get("grad_norm") for r in steps[-300:]],
        "median_grad_norm": _median([r.get("grad_norm") for r in steps[len(steps) // 20:]]),
        "evals": _thin([{"tokens": r.get("tokens"), "val": r.get("val_loss"), "tok_s": r.get("tok_s"),
                         "t": r.get("t")} for r in evals], max_points),
        "speed": _thin(speed, max_points),
        "bpb": [{k: v for k, v in r.items() if k == "tokens" or k.startswith("bpb_")} for r in bpb],
        "nodes": nodes, "energy": energy,
        "ledger": ledger, "gpu_hours_ledger": gpu_h_ledger, "alerts": alerts,
        "samples": samples,
    }


def check(run, stall_min=15.0, budget_gpu_h=0.0, stop_rule=None, node_stale_min=10.0, min_util=30.0,
          max_temp=85.0, slow=0.8):
    """Problems of a run, worst first: [(level, code, message)]. `stop_rule`: (slice, threshold, tokens), e.g.
    ("code_new", 0.512, 20e9): at or after `tokens`, bits per byte above the threshold on that slice."""
    out = []
    running = run["state"] in RUNNING
    age = run["heartbeat_age_s"]
    mirror = run.get("mirror_age_s")
    if mirror is not None and mirror > stall_min * 60:
        out.append((WARN, "mirror_stale", f"the local copy was last updated {mirror / 60:.0f} min ago: pull.sh cannot "
                                          f"reach the cluster (ssh certificate, network). The run itself is unknown"))
        running = False                             # its heartbeat is as old as the copy: no verdict on the run
    if running and age is not None and age > stall_min * 60:
        out.append((ALARM, "stalled", f"no heartbeat for {age / 60:.0f} min (state '{run['state']}', step "
                                      f"{run['step']}): a hung collective, a dead node, or the job is gone"))
    losses = run["recent_losses"]
    if any(x is not None and not _finite(x) for x in losses[-50:]):
        out.append((ALARM, "nonfinite", "the training loss is NaN or inf in the last steps"))
    fin = [x for x in losses if _finite(x)]
    if len(fin) >= 120:
        base, now_ = _median(fin[:-20]), sum(fin[-20:]) / 20
        if base and now_ > base * 1.15:
            out.append((WARN, "loss_up", f"training loss {now_:.3f} over the last 20 logged steps, against a "
                                         f"median of {base:.3f} before: a spike or a divergence"))
    gn, mgn = [x for x in run["recent_grad_norms"][-50:] if _finite(x)], run["median_grad_norm"]
    if gn and mgn and max(gn) > 10 * mgn:
        out.append((WARN, "grad_spike", f"grad norm up to {max(gn):.2f} in the last steps, run median {mgn:.2f}"))
    if running and run["tok_s_now"] and run["tok_s_run"] and run["tok_s_now"] < slow * run["tok_s_run"]:
        out.append((WARN, "slow", f"{run['tok_s_now']:,.0f} tokens/s in the last 10 minutes against "
                                  f"{run['tok_s_run']:,.0f} for the run: a slow node, the disk, or the network"))
    for n in run["nodes"]:
        if running and n["age_s"] > node_stale_min * 60:
            out.append((WARN, "node_silent", f"{n['host']}: no telemetry for {n['age_s'] / 60:.0f} min"))
            continue
        for g in n["gpus"]:
            if running and (g.get("util_pct") or 0) < min_util:
                out.append((WARN, "gpu_idle", f"{n['host']} GPU {g.get('idx')}: {g.get('util_pct')}% busy"))
            if (g.get("temp_c") or 0) >= max_temp:
                out.append((WARN, "gpu_hot", f"{n['host']} GPU {g.get('idx')}: {g.get('temp_c')} °C"))
    if budget_gpu_h:
        used = run["gpu_hours_ledger"] or run["energy"]["gpu_hours"]
        gpus = sum(len(n["gpus"]) for n in run["nodes"]) or (run["world"] or 0)
        need = used + (run["eta_s"] or 0) / 3600 * gpus
        if used > 0.9 * budget_gpu_h:
            out.append((WARN, "budget", f"{used:,.0f} of {budget_gpu_h:,.0f} GPU hours used"))
        elif run["eta_s"] and need > budget_gpu_h:
            out.append((WARN, "budget_projection", f"at this speed the run needs ~{need:,.0f} GPU hours, the budget "
                                                   f"is {budget_gpu_h:,.0f}"))
    if stop_rule and run["bpb"]:
        name, thr, at = stop_rule
        last = [r for r in run["bpb"] if (r.get("tokens") or 0) >= at and f"bpb_{name}" in r]
        if last and last[0][f"bpb_{name}"] > thr:
            out.append((ALARM, "stop_rule", f"bpb {name} = {last[0][f'bpb_{name}']:.4f} at "
                                            f"{_tok(last[0]['tokens'])} tokens, above {thr}: the stopping rule "
                                            f"says stop and look"))
    if run.get("alerts"):
        out.append((ALARM, "chain_alert", f"the Slurm chain wrote: {run['alerts'][-1]}"
                                          + (f" (and {len(run['alerts']) - 1} earlier)" if len(run["alerts"]) > 1 else "")))
    if run["state"] == "crashed":
        err = (run["status"] or {}).get("error", "")
        out.append((ALARM, "crashed", f"the trainer crashed at step {run['step']}: {err}"))
    elif run["state"] in PAUSED:
        out.append((INFO, "paused", f"stopped cleanly ('{run['state']}') at step {run['step']}, "
                                    f"{(age or 0) / 60:.0f} min ago: the next job of the chain resumes it"))
    elif run["state"] == "done":
        out.append((INFO, "done", f"the run is done at step {run['step']}"))
    order = {ALARM: 0, WARN: 1, INFO: 2}
    return sorted(out, key=lambda x: order[x[0]])
