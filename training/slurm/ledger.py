# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
The ledger of a chained Slurm run: one line per job in <out>/ledger.jsonl (written by train.sbatch), plus
the per-node energy summaries the trainer writes in <out>/telemetry/. It answers three questions:

  how far is the run, and is it moving?    python training/slurm/ledger.py <out>
  how many GPU hours and kWh has it used?  (same, the totals at the bottom)
  may the chain submit one more job?       python training/slurm/ledger.py <out> --can-chain \\
                                               --budget 1650 --next-gpu-h 768 --max-stalled 2

--can-chain exits 0 (yes) or 1 (no, and prints why): the budget would be exceeded, or the last
--max-stalled jobs made no progress (a crash loop that would burn the allocation), or the run is done.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def read_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def energy_by_job(out):
    """kWh per Slurm job id, summed over the nodes' telemetry summaries."""
    kwh = {}
    for f in (out / "telemetry").glob("*.jsonl"):
        for r in read_jsonl(f):
            if r.get("summary"):
                kwh[str(r.get("job", ""))] = kwh.get(str(r.get("job", "")), 0.0) + float(r.get("energy_kwh") or 0)
    return kwh


def summary(out):
    jobs = read_jsonl(out / "ledger.jsonl")
    kwh = energy_by_job(out)
    status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    hb = json.loads((out / "heartbeat.json").read_text()) if (out / "heartbeat.json").exists() else {}
    hb_age = time.time() - (out / "heartbeat.json").stat().st_mtime if (out / "heartbeat.json").exists() else None
    return jobs, kwh, status, hb, hb_age


def stalled_tail(jobs):
    """How many of the most recent jobs ended without advancing (a backfilled job, whose progress is unknown,
    neither counts nor breaks the series). Progress is counted in tokens: the step number is rescaled when a job
    resumes on a different number of GPUs (step 902 on 4 GPUs is step 451 on 8), so by its steps a job that
    resumed on more nodes would look stuck. Lines without tokens fall back to the steps."""
    n = 0
    for j in reversed(jobs):
        a, b = ("tokens_start", "tokens_end") if j.get("tokens_end") is not None else ("step_start", "step_end")
        if j.get(b) is None:
            continue
        if (j.get(b) or 0) > (j.get(a) or 0):
            break
        n += 1
    return n


def _slurm_seconds(s):
    d = 0
    if "-" in s:
        d, s = s.split("-", 1)
        d = int(d)
    p = [int(float(x)) for x in s.split(":")]
    p = [0] * (3 - len(p)) + p
    return d * 86400 + p[0] * 3600 + p[1] * 60 + p[2]


def backfill(out, current_job, gpus_per_node):
    """A job killed hard (time limit without a clean stop, its batch node lost) never writes its ledger line.
    Recover its GPU hours from sacct, so the budget does not undercount, and its progress from the resume
    points: every job writes chain/start.<id> ("step tokens") when it starts, so a lost job went from its own
    start point to the start point of the job after it. A lost job that did not advance counts as stalled."""
    import subprocess
    known = {str(j.get("job")) for j in read_jsonl(out / "ledger.jsonl")}
    ids = [l.strip() for l in (out / "chain" / "jobs.txt").read_text().splitlines()] \
        if (out / "chain" / "jobs.txt").exists() else []
    def start_point(jid):
        f = out / "chain" / f"start.{jid}"
        try:
            step, tok = f.read_text().split()[:2]
            return int(step), int(float(tok))
        except Exception:
            return None, None

    for i, jid in enumerate(ids):
        if not jid or jid == str(current_job) or jid in known:
            continue
        try:
            row = subprocess.run(["sacct", "-n", "-P", "-X", "-j", jid, "-o", "Elapsed,NNodes,State,Start"],
                                 capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        except Exception:
            return
        if not row:
            continue
        elapsed, nodes, state, _ = (row[0].split("|") + ["", "", "", ""])[:4]
        sec, nodes = _slurm_seconds(elapsed or "0"), int(nodes or 0)
        s0, t0 = start_point(jid)
        s1, t1 = start_point(ids[i + 1]) if i + 1 < len(ids) else (None, None)
        if s0 is None or s1 is None:
            s0 = s1 = t0 = t1 = None
        rec = {"job": jid, "nodes": nodes, "gpus": nodes * gpus_per_node, "start": None, "end": None,
               "elapsed_s": sec, "gpu_h": round(sec * nodes * gpus_per_node / 3600, 3), "step_start": s0,
               "step_end": s1, "tokens_start": t0, "tokens_end": t1,
               "state": f"lost ({state.split()[0].lower() if state else 'unknown'})", "exit": None}
        with (out / "ledger.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        known.add(jid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="run directory (--out of the trainer)")
    ap.add_argument("--can-chain", action="store_true")
    ap.add_argument("--budget", type=float, default=0.0, help="GPU-hour cap of the whole run (0 = none)")
    ap.add_argument("--next-gpu-h", type=float, default=0.0, help="GPU hours the next job may use at most")
    ap.add_argument("--max-stalled", type=int, default=2, help="stop chaining after N jobs in a row without progress")
    ap.add_argument("--json", action="store_true", help="one JSON object (for a remote monitor)")
    ap.add_argument("--backfill", default=None, metavar="CURRENT_JOB",
                    help="first add the jobs of chain/jobs.txt missing from the ledger, from sacct")
    ap.add_argument("--gpus-per-node", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    if a.backfill:
        backfill(out, a.backfill, a.gpus_per_node)
    jobs, kwh, status, hb, hb_age = summary(out)
    used = sum(float(j.get("gpu_h") or 0) for j in jobs)
    energy = sum(kwh.values())

    if a.can_chain:
        if status.get("state") == "done":
            print("no: the run is done")
            sys.exit(1)
        if a.budget and used + a.next_gpu_h > a.budget:
            print(f"no: budget {a.budget:,.0f} GPU-h, used {used:,.1f}, next job up to {a.next_gpu_h:,.1f}")
            sys.exit(1)
        st = stalled_tail(jobs)
        if st >= a.max_stalled:
            print(f"no: the last {st} jobs made no progress (crash loop?)")
            sys.exit(1)
        print(f"yes: used {used:,.1f} GPU-h" + (f" of {a.budget:,.0f}" if a.budget else ""))
        sys.exit(0)

    if a.json:
        print(json.dumps({"jobs": len(jobs), "gpu_h": round(used, 2), "energy_kwh": round(energy, 3),
                          "status": status, "heartbeat": hb, "heartbeat_age_s": hb_age,
                          "stalled_tail": stalled_tail(jobs),
                          "alert": (out / "ALERT").read_text().strip() if (out / "ALERT").exists() else None}))
        return

    print(f"{'job':>10} {'nodes':>5} {'start':>16} {'hours':>6} {'GPU-h':>8} {'kWh':>8} {'steps':>15} "
          f"{'tokens (B)':>15}  state")
    for j in jobs:
        t0 = time.strftime("%Y-%m-%d %H:%M", time.localtime(j["start"])) if j.get("start") else "?"
        print(f"{j.get('job', ''):>10} {j.get('nodes', 0):>5} {t0:>16} {j.get('elapsed_s', 0) / 3600:>6.2f} "
              f"{j.get('gpu_h', 0):>8.1f} {kwh.get(str(j.get('job', '')), 0):>8.2f} "
              f"{j.get('step_start') or 0:>7}-{j.get('step_end') or 0:<7} "
              f"{(j.get('tokens_start') or 0) / 1e9:>7.2f}-{(j.get('tokens_end') or 0) / 1e9:<7.2f}  {j.get('state', '')}")
    print(f"total: {len(jobs)} jobs, {used:,.1f} GPU-h, {energy:,.2f} kWh")
    if status:
        print(f"status: {status.get('state')} at step {status.get('step')}/{status.get('total_steps')}, "
              f"{(status.get('tokens') or 0) / 1e9:.2f}B tokens")
    if hb:
        tps = hb.get("tok_s")
        print(f"heartbeat: {hb.get('state')} step {hb.get('step')} loss {hb.get('loss')} "
              f"{(tps or 0) / 1e3:.1f}k tok/s, {hb_age:.0f}s ago")
    if (out / "ALERT").exists():
        print(f"ALERT: {(out / 'ALERT').read_text().strip()}")


if __name__ == "__main__":
    main()
