# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Watch a pretraining run: the directory the trainer writes (--out), or a mirror of it pulled from a cluster with
training/monitor/pull.sh. Three ways, all read-only:

  python training/bin.monitor.py check <out> [--budget 4500] [--stop_rule code_new:0.512@20e9] [--json]
      one look: the problems, worst first. Exit 0 = fine, 1 = warnings, 2 = alarm (for a cron job or a watchdog)
  python training/bin.monitor.py top <out>
      live view in the terminal, like btop: progress, speed, loss and bpb trends, every GPU of every node
  python training/bin.monitor.py serve <out> [<out2> ...] [--port 8020]
      dashboard in the browser at http://127.0.0.1:8020: curves, GPUs per node, GPU hours, samples, problems
  python training/bin.monitor.py watch <out> [--every 600] [--report_every 3600] [--notify "cmd"]
      a watchdog: checks every --every seconds and prints a line when the problems change, plus a summary every
      --report_every seconds. --notify runs a command with the message in $MONITOR_MESSAGE (ntfy, a webhook, mail)

A run on a cluster is watched through a local copy of its small files: training/monitor/pull.sh.

What it reads and what it checks: training/monitor/core.py.
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.monitor.core import ALARM, WARN, check, read_run  # noqa: E402

HERE = Path(__file__).resolve().parent / "monitor"


def jsonable(o):
    """NaN and inf (a diverged loss) are not JSON: they travel as null."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def parse_stop_rule(spec):
    """'code_new:0.512@20e9' -> ('code_new', 0.512, 20e9)."""
    if not spec:
        return None
    name, rest = spec.split(":", 1)
    thr, at = rest.split("@", 1)
    return name, float(thr), float(at)


def check_kwargs(args):
    return {"stall_min": args.stall_min, "budget_gpu_h": args.budget, "stop_rule": parse_stop_rule(args.stop_rule)}


def fmt_dur(s):
    if s is None:
        return "-"
    s = int(s)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m = s // 60
    return f"{d}d {h}h" if d else (f"{h}h {m:02d}m" if h else f"{m}m")


def cmd_check(args):
    worst = 0
    report = []
    for out in args.out:
        run = read_run(out)
        problems = check(run, **check_kwargs(args))
        report.append({"out": run["out"], "state": run["state"], "step": run["step"], "tokens": run["tokens"],
                       "tok_s": run["tok_s_now"], "eta_s": run["eta_s"],
                       "problems": [{"level": a, "code": b, "message": c} for a, b, c in problems]})
        worst = max([worst] + [2 if a == ALARM else 1 if a == WARN else 0 for a, _, _ in problems])
    if args.json:
        print(json.dumps(jsonable(report), indent=1))
    else:
        for r in report:
            tok = f"{r['tokens'] / 1e9:.2f}B" if r["tokens"] else "-"
            spd = f"{r['tok_s']:,.0f} tok/s" if r["tok_s"] else "-"
            print(f"{r['out']}: {r['state']}, step {r['step']}, {tok} tokens, {spd}, ETA {fmt_dur(r['eta_s'])}")
            for p in r["problems"]:
                print(f"  [{p['level']}] {p['message']}")
            if not r["problems"]:
                print("  ok")
    return worst


SPARK = "▁▂▃▄▅▆▇█"


def spark(xs, width=60):
    xs = [x for x in xs if x is not None]
    if len(xs) > width:
        k = len(xs) / width
        xs = [sum(xs[int(i * k):int((i + 1) * k)]) / max(1, len(xs[int(i * k):int((i + 1) * k)]))
              for i in range(width)]
    if not xs:
        return ""
    lo, hi = min(xs), max(xs)
    return "".join(SPARK[int((x - lo) / (hi - lo) * 7) if hi > lo else 3] for x in xs)


def cmd_top(args):
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    def render():
        run = read_run(args.out[0])
        probs = check(run, **check_kwargs(args))
        head = Table.grid(expand=True)
        head.add_column(); head.add_column(justify="right")
        tok, tgt = run["tokens"] or 0, run["target_tokens"] or 0
        head.add_row(Text(f"{run['name']}  ·  {run['state']}  ·  step {run['step']}/{run['total_steps']}", style="bold"),
                     Text(time.strftime("%H:%M:%S")))
        head.add_row(f"{tok / 1e9:.2f}B / {tgt / 1e9:.1f}B tokens  ·  "
                     f"{(run['tok_s_now'] or 0):,.0f} tok/s now ({(run['tok_s_run'] or 0):,.0f} run)  ·  "
                     f"ETA {fmt_dur(run['eta_s'])}  ·  {run['world'] or '?'} GPUs  ·  job {run['job'] or '-'}",
                     f"heartbeat {fmt_dur(run['heartbeat_age_s'])} ago")
        bar = ProgressBar(total=tgt or 1, completed=tok, width=None)
        curves = Table.grid(padding=(0, 1))
        curves.add_column(style="cyan"); curves.add_column(); curves.add_column(justify="right")
        losses = [r["loss"] for r in run["steps"]]
        curves.add_row("loss", spark(losses), f"{losses[-1]:.4f}" if losses and losses[-1] is not None else "-")
        vals = [r["val"] for r in run["evals"]]
        curves.add_row("val", spark(vals), f"{vals[-1]:.4f}" if vals and vals[-1] is not None else "-")
        sp = [r["tok_s"] for r in run["speed"]]
        curves.add_row("tok/s", spark(sp), f"{sp[-1]:,.0f}" if sp else "-")
        for k in sorted({k for r in run["bpb"] for k in r if k.startswith("bpb_")}):
            ys = [r.get(k) for r in run["bpb"]]
            curves.add_row(k, spark(ys), f"{ys[-1]:.4f}" if ys and ys[-1] is not None else "-")
        gpus = Table(expand=True, box=None, header_style="bold")
        for c in ("node", "gpu", "util", "power", "memory", "temp", "seen"):
            gpus.add_column(c, justify="right" if c not in ("node",) else "left")
        for n in run["nodes"]:
            for g in n["gpus"]:
                u = g.get("util_pct") or 0
                ustyle = "green" if u >= 80 else "yellow" if u >= 30 else "red"
                gpus.add_row(n["host"][-24:], str(g.get("idx")),
                             Text(f"{'█' * int(u / 10):<10} {u:3d}%", style=ustyle),
                             f"{(g.get('power_w') or 0):.0f} W", f"{(g.get('mem_used_mb') or 0) / 1024:.1f} GB",
                             f"{g.get('temp_c', '-')} °C", fmt_dur(n["age_s"]))
        en = run["energy"]
        foot = Text(f"GPU hours {run['gpu_hours_ledger'] or en['gpu_hours']:,.1f}  ·  energy {en['kwh']:,.1f} kWh"
                    f"  ·  samples: {', '.join(run['samples']) or '-'}")
        if probs:
            ptxt = Text("\n".join(f"[{a}] {c}" for a, _, c in probs))
            ptxt.stylize("bold red" if probs[0][0] == ALARM else "yellow" if probs[0][0] == WARN else "dim")
        else:
            ptxt = Text("no problems", style="green")
        return Group(Panel(Group(head, bar)), Panel(curves, title="trends"), Panel(gpus, title="GPUs"),
                     Panel(ptxt, title="checks"), foot)

    with Live(render(), refresh_per_second=1, screen=True) as live:
        try:
            while True:
                time.sleep(args.every)
                live.update(render())
        except KeyboardInterrupt:
            pass
    return 0


def cmd_watch(args):
    import os
    import subprocess

    def emit(msg, notify):
        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] {msg}", flush=True)
        if notify and args.notify:
            subprocess.run(args.notify, shell=True, env={**os.environ, "MONITOR_MESSAGE": msg}, timeout=60)

    seen, last_report = {}, 0.0
    while True:
        worst, report = 0, time.time() - last_report >= args.report_every
        for out in args.out:
            run = read_run(out)
            problems = [p for p in check(run, **check_kwargs(args)) if p[0] in (ALARM, WARN)]
            key = sorted((a, b) for a, b, _ in problems)
            name = Path(out).resolve().name
            if key != seen.get(out):
                if problems:
                    emit(f"{name}: " + " | ".join(f"[{a}] {c}" for a, _, c in problems), notify=True)
                elif out in seen:
                    emit(f"{name}: problems cleared", notify=True)
                seen[out] = key
            if report:
                tok = f"{run['tokens'] / 1e9:.2f}B" if run["tokens"] else "-"
                emit(f"{name}: {run['state']}, step {run['step']}/{run['total_steps']}, {tok} tokens, "
                     f"{(run['tok_s_now'] or 0):,.0f} tok/s, ETA {fmt_dur(run['eta_s'])}, "
                     f"{len(problems)} problem(s)", notify=False)
            worst = max(worst, 2 if any(p[0] == ALARM for p in problems) else 1 if problems else 0)
        if report:
            last_report = time.time()
        if args.once:
            return worst
        time.sleep(args.every)


def cmd_serve(args):
    outs = [Path(o) for o in args.out]
    html = (HERE / "dashboard.html").read_bytes()
    kw = check_kwargs(args)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, html, "text/html; charset=utf-8")
            if self.path.startswith("/api/runs"):
                return self._send(200, json.dumps([o.resolve().name for o in outs]).encode(), "application/json")
            if self.path.startswith("/api/run"):
                i = 0
                if "?i=" in self.path:
                    try:
                        i = int(self.path.split("?i=", 1)[1])
                    except ValueError:
                        pass
                run = read_run(outs[min(max(i, 0), len(outs) - 1)])
                run["problems"] = [{"level": a, "code": b, "message": c} for a, b, c in check(run, **kw)]
                run["stop_rule"] = kw["stop_rule"]
                run["budget_gpu_h"] = kw["budget_gpu_h"]
                return self._send(200, json.dumps(jsonable(run), allow_nan=False, default=str).encode(),
                                  "application/json")
            self._send(404, b"not found", "text/plain")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard: http://{args.host}:{args.port}  ({', '.join(str(o) for o in outs)})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "top", "serve", "watch"])
    ap.add_argument("out", nargs="+", help="run directory (the trainer's --out, or its mirror)")
    ap.add_argument("--budget", type=float, default=0.0, help="GPU-hour budget of the run (0 = no check)")
    ap.add_argument("--stop_rule", default=None, help="slice:threshold@tokens, e.g. code_new:0.512@20e9")
    ap.add_argument("--stall_min", type=float, default=15.0, help="minutes without a heartbeat = stalled")
    ap.add_argument("--json", action="store_true", help="check: one JSON report")
    ap.add_argument("--every", type=float, default=5.0, help="top: seconds between refreshes")
    ap.add_argument("--report_every", type=float, default=3600.0, help="watch: seconds between summaries")
    ap.add_argument("--notify", default=None, help="watch: shell command run on a change, message in $MONITOR_MESSAGE")
    ap.add_argument("--once", action="store_true", help="watch: one pass and exit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    args = ap.parse_args()
    sys.exit({"check": cmd_check, "top": cmd_top, "serve": cmd_serve, "watch": cmd_watch}[args.mode](args))


if __name__ == "__main__":
    main()
