#!/usr/bin/env python3
"""Skylar COBOL demo runner — OUTPUT 100% REALE (nessun COBOL scriptato).

Per ogni task COBOLEval-style:
  1) mostra lo stub di input (commenti = specifica del problema)
  2) STREAMA il COBOL generato dal modello locale (sk.stream, greedy det.)
  3) RIASSEMBLA il programma intero e lo COMPILA con GnuCOBOL vero (cobc)
  4) ESEGUE il programma linkando il caller ufficiale COBOLEval e mostra
     l'output REALE (deve combaciare col risultato atteso)

Modello: checkpoint LOCALE gold 386M from-scratch (`checkpoints_sft/gold_sft_v5`),
mai warm-start, mai pesi altrui. cobc = GnuCOBOL 3.2 in tools/gnucobol-env.
Riproducibile:  python assets/cobol_demo_runner.py
Usato per registrare assets/skylar-cobol-demo.gif con charmbracelet/vhs.
"""
import json
import os
import subprocess
import sys
import time

ROOT = "/home/mwspace/htdocs/skylar/projects/skylar-cobol"
GNU = f"{ROOT}/tools/gnucobol-env"
COBC = f"{GNU}/bin/cobc"
CKPT = f"{ROOT}/checkpoints_sft/gold_sft_v5"
DEMO_DIR = os.path.dirname(os.path.abspath(__file__)) + "/cobol_demo"
WORK = "/tmp/skylar_cobol_gif"

ENV = dict(os.environ)
ENV["COBC"] = COBC
ENV["COB_CC"] = "/usr/bin/gcc"
ENV["COB_CFLAGS"] = f"-I{GNU}/include"
ENV["COB_LDFLAGS"] = f"-L{GNU}/lib"
ENV["LD_LIBRARY_PATH"] = f"{GNU}/lib:" + ENV.get("LD_LIBRARY_PATH", "")

# colors
C = "\033[36m"; G = "\033[32m"; D = "\033[90m"; B = "\033[1m"
Y = "\033[33m"; M = "\033[35m"; W = "\033[97m"; R = "\033[0m"

# (entry_point, which test index to RUN live, human label of the task)
TASKS = [
    ("max_element", 1, "trova il massimo di una lista"),
    ("sum_to_n", 4, "somma i numeri da 1 a n"),
]


def reveal(text, cps=0.006, color=W):
    sys.stdout.write(color)
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if ch != " ":
            time.sleep(cps)
    sys.stdout.write(R)
    sys.stdout.flush()


def main():
    os.makedirs(WORK, exist_ok=True)
    sys.path.insert(0, "/home/mwspace/htdocs/skylar/pip-skylar")
    import skylar
    from skylar.core import COBOL_SYSTEM
    from skylar.cobol import _EVAL_USER, construct, extract_code_block

    # GATE: when SKYLAR_DEMO_GATED=1, block on stdin before each task so the
    # vhs recorder controls exactly what is captured (load happens off-camera).
    gated = os.environ.get("SKYLAR_DEMO_GATED") == "1"
    _tty = None
    if gated:
        try:
            _tty = open("/dev/tty")
        except Exception:
            _tty = sys.stdin

    def gate():
        if gated and _tty is not None:
            try:
                line = _tty.readline()
                if line == "":          # EOF -> don't busy-spin; fall back to a pause
                    time.sleep(6)
            except Exception:
                time.sleep(6)

    print(f"{D}# Skylar COBOL — carico il modello locale gold 386M (from-scratch) sulla RTX 4090 ...{R}", flush=True)
    sk = skylar.load(CKPT, device="cuda")
    cobc_ver = subprocess.run([COBC, "--version"], capture_output=True, text=True).stdout.split(chr(10))[0]
    print(f"{G}# pronto.{R} {D}pesi su GPU, {cobc_ver}.{R}", flush=True)
    time.sleep(1.0)
    gate()  # (gated mode only) wait for recorder before task 1

    for ti, (ep, run_idx, label) in enumerate(TASKS):
        if ti > 0:
            gate()  # recorder advances to the next task on its own clock
        sys.stdout.write("\033[H\033[2J")  # clear so each task fits the viewport
        print(f"{D}# Skylar COBOL · gold 386M · 100% from-scratch · {cobc_ver} · RTX 4090{R}\n")
        name = ep.upper().replace("_", "-")
        stub = open(f"{DEMO_DIR}/{ep}.prompt.cbl").read()
        meta = json.load(open(f"{DEMO_DIR}/{ep}.meta.json"))
        expected = meta["tests"][run_idx]["expected"]

        print(f"{G}${R} {B}skylar cobol{R} --model gold_sft_v5 --device cuda  {D}# {label}{R}")
        time.sleep(0.4)

        # 1) the task spec (compact: description + one example)
        spec = [l for l in stub.splitlines() if l.lstrip().startswith("* ") and "Complete" not in l and "Store" not in l]
        spec = [l.replace("      *", "  ").rstrip() for l in spec if l.strip()][:3]
        print(f"{D}  task:{R}")
        for l in spec:
            print(f"{M}{l}{R}")
        time.sleep(0.5)

        # 2) live generation (REAL token stream)
        print(f"\n{C}  ┌─ Skylar genera il COBOL ──────────────────────────────{R}")
        acc = ""
        sys.stdout.write("  ")
        for delta in sk.stream(_EVAL_USER.format(stub=stub), system=COBOL_SYSTEM,
                               max_new_tokens=300, temperature=0.0):
            acc += delta
            out = delta.replace("\n", "\n  ")
            sys.stdout.write(f"{W}{out}{R}")
            sys.stdout.flush()
            time.sleep(0.012)
        print(f"\n{C}  └────────────────────────────────────────────────────────{R}")
        time.sleep(0.4)

        # 3) reassemble + compile (REAL cobc)
        prog = construct(stub, extract_code_block(acc), ep)
        sol_path = f"{WORK}/{name}.cbl"
        open(sol_path, "w").write(prog)
        call_path = f"{WORK}/call_{name}.cbl"
        open(call_path, "w").write(open(f"{DEMO_DIR}/{ep}.call{run_idx}.cbl").read())
        exe = f"{WORK}/run_{ep}"
        txt = f"{WORK}/{name}.TXT"
        for f in (exe, txt):
            if os.path.exists(f):
                os.remove(f)

        print(f"\n{D}  $ cobc -x -fformat=variable call_{name}.cbl {name}.cbl{R}")
        time.sleep(0.3)
        cp = subprocess.run([COBC, "-w", "-fformat=variable", "-x", "-o", exe,
                             call_path, sol_path], env=ENV, cwd=WORK,
                            capture_output=True, text=True)
        if cp.returncode == 0:
            print(f"  {G}✓ GnuCOBOL: compila{R}")
        else:
            print(f"  \033[31m✗ errore di compilazione{R}\n{cp.stderr}")
            continue
        time.sleep(0.4)

        # 4) RUN it (REAL execution, linked with official caller)
        print(f"{D}  $ ./run_{ep}{R}")
        time.sleep(0.3)
        subprocess.run([exe], env=ENV, cwd=WORK, capture_output=True, timeout=15)
        got = [l.lstrip("0") or "0" for l in open(txt).read().splitlines()] if os.path.exists(txt) else []
        # keep only the meaningful values (trim trailing zero padding for list results)
        typ = meta["tests"][run_idx]["type"]
        if isinstance(typ, dict):  # list result
            n = len(eval(expected))
            got_show = "[" + ", ".join(got[:n]) + "]"
        else:
            got_show = got[0] if got else "(vuoto)"
        ok = got_show.strip("[]").replace(" ", "") == expected.replace(" ", "").strip("[]") or got_show == expected
        mark = f"{G}✓ corretto{R}" if ok else f"\033[31m✗{R}"
        print(f"  {Y}output:{R} {B}{got_show}{R}   {D}(atteso {expected}){R}   {mark}")
        print()
        time.sleep(2.2 if ti == len(TASKS) - 1 else 1.4)

    print(f"{B}Skylar{R} — scrive {C}COBOL{R} che compila ed esegue · in locale · {M}100% from-scratch{R}")
    time.sleep(2.0)


if __name__ == "__main__":
    main()
