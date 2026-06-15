"""COBOL-aware helpers for the Skylar-390M-Cobol model.

The model is trained to *complete* a COBOLEval-style stub: given a fixed-format COBOL skeleton
(IDENTIFICATION/ENVIRONMENT/DATA/LINKAGE divisions + the task as comments, ending at
`WORKING-STORAGE SECTION.`), it emits the WORKING-STORAGE entries + PROCEDURE DIVISION as a
fenced ```cobol block. `complete_cobol()` wraps the prompt exactly like the training/eval harness
and reassembles a full, compilable program — so you get real COBOL, not a fragment.
"""
import re

from .core import COBOL_SYSTEM

_EVAL_USER = (
    "Complete the following COBOL subprogram. Output ONLY the WORKING-STORAGE SECTION "
    "entries and the PROCEDURE DIVISION USING LINKED-ITEMS (do NOT repeat IDENTIFICATION/"
    "ENVIRONMENT/DATA/LINKAGE), storing the answer in RESULT, ending with END PROGRAM.\n"
    "```cobol\n{stub}\n```"
)


def extract_code_block(src):
    m = re.search(r"```(?:cobol)?\s*\n(.*?)```", src, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else src


def swap_sections(src):
    ws, lk, proc, begin = [], [], [], []
    cur = begin
    for line in src.split("\n"):
        s = line.strip().upper()
        if s.startswith("WORKING-STORAGE SECTION."):
            cur = ws
        elif s.startswith("LINKAGE SECTION."):
            cur = lk
        elif s.startswith("PROCEDURE DIVISION"):
            cur = proc
            line = "       PROCEDURE DIVISION USING LINKED-ITEMS."
        cur.append(line)
    return "\n".join(begin + ws + lk + proc)


def _program_id(stub):
    m = re.search(r"(?im)^\s*PROGRAM-ID\.\s*([A-Za-z0-9-]+)", stub)
    return m.group(1) if m else "SOLUTION"


def construct(stub, completion, entry_point):
    if "IDENTIFICATION DIVISION" in completion.upper():
        prog = completion
    else:
        sol = completion
        if sol.strip().startswith("WORKING-STORAGE SECTION."):
            sol = sol.replace("WORKING-STORAGE SECTION.", "", 1)
        prog = f"{stub}\n{sol}"
    prog = swap_sections(prog)
    name = entry_point.upper().replace("_", "-")
    prog = re.sub(r"(?im)^[ \t]*END[ \t]+PROGRAM\b.*$", "", prog).rstrip()
    prog += f"\n       END PROGRAM {name}.\n"
    return prog


def complete_cobol(sk, stub, entry_point=None, max_new_tokens=900, temperature=0.0):
    """Return a full, reassembled COBOL program for a COBOLEval-style stub."""
    name = entry_point or _program_id(stub)
    raw = sk.generate(_EVAL_USER.format(stub=stub), system=COBOL_SYSTEM,
                      max_new_tokens=max_new_tokens, temperature=temperature)
    return construct(stub, extract_code_block(raw), name)


def syntax_ok(program_text):
    """Best-effort: does GnuCOBOL accept it syntactically? (None if cobc missing).

    Uses fixed-format (COBOLEval programs are column-sensitive). Honors a COBC env var so a
    non-PATH GnuCOBOL build can be pointed at explicitly."""
    import shutil, subprocess, tempfile, os
    cobc = os.environ.get("COBC", "cobc")
    if not (os.path.isfile(cobc) or shutil.which(cobc)):
        return None
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "prog.cbl")
        open(f, "w").write(program_text)
        r = subprocess.run([cobc, "-fsyntax-only", "-fformat=fixed", "-w", f],
                           capture_output=True, text=True)
        return r.returncode == 0


# sample stub for `skylar cobol --example` — a task the model handles well (increment a list).
# (COBOLEval-style fixed format; the model emits WORKING-STORAGE + PROCEDURE, we reassemble.)
EXAMPLE_STUB = """\
       IDENTIFICATION DIVISION.
       PROGRAM-ID. INCR-LIST.

       ENVIRONMENT DIVISION.

       INPUT-OUTPUT SECTION.

       DATA DIVISION.

       LINKAGE SECTION.

       01 LINKED-ITEMS.
           05 L-L OCCURS 3 TIMES INDEXED BY NI PIC S9(10).
           05 RESULT OCCURS 100 TIMES INDEXED BY NJ PIC S9(10).

      * Return list with elements incremented by 1.
      * >>> incr_list([1, 2, 3])
      * [2, 3, 4]
      * >>> incr_list([5, 3, 5])
      * [6, 4, 6]

      * Complete the WORKING-STORAGE SECTION and the PROCEDURE DIVISION
      * Store the result in the RESULT variable and mark the end of your program with END PROGRAM

       WORKING-STORAGE SECTION.
"""
