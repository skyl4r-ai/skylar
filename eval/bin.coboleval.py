# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
COBOLEval: does the generated COBOL compile, run, and return the right answer?

COBOLEval (BloopAI, MIT) turns the 146 HumanEval problems into COBOL subprograms. For each test the
candidate is linked with a calling program, both are compiled by GnuCOBOL, the binary runs, and the value it
writes is compared with the expected one. Assembly and scoring follow the original (`generate.py`,
`evaluation.py`), with execution enabled (the original leaves it commented out), each test in its own
temporary directory and under a timeout.

Two numbers per run:
  CSR     compile success rate: problems whose program compiles
  pass@1  problems whose program compiles and passes every test

    git clone https://github.com/BloopAI/COBOLEval eval/COBOLEval     # the problems (not vendored here)
    python eval/bin.coboleval.py --model Skyl4r-Ai/Skylar-980M-Cobol --greedy --out samples.jsonl
    python eval/bin.coboleval.py --samples samples.jsonl              # rescore, or score any other model

`--samples` takes one JSON object per line with `task_id` and `completion` (a full program, or only the
WORKING-STORAGE and PROCEDURE DIVISION that follow the prompt), so baselines generated elsewhere are
scored by the same code. Needs GnuCOBOL 3.2 or newer (`cobc` on PATH, or COBC=/path/to/cobc): the
published numbers use 3.2.0, and 3.1 does not accept the source format COBOLEval compiles with.
"""
import argparse, ast, json, math, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "eval/COBOLEval/data/CobolEval.jsonl"
SYSTEM = "Sei un esperto programmatore COBOL."
INSTRUCTION = ("Complete the following COBOL subprogram. Output ONLY the WORKING-STORAGE SECTION "
               "entries and the PROCEDURE DIVISION USING LINKED-ITEMS (do NOT repeat IDENTIFICATION/"
               "ENVIRONMENT/DATA/LINKAGE), storing the answer in RESULT, ending with END PROGRAM.\n")


def cobc_path():
    path = os.environ.get("COBC") or shutil.which("cobc")
    if not path:
        raise SystemExit("GnuCOBOL not found: install it (apt install gnucobol) or set COBC=/path/to/cobc")
    return path


def cobc_env(cobc):
    """A GnuCOBOL outside the system paths (e.g. a conda env) needs its runtime library and a C compiler."""
    env = dict(os.environ)
    lib = Path(cobc).resolve().parent.parent / "lib"
    if lib.is_dir():
        env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env.setdefault("COB_CC", "gcc")
    return env


# ── assembly, as in COBOLEval generate.py ─────────────────────────────────────────
def extract_code_block(text):
    m = re.search(r"```(?:cobol)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else text


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


def construct(prompt, completion, entry_point):
    """The program to compile: the model's full program, or the prompt followed by the model's tail."""
    if "IDENTIFICATION DIVISION" in completion.upper():
        prog = completion
    else:
        sol = completion
        if sol.strip().startswith("WORKING-STORAGE SECTION."):
            sol = sol.replace("WORKING-STORAGE SECTION.", "", 1)
        prog = f"{prompt}\n{sol}"
    prog = swap_sections(prog)
    # a missing or garbled END PROGRAM is replaced by the entry point's name
    prog = re.sub(r"(?im)^[ \t]*END[ \t]+PROGRAM\b.*$", "", prog).rstrip()
    return prog + f"\n       END PROGRAM {entry_point.upper().replace('_', '-')}.\n"


# ── scoring, as in COBOLEval evaluation.py, with execution ────────────────────────
def parse_result(lines, type_, expected):
    def num(cast, x):
        x = x.strip()
        return -cast(x[1:]) if x[:1] in ("p", "y") else cast(x)   # COBOL signed display: p/y = negative
    try:
        if type_ == "Bool":
            return lines[0].strip() == "1"
        if type_ == "Int":
            return num(int, lines[0])
        if type_ == "Float":
            return num(float, lines[0])
        if type_ == "String":
            return lines[0].strip()
        if isinstance(type_, dict):
            cast = {"Int": lambda x: num(int, x), "Float": lambda x: num(float, x),
                    "String": lambda x: x.strip()}[type_["List"]]
            return [cast(x) for x in lines][:len(expected)]
    except Exception:
        return None
    return None


def is_equal(type_, got, expected):
    if got is None:
        return False
    if type_ == "Float":
        return isinstance(got, float) and math.isclose(got, expected, abs_tol=1e-3)
    if isinstance(type_, dict) and type_.get("List") == "Float":
        return len(got) == len(expected) and all(math.isclose(a, b, abs_tol=1e-3) for a, b in zip(got, expected))
    return got == expected


def score_problem(problem, program, cobc, env, timeout=10):
    """(compiled, passed): compiled for at least one test; passed = compiled and every test right."""
    name = problem["entry_point"]
    out_name = name.upper().replace("_", "-") + ".TXT"
    compiled, passed = False, True
    for test in problem["tests"]:
        expected = ast.literal_eval(test["result"]["value"])
        if isinstance(expected, tuple):
            expected = list(expected)
        d = Path(tempfile.mkdtemp(prefix="coboleval_"))
        try:
            (d / f"{name}.cbl").write_text(program)
            (d / f"call_{name}.cbl").write_text(test["test"])
            c = subprocess.run([cobc, "-w", "-fformat=variable", "-x", f"call_{name}.cbl", f"{name}.cbl", "-o", "run"],
                               capture_output=True, text=True, timeout=30, cwd=d, env=env)
            if c.returncode != 0:
                passed = False
                continue
            compiled = True
            try:
                subprocess.run([str(d / "run")], capture_output=True, text=True, timeout=timeout, cwd=d, env=env)
            except subprocess.TimeoutExpired:
                passed = False
                continue
            if not (d / out_name).exists():
                passed = False
                continue
            got = parse_result((d / out_name).read_text().splitlines(), test["result"]["type_"], expected)
            passed &= is_equal(test["result"]["type_"], got, expected)
        except Exception:
            passed = False
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return compiled, compiled and passed


# ── generation with a Skylar checkpoint ───────────────────────────────────────────
class LocalModel:
    def __init__(self, model_dir, greedy, seed, max_new_tokens):
        import torch
        from tokenizers import Tokenizer
        from models.decoder import Skylar2ForCausalLM
        from utils.chatML import encode_chatml
        self.torch, self.encode = torch, encode_chatml
        if not Path(model_dir).is_dir():           # a Hugging Face id, e.g. Skyl4r-Ai/Skylar-980M-Cobol
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(model_dir)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Skylar2ForCausalLM.from_pretrained(model_dir).to(self.dev).eval()
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.greedy, self.seed, self.max_new_tokens = greedy, seed, max_new_tokens

    def __call__(self, prompt, entry_point):
        torch = self.torch
        ids = self.encode([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": INSTRUCTION + f"```cobol\n{prompt}\n```"}],
                          self.tok, add_generation_prompt=True)
        torch.manual_seed(self.seed)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            out = self.model.generate(torch.tensor([ids], device=self.dev), max_new_tokens=self.max_new_tokens,
                                      temperature=0.0 if self.greedy else 0.2, top_k=40, repetition_penalty=1.0,
                                      eos_token_id=self.tok.token_to_id("<|im_end|>"))
        text = self.tok.decode(out[0].tolist()[len(ids):])
        if "</think>" in text:                     # reasoning models: keep the answer only
            text = text.split("</think>", 1)[1]
        return extract_code_block(text)


def main():
    ap = argparse.ArgumentParser(description="COBOLEval: compile, run and check generated COBOL")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="Skylar checkpoint to generate with: a directory or a Hugging Face id")
    src.add_argument("--samples", help="JSONL of {task_id, completion} to score")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="CobolEval.jsonl")
    ap.add_argument("--out", help="with --model: write the generated samples here")
    ap.add_argument("--results", help="write one line per problem: task_id, compiled, passed")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--greedy", action="store_true", help="argmax decoding, as for the published numbers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=1500)
    ap.add_argument("--timeout", type=int, default=10, help="seconds per test run")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} not found: git clone https://github.com/BloopAI/COBOLEval eval/COBOLEval")
    problems = [json.loads(line) for line in open(args.data)][:args.limit]
    cobc = cobc_path()
    env = cobc_env(cobc)
    version = subprocess.run([cobc, "--version"], capture_output=True, text=True, env=env).stdout.splitlines()[0]
    print(version)
    m = re.search(r"(\d+)\.(\d+)", version)
    if not m or (int(m.group(1)), int(m.group(2))) < (3, 2):
        # 3.1 rejects -fformat=variable: every program would "fail to compile" and the scores would read 0
        raise SystemExit("COBOLEval needs GnuCOBOL 3.2 or newer (Ubuntu's apt package is 3.1). Without root:\n"
                         "  micromamba create -y -p tools/gnucobol-env -c conda-forge gnucobol=3.2\n"
                         "  COBC=tools/gnucobol-env/bin/cobc python eval/bin.coboleval.py ...")

    if args.samples:
        completions = {}
        for line in open(args.samples):
            o = json.loads(line)
            completions[o["task_id"]] = o["completion"]
    else:
        gen = LocalModel(args.model, args.greedy, args.seed, args.max_new_tokens)
        completions = {}
        out = open(args.out, "w") if args.out else None
        for i, p in enumerate(problems, 1):
            try:
                completions[p["task_id"]] = construct(p["prompt"], gen(p["prompt"], p["entry_point"]), p["entry_point"])
            except Exception as e:
                print(f"  generation failed on {p['task_id']}: {type(e).__name__}: {str(e)[:80]}")
                completions[p["task_id"]] = ""
            if out:
                out.write(json.dumps({"sample_id": 0, "task_id": p["task_id"],
                                      "completion": completions[p["task_id"]]}) + "\n")
            print(f"  [{i}/{len(problems)}] generated {p['task_id']}")
        if out:
            out.close()

    n = n_compiled = n_passed = 0
    results = open(args.results, "w") if args.results else None
    for p in problems:
        if p["task_id"] not in completions:
            continue
        program = completions[p["task_id"]]
        if args.samples:
            program = construct(p["prompt"], program, p["entry_point"])
        compiled, passed = score_problem(p, program, cobc, env, args.timeout)
        n += 1
        n_compiled += compiled
        n_passed += passed
        print(f"  {p['task_id']:16s} compiled={int(compiled)} passed={int(passed)}")
        if results:
            results.write(json.dumps({"task_id": p["task_id"], "compiled": compiled, "passed": passed}) + "\n")
    if results:
        results.close()
    print("=" * 50)
    print(f"  n={n}  CSR={n_compiled / max(1, n):.3f}  pass@1={n_passed / max(1, n):.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
