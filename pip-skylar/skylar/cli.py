"""`skylar` command-line interface: chat · generate · serve."""
import argparse
import sys

DEFAULT_MODEL = "Sophia-AI/Skylar-236M-Chat"
COBOL_MODEL = "Sophia-AI/Skylar-390M-Cobol"   # the published COBOL specialist (used by `skylar cobol`)


def _add_common(sp):
    sp.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF repo id (es. Sophia-AI/Skylar-236M-Chat) o cartella locale")
    sp.add_argument("--device", default=None, help="cuda | cpu (auto se omesso)")
    sp.add_argument("--system", default=None, help="system prompt (default: nessuno — usa --system per impostarlo)")
    sp.add_argument("--max-new", dest="max_new", type=int, default=512)
    sp.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy deterministico (default)")


def cmd_generate(args):
    from .core import Skylar
    sk = Skylar.load(args.model, device=args.device)
    print(sk.generate(args.prompt, system=args.system, max_new_tokens=args.max_new,
                      temperature=args.temperature, seed=args.seed))


def cmd_chat(args):
    from .core import Skylar
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
    except Exception:
        console = None

    print(f"Carico {args.model} ...", file=sys.stderr)
    sk = Skylar.load(args.model, device=args.device)
    system = args.system  # default None: no forced persona (use --system to steer)
    sys_label = repr(system) if system else "(nessuno — usa --system per impostarlo)"
    head = f"Skylar · {args.model}\ndevice: {sk.device} · system: {sys_label}\n('exit' o Ctrl-D per uscire)"
    if console:
        console.print(Panel(head, title="skylar chat", border_style="cyan"))
    else:
        print(head)

    while True:
        try:
            user = input("\n\033[1m›\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in ("exit", "quit", ":q"):
            break
        if not user:
            continue
        for delta in sk.stream(user, system=system, max_new_tokens=args.max_new,
                               temperature=args.temperature):
            sys.stdout.write(delta)
            sys.stdout.flush()
        print()


def cmd_cobol(args):
    from .core import Skylar
    from .cobol import EXAMPLE_STUB, complete_cobol, syntax_ok
    # `skylar cobol` defaults to the published COBOL specialist (override with --model)
    model = COBOL_MODEL if args.model == DEFAULT_MODEL else args.model
    if args.example:
        stub = EXAMPLE_STUB
    elif args.stub_file:
        stub = open(args.stub_file).read()
    else:
        print("usa --example oppure --stub-file FILE", file=sys.stderr)
        sys.exit(1)
    print(f"Carico {model} ...", file=sys.stderr)
    sk = Skylar.load(model, device=args.device)
    prog = complete_cobol(sk, stub, max_new_tokens=args.max_new, temperature=args.temperature)
    print(prog)
    if args.compile:
        ok = syntax_ok(prog)
        if ok is None:
            print("\n[cobc non installato — salto il check sintassi]", file=sys.stderr)
        else:
            print(f"\n[cobc -fsyntax-only: {'OK, compila ✓' if ok else 'errori di sintassi ✗'}]",
                  file=sys.stderr)


def cmd_embed(args):
    from .embed import SkylarEmbed
    e = SkylarEmbed.load(args.model, device=args.device)
    if args.query is not None and args.docs:
        for doc, score in e.rank(args.query, args.docs):
            print(f"{score:+.4f}  {doc}")
    elif args.text is not None:
        v = e.encode(args.text)
        print(f"dim={len(v)}  ||v||=1.0")
        if args.show_vector:
            print(v)
    else:
        print("usa --text 'frase'  oppure  --query '...' --docs 'a' 'b' 'c'", file=sys.stderr)
        sys.exit(1)


def _serve_embeddings(args):
    import uvicorn
    from fastapi import FastAPI
    from .embed import SkylarEmbed
    e = SkylarEmbed.load(args.model, device=args.device)
    app = FastAPI(title="Skylar Embeddings", version="0.2.3")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": args.model, "device": e.device, "type": "embedding"}

    @app.post("/v1/embeddings")
    def embeddings(body: dict):
        inp = body.get("input", [])
        texts = [inp] if isinstance(inp, str) else list(inp)
        vecs = e.encode(texts)
        if isinstance(vecs[0], float):
            vecs = [vecs]
        data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]
        return {"object": "list", "model": args.model, "data": data}

    print(f"Skylar EMBEDDINGS su http://{args.host}:{args.port}  (POST /v1/embeddings)")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_serve(args):
    try:
        import uvicorn
        from fastapi import FastAPI
        from pydantic import BaseModel
    except Exception:
        print("`skylar serve` richiede gli extra: pip install 'skylar[serve]'", file=sys.stderr)
        sys.exit(1)
    from .core import Skylar
    from .embed import is_embedder

    # auto-detect: an embedder model serves /v1/embeddings, a generative one serves chat/generate
    if is_embedder(args.model):
        return _serve_embeddings(args)

    sk = Skylar.load(args.model, device=args.device)
    app = FastAPI(title="Skylar", version="0.2.3")

    class GenReq(BaseModel):
        prompt: str
        system: str = ""          # no forced persona — caller steers with `system`
        max_new_tokens: int = 512
        temperature: float = 0.0

    @app.get("/health")
    def health():
        return {"status": "ok", "model": args.model, "device": sk.device}

    @app.post("/generate")
    def generate(r: GenReq):
        return {"completion": sk.generate(r.prompt, system=r.system,
                                          max_new_tokens=r.max_new_tokens,
                                          temperature=r.temperature)}

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict):
        msgs = body.get("messages", [])
        system = next((m["content"] for m in msgs if m.get("role") == "system"), "")
        user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
        text = sk.generate(user, system=system,
                           max_new_tokens=body.get("max_tokens", 512),
                           temperature=body.get("temperature", 0.0))
        return {"object": "chat.completion", "model": args.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}]}

    print(f"Skylar serve su http://{args.host}:{args.port}  (POST /generate, /v1/chat/completions)")
    uvicorn.run(app, host=args.host, port=args.port)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="skylar",
        description="Skylar — LLM locali e sovrani, from-scratch: chat + embeddings (specialista COBOL in arrivo).")
    p.add_argument("--version", action="store_true", help="stampa la versione ed esci")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="una risposta singola a un prompt")
    _add_common(g)
    g.add_argument("--prompt", required=True)
    g.add_argument("--seed", type=int, default=None)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("chat", help="REPL interattiva in streaming")
    _add_common(c)
    c.set_defaults(func=cmd_chat)

    co = sub.add_parser("cobol", help="completa uno stub COBOL in un programma intero")
    _add_common(co)
    co.add_argument("--stub-file", default=None, help="file con uno stub COBOLEval-style")
    co.add_argument("--example", action="store_true", help="usa lo stub d'esempio incluso")
    co.add_argument("--compile", action="store_true", help="verifica la sintassi con GnuCOBOL")
    co.set_defaults(func=cmd_cobol, max_new=900)

    e = sub.add_parser("embed", help="embeddings: vettorizza testo o ranka documenti")
    e.add_argument("--model", default="Sophia-AI/Skylar-236M-Embed",
                   help="HF repo id di un modello SkylarEmbedder, o cartella locale")
    e.add_argument("--device", default=None)
    e.add_argument("--text", default=None, help="frase singola da vettorizzare")
    e.add_argument("--query", default=None, help="query per il ranking")
    e.add_argument("--docs", nargs="*", default=None, help="documenti da rankare vs --query")
    e.add_argument("--show-vector", dest="show_vector", action="store_true")
    e.set_defaults(func=cmd_embed)

    s = sub.add_parser("serve", help="server HTTP OpenAI-compatibile (auto-rileva chat vs embeddings) — extra [serve]")
    _add_common(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    if args.version:
        from . import __version__
        print(f"skylar {__version__}")
        return
    if not getattr(args, "func", None):
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
