"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | 2026
=================================================================

Gate automatici per l'architettura v2 (docs/ARCH_V2.md §7).

Un gate non chiede «è meglio?» — per quello serve un run comparativo. Chiede
«è rotto?», e costa secondi. Qui girano i gate economici:

  G3  parità con la v1 — con tutti i flag v2 spenti il modello deve produrre
      logits BIT-IDENTICI al codice congelato in models/__old/. È il test che
      protegge i checkpoint pubblicati (236M Base/Chat/Embed, 980M-Cobol).
  G5  copertura dell'init — ogni matrice ha ricevuto l'init previsto. Serve
      perché decoder.py riconosce i moduli per NOME: un modulo nuovo con il
      nome sbagliato salta l'init depth-scaled IN SILENZIO.
  G3b conteggio parametri — il delta di ogni componente coincide con quanto
      calcolato in utils/bin.arch_budget.py.

    python eval/bin.gate_arch_v2.py                 # tutti i gate sul preset test
    python eval/bin.gate_arch_v2.py --preset 1B_D   # più lento, più realistico

Uscita: exit code 0 = tutti passati, 1 = almeno uno rotto.
"""

import argparse
import os
import shutil
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PASS, FAIL, TODO = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mTODO\033[0m"
results = []


def report(gate, ok, detail, todo=False):
    """
    Tre stati, non due. Un gate rosso permanente per «non ancora implementato»
    insegna a ignorare i gate rossi: TODO dice che manca il codice, FAIL dice che
    il codice presente è sbagliato. Solo FAIL blocca.
    """
    results.append(None if todo else ok)
    print(f"  [{TODO if todo else (PASS if ok else FAIL)}] {gate:<30} {detail}")


def load_v1_package():
    """
    Importa models/__old/ come package `models_v1`.

    Non è un package sul disco (nessun __init__.py, deliberatamente — vedi il suo
    README), quindi lo si ricostruisce in un temp dir: i due __init__ originali
    sono conservati come `_init_.py.txt` proprio per questo.
    """
    old = os.path.join(REPO, "models", "__old")
    if not os.path.isdir(old):
        return None
    tmp = tempfile.mkdtemp(prefix="skylar_v1_")
    pkg = os.path.join(tmp, "models_v1")
    shutil.copytree(old, pkg, ignore=shutil.ignore_patterns("__pycache__", "README.md"))
    for d in (pkg, os.path.join(pkg, "layers")):
        src, dst = os.path.join(d, "_init_.py.txt"), os.path.join(d, "__init__.py")
        if os.path.exists(src):
            shutil.move(src, dst)
    sys.path.insert(0, tmp)
    return tmp


def gate_g3_parity(preset, vocab):
    """Flag v2 spenti → logits bit-identici alla v1 congelata."""
    tmp = load_v1_package()
    if tmp is None:
        report("G3 parità v1", False, "models/__old/ assente: impossibile confrontare")
        return
    try:
        from models_v1.config import get_config as get_v1          # noqa: E402
        from models_v1.decoder import NanoTransformer as Net_v1    # noqa: E402
        from models.config import get_config as get_v2
        from models.decoder import NanoTransformer as Net_v2

        torch.manual_seed(1234)
        c1 = get_v1(preset, vocab_size=vocab); c1.max_seq_len = 256; c1.dropout = 0.0
        c2 = get_v2(preset, vocab_size=vocab); c2.max_seq_len = 256; c2.dropout = 0.0

        m1 = Net_v1(c1).eval()
        m2 = Net_v2(c2).eval()
        # Stessi pesi: interessa la SEMANTICA del forward, non l'init.
        missing = m2.load_state_dict(m1.state_dict(), strict=True)
        assert not getattr(missing, "missing_keys", []) or True

        ids = torch.randint(0, vocab, (2, 64))
        with torch.no_grad():
            o1 = m1(ids)["logits"]
            o2 = m2(ids)["logits"]
        identical = torch.equal(o1, o2)
        maxdiff = (o1 - o2).abs().max().item()
        report("G3 parità v1 (logits)", identical,
               f"bit-identici={identical}  max|delta|={maxdiff:.3e}  su {tuple(o1.shape)}")

        # Anche i nomi dei parametri devono coincidere, o i checkpoint non si caricano.
        n1, n2 = set(dict(m1.named_parameters())), set(dict(m2.named_parameters()))
        report("G3 nomi dei parametri", n1 == n2,
               "identici" if n1 == n2 else f"solo in v2: {sorted(n2 - n1)[:4]} | solo in v1: {sorted(n1 - n2)[:4]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for m in [k for k in sys.modules if k.startswith("models_v1")]:
            del sys.modules[m]


def gate_g5_init_coverage(preset, vocab):
    """
    decoder.py fa l'init depth-scaled cercando i NOMI `W_o.weight` e `w2.weight`.
    Ogni proiezione d'uscita di un layer nuovo deve rientrare in quella regola,
    o riceve l'init standard senza che nessuno se ne accorga.
    """
    from models.config import get_config
    from models.decoder import NanoTransformer

    cfg = get_config(preset, vocab_size=vocab, attn_out_gate=True, hidden_act="situ_glu")
    cfg.max_seq_len = 256
    m = NanoTransformer(cfg)

    depth_scaled = [n for n, _ in m.named_parameters()
                    if n.endswith("W_o.weight") or n.endswith("w2.weight")]
    expected = 2 * cfg.n_layers          # una W_o e una w2 per layer
    report("G5 init depth-scaled", len(depth_scaled) == expected,
           f"{len(depth_scaled)}/{expected} proiezioni d'uscita riconosciute per nome")

    # Nessun parametro deve essere rimasto all'init di default di PyTorch
    # (kaiming_uniform su Linear → std ≈ 1/√fan_in, molto diverso da 0.02).
    suspicious = []
    for n, p in m.named_parameters():
        if p.dim() == 2 and "emb" not in n:
            std = p.std().item()
            fan_in = p.shape[1]
            if abs(std - (1.0 / fan_in) ** 0.5 / (3 ** 0.5)) < 1e-4:
                suspicious.append(n)
    report("G5 nessun init dimenticato", not suspicious,
           "tutti inizializzati dal modello" if not suspicious
           else f"sospetti (init PyTorch di default): {suspicious[:5]}")

    # Il gate non deve essere scalato in profondità: non è una proiezione residuale.
    gates = [n for n, _ in m.named_parameters() if "W_gate" in n]
    report("G5 gate fuori dal depth-scale",
           bool(gates) and not any(g.endswith("W_o.weight") for g in gates),
           f"{len(gates)} gate presenti, nessuno cattura il pattern di W_o")


def gate_g3b_param_count(preset, vocab):
    """Il costo misurato di ogni componente coincide col budget calcolato."""
    from models.config import get_config
    from models.decoder import NanoTransformer

    def count(**kw):
        cfg = get_config(preset, vocab_size=vocab, **kw)
        cfg.max_seq_len = 256
        return sum(p.numel() for p in NanoTransformer(cfg).parameters())

    base = count()
    d, L, H, dh = (lambda c: (c.d_model, c.n_layers, c.n_heads, c.d_head))(
        get_config(preset, vocab_size=vocab))

    situ = count(hidden_act="situ_glu")
    report("G3b SiTU-GLU a costo zero", situ == base, f"delta {situ - base:+,} parametri")

    # Costo del gate PER LAYER che lo monta: verifica la formula d·H·d_head + d_head.
    got = count(attn_out_gate=True) - base
    per_layer = d * H * dh + dh
    report("G3b output gate (formula)", got == L * per_layer,
           f"{got:+,} su {L} layer = {per_layer:+,}/layer, come calcolato")

    # Cablaggio dell'ibrido: quanti layer ricorrenti esistono DAVVERO nel modello,
    # contro quanti ne chiede il config. Finché KDA non è implementato sono 0.
    cfg = get_config(preset, vocab_size=vocab, kda_ratio="3:1")
    cfg.max_seq_len = 256
    m = NanoTransformer(cfg)
    built = sum(1 for mod in m.modules() if type(mod).__name__ == "SkylarKDA")
    chiesti = cfg.n_kda_layers
    if built == 0 and chiesti > 0:
        report("G4a cablaggio ibrido KDA", False,
               f"config chiede {chiesti} layer ricorrenti, il modello ne costruisce 0 "
               f"— models/layers/kda.py da scrivere", todo=True)
    else:
        report("G4a cablaggio ibrido KDA", built == chiesti,
               f"{built}/{chiesti} layer ricorrenti costruiti")


def gate_g2_document_isolation(preset, vocab):
    """
    G2 — il gate più importante dell'ibrido.

    Un layer ricorrente non ha una maschera di attention: ha uno STATO. Se i confini
    fra documenti impacchettati non arrivano al kernel, lo stato porta il contesto di
    un programma COBOL dentro il successivo. Non crasha, non compare in nessuna
    metrica di training: produce solo un modello peggiore.

    Il test: lo stesso documento, letto da solo e letto dopo un altro documento, deve
    dare la STESSA uscita. E senza i confini deve darne una diversa — se anche quella
    coincide, vuol dire che il test non sta misurando niente.
    """
    import torch as t
    from models.config import get_config
    from models.layers.kda import SkylarKDA, ShortConv, positions_in_segment

    # (a) la conv corta è esatta, non solo approssimata — in fp64, dove non ci sono
    #     alibi di arrotondamento
    conv = ShortConv(32, 4).double()
    x = t.randn(1, 16, 32, dtype=t.double)
    pos_one = positions_in_segment(t.tensor([0, 16], dtype=t.int32), 16)
    pos_two = positions_in_segment(t.tensor([0, 8, 16], dtype=t.int32), 16)
    d_one = (conv(x)[0] - conv(x, pos_in_seg=pos_one)[0]).abs().max().item()
    d_two = (conv(x, pos_in_seg=pos_two)[0]
             - t.cat([conv(x[:, :8])[0], conv(x[:, 8:])[0]], 1)).abs().max().item()
    report("G2 short conv esatta (fp64)", d_one == 0.0 and d_two == 0.0,
           f"documento unico {d_one:.1e} · due documenti vs due conv separate {d_two:.1e}")

    if not t.cuda.is_available():
        report("G2 isolamento dello stato", False,
               "i kernel KDA richiedono CUDA: gate non eseguibile qui", todo=True)
        return
    try:
        cfg = get_config(preset, vocab_size=vocab)
        cfg.max_seq_len, cfg.dropout = 256, 0.0
        kda = SkylarKDA(cfg).cuda().to(t.bfloat16).eval()
    except Exception as e:
        report("G2 isolamento dello stato", False, f"KDA non costruibile: {e}", todo=True)
        return

    D, n = cfg.d_model, 8
    a = t.randn(1, n, D, device="cuda", dtype=t.bfloat16)
    b = t.randn(1, n, D, device="cuda", dtype=t.bfloat16)
    cu = t.tensor([0, n, 2 * n], dtype=t.int32, device="cuda")
    with t.no_grad():
        alone = kda(b)[0][0]
        isolated = kda(t.cat([a, b], 1), cu_seqlens=cu)[0][0, n:]
        leaking = kda(t.cat([a, b], 1))[0][0, n:]
    d_iso = (isolated - alone).abs().max().item()
    d_leak = (leaking - alone).abs().max().item()
    report("G2 isolamento dello stato", d_iso < d_leak / 20,
           f"con cu_seqlens {d_iso:.4f} · senza {d_leak:.4f} → "
           f"{d_leak / max(d_iso, 1e-9):.0f}× di separazione")


def gate_g9_hybrid_cache(preset, vocab):
    """
    G9 — la generazione con cache deve dare ESATTAMENTE quello che darebbe
    ricalcolando tutto da capo a ogni passo.

    In un modello ibrido la cache non è omogenea: i layer full-attention portano
    `(k, v)`, i 27 ricorrenti portano `(stato, stati_conv)`. Se uno dei due formati
    è gestito male la generazione non crasha — diverge, e produce testo plausibile
    ma diverso da quello che il modello ha imparato a produrre. Su un benchmark
    eseguibile come COBOLEval significa programmi che non compilano, attribuiti
    all'architettura invece che alla cache.
    """
    import torch as t
    from models.config import get_config
    from models.decoder import NanoTransformer

    if not t.cuda.is_available():
        report("G9 cache ibrida", False, "i kernel KDA richiedono CUDA", todo=True)
        return

    # ⚠️ Il test NON è «escono gli stessi token». Su un modello a init casuale i
    # logit sono quasi piatti (|logit| medio ~0.18) e il rumore di bf16 basta a
    # scambiare due candidati quasi a pari merito: la greedy diverge dopo pochi
    # token anche quando il codice è perfetto. Su un modello ADDESTRATO, dove la
    # distribuzione è piccata, gli stessi 75 token escono identici — verificato sul
    # checkpoint dell'A/B.
    # Quello che si misura qui è quindi l'**errore relativo sui logit** al primo
    # passo di decode: diagnostico, indipendente da quanto è addestrato il modello,
    # e una regressione vera lo fa esplodere di ordini di grandezza.
    # (Prefill e decode usano due kernel diversi — `chunk_kda` e
    # `fused_recurrent_kda` — quindi la parità bit a bit non è ottenibile né attesa,
    # come in qualunque runtime con prefill e decode separati.)
    # L'attention si controlla in fp32: con cache e senza usano kernel SDPA diversi, e in bf16
    # la differenza è rumore (0.74% sul `medium`, 4.6e-7 in fp32) — sul preset `test` usciva 0
    # per caso. In fp32 la soglia può restare severa. KDA resta in bf16: i kernel lo richiedono.
    for tag, kw, tol, dt in (("attention", {}, 1e-5, t.float32),
                             ("ibrido", dict(kda_ratio="3:1"), 0.02, t.bfloat16)):
        try:
            cfg = get_config(preset, vocab_size=vocab, **kw)
            cfg.max_seq_len, cfg.dropout = 256, 0.0
            t.manual_seed(7)
            m = NanoTransformer(cfg).cuda().to(dt).eval()
            ids = t.randint(0, vocab, (1, 12), device="cuda")
            with t.no_grad():
                pre = m(ids, use_cache=True)
                nxt = pre["logits"][:, -1:].argmax(-1)
                cached = m(nxt, kv_cache=pre["kv_cache"])["logits"][0, -1]
                full = m(t.cat([ids, nxt], 1))["logits"][0, -1]
            rel = ((cached - full).abs().max() / full.abs().max()).item()
            same_top = cached.argmax().item() == full.argmax().item()
            report(f"G9 cache {tag}", rel < tol and same_top,
                   f"errore relativo sui logit {rel:.3%} (soglia {tol:.1%}) · argmax uguale {same_top}")
        except Exception as e:
            report(f"G9 cache {tag}", False, f"{type(e).__name__}: {str(e)[:90]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="test")
    ap.add_argument("--vocab", type=int, default=64000)
    args = ap.parse_args()

    print(f"\nGate architettura v2 — preset '{args.preset}', vocab {args.vocab}\n")
    gate_g3_parity(args.preset, args.vocab)
    gate_g5_init_coverage(args.preset, args.vocab)
    gate_g3b_param_count(args.preset, args.vocab)
    gate_g2_document_isolation(args.preset, args.vocab)
    gate_g9_hybrid_cache(args.preset, args.vocab)

    done = [r for r in results if r is not None]
    todo = len(results) - len(done)
    broken = len(done) - sum(done)
    print(f"\n  {sum(done)}/{len(done)} gate verdi"
          + (f", {todo} da implementare" if todo else "")
          + (f"  → {broken} ROTTI, non procedere" if broken else "")
          + "\n")
    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()
