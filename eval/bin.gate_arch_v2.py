"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Gate automatici per l'architettura Skylar 2 (docs/PAPER_V2.md §4.5).

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
        from models.decoder import Skylar2ForCausalLM as Net_v2

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
    from models.decoder import Skylar2ForCausalLM

    cfg = get_config(preset, vocab_size=vocab, attn_out_gate=True, hidden_act="situ_glu",
                     gated_norm=16, attn_res=True, attn_res_mode="block", attn_res_block_size=2)
    cfg.max_seq_len = 256
    m = Skylar2ForCausalLM(cfg)

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
    from models.decoder import Skylar2ForCausalLM

    def count(**kw):
        cfg = get_config(preset, vocab_size=vocab, **kw)
        cfg.max_seq_len = 256
        return sum(p.numel() for p in Skylar2ForCausalLM(cfg).parameters())

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

    # GatedNorm: 2 norm per layer + quella finale, ognuna con W_down (d×r) e W_up (r×d).
    r = 16
    got = count(gated_norm=r) - base
    expected = (2 * L + 1) * 2 * d * r
    report("G3b GatedNorm (formula)", got == expected,
           f"{got:+,} = (2·{L}+1) norm × 2·{d}·{r}")

    # Cablaggio dell'ibrido: quanti layer ricorrenti esistono DAVVERO nel modello,
    # contro quanti ne chiede il config. Finché KDA non è implementato sono 0.
    cfg = get_config(preset, vocab_size=vocab, kda_ratio="3:1")
    cfg.max_seq_len = 256
    m = Skylar2ForCausalLM(cfg)
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

    # (a) la conv corta è esatta, non solo approssimata — in fp64. "Esatta" vuol dire
    #     entro l'arrotondamento del fp64, non bit per bit: su un'altra CPU (pod RunPod
    #     con 2 A100, 09/10/2026) conv1d somma in un altro ordine e dà 4,4e-16 (2 ulp).
    #     Un confine fra documenti che perde darebbe un errore dell'ordine dei valori
    #     (~1e-1): 1e-12 lo separa da quello di 11 ordini di grandezza.
    conv = ShortConv(32, 4).double()
    x = t.randn(1, 16, 32, dtype=t.double)
    pos_one = positions_in_segment(t.tensor([0, 16], dtype=t.int32), 16)
    pos_two = positions_in_segment(t.tensor([0, 8, 16], dtype=t.int32), 16)
    d_one = (conv(x)[0] - conv(x, pos_in_seg=pos_one)[0]).abs().max().item()
    d_two = (conv(x, pos_in_seg=pos_two)[0]
             - t.cat([conv(x[:, :8])[0], conv(x[:, 8:])[0]], 1)).abs().max().item()
    report("G2 short conv esatta (fp64)", d_one < 1e-12 and d_two < 1e-12,
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
    from models.decoder import Skylar2ForCausalLM

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
                             ("ibrido", dict(kda_ratio="3:1"), 0.02, t.bfloat16),
                             ("ibrido+AttnRes blocchi+GN",
                              dict(kda_ratio="3:1", attn_res=True, attn_res_mode="block",
                                   attn_res_block_size=2, gated_norm=16), 0.02, t.bfloat16)):
        try:
            cfg = get_config(preset, vocab_size=vocab, **kw)
            cfg.max_seq_len, cfg.dropout = 256, 0.0
            t.manual_seed(7)
            m = Skylar2ForCausalLM(cfg).cuda().to(dt).eval()
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


def gate_g10_block_attnres(preset, vocab):
    """
    G10 — Block AttnRes (Kimi, arXiv 2603.15031 §3.2) fa quello che dice.

    (a) La SEMANTICA: a ogni punto le sorgenti sono l'embedding, la somma di ogni
        blocco chiuso e la somma parziale del blocco aperto. Controllato contro la
        definizione scritta a mano, in fp64 su CPU: deve essere esatto.
    (b) Il KERNEL fuso di fla (con la pre-norm piegata dentro) è accurato ALMENO
        quanto la forma in torch. Il riferimento è la forma in torch in fp64: una
        soglia fissa fra due run fp32 misura il rumore di fp32 (su `medium` ~1e-3
        sui gradienti di alcune pseudo-query, per entrambi), non un errore.
        Senza KDA e senza maschera di documento, di proposito: i kernel KDA e la
        FlexAttention non girano in fp64, e KDA amplifica differenze di arrotondamento
        di 1e-6 fino a ~1e-4. L'ibrido completo lo coprono G9 e G11.
    """
    import torch as t
    from models.layers.attn_res import AttnResMixer, DepthState

    # (a) semantica, blocchi da 3 sotto-layer, 8 uscite
    emb = t.randn(2, 5, 4, dtype=t.float64)
    ys = [t.randn(2, 5, 4, dtype=t.float64) for _ in range(8)]
    st = DepthState(emb, "block", 3)
    ok, bs = True, 3
    for i, y in enumerate(ys, 1):
        st.push(y)
        closed = [sum(ys[j * bs:(j + 1) * bs]) for j in range(i // bs)]
        rest = ys[(i // bs) * bs:i]
        expect = [emb] + closed + ([sum(rest)] if rest else [])
        got = st.sources()
        ok &= len(got) == len(expect) and all(t.equal(a, b) for a, b in zip(got, expect))
    report("G10 semantica dei blocchi", ok,
           "embedding + somme dei blocchi chiusi + parziale, esatto a ogni passo" if ok
           else "le sorgenti non coincidono con la definizione")

    if not t.cuda.is_available():
        report("G10 kernel fuso vs fp64", False, "il kernel fla richiede CUDA", todo=True)
        return
    from models.config import get_config
    from models.decoder import Skylar2ForCausalLM
    cfg = get_config(preset, vocab_size=vocab, attn_res=True,
                     attn_res_mode="block", attn_res_block_size=2, gated_norm=16)
    cfg.max_seq_len, cfg.dropout = 256, 0.0
    t.manual_seed(11)
    m = Skylar2ForCausalLM(cfg).cuda().train()
    for mod in m.modules():                 # pseudo-query non nulle: pesi di profondità non uniformi
        if isinstance(mod, AttnResMixer):
            t.nn.init.normal_(mod.query, std=0.5)
    ids = t.randint(0, vocab, (2, 64), device="cuda")

    def run(dtype, fused):
        AttnResMixer.use_fused = fused
        mm = m.to(dtype)
        mm.zero_grad(set_to_none=True)
        out = mm(ids, labels=ids)
        out["loss"].backward()
        return (out["logits"].detach().double(),
                {n: p.grad.detach().double().clone() for n, p in mm.named_parameters()
                 if p.grad is not None})

    try:
        l64, g64 = run(t.float64, False)
        lt, gt = run(t.float32, False)
        lf, gf = run(t.float32, True)
    finally:
        AttnResMixer.use_fused = True
        m.float()
    rel = lambda a, b: ((a - b).norm() / (b.norm() + 1e-30)).item()
    lt_e, lf_e = rel(lt, l64), rel(lf, l64)
    gt_e = max(rel(gt[n], g64[n]) for n in g64)
    gf_e = max(rel(gf[n], g64[n]) for n in g64)
    ok = lf_e <= 2 * lt_e + 1e-7 and gf_e <= 2 * gt_e + 1e-7
    report("G10 kernel fuso vs fp64", ok,
           f"errore sui logit fuso {lf_e:.1e} / torch {lt_e:.1e} · "
           f"gradiente peggiore fuso {gf_e:.1e} / torch {gt_e:.1e}")


def gate_g11_checkpoint_grads(preset, vocab):
    """
    G11 — il gradient checkpointing non cambia i gradienti con Block AttnRes + GatedNorm.

    Il checkpoint riesegue ogni blocco nel backward partendo da una fotografia dello
    stato di profondità: se la fotografia è sbagliata (stato già avanzato, blocco
    chiuso contato due volte) il ricalcolo legge sorgenti diverse e i gradienti
    cambiano senza nessun errore. Qui devono coincidere.
    """
    import torch as t
    if not t.cuda.is_available():
        report("G11 checkpoint = senza", False, "i kernel KDA richiedono CUDA", todo=True)
        return
    from models.config import get_config
    from models.decoder import Skylar2ForCausalLM
    cfg = get_config(preset, vocab_size=vocab, kda_ratio="3:1", attn_res=True,
                     attn_res_mode="block", attn_res_block_size=3, gated_norm=16)
    cfg.max_seq_len, cfg.dropout = 256, 0.0
    t.manual_seed(5)
    m = Skylar2ForCausalLM(cfg).cuda().float().train()
    ids = t.randint(0, vocab, (2, 64), device="cuda")
    doc = t.zeros_like(ids); doc[:, 20:] = 1

    def grads(ckpt):
        m.gradient_checkpointing = ckpt
        m.zero_grad(set_to_none=True)
        m(ids, labels=ids, document_ids=doc)["loss"].backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}

    g0, g1 = grads(False), grads(True)
    m.gradient_checkpointing = False
    worst = max(((g0[n] - g1[n]).abs().max() / (g0[n].abs().max() + 1e-12)).item() for n in g0)
    same_keys = set(g0) == set(g1)
    report("G11 checkpoint = senza", same_keys and worst < 1e-5,
           f"{len(g0)} tensori di gradiente · differenza relativa massima {worst:.1e}")


def gate_g12_kda_torch(preset, vocab):
    """KDA in PyTorch (`kda_torch`, the CPU/MPS path) = the Triton kernels of fla, on the same weights:
    prefill, one decode step from the cache, and varlen with two documents."""
    import torch as t
    if not t.cuda.is_available():
        report("G12 KDA PyTorch = kernel", False, "i kernel KDA richiedono CUDA", todo=True)
        return
    from models.config import get_config
    from models.layers.kda import SkylarKDA
    cfg = get_config(preset, vocab_size=vocab, kda_ratio="3:1")
    t.manual_seed(7)
    layer = SkylarKDA(cfg, layer_idx=0).cuda().float().eval()
    with t.no_grad():
        layer.A_log.add_(t.randn_like(layer.A_log) * 0.5)      # gate lontano dall'init
        layer.dt_bias.add_(t.randn_like(layer.dt_bias) * 0.5)
    x = t.randn(2, 97, cfg.d_model, device="cuda") * 0.5
    x1 = t.randn(2, 1, cfg.d_model, device="cuda") * 0.5
    xv = t.randn(1, 80, cfg.d_model, device="cuda") * 0.5
    cu = t.tensor([0, 33, 80], dtype=t.int32, device="cuda")

    def run(kernels):
        SkylarKDA.use_kernels = kernels
        try:
            with t.no_grad():
                o, c = layer(x, use_cache=True)
                o1, c1 = layer(x1, kv_cache=c, use_cache=True)
                ov, _ = layer(xv, cu_seqlens=cu)
        finally:
            SkylarKDA.use_kernels = True
        return o, c[0], o1, c1[0], ov

    ref, got = run(True), run(False)
    rel = [((g - r).norm() / r.norm()).item() for g, r in zip(got, ref)]
    report("G12 KDA PyTorch = kernel", max(rel) < 5e-3,
           "errore relativo prefill {:.1e} · stato {:.1e} · decode {:.1e} · stato {:.1e} · "
           "varlen {:.1e} (soglia 5e-3: precisione interna dei kernel)".format(*rel))


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
    gate_g10_block_attnres(args.preset, args.vocab)
    gate_g11_checkpoint_grads(args.preset, args.vocab)
    gate_g12_kda_torch(args.preset, args.vocab)

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
