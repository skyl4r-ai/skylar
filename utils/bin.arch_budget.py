"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Budget di parametri e learning rate per l'architettura Skylar 2 (docs/PAPER_V2.md).

Risponde a due domande che NON vanno stimate a occhio prima di spendere
GPU-settimane (regola delle fonti, CLAUDE.md):

  1. quanto costa in parametri ogni componente v2 (ibrido KDA, AttnRes a
     blocchi, GatedNorm, output gate, SiTU-GLU, MTP) su ciascun preset;
  2. quale peak LR esce dalla scaling-law 2601.05049 ancorata al nostro
     punto noto (980M @ 3e-4 su 20.37B token), dato N e D reali.

Il conto dei parametri è ARITMETICO, non misurato istanziando il modello:
serve a decidere PRIMA di scrivere il codice. Dopo, `--check` confronta il
baseline calcolato con quello reale di `get_config` + `Skylar2ForCausalLM`.

    python utils/bin.arch_budget.py                       # tabella completa
    python utils/bin.arch_budget.py --tokens 300e9        # LR a 300B token
    python utils/bin.arch_budget.py --preset 4b --check   # verifica vs modello reale
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────
# Preset. `2b` NON esiste ancora in models/config.py: è la proposta v2.
# ─────────────────────────────────────────────────────────────────────
PRESETS = {
    "1B_D": dict(d_model=1536, n_layers=36, n_heads=12, n_kv_heads=4, d_head=128, d_ff=4096),
    "2b":   dict(d_model=2048, n_layers=36, n_heads=16, n_kv_heads=4, d_head=128, d_ff=7168),
    "4b":   dict(d_model=2560, n_layers=36, n_heads=32, n_kv_heads=8, d_head=128, d_ff=9728),
}
VOCAB = 64000

# Punto di ancoraggio: il 980M che ha trainato SANO (val ppl 3.01, gnorm 0.07).
# Le costanti assolute della formula 2601.05049 vengono da un altro codebase
# (batch 4M tok, seq 4096) → si usa il RAPPORTO, non il valore assoluto.
ANCHOR_N, ANCHOR_D, ANCHOR_LR = 0.98e9, 20.37e9, 3e-4
EXP_N, EXP_D = -0.2219, -0.3509     # arXiv 2601.05049, fit R² ≈ 0.96


def baseline(p, vocab=VOCAB, tie=True):
    """Parametri del transformer v1 (GQA + SwiGLU + QK-Norm, pesi legati)."""
    d, L, H, kv, dh, dff = p["d_model"], p["n_layers"], p["n_heads"], p["n_kv_heads"], p["d_head"], p["d_ff"]
    q_dim, kv_dim = H * dh, kv * dh
    attn = d * q_dim + 2 * d * kv_dim + q_dim * d + 2 * dh    # Wq,Wk,Wv,Wo + q_norm,k_norm
    ffn = 3 * d * dff                                          # SwiGLU: w1,w2,w3
    norms = 2 * d                                              # ln1, ln2
    emb = vocab * d * (1 if tie else 2)
    return dict(attn=attn, ffn=ffn, per_layer=attn + ffn + norms,
                total=L * (attn + ffn + norms) + emb + d, emb=emb)


def kda_params(p, n_kda_heads, gate="lowrank", dh=128):
    """
    KDA (variante Kimi Linear / fla.ops.kda). Nessun GQA: q,k,v,o tutte d × key_dim.
      q,k,v,o        4 · d · key_dim
      short conv     3 · key_dim · 4
      b_proj (beta)  d · H
      f_a/f_b        d·128 + 128·key_dim        (log-decay per canale, low-rank)
      g gate         d·128 + 128·key_dim  (lowrank)  |  d · key_dim  (fullrank)
      A_log, dt_bias 2 · H ;  o_norm  key_dim
    """
    d, H = p["d_model"], n_kda_heads
    key_dim = H * dh
    tot = 4 * d * key_dim                                           # W_q, W_k, W_v, W_o
    tot += 3 * key_dim * 4                                          # short conv depthwise, kernel 4
    tot += d * H                                                    # b_proj (beta)
    tot += d * dh + dh * key_dim                                    # f_a, f_b (decay low-rank)
    tot += (d * dh + dh * key_dim + key_dim) if gate == "lowrank" else (d * key_dim + key_dim)
    tot += H + key_dim + dh                                         # A_log, dt_bias, o_norm
    return tot, key_dim


def h_kda_parity(p):
    """
    Regola di dimensionamento: H_kda = (n_heads + n_kv_heads) / 2.

    L'attention GQA costa 2·d·dh·(H+kv); KDA costa 4·d·dh·H_kda. Uguagliando i
    termini dominanti i due si annullano → l'ibrido non gonfia il modello.
    """
    return (p["n_heads"] + p["n_kv_heads"]) / 2


def lr_from_scaling_law(n_params, tokens):
    """arXiv 2601.05049 usata come RAPPORTO dal punto noto (vedi ANCHOR_*)."""
    return ANCHOR_LR * (n_params / ANCHOR_N) ** EXP_N * (tokens / ANCHOR_D) ** EXP_D


def components(name, p, ratio_kda=(27, 9), attnres_points=None, gate="lowrank", with_mtp=False,
               gated_norm=16):
    """Delta di parametri di ogni componente v2, in assoluto e in % sul baseline."""
    base = baseline(p)
    d, L = p["d_model"], p["n_layers"]
    n_kda, n_attn = ratio_kda
    assert n_kda + n_attn == L, f"{n_kda}+{n_attn} != {L} layer"
    # Punti di lettura: due per layer + quello prima della testa, meno il primo, che ha
    # una sola sorgente (l'embedding) e nessun parametro.
    pts = attnres_points if attnres_points else 2 * L

    hk = h_kda_parity(p)
    assert hk == int(hk), f"H_kda non intero per {name}: {hk}"
    kda, key_dim = kda_params(p, int(hk), gate=gate)

    # MTP stile DeepSeek-V3: un blocco completo + proiezione Linear(2d, d) + 2 norm.
    mtp = base["per_layer"] + 2 * d * d + 2 * d

    out = [
        ("AttnRes (blocchi, %d punti)" % pts, pts * 2 * d),
        ("Ibrido KDA %d:%d, H_kda=%d" % (n_kda // math.gcd(n_kda, n_attn),
                                         n_attn // math.gcd(n_kda, n_attn), hk),
         n_kda * (kda - base["attn"])),
        # GatedNorm (2601.22966): gate a basso rango dopo ogni pre-norm e la norm finale,
        # Linear(d, r) + Linear(r, d) senza bias, (2L+1) norm.
        ("GatedNorm (rango %d)" % gated_norm, (2 * L + 1) * 2 * d * gated_norm),
        # Gate per-testa: Linear(d, H) + o_norm. La variante per-canale costa 127x
        # di piu' e su 5 seed e' indistinguibile (docs/PAPER_V2.md §3.5).
        ("Output gate per-testa (%d layer)" % n_attn,
         n_attn * (d * p["n_heads"] + p["d_head"])),
        ("SiTU-GLU (al posto di SwiGLU)", 0),
        # MTP TAGLIATO dopo misura (docs/PAPER_V2.md §3.7): zero beneficio sulla CE,
        # -11% di velocita'. Il conto resta calcolabile con --with_mtp per riaprire
        # la decisione se un giorno cambiassero le condizioni.
    ] + ([("MTP (1 layer + proiezione)", mtp)] if with_mtp else [])
    return base, out, dict(h_kda=int(hk), key_dim=key_dim, kda_per_layer=kda,
                           state_bytes=int(hk) * 128 * 128 * 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=300e9)
    ap.add_argument("--preset", default=None, help="limita a un preset")
    ap.add_argument("--gate", default="lowrank", choices=["lowrank", "fullrank"])
    ap.add_argument("--with_mtp", action="store_true",
                    help="includi MTP nel conto (tagliato dopo misura, docs/PAPER_V2.md §3.7)")
    ap.add_argument("--gated_norm", type=int, default=16, help="rango di GatedNorm (0 = senza)")
    ap.add_argument("--check", action="store_true",
                    help="confronta il baseline calcolato col modello reale (serve torch)")
    args = ap.parse_args()

    names = [args.preset] if args.preset else list(PRESETS)
    summary = []

    for name in names:
        p = PRESETS[name]
        base, comps, kinfo = components(name, p, gate=args.gate, with_mtp=args.with_mtp,
                                        gated_norm=args.gated_norm)
        b = base["total"]

        print("\n" + "=" * 78)
        print(f"  {name}   d={p['d_model']} L={p['n_layers']} H={p['n_heads']}/{p['n_kv_heads']} "
              f"d_ff={p['d_ff']}   vocab {VOCAB}")
        print("=" * 78)
        print(f"  baseline v1                        {b/1e6:>10.2f}M    "
              f"(attn {base['attn']/1e6:.2f}M + ffn {base['ffn']/1e6:.2f}M per layer)")
        print(f"  KDA a parità: H_kda={kinfo['h_kda']}, key_dim={kinfo['key_dim']}, "
              f"{kinfo['kda_per_layer']/1e6:.2f}M/layer, stato {kinfo['state_bytes']/1024:.0f} KB/seq")
        print("  " + "-" * 74)
        tot = 0
        for label, delta in comps:
            tot += delta
            print(f"  {label:<38} {delta/1e6:>+9.2f}M   {100*delta/b:>+7.2f}%")
        print("  " + "-" * 74)
        v2 = b + tot
        print(f"  {'TOTALE v2':<38} {v2/1e6:>10.2f}M   {100*tot/b:>+7.2f}%")

        lr_v1 = lr_from_scaling_law(b, args.tokens)
        lr_v2 = lr_from_scaling_law(v2, args.tokens)
        print(f"  peak LR @ {args.tokens/1e9:.0f}B token (2601.05049, rapporto su 980M@3e-4): "
              f"{lr_v2:.2e}")
        summary.append((name, b, v2, lr_v2))

        if args.check:
            from models.config import get_config
            cfg = get_config(name if name in ("1B_D", "4b") else "1B_D", vocab_size=VOCAB)
            if name in ("1B_D", "4b"):
                from models.decoder import Skylar2ForCausalLM
                real = sum(x.numel() for x in Skylar2ForCausalLM(cfg).parameters())
                delta = real - b
                flag = "OK" if abs(delta) < 0.001 * b else "DISCREPANZA"
                print(f"  [check] modello reale {real/1e6:.2f}M vs calcolato {b/1e6:.2f}M "
                      f"→ {delta:+,} ({flag})")

    print("\n" + "=" * 78)
    print("  RIEPILOGO — stessa pipeline, cambiano solo questi numeri")
    print("=" * 78)
    print(f"  {'preset':<8} {'v1':>12} {'v2':>12} {'delta':>8}   {'peak LR':>10}")
    for name, b, v2, lr in summary:
        print(f"  {name:<8} {b/1e6:>11.2f}M {v2/1e6:>11.2f}M {100*(v2-b)/b:>+7.2f}%   {lr:>10.2e}")
    print(f"\n  D = {args.tokens/1e9:.0f}B token per TUTTI i modelli (dataset intero).")
    print("  ⚠ i LR sono di AdamW. Muon (update in scala AdamW) usa 2x questo valore: "
          "docs/PAPER_V2.md §7.")


if __name__ == "__main__":
    main()
