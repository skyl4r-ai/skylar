"""
Italian preference pairs (prompt, chosen, rejected) for ORPO/SimPO.

Targets the exact failure modes seen in SFT validation — for the SAME prompt,
`chosen` is the grounded / format-correct answer and `rejected` is a realistic
bad answer (hallucinated, format-violating, or non-refusing). Preference
optimization then pushes the policy toward chosen and away from rejected.

This is DIFFERENT from contrastive data: contrastive = (query, positive passage)
for the embedder; preference = (prompt, good answer, bad answer) for the chat
policy. Same base model, different post-training signal.

Fully local, deterministic. Output JSONL: {"system","user","chosen","rejected"}.

  .venv/bin/python data/gen_preference_data.py -o .datasets/pref/preference_it.jsonl
"""
import argparse, json, random
from pathlib import Path

ENTI = ["la Banca d'Italia", "la CONSOB", "la Banca Centrale Europea", "l'IVASS",
        "il Comitato di Basilea", "il Fondo Interbancario di Tutela dei Depositi"]
CITTA = ["Roma", "Milano", "Francoforte", "Basilea", "Bruxelles"]
RUOLI = ["la vigilanza prudenziale", "la tutela del risparmio", "la stabilità finanziaria",
         "la trasparenza delle operazioni", "la prevenzione del riciclaggio"]
MESI = ["gennaio","febbraio","marzo","aprile","maggio","giugno","luglio","agosto",
        "settembre","ottobre","novembre","dicembre"]
CLASSI = {
    "credito": "un mutuo ipotecario per la prima casa",
    "assicurazione": "una polizza RC auto",
    "investimenti": "un fondo comune di investimento",
    "conto": "un conto corrente cointestato",
    "pagamenti": "un bonifico SEPA verso l'estero",
}
NORME = ["il regolamento DORA", "la direttiva PSD2", "il regolamento CRR", "la direttiva MiFID II"]


def msg(system, user, chosen, rejected):
    return {"system": system, "user": user, "chosen": chosen, "rejected": rejected}


def gen_ctxqa(r):  # grounded vs hallucinated
    ente, anno, citta = r.choice(ENTI), r.randint(1850, 2024), r.choice(CITTA)
    cap = ente[0].upper() + ente[1:]
    ctx = f"{cap} è stata istituita nel {anno} e ha sede a {citta}."
    sys = "Rispondi usando SOLO il contesto fornito."
    user = f"Contesto: {ctx}\n\nDomanda: In che anno è stata istituita {ente} e dove ha sede?"
    chosen = f"{cap} è stata istituita nel {anno} e ha sede a {citta}."
    wrong_year = r.randint(1850, 2024)
    wrong_city = r.choice([c for c in CITTA if c != citta])
    rejected = f"{cap} è stata istituita nel {wrong_year} e ha sede a {wrong_city}."
    return msg(sys, user, chosen, rejected)


def gen_classify(r):  # one word vs JSON dump
    label = r.choice(list(CLASSI.keys()))
    sys = "Classifica la richiesta in UNA categoria tra [credito, assicurazione, investimenti, conto, pagamenti]. Una sola parola."
    user = f"Vorrei {CLASSI[label]}."
    rejected = json.dumps({"categoria": label, "dettagli": {"tipo": "CET1", "peso": r.randint(1, 9)}}, ensure_ascii=False)
    return msg(sys, user, label, rejected)


def gen_refuse(r):  # refusal vs invented answer
    fatto = r.choice([
        "L'IBAN identifica univocamente un conto corrente.",
        "Il bonifico SEPA trasferisce denaro in euro.",
        "La carta di debito addebita le spese sul conto.",
    ])
    dom = r.choice(["Qual è il tasso di interesse del mutuo?", "Quanti dipendenti ha la banca?",
                    "Qual è il rendimento dell'investimento?"])
    sys = "Rispondi SOLO se l'informazione è nel contesto, altrimenti scrivi 'Non presente nel contesto'."
    user = f"Contesto: {fatto}\n\nDomanda: {dom}"
    rejected = f"Il valore richiesto è circa {r.randint(2, 15)}%, secondo le condizioni standard di mercato."
    return msg(sys, user, "Non presente nel contesto", rejected)


def gen_summary(r):  # concise vs degenerate repetition
    norma = r.choice(NORME); p = r.sample(RUOLI, 2)
    cap = norma[0].upper() + norma[1:]
    ctx = f"{cap} introduce obblighi in materia di {p[0]} e {p[1]} per gli enti vigilati."
    sys = "Riassumi il contesto in una sola frase."
    chosen = f"{cap} stabilisce obblighi su {p[0]} e {p[1]} per gli enti vigilati."
    rejected = f"{cap} {p[0]} {p[0]} {p[0]} obblighi obblighi {p[1]} {p[1]} vigilati vigilati vigilati."
    return msg(sys, f"Contesto: {ctx}", chosen, rejected)


def gen_extract(r):  # correct JSON vs wrong values
    importo = f"{r.randint(1,9)}.{r.randint(0,999):03d}"
    gg, mm, aaaa = r.randint(1, 28), r.randint(1, 12), r.randint(2023, 2027)
    scad = f"{gg} {MESI[mm-1]} {aaaa}"
    ctx = f"Si comunica che il pagamento di {importo} euro è dovuto entro il {scad}."
    sys = "Estrai importo e scadenza in JSON con chiavi 'importo' e 'scadenza'."
    chosen = json.dumps({"importo": f"{importo} euro", "scadenza": scad}, ensure_ascii=False)
    rejected = json.dumps({"importo": f"{r.randint(1,9)}.{r.randint(0,999):03d} euro",
                           "scadenza": f"{r.randint(1,28)} {r.choice(MESI)} {r.randint(2023,2027)}"}, ensure_ascii=False)
    return msg(sys, ctx, chosen, rejected)


PLAN = [(gen_ctxqa, 1200), (gen_classify, 900), (gen_refuse, 900), (gen_summary, 800), (gen_extract, 1000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    r = random.Random(args.seed)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for gen, n in PLAN:
        seen = set(); made = 0; att = 0
        while made < n and att < n * 40:
            att += 1
            ex = gen(r)
            key = ex["user"] + "||" + ex["chosen"] + "||" + ex["rejected"]
            if key in seen:
                continue
            seen.add(key); rows.append(ex); made += 1
    r.shuffle(rows)
    with open(out, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} preference pairs -> {out}")


if __name__ == "__main__":
    main()
