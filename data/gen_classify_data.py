"""
Synthetic Italian intent-classification dataset (banking) for SkylarClassifier.

Demonstrates the discriminative / non-generative path: text → single class label.
5 classes (credito, assicurazione, investimenti, conto, pagamenti). Fully local.

Output JSONL: {"text": "...", "label": <int>, "label_name": "..."}.
A labels.json with the id→name map is written next to it.

  .venv/bin/python data/gen_classify_data.py -o .datasets/cls/intent_it.jsonl
"""
import argparse, json, random
from pathlib import Path

CLASSES = {
    "credito": ["un mutuo per la prima casa", "un prestito personale", "una cessione del quinto",
                "un fido per l'azienda", "un finanziamento auto", "un microcredito", "surroga del mutuo",
                "un prestito per ristrutturare", "anticipo su fatture"],
    "assicurazione": ["una polizza RC auto", "un'assicurazione vita", "una polizza casa",
                      "una copertura sanitaria", "una polizza viaggio", "assicurazione professionale",
                      "una polizza infortuni", "garanzia furto e incendio"],
    "investimenti": ["investire in un fondo comune", "comprare azioni", "obbligazioni del tesoro",
                     "un portafoglio di ETF", "un piano di accumulo", "consulenza sui titoli di stato",
                     "diversificare il portafoglio", "investire in BTP"],
    "conto": ["aprire un conto corrente", "un conto deposito", "una carta di debito",
              "trasferire il conto", "un conto online a zero spese", "un conto per l'impresa",
              "chiudere il conto", "un conto cointestato"],
    "pagamenti": ["fare un bonifico SEPA", "un addebito diretto per le bollette", "pagare col POS",
                  "un pagamento istantaneo", "domiciliare lo stipendio", "i pagamenti contactless",
                  "un bonifico estero", "ricaricare la carta prepagata"],
}
WRAP = ["Vorrei {x}.", "Ho bisogno di {x}.", "Come faccio ad avere {x}?", "Mi interessa {x}.",
        "Posso richiedere {x}?", "Sto cercando {x}.", "Avrei bisogno di {x}, grazie.",
        "Potete aiutarmi con {x}?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--per_class", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    r = random.Random(args.seed)
    names = list(CLASSES.keys())
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for li, name in enumerate(names):
        seen = set(); made = 0; att = 0
        while made < args.per_class and att < args.per_class * 50:
            att += 1
            text = r.choice(WRAP).format(x=r.choice(CLASSES[name]))
            if text in seen:
                continue
            seen.add(text); rows.append({"text": text, "label": li, "label_name": name}); made += 1
    r.shuffle(rows)
    with open(out, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    (out.parent / "labels.json").write_text(json.dumps({i: n for i, n in enumerate(names)}, ensure_ascii=False, indent=2))
    print(f"Wrote {len(rows)} rows, {len(names)} classes -> {out}")


if __name__ == "__main__":
    main()
