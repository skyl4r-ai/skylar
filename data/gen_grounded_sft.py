# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Templated grounded-instruction SFT generator (Italian, banking/legal).

Teaches a SMALL model the *patterns* the RAG prod use needs — read the provided
context and answer FROM it, output a single label, extract key/values, give a
one-line summary, refuse when the answer is absent, reformulate into queries.
Answers are always derivable from the prompt (extractive / closed-set), so the
model learns grounding & format discipline, not parametric facts.

Fully local, deterministic (fixed seed), zero external calls. Output: ChatML
JSONL  {"messages":[{role,content}...]}  ready for training/bin.sft.py.

Per-generator targets with sampling; closed-set tasks (classify/refuse/query)
allow repetition on purpose — repetition teaches the mapping, and there is no
global round-robin to deadlock on a small-space generator.

  .venv/bin/python data/gen_grounded_sft.py -o .datasets/sft/grounded_synth.jsonl
"""
import argparse, json, random
from pathlib import Path
from collections import Counter

# ── slot banks ────────────────────────────────────────────────────────────
ENTI = [
    "la Banca d'Italia", "la Banca Centrale Europea", "la CONSOB", "l'IVASS",
    "l'Autorità Bancaria Europea", "il Fondo Interbancario di Tutela dei Depositi",
    "la Cassa Depositi e Prestiti", "il Comitato di Basilea", "l'ESMA",
    "il Meccanismo di Vigilanza Unico", "la Corte dei Conti", "il Ministero dell'Economia",
    "l'Unità di Informazione Finanziaria", "il Single Resolution Board",
]
CITTA = ["Roma", "Milano", "Francoforte", "Torino", "Bruxelles", "Parigi", "Bologna",
         "Napoli", "Basilea", "Londra", "Lussemburgo", "Bari", "Firenze"]
NORME = [
    "il regolamento DORA", "la direttiva PSD2", "la Circolare 285 della Banca d'Italia",
    "il regolamento CRR", "la direttiva CRD IV", "gli accordi di Basilea III",
    "la normativa antiriciclaggio (AML)", "il principio contabile IFRS 9",
    "la direttiva MiFID II", "il Testo Unico Bancario (TUB)",
    "il regolamento GDPR", "la direttiva sui mercati degli strumenti finanziari",
]
RUOLI = ["la vigilanza prudenziale", "la tutela del risparmio", "la stabilità finanziaria",
         "la trasparenza delle operazioni", "la gestione del rischio di credito",
         "la resilienza operativa digitale", "la prevenzione del riciclaggio",
         "la supervisione dei mercati", "la risoluzione delle crisi bancarie",
         "la tutela dei depositanti"]

CLASSI = {
    "credito": ["un mutuo ipotecario per l'acquisto della prima casa",
                "un prestito personale di diecimila euro", "una cessione del quinto dello stipendio",
                "una linea di fido per la mia azienda", "un finanziamento auto a tasso fisso",
                "un prestito per ristrutturare casa", "un microcredito per la mia attività"],
    "assicurazione": ["una polizza RC auto", "un'assicurazione sulla vita per la mia famiglia",
                      "una copertura contro furto e incendio della casa", "una polizza sanitaria integrativa",
                      "un preventivo per una polizza viaggio", "un'assicurazione per la responsabilità professionale"],
    "investimenti": ["investire i risparmi in un fondo comune", "acquistare azioni di una società quotata",
                     "sottoscrivere obbligazioni del tesoro", "un portafoglio diversificato con ETF",
                     "avviare un piano di accumulo", "consulenza per investire in titoli di stato"],
    "conto": ["aprire un conto corrente cointestato", "attivare un conto deposito vincolato",
              "una carta di debito per i prelievi", "trasferire il saldo su un nuovo conto",
              "passare a un conto online a zero spese", "aprire un conto per la mia impresa"],
    "pagamenti": ["effettuare un bonifico SEPA verso l'estero", "impostare un addebito diretto per le bollette",
                  "pagare un fornitore tramite POS", "inviare un pagamento istantaneo",
                  "domiciliare lo stipendio sul conto", "attivare i pagamenti contactless"],
}
CLASS_WRAP = ["Vorrei {x}.", "Ho bisogno di {x}.", "Come faccio a ottenere {x}?",
              "Posso richiedere {x}?", "Mi interessa {x}.", "Sto cercando {x}."]

REFUSE_FATTI = [
    "L'IBAN identifica univocamente un conto corrente bancario.",
    "Il bonifico SEPA consente trasferimenti in euro nell'area unica dei pagamenti.",
    "La carta di debito permette prelievi e pagamenti addebitati sul conto.",
    "Il fido è una linea di credito concessa dalla banca entro un limite prestabilito.",
    "Lo spread è la differenza tra il tasso applicato e il tasso di riferimento.",
    "Il TAEG indica il costo totale annuo di un finanziamento.",
    "Il Fondo Interbancario tutela i depositi fino a centomila euro per depositante.",
    "La firma digitale ha lo stesso valore legale della firma autografa.",
]
REFUSE_DOMANDE = [
    "Qual è il tasso di interesse del mutuo?", "Quanti dipendenti ha la banca?",
    "In che anno è stata fondata la società?", "Qual è il rendimento atteso dell'investimento?",
    "Quante filiali ci sono in Italia?", "Qual è l'importo massimo finanziabile?",
    "Chi è l'amministratore delegato?", "Qual è la durata del contratto?",
]

# (richiesta utente, [3 query di ricerca PULITE e distinte — scritte a mano, no stutter])
QUERY_ITEMS = [
    ("obblighi di segnalazione degli incidenti informatici per le banche",
     ["segnalazione incidenti ICT banche normativa", "notifica incidenti gravi DORA tempistiche",
      "obblighi reporting cyber settore bancario"]),
    ("requisiti di adeguatezza patrimoniale delle banche",
     ["requisiti patrimoniali minimi CRR", "coefficiente CET1 capitale primario",
      "fondi propri vigilanza prudenziale Basilea"]),
    ("regole antiriciclaggio per l'adeguata verifica della clientela",
     ["adeguata verifica clientela AML", "identificazione titolare effettivo antiriciclaggio",
      "segnalazione operazioni sospette UIF"]),
    ("trasparenza delle condizioni dei contratti bancari",
     ["trasparenza condizioni contrattuali banche", "documento di sintesi conto corrente",
      "obblighi informativi precontrattuali credito"]),
    ("gestione del rischio di credito negli enti finanziari",
     ["misurazione rischio di credito IFRS 9", "accantonamenti perdite attese ECL",
      "modelli interni rating credito vigilanza"]),
    ("tutela dei depositi e garanzia interbancaria",
     ["garanzia depositi fino a 100000 euro", "Fondo Interbancario Tutela Depositi",
      "rimborso depositanti banca insolvente"]),
    ("resilienza operativa digitale degli intermediari",
     ["resilienza operativa digitale DORA", "test di resilienza ICT intermediari",
      "gestione rischi terze parti tecnologiche"]),
    ("obblighi informativi precontrattuali per i mutui",
     ["informazioni precontrattuali mutuo ipotecario", "prospetto informativo europeo standardizzato PIES",
      "TAEG e costi mutuo trasparenza"]),
    ("vigilanza prudenziale sulle imprese di investimento",
     ["vigilanza imprese di investimento MiFID", "requisiti prudenziali SIM",
      "autorizzazione servizi di investimento CONSOB"]),
    ("disciplina dei servizi di pagamento",
     ["servizi di pagamento direttiva PSD2", "autenticazione forte SCA pagamenti",
      "open banking accesso conto terze parti"]),
    ("normativa sui crediti deteriorati",
     ["gestione crediti deteriorati NPL", "cessione sofferenze bancarie",
      "calendar provisioning crediti deteriorati"]),
    ("governance e controlli interni delle banche",
     ["sistema controlli interni banche", "requisiti organi amministrazione idoneità",
      "funzioni di controllo compliance e risk"]),
]

SOGG_PAG = ["il pagamento", "il versamento", "il rimborso", "la rata", "il saldo", "il canone"]
MESI = ["gennaio","febbraio","marzo","aprile","maggio","giugno","luglio","agosto",
        "settembre","ottobre","novembre","dicembre"]


def msg(system, user, assistant):
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def gen_context_qa(r):
    ente, anno, citta = r.choice(ENTI), r.randint(1850, 2024), r.choice(CITTA)
    ruolo = r.choice(RUOLI)
    cap = ente[0].upper() + ente[1:]
    ctx = f"{cap} è stata istituita nel {anno}, ha sede a {citta} e si occupa di {ruolo}."
    v = r.choice(["anno_sede", "ruolo", "tutto"])
    if v == "anno_sede":
        q, a = f"In che anno è stata istituita {ente} e dove ha sede?", f"{cap} è stata istituita nel {anno} e ha sede a {citta}."
    elif v == "ruolo":
        q, a = f"Di cosa si occupa {ente}?", f"{cap} si occupa di {ruolo}."
    else:
        q, a = f"Riassumi le informazioni su {ente}.", ctx
    return msg("Rispondi alla domanda usando SOLO il contesto fornito. Sii conciso.",
               f"Contesto: {ctx}\n\nDomanda: {q}", a)


def gen_classify(r):
    label = r.choice(list(CLASSI.keys()))
    x = r.choice(CLASSI[label])
    user = r.choice(CLASS_WRAP).format(x=x)
    sys = ("Classifica la richiesta del cliente in UNA categoria tra: "
           "[credito, assicurazione, investimenti, conto, pagamenti]. Rispondi con una sola parola.")
    return msg(sys, user, label)


def gen_extract(r):
    sogg = r.choice(SOGG_PAG)
    importo = r.choice([f"{r.randint(1,9)}.{r.randint(0,999):03d}", f"{r.randint(10,99)}.{r.randint(0,999):03d}", str(r.randint(50,990))])
    gg, mm, aaaa = r.randint(1, 28), r.randint(1, 12), r.randint(2023, 2027)
    scad = f"{gg} {MESI[mm-1]} {aaaa}"
    ctx = f"Si comunica che {sogg} di {importo} euro è dovuto entro il {scad}."
    sys = "Estrai importo e scadenza dal testo e restituiscili in JSON con chiavi 'importo' e 'scadenza'. Rispondi solo con il JSON."
    return msg(sys, ctx, json.dumps({"importo": f"{importo} euro", "scadenza": scad}, ensure_ascii=False))


def gen_summary(r):
    norma = r.choice(NORME); p = r.sample(RUOLI, 2)
    cap = norma[0].upper() + norma[1:]
    ctx = f"{cap} introduce obblighi in materia di {p[0]} e {p[1]} per gli enti finanziari vigilati."
    return msg("Riassumi il contesto in una sola frase, senza aggiungere informazioni.",
               f"Contesto: {ctx}", f"{cap} stabilisce obblighi su {p[0]} e {p[1]} per gli enti vigilati.")


def gen_refuse(r):
    fatto, dom = r.choice(REFUSE_FATTI), r.choice(REFUSE_DOMANDE)
    sys = "Rispondi alla domanda SOLO se l'informazione è nel contesto, altrimenti scrivi esattamente 'Non presente nel contesto'."
    return msg(sys, f"Contesto: {fatto}\n\nDomanda: {dom}", "Non presente nel contesto")


def gen_query(r):
    richiesta, queries = r.choice(QUERY_ITEMS)
    # light shuffle so the 3 lines aren't always in the same order
    qs = queries[:]; r.shuffle(qs)
    return msg("Genera esattamente 3 query di ricerca, una per riga, per recuperare documenti pertinenti alla richiesta.",
               richiesta.capitalize() + ".", "\n".join(qs))


# (generator, target count, dedup-within-type?)
PLAN = [
    (gen_context_qa, 1600, True),
    (gen_extract,    1400, True),
    (gen_summary,    1000, True),
    (gen_classify,    900, False),
    (gen_refuse,      700, False),
    (gen_query,       700, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    r = random.Random(args.seed)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for gen, target, dedup in PLAN:
        seen = set(); made = 0; attempts = 0
        while made < target and attempts < target * 40:
            attempts += 1
            ex = gen(r)
            if dedup:
                key = ex["messages"][1]["content"] + "||" + ex["messages"][2]["content"]
                if key in seen:
                    continue
                seen.add(key)
            rows.append(ex); made += 1
        if made < target:
            print(f"  ! {gen.__name__}: only {made}/{target} unique (space exhausted)")
    r.shuffle(rows)
    with open(out, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    c = Counter(ex["messages"][0]["content"][:32] for ex in rows)
    print(f"Wrote {len(rows)} grounded examples -> {out}")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  {k}...")


if __name__ == "__main__":
    main()
