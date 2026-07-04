"""
Italian contrastive pairs (query ↔ positive passage) for the embedder.

The embedder is trained with InfoNCE / in-batch negatives: each (query, positive)
pair is pulled together, every OTHER positive in the batch is a negative. So we
only need (query, positive) pairs here — negatives come for free from the batch.

Domain: Italian banking / legal (production RAG). Definitions are real so the
embedder learns true domain semantics. Query paraphrases give surface variety so
the model learns meaning, not lexical overlap.

Fully local, deterministic. Output JSONL: {"query": "...", "positive": "..."}.

  .venv/bin/python data/gen_contrastive_data.py -o .datasets/embed/contrastive_it.jsonl
"""
import argparse, json, random
from pathlib import Path

# (termine, definizione/passaggio reale) — il "positive" da recuperare.
CONCEPTS = [
    ("IBAN", "L'IBAN (International Bank Account Number) è il codice che identifica in modo univoco un conto corrente bancario a livello internazionale, comprendendo il paese, le coordinate della banca e il numero di conto."),
    ("TAEG", "Il TAEG (Tasso Annuo Effettivo Globale) esprime il costo totale di un finanziamento in percentuale annua, includendo interessi, spese, commissioni e oneri accessori obbligatori."),
    ("TAN", "Il TAN (Tasso Annuo Nominale) è il tasso di interesse puro applicato a un finanziamento, senza considerare le spese accessorie incluse invece nel TAEG."),
    ("mutuo ipotecario", "Il mutuo ipotecario è un finanziamento a medio-lungo termine garantito da un'ipoteca su un immobile, tipicamente concesso per l'acquisto della casa."),
    ("DORA", "Il regolamento DORA (Digital Operational Resilience Act, Reg. UE 2022/2554) impone agli enti finanziari requisiti di resilienza operativa digitale, gestione dei rischi ICT, segnalazione degli incidenti e test di resilienza."),
    ("PSD2", "La direttiva PSD2 (Payment Services Directive 2) disciplina i servizi di pagamento nell'Unione europea, introducendo l'autenticazione forte del cliente e l'open banking con accesso ai conti da parte di terze parti autorizzate."),
    ("CET1", "Il Common Equity Tier 1 (CET1) è la componente di massima qualità del capitale di una banca, costituita principalmente da capitale sociale e riserve, usata per misurare la solidità patrimoniale."),
    ("antiriciclaggio", "La normativa antiriciclaggio (AML) impone agli intermediari obblighi di adeguata verifica della clientela, identificazione del titolare effettivo e segnalazione delle operazioni sospette all'UIF."),
    ("Basilea III", "Gli accordi di Basilea III sono standard internazionali di vigilanza prudenziale che rafforzano i requisiti di capitale e introducono indici di liquidità e leva finanziaria per le banche."),
    ("MiFID II", "La direttiva MiFID II disciplina i mercati degli strumenti finanziari e i servizi di investimento, rafforzando la trasparenza e la tutela degli investitori."),
    ("spread", "Lo spread bancario è la differenza tra il tasso di interesse applicato dalla banca e un tasso di riferimento, come l'Euribor, e rappresenta il margine della banca."),
    ("Euribor", "L'Euribor è il tasso medio interbancario a cui le principali banche europee si prestano denaro, usato come riferimento per i mutui a tasso variabile."),
    ("fido bancario", "Il fido bancario è una linea di credito che la banca mette a disposizione del cliente entro un limite prestabilito, utilizzabile anche andando in rosso sul conto."),
    ("conto corrente", "Il conto corrente è un contratto con cui la banca custodisce le somme del cliente e fornisce servizi di pagamento e incasso come bonifici, prelievi e addebiti."),
    ("bonifico SEPA", "Il bonifico SEPA è un trasferimento di denaro in euro all'interno dell'area unica dei pagamenti europei, eseguito tramite il codice IBAN del beneficiario."),
    ("Fondo Interbancario", "Il Fondo Interbancario di Tutela dei Depositi garantisce il rimborso dei depositi fino a centomila euro per depositante in caso di insolvenza della banca aderente."),
    ("IFRS 9", "Il principio contabile IFRS 9 disciplina la classificazione e la misurazione degli strumenti finanziari e introduce il modello delle perdite attese per gli accantonamenti sui crediti."),
    ("NPL", "I crediti deteriorati (Non Performing Loans, NPL) sono esposizioni per cui il debitore non è più in grado di adempiere regolarmente, suddivise in sofferenze, inadempienze probabili e scaduti."),
    ("vigilanza prudenziale", "La vigilanza prudenziale è l'attività con cui le autorità verificano che gli intermediari rispettino i requisiti di capitale, liquidità e governance per garantire la stabilità del sistema."),
    ("titolare effettivo", "Il titolare effettivo è la persona fisica che possiede o controlla in ultima istanza un cliente, individuata negli obblighi antiriciclaggio di adeguata verifica."),
    ("leasing", "Il leasing è un contratto con cui una società concede l'uso di un bene a fronte di un canone periodico, con facoltà di riscatto finale del bene."),
    ("cessione del quinto", "La cessione del quinto è un prestito personale rimborsato tramite trattenuta diretta sullo stipendio o sulla pensione, fino a un quinto dell'importo netto."),
    ("garanzia reale", "La garanzia reale è una forma di garanzia che grava su un bene determinato, come l'ipoteca sugli immobili o il pegno sui beni mobili, a tutela del creditore."),
    ("polizza assicurativa", "La polizza assicurativa è il contratto con cui l'assicuratore, dietro pagamento di un premio, si impegna a risarcire un danno o a versare un capitale al verificarsi di un evento."),
    ("CONSOB", "La CONSOB è l'autorità che vigila sui mercati finanziari italiani, tutela gli investitori e controlla la trasparenza e la correttezza delle informazioni societarie."),
    ("Banca d'Italia", "La Banca d'Italia è la banca centrale nazionale che concorre alle decisioni di politica monetaria dell'Eurosistema e vigila sulle banche e sugli intermediari finanziari."),
    ("liquidità", "La liquidità indica la capacità di un intermediario di far fronte ai propri impegni di pagamento a breve termine, misurata da indici come l'LCR e l'NSFR."),
    ("leva finanziaria", "La leva finanziaria misura il rapporto tra le esposizioni complessive di una banca e il suo capitale, limitando l'eccessivo indebitamento."),
    ("open banking", "L'open banking consente, con il consenso del cliente, a fornitori terzi autorizzati di accedere ai dati del conto per offrire servizi di pagamento e informazione."),
    ("KYC", "Il processo di Know Your Customer (KYC) consiste nell'identificare e verificare l'identità del cliente prima di instaurare un rapporto, nell'ambito degli obblighi antiriciclaggio."),
]

# template di query per ciascun concetto (il "lato sinistro" della coppia)
Q_TEMPLATES = [
    "Cos'è {t}?",
    "Che cosa significa {t}?",
    "Definizione di {t}",
    "{t} significato",
    "Mi spieghi {t}?",
    "A cosa serve {t}?",
    "{t}",
    "Vorrei capire {t}",
    "Puoi descrivere {t}?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--per_concept", type=int, default=120,
                    help="quante coppie (query paraphrase) per concetto")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    r = random.Random(args.seed)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for term, passage in CONCEPTS:
        for _ in range(args.per_concept):
            q = r.choice(Q_TEMPLATES).format(t=term)
            rows.append({"query": q, "positive": passage})
    r.shuffle(rows)
    with open(out, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} contrastive pairs over {len(CONCEPTS)} concepts -> {out}")


if __name__ == "__main__":
    main()
