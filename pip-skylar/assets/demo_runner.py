#!/usr/bin/env python3
"""Skylar demo runner — OUTPUT 100% REALE del modello (nessuna risposta scriptata).

Carica i modelli pubblici Skylar-236M (Chat + Embed) una sola volta e mostra:
  1) due Q&A GROUNDED bancarie/legali in italiano  -> sk.stream()  (streaming reale)
  2) un retrieval con embeddings                    -> embed.rank() (cosine reali)

Tutto ciò che appare a schermo è generato dal modello in greedy deterministico
(temperature=0.0). Riproducibile: `python assets/demo_runner.py`.
Usato per registrare assets/skylar-demo.gif con charmbracelet/vhs.
"""
import sys
import time

# ---- estetica terminale (solo cosmetica: prompt/etichette, non le risposte) ----
C = "\033[36m"   # cyan
G = "\033[32m"   # green
D = "\033[90m"   # dim
B = "\033[1m"    # bold
Y = "\033[33m"   # yellow
R = "\033[0m"    # reset


def typeline(prompt, cmd, cps=0.012):
    """Mostra un comando come se fosse digitato (solo l'eco del comando)."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    for ch in cmd:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(cps)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main():
    from skylar.core import Skylar
    from skylar.embed import SkylarEmbed

    PROMPT = f"{G}${R} "

    print(f"{D}# carico i pesi pubblici da HuggingFace (RTX 4090, bf16)...{R}")
    chat = Skylar.load("Sophia-AI/Skylar-236M-Chat", device="cuda")
    emb = SkylarEmbed.load("Sophia-AI/Skylar-236M-Embed", device="cuda")
    print(f"{D}# pronti. 236M parametri, gira tutto in locale.{R}\n")
    time.sleep(0.5)

    # --- esempi GROUNDED (system = contesto, poi la domanda) ---
    qa = [
        (
            "Il Testo Unico Bancario (TUB) e' il decreto legislativo 385 del 1993 "
            "che disciplina l'attivita' bancaria e creditizia in Italia.",
            "Cos'e' il Testo Unico Bancario?",
        ),
        (
            "L'IBAN e' il codice internazionale che identifica un conto corrente "
            "bancario. In Italia e' composto da 27 caratteri e serve per bonifici e pagamenti.",
            "A cosa serve l'IBAN?",
        ),
    ]

    for ctx, question in qa:
        cmd = (f"skylar generate --device cuda \\\n"
               f"  --system \"Rispondi solo dal contesto: {ctx[:46]}...\" \\\n"
               f"  --prompt \"{question}\"")
        typeline(PROMPT, cmd)
        time.sleep(0.3)
        sys.stdout.write(f"{C}skylar ›{R} ")
        sys.stdout.flush()
        for delta in chat.stream(question, system="Rispondi in modo conciso usando solo il contesto. Contesto: " + ctx,
                                 max_new_tokens=80, temperature=0.0):
            sys.stdout.write(delta)
            sys.stdout.flush()
            time.sleep(0.015)
        sys.stdout.write("\n\n")
        sys.stdout.flush()
        time.sleep(0.6)

    # --- retrieval / embeddings ---
    query = "come cambia la rata del mutuo"
    docs = [
        "il mutuo a tasso variabile cambia rata con l'Euribor",
        "classifica del campionato di calcio",
        "il tasso fisso mantiene la rata costante",
    ]
    cmd = (f"skylar embed --device cuda --query \"{query}\" \\\n"
           f"  --docs \"{docs[0]}\" \\\n"
           f"        \"{docs[1]}\" \\\n"
           f"        \"{docs[2]}\"")
    typeline(PROMPT, cmd)
    time.sleep(0.3)
    ranked = emb.rank(query, docs)
    for doc, score in ranked:
        bars = int(max(0, score) * 30)
        bar = "█" * bars
        print(f"  {Y}{score:+.3f}{R}  {C}{bar}{R} {doc}")
    sys.stdout.write("\n")
    sys.stdout.flush()
    time.sleep(0.8)

    print(f"{B}Skylar{R} — SLM sovrano italiano · 236M · gira in locale 🇮🇹")
    time.sleep(1.2)


if __name__ == "__main__":
    main()
