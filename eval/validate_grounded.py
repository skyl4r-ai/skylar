"""
Grounded / format-task validation — the SFT model's REAL intended role.

A 101M model can't be a factual oracle, but the prod use (sophia-vector RAG) is
grounded: answer-from-context, classify, extract-to-JSON, query-gen. Those need
fluency + instruction-following + format discipline, not parametric knowledge.
This probes exactly that, at low temperature for task reliability.

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python eval/validate_grounded.py --model checkpoints_sft/skylar-medium-chat/best
"""

import argparse, sys, torch
from pathlib import Path
from tokenizers import Tokenizer, decoders

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.decoder import NanoTransformer
from utils.chatML import encode_chatml

TESTS = [
    ("context_qa",
     "Rispondi alla domanda usando SOLO il contesto fornito. Sii conciso.",
     "Contesto: La Banca d'Italia è stata fondata nel 1893 e ha sede a Roma in via Nazionale.\n\nDomanda: In che anno è stata fondata la Banca d'Italia e dove ha sede?"),
    ("classify",
     "Classifica la richiesta del cliente in UNA categoria tra: [credito, assicurazione, investimenti]. Rispondi con una sola parola.",
     "Vorrei richiedere un mutuo ipotecario per l'acquisto della mia prima casa."),
    ("extract_json",
     "Estrai importo e scadenza dal testo e restituiscili in formato JSON con chiavi 'importo' e 'scadenza'.",
     "Si comunica che il pagamento di 1.500 euro è dovuto entro il 31 dicembre 2024."),
    ("summarize_ctx",
     "Riassumi il contesto in una sola frase.",
     "Contesto: Il regolamento DORA impone agli enti finanziari requisiti di resilienza operativa digitale, tra cui la gestione dei rischi ICT, la segnalazione degli incidenti gravi e i test di resilienza."),
    ("query_gen",
     "Genera esattamente 3 query di ricerca, una per riga, per recuperare documenti pertinenti alla richiesta.",
     "Obblighi di segnalazione degli incidenti informatici per le banche."),
    ("refuse_ungrounded",
     "Rispondi SOLO se la risposta è nel contesto, altrimenti scrivi 'Non presente nel contesto'.",
     "Contesto: L'IBAN identifica univocamente un conto corrente bancario.\n\nDomanda: Qual è il tasso di interesse del mutuo?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()
    torch.manual_seed(getattr(args, "seed", 0))   # R15: eval riproducibile run-to-run

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} (temp={args.temperature}) ...")
    model = NanoTransformer.from_pretrained(args.model).to(device).eval()
    tok = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    if tok.decoder is None:
        tok.decoder = decoders.ByteLevel()
    eos_ids = [i for i in (tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>"), tok.token_to_id("<eos>")) if i is not None]

    n_stop = 0
    print("=" * 78)
    for label, system, user in TESTS:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = encode_chatml(messages, tok, add_generation_prompt=True)
        inp = torch.tensor([ids], device=device)
        plen = inp.shape[1]
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                 top_k=40, top_p=0.9, repetition_penalty=1.15, eos_token_id=eos_ids, use_cache=True)
        gen = out[0][plen:].tolist()
        stopped = any(t in eos_ids for t in gen)
        n_stop += int(stopped)
        text = tok.decode([t for t in gen if t not in eos_ids]).strip()
        print(f"\n[{label}]  {'✅ STOP' if stopped else '⚠️ RUNAWAY'}")
        print(f"  Q: {user.replace(chr(10), ' / ')}")
        print(f"  A: {text}")
    print("\n" + "=" * 78)
    print(f"  Stopped cleanly: {n_stop}/{len(TESTS)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
