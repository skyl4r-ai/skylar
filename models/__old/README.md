# `models/__old/` — architettura v1, congelata

Copia **byte-identica** di `models/` come era prima del lavoro su **Skylar v2**
(ibrido KDA + AttnRes + SiTU-GLU + output gate + MTP). Snapshot preso il **2026-07-30**
dal commit `85b70de`, branch di lavoro `feat/arch-v2`.

## A cosa serve

Ogni file qui dentro è la **versione con cui sono stati addestrati i modelli pubblicati**:

| modello | architettura |
|---|---|
| `Skyl4r-Ai/Skylar-236M-{Base,Chat,Embed}` | questa |
| `Skyl4r-Ai/Skylar-980M-Cobol-Base` | questa |
| `Skyl4r-Ai/Skylar-980M-Cobol` (SFT) | questa |

Quindi: se un checkpoint v1 non si carica più con i `models/` nuovi, **la verità è qui**.
I file sono riusabili tali e quali — nessuno è stato modificato, solo copiato. Unica eccezione, il
30/09/2026: la riga di copyright in testa ai file (`CEO MwSpace` → `CEO SKYL4R`): il codice è lo stesso.

## Come si riusa

Questa cartella **non è un package Python** (nessun `__init__.py` — i due originali sono
salvati come `_init_.py.txt` proprio per non renderla importabile per sbaglio). Non viene
raccolta da `setuptools.packages.find`, quindi non finisce nel wheel `skylar`.

Per tornare all'architettura v1:

```bash
git checkout 85b70de -- models/          # via git, la strada pulita
# oppure, a mano:
cp models/__old/config.py models/config.py
cp models/__old/layers/attention.py models/layers/attention.py
# ... e per gli __init__: cp models/__old/_init_.py.txt models/__init__.py
```

## Regola

Le modifiche v2 sono **additive e dietro flag**: il default di ogni nuovo campo di
`NanoTransformerConfig` riproduce il comportamento v1. Un checkpoint v1 caricato con i
`models/` v2 deve continuare a funzionare senza toccare il suo `config.json`.
Se così non è, è un bug — non una scelta.
