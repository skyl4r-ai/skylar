# NanoTransformer — Stime Reali di Training

Basate su benchmark misurati su RTX 4090:
- small (40M): **100K tok/s** (misurato)  
- medium (107M): **51K tok/s** (misurato, in corso)

Scaling: `tok/s ≈ 51K × (107M / params)^0.7` per stessa GPU.  
B200 ≈ 8-10× RTX 4090. Multi-GPU scala ~linearmente con buon interconnect.

## Tabella riassuntiva

| Preset | Parametri | GPU minima | tok/s stimati | Dati necessari | Tempo training |
|--------|-----------|------------|---------------|----------------|----------------|
| **test** | 5M | Qualsiasi | 150K | 500MB | Minuti |
| **small** | 40M | 1x 4090 | 100K | 1-5GB | **5 ore** |
| **medium** | 107M | 1x 4090 | 51K | 5-10GB | **18 ore** |
| **large** | 350M | 1x 4090 | 20K | 20-50GB | **2 giorni** |
| **xl** | 1B | 1x B200 | 55K | 50-100GB | **4 giorni** |
| **3b** | 3B | 2x B200 | 40K | 200-500GB | **17 giorni** |
| **4b** | 4B | 2x B200 | 28K | 500GB-1TB | **1 mese** |
| **7b** | 7B | 4x B200 | 30K | 1-2TB | **2 mesi** |
| **14b** | 14B | 8x B200 | 22K | 2-5TB | **5 mesi** |
| **32b** | 32B | 32x B200 | 28K | 5-10TB | **9 mesi** |
| **64b** | 64B | 128x B200 | 32K | 10-20TB | **15 mesi** |
| **96b** | 96B | 256x B200 | 32K | 15-30TB | **23 mesi** |
| **128b** | 128B | 512x B200 | 40K | 20-40TB | **24 mesi** |

## Cosa possiamo fare ORA

| Setup | Modello max | Tempo | Qualità attesa |
|-------|-------------|-------|----------------|
| **1x RTX 4090** (attuale) | medium (107M) | 18 ore | Testo coerente, SFT base |
| **1x RTX 4090** | large (350M) | 2 giorni | SFT chat solida, JSON ok |
| **1x B200** | xl (1B) | 4 giorni | Chat fluente, istruzioni complesse |
| **2x B200** | 4b (4B) | 1 mese | Qualità Qwen3-4B, produzione reale |

## Note per investitori

### Costo compute stimato (cloud B200 ~$5/h/GPU)
| Modello | GPU | Mesi | Costo compute |
|---------|-----|------|---------------|
| 4b | 2x B200 | 1 | ~$7K |
| 7b | 4x B200 | 2 | ~$30K |
| 14b | 8x B200 | 5 | ~$150K |
| 32b | 32x B200 | 9 | ~$1.2M |
| 64b | 128x B200 | 15 | ~$7M |
| 128b | 512x B200 | 24 | ~$44M |

### Il vero collo di bottiglia: i dati
Il modello 128B necessita di 20-40TB di testo curato. Raccogliere, pulire e deduplicare questo volume di dati è un progetto a sé che richiede mesi di lavoro indipendentemente dal compute. La qualità dei dati conta più del numero di parametri — un 7B allenato su dati eccellenti batte un 32B allenato su dati mediocri.

### Architettura
NanoTransformer usa la stessa architettura di Qwen3-4B-Instruct: RMSNorm, RoPE, SwiGLU, GQA, QK-Norm. La differenza è solo nei dati e nel compute. Il preset `4b` ha dimensioni **identiche** a Qwen3-4B. Scalare a 128B è un cambio di numeri nella config, non di codice.