import { useState } from "react";

const COLORS = {
  bg: "#0a0a0f",
  surface: "#12121a",
  surfaceHover: "#1a1a25",
  border: "#2a2a3a",
  borderActive: "#6366f1",
  text: "#e2e2f0",
  textMuted: "#8888a0",
  accent: "#6366f1",
  accentGlow: "rgba(99,102,241,0.15)",
  green: "#22c55e",
  greenGlow: "rgba(34,197,94,0.15)",
  orange: "#f59e0b",
  orangeGlow: "rgba(245,158,11,0.15)",
  pink: "#ec4899",
  pinkGlow: "rgba(236,72,153,0.15)",
  cyan: "#06b6d4",
  cyanGlow: "rgba(6,182,212,0.15)",
  red: "#ef4444",
  redGlow: "rgba(239,68,68,0.15)",
};

const COMPONENTS = {
  input: {
    label: "Input Tokens",
    icon: "📝",
    color: COLORS.textMuted,
    glow: "rgba(136,136,160,0.1)",
    short: "La frase entra come numeri",
    detail: `Ogni parola (o pezzo di parola) viene convertita in un numero intero dal tokenizer BPE.

"Il gatto dorme" → [142, 8823, 4901]

Il tokenizer ha un vocabolario di 40,960 token. Ogni token è un ID numerico. Il modello non vede mai le parole — vede solo numeri.

Dimensione: (batch=16, seq_len=16384)
= 16 frasi in parallelo, ognuna fino a 16K token.`,
    limits: `• seq_len=16384 è il contesto massimo. Oltre, devi aumentare la RoPE base o usare tecniche come YaRN.
• Vocab 40K è un compromesso: troppo piccolo = tanti token per frase, troppo grande = embedding enorme.`
  },
  embedding: {
    label: "Token Embedding",
    icon: "🎨",
    color: COLORS.cyan,
    glow: COLORS.cyanGlow,
    short: "Ogni numero diventa un vettore",
    detail: `Una tabella gigante (40960 × 768) dove ogni riga è il "significato" di un token.

Token 8823 ("gatto") → vettore di 768 numeri decimali:
[0.12, -0.45, 0.78, 0.03, -0.91, ...]

Questi 768 numeri sono il DNA del token — codificano significato, grammatica, contesto. All'inizio sono random, il training li ottimizza.

Peso: nn.Embedding(40960, 768)
Parametri: 40960 × 768 = 31.5M (≈30% del modello!)`,
    limits: `• d_model=768 è la "larghezza" del modello. Più è grande, più sfumature cattura.
• Weight tying: lm_head condivide questi pesi → risparmia 31.5M parametri.
• Nella tua architettura, l'embedding è ≈30% dei parametri totali. Modelli più grandi riducono questa proporzione.`
  },
  rope: {
    label: "RoPE",
    icon: "🌀",
    color: COLORS.orange,
    glow: COLORS.orangeGlow,
    short: "Dice al modello DOVE è ogni token",
    detail: `Rotary Position Embedding — codifica la posizione senza parametri extra.

Funziona così: ruota i vettori Q e K di un angolo proporzionale alla posizione. Token vicini hanno angoli simili → dot product alto → più attenzione tra loro.

Posizione 0: ruota di 0°
Posizione 1: ruota di θ
Posizione 2: ruota di 2θ
...dove θ dipende dalla dimensione (base=10000)

Non aggiunge parametri — è una trasformazione matematica pura. È relativo: il modello vede la DISTANZA tra token, non la posizione assoluta.`,
    limits: `• rope_theta=10000 è lo standard. Aumentarlo (es. 500K come LLaMA 3) estende il contesto.
• Funziona solo su coppie di dimensioni → d_head deve essere pari.
• Oltre ~4× il training seq_len, la qualità degrada senza fine-tuning con contesto esteso.`
  },
  attention: {
    label: "Self-Attention (GQA)",
    icon: "👀",
    color: COLORS.accent,
    glow: COLORS.accentGlow,
    short: "Ogni token guarda tutti gli altri",
    detail: `Il cuore del transformer. Ogni token "chiede" (Query) cosa è importante, ogni altro token "offre" (Key) la sua identità, e manda il suo contenuto (Value) se c'è match.

Q = x × W_q → "Cosa cerco?"
K = x × W_k → "Cosa sono?"  
V = x × W_v → "Cosa offro?"

Attention(Q,K,V) = softmax(Q·Kᵀ / √d) · V

GQA (Grouped Query Attention): 12 teste Q ma solo 4 teste KV.
Ogni 3 teste Q condividono 1 testa KV → 3× meno memoria per la KV cache.

Nel tuo modello:
• 12 Q heads × 64 dim = 768 → W_q è (768→768)
• 4 KV heads × 64 dim = 256 → W_k, W_v sono (768→256)
• W_o riproietta: (768→768)`,
    limits: `• Complessità O(T²) — raddoppiare seq_len quadruplica il costo dell'attention.
• FlexAttention maschera cross-documento senza allocare la matrice T×T.
• GQA 12:4 riduce KV cache del 66%. LLaMA usa rapporti simili.
• La KV cache a inference scala come: batch × layers × seq × d_head × n_kv × 2 × dtype.`
  },
  flexattn: {
    label: "FlexAttention Mask",
    icon: "🎭",
    color: COLORS.pink,
    glow: COLORS.pinkGlow,
    short: "Impedisce ai documenti di vedersi",
    detail: `In training, 16K token = molti documenti packed insieme.

SENZA mask:
  Doc A: "Il gatto..."  Doc B: "Roma è..."
  "gatto" può guardare "Roma" → correlazione spuria!

CON FlexAttention mask:
  "gatto" vede SOLO token del Doc A.
  "Roma" vede SOLO token del Doc B.
  + causal: ogni token vede solo token PRECEDENTI nel suo documento.

Il mask è una funzione, non una matrice:
  mask(b, h, q_idx, kv_idx) = causal AND same_document

FlexAttention compila questa funzione in un kernel CUDA block-sparse.
Memoria: O(T) invece di O(T²). Con T=16384, questo salva ~1GB di VRAM.`,
    limits: `• Richiede PyTorch ≥ 2.5 con torch.compile.
• Senza FlexAttention, il fallback è una matrice densa (16K×16K) = ~1GB per batch.
• Il mask è ricalcolato ogni batch (diversi documenti ogni volta). Il costo è trascurabile.
• Non attivo durante generation (si usa KV cache, un token alla volta).`
  },
  ffn: {
    label: "SwiGLU FFN",
    icon: "⚡",
    color: COLORS.green,
    glow: COLORS.greenGlow,
    short: "Elabora e trasforma ogni token",
    detail: `Feed-Forward Network con gating — dove il modello "ragiona" su ogni token individualmente.

FFN(x) = W₂ · (SiLU(W₁·x) ⊙ W₃·x)

Tre matrici:
• W₁: (768 → 2048) — espande
• W₃: (768 → 2048) — gate (decide cosa passa)
• ⊙: moltiplicazione elemento per elemento (il gate!)
• SiLU: attivazione smooth (simile a ReLU ma derivabile)
• W₂: (2048 → 768) — comprime di nuovo

Perché SwiGLU e non ReLU?
SiLU(x)·gate è più espressivo: il gate impara COSA è importante, SiLU impara QUANTO. Due decisioni invece di una.

Parametri per layer: 768×2048×3 = 4.7M`,
    limits: `• d_ff=2048 = 2.67× d_model. Lo standard è 8/3× ≈ 2.67. LLaMA usa questo rapporto.
• SwiGLU ha 3 matrici invece di 2 (rispetto a FFN classico), ma è più efficiente per parametro.
• Questa è la parte più "densa" di parametri: 4.7M × 12 layer = 56.6M (53% del modello).
• Aumentare d_ff è il modo più semplice per aggiungere capacità.`
  },
  residual: {
    label: "Residual + RMSNorm",
    icon: "🔄",
    color: COLORS.textMuted,
    glow: "rgba(136,136,160,0.08)",
    short: "Scorciatoia + normalizzazione",
    detail: `Ogni blocco ha due residual connections:

x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))

La "scorciatoia" (x + ...) è fondamentale:
• Senza: il gradiente deve attraversare 12 layer → svanisce
• Con: il gradiente può "saltare" direttamente → training stabile

RMSNorm (invece di LayerNorm):
• LayerNorm: sottrae media, divide per deviazione standard
• RMSNorm: divide solo per la root mean square. Più veloce, stessi risultati.

RMSNorm(x) = x / √(mean(x²) + ε) × γ

Pre-Norm: normalizza PRIMA di attention/FFN (non dopo). Il tuo modello usa Pre-Norm, come LLaMA.`,
    limits: `• Pre-Norm è più stabile di Post-Norm per modelli profondi.
• Il residual scaling (0.02/√(2×n_layers)) previene esplosione nei layer profondi.
• Con µP, lo scaling si adatta anche alla larghezza del modello.`
  },
  layers: {
    label: "× 12 Layers",
    icon: "📚",
    color: COLORS.accent,
    glow: COLORS.accentGlow,
    short: "Ripeti tutto 12 volte",
    detail: `Lo stesso blocco (Attention + FFN) ripetuto 12 volte in sequenza.

Ogni layer aggiunge un livello di comprensione:
• Layer 1-3: pattern locali (grammatica, sintassi)
• Layer 4-8: semantica (significato, relazioni)
• Layer 9-12: ragionamento (inferenza, generazione)

Questo è intuitivo ma approssimativo — in realtà ogni layer fa un po' di tutto, e il modello trova la sua distribuzione ottimale durante il training.

Parametri per layer:
  Attention: 768² × 4 proiezioni ≈ 2.4M
  FFN: 768 × 2048 × 3 ≈ 4.7M
  Norms: 768 × 2 ≈ 1.5K
  Totale: ~7.1M × 12 = ~85M`,
    limits: `• 12 layer è il punto dolce per ~100M parametri.
• Raddoppiare i layer (24) senza cambiare d_model = modello "stretto e profondo" → meno efficiente.
• Lo sweet spot è d_model ≈ 64-128 × n_layers.
• Gradient checkpointing salva VRAM ripetendo il forward invece di salvare le attivazioni.`
  },
  lmhead: {
    label: "LM Head",
    icon: "🎯",
    color: COLORS.red,
    glow: COLORS.redGlow,
    short: "Predice il prossimo token",
    detail: `La proiezione finale: da vettore 768-dim a probabilità su 40,960 token.

logits = LN(x) × W_lm    →  (768 → 40960)

Poi softmax per avere probabilità:
P("gatto") = 0.02
P("cane")  = 0.15  ← più probabile!
P("casa")  = 0.01
...40,957 altri token...

Weight tying: W_lm = W_embedding (trasposta)
Stessi pesi usati al contrario! Token simili nel significato → punteggi simili in output.

Durante il training: Cross-Entropy loss tra logits e token reale.
Durante generation: sampling (temperature, top-k, top-p) dai logits.`,
    limits: `• Con µP, i logits sono scalati da 1/width_mult per stabilità.
• Weight tying risparmia 31.5M parametri ma vincola embedding = unembedding.
• Senza tying avresti ~138M parametri invece di 107M.
• Il vocabolario grande (40K) rende questa proiezione costosa — ogni forward calcola 40K logits.`
  },
  kvcache: {
    label: "KV Cache",
    icon: "💾",
    color: COLORS.orange,
    glow: COLORS.orangeGlow,
    short: "Velocizza la generazione",
    detail: `A inference, il modello genera 1 token alla volta. Senza cache:

Token 1: computa K,V per posizione 0
Token 2: RICOMPUTA K,V per pos 0 + computa pos 1
Token 3: RICOMPUTA 0,1 + computa 2
→ O(T²) computazione totale!

Con KV cache:
Token 1: computa e SALVA K,V per pos 0
Token 2: carica cache + computa SOLO pos 1, appendi alla cache
Token 3: carica cache + computa SOLO pos 2, appendi
→ O(T) computazione totale!

Nel tuo modello la cache pesa:
12 layers × 4 kv_heads × seq_len × 64 d_head × 2 (K+V) × 2 bytes (bf16)
= 12 × 4 × T × 64 × 2 × 2 = 12,288 × T bytes

Per T=16K: ~192 MB per utente. Piccolo per una B200, ma scala con batch.`,
    limits: `• La KV cache è il bottleneck per servire tanti utenti.
• GQA (4 KV heads invece di 12) riduce la cache del 66%.
• Per modelli grossi, la cache può essere offloadata in RAM o SSD.
• Con 107M params, la cache è piccola. A 7B+ diventa il fattore dominante.`
  },
};

const FLOW = ["input", "embedding", "rope", "attention", "flexattn", "ffn", "residual", "layers", "lmhead", "kvcache"];

const Arrow = ({ flip }) => (
  <div style={{
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: 32,
    color: COLORS.border,
    fontSize: 20,
    transform: flip ? "rotate(180deg)" : "none",
    userSelect: "none",
  }}>
    ↓
  </div>
);

const ComponentCard = ({ id, selected, onClick }) => {
  const c = COMPONENTS[id];
  const isActive = selected === id;
  return (
    <button
      onClick={() => onClick(id)}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "12px 16px",
        background: isActive ? c.glow : COLORS.surface,
        border: `1.5px solid ${isActive ? c.color : COLORS.border}`,
        borderRadius: 10,
        cursor: "pointer",
        transition: "all 0.2s ease",
        width: "100%",
        textAlign: "left",
        boxShadow: isActive ? `0 0 20px ${c.glow}` : "none",
        outline: "none",
      }}
    >
      <span style={{ fontSize: 22, flexShrink: 0 }}>{c.icon}</span>
      <div style={{ minWidth: 0 }}>
        <div style={{
          fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
          fontSize: 13,
          fontWeight: 600,
          color: isActive ? c.color : COLORS.text,
          letterSpacing: "-0.02em",
        }}>
          {c.label}
        </div>
        <div style={{
          fontSize: 11,
          color: COLORS.textMuted,
          marginTop: 2,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}>
          {c.short}
        </div>
      </div>
    </button>
  );
};

const DetailPanel = ({ id }) => {
  const [tab, setTab] = useState("come");
  const c = COMPONENTS[id];
  if (!c) return null;

  return (
    <div style={{
      background: COLORS.surface,
      border: `1.5px solid ${c.color}`,
      borderRadius: 14,
      overflow: "hidden",
      boxShadow: `0 0 30px ${c.glow}`,
    }}>
      <div style={{
        padding: "16px 20px",
        background: c.glow,
        borderBottom: `1px solid ${c.color}33`,
        display: "flex",
        alignItems: "center",
        gap: 12,
      }}>
        <span style={{ fontSize: 28 }}>{c.icon}</span>
        <div>
          <div style={{
            fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
            fontSize: 18,
            fontWeight: 700,
            color: c.color,
          }}>
            {c.label}
          </div>
          <div style={{ fontSize: 13, color: COLORS.textMuted, marginTop: 2 }}>
            {c.short}
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 0, borderBottom: `1px solid ${COLORS.border}` }}>
        {[["come", "Come funziona"], ["limiti", "Limiti"]].map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            style={{
              flex: 1,
              padding: "10px 0",
              background: tab === key ? `${c.color}15` : "transparent",
              border: "none",
              borderBottom: tab === key ? `2px solid ${c.color}` : "2px solid transparent",
              color: tab === key ? c.color : COLORS.textMuted,
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
              fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
              letterSpacing: "0.05em",
              textTransform: "uppercase",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      <div style={{
        padding: "18px 20px",
        fontSize: 13,
        lineHeight: 1.7,
        color: COLORS.text,
        whiteSpace: "pre-wrap",
        fontFamily: "'IBM Plex Sans', -apple-system, sans-serif",
        maxHeight: 420,
        overflowY: "auto",
      }}>
        {tab === "come" ? c.detail : c.limits}
      </div>
    </div>
  );
};

const ModelStats = () => (
  <div style={{
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 8,
    padding: "12px 0",
  }}>
    {[
      ["107M", "Parametri"],
      ["768", "d_model"],
      ["12", "Layers"],
      ["12:4", "Q:KV heads"],
      ["16K", "Contesto"],
      ["40K", "Vocab"],
    ].map(([val, label]) => (
      <div key={label} style={{
        textAlign: "center",
        padding: "8px 4px",
        background: COLORS.surface,
        borderRadius: 8,
        border: `1px solid ${COLORS.border}`,
      }}>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 15,
          fontWeight: 700,
          color: COLORS.accent,
        }}>{val}</div>
        <div style={{
          fontSize: 10,
          color: COLORS.textMuted,
          marginTop: 2,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}>{label}</div>
      </div>
    ))}
  </div>
);

const FlowDiagram = ({ selected, onSelect }) => {
  const mainFlow = ["input", "embedding", "attention", "ffn", "layers", "lmhead"];
  const sideComponents = { rope: 2, flexattn: 2, residual: 3, kvcache: 5 };

  return (
    <div style={{ position: "relative", display: "flex", flexDirection: "column", alignItems: "center", gap: 0, padding: "8px 0" }}>
      {mainFlow.map((id, i) => (
        <div key={id} style={{ width: "100%", display: "flex", flexDirection: "column", alignItems: "center" }}>
          <ComponentCard id={id} selected={selected} onClick={onSelect} />
          {Object.entries(sideComponents)
            .filter(([, idx]) => idx === i)
            .map(([sideId]) => (
              <div key={sideId} style={{
                width: "85%",
                marginTop: 4,
                marginBottom: 4,
                opacity: 0.85,
              }}>
                <ComponentCard id={sideId} selected={selected} onClick={onSelect} />
              </div>
            ))}
          {i < mainFlow.length - 1 && <Arrow />}
        </div>
      ))}
    </div>
  );
};

const ParamBreakdown = () => {
  const data = [
    { label: "Embedding", value: 31.5, color: COLORS.cyan },
    { label: "Attention ×12", value: 28.3, color: COLORS.accent },
    { label: "FFN ×12", value: 56.6, color: COLORS.green },
    { label: "Norms + altro", value: 0.02, color: COLORS.textMuted },
  ];
  const total = data.reduce((s, d) => s + d.value, 0);

  return (
    <div style={{
      background: COLORS.surface,
      borderRadius: 12,
      padding: 16,
      border: `1px solid ${COLORS.border}`,
    }}>
      <div style={{
        fontSize: 11,
        fontWeight: 700,
        color: COLORS.textMuted,
        textTransform: "uppercase",
        letterSpacing: "0.08em",
        marginBottom: 12,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        Distribuzione parametri (107M)
      </div>
      <div style={{
        display: "flex",
        height: 8,
        borderRadius: 4,
        overflow: "hidden",
        marginBottom: 12,
      }}>
        {data.map((d) => (
          <div key={d.label} style={{
            width: `${(d.value / total) * 100}%`,
            background: d.color,
          }} />
        ))}
      </div>
      {data.map((d) => (
        <div key={d.label} style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "4px 0",
          fontSize: 12,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{
              width: 8,
              height: 8,
              borderRadius: 2,
              background: d.color,
            }} />
            <span style={{ color: COLORS.text }}>{d.label}</span>
          </div>
          <span style={{
            fontFamily: "'JetBrains Mono', monospace",
            color: COLORS.textMuted,
            fontSize: 11,
          }}>
            {d.value.toFixed(1)}M ({(d.value / total) * 100 < 1 ? "<1" : ((d.value / total) * 100).toFixed(0)}%)
          </span>
        </div>
      ))}
    </div>
  );
};

const ForwardPassViz = () => {
  const [step, setStep] = useState(0);
  const steps = [
    { label: "Input", shape: "(16, 16384)", desc: "batch × seq_len — token IDs interi", color: COLORS.textMuted },
    { label: "Embedding", shape: "(16, 16384, 768)", desc: "ogni ID → vettore 768-dim", color: COLORS.cyan },
    { label: "After Attention", shape: "(16, 16384, 768)", desc: "stessa forma! solo il contenuto cambia", color: COLORS.accent },
    { label: "After FFN", shape: "(16, 16384, 768)", desc: "elaborato, pronto per il prossimo layer", color: COLORS.green },
    { label: "× 12 layers", shape: "(16, 16384, 768)", desc: "la forma non cambia mai tra i layer", color: COLORS.accent },
    { label: "LM Head", shape: "(16, 16384, 40960)", desc: "esplode a 40K logits per posizione!", color: COLORS.red },
  ];

  return (
    <div style={{
      background: COLORS.surface,
      borderRadius: 12,
      padding: 16,
      border: `1px solid ${COLORS.border}`,
    }}>
      <div style={{
        fontSize: 11,
        fontWeight: 700,
        color: COLORS.textMuted,
        textTransform: "uppercase",
        letterSpacing: "0.08em",
        marginBottom: 12,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        Tensor shapes nel forward pass
      </div>
      <div style={{
        display: "flex",
        gap: 4,
        marginBottom: 14,
      }}>
        {steps.map((s, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            style={{
              flex: 1,
              height: 4,
              borderRadius: 2,
              background: i <= step ? s.color : COLORS.border,
              border: "none",
              cursor: "pointer",
              transition: "all 0.2s",
              padding: 0,
            }}
          />
        ))}
      </div>
      <div style={{
        display: "flex",
        alignItems: "baseline",
        gap: 10,
        marginBottom: 6,
      }}>
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 13,
          fontWeight: 700,
          color: steps[step].color,
        }}>
          {steps[step].label}
        </span>
        <code style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 14,
          color: COLORS.text,
          background: `${steps[step].color}15`,
          padding: "2px 8px",
          borderRadius: 4,
        }}>
          {steps[step].shape}
        </code>
      </div>
      <div style={{ fontSize: 12, color: COLORS.textMuted }}>
        {steps[step].desc}
      </div>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        marginTop: 10,
      }}>
        <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}
          style={{
            background: "none", border: `1px solid ${COLORS.border}`, color: step === 0 ? COLORS.border : COLORS.text,
            borderRadius: 6, padding: "4px 12px", cursor: step === 0 ? "default" : "pointer",
            fontSize: 12, fontFamily: "'JetBrains Mono', monospace",
          }}>← prev</button>
        <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} disabled={step === steps.length - 1}
          style={{
            background: "none", border: `1px solid ${COLORS.border}`, color: step === steps.length - 1 ? COLORS.border : COLORS.text,
            borderRadius: 6, padding: "4px 12px", cursor: step === steps.length - 1 ? "default" : "pointer",
            fontSize: 12, fontFamily: "'JetBrains Mono', monospace",
          }}>next →</button>
      </div>
    </div>
  );
};

export default function TransformerExplainer() {
  const [selected, setSelected] = useState("attention");

  return (
    <div style={{
      minHeight: "100vh",
      background: COLORS.bg,
      color: COLORS.text,
      fontFamily: "'IBM Plex Sans', -apple-system, sans-serif",
      padding: "24px 16px",
    }}>
      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: 20 }}>
          <h1 style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 22,
            fontWeight: 800,
            color: COLORS.text,
            margin: 0,
            letterSpacing: "-0.03em",
          }}>
            🧠 NanoTransformer — 107M
          </h1>
          <p style={{
            fontSize: 12,
            color: COLORS.textMuted,
            margin: "6px 0 0",
          }}>
            Clicca su ogni componente per capire cosa fa e dove sono i limiti
          </p>
        </div>

        <ModelStats />

        <div style={{
          display: "grid",
          gridTemplateColumns: "minmax(240px, 300px) 1fr",
          gap: 20,
          marginTop: 16,
          alignItems: "start",
        }}>
          <div>
            <FlowDiagram selected={selected} onSelect={setSelected} />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 16, position: "sticky", top: 24 }}>
            <DetailPanel id={selected} />
            <ForwardPassViz />
            <ParamBreakdown />
          </div>
        </div>
      </div>
    </div>
  );
}