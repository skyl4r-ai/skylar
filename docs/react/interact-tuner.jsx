// =================================================================
// @copyright: A. Ivanovitch | CEO SKYL4R | 2026
// =================================================================
import { useState, useMemo } from "react";

// ═══════════════════════════════════════════════════════════════
// MATHEMATICAL FOUNDATIONS
// ═══════════════════════════════════════════════════════════════
// PARAMETER COUNT: exact from model_explain.py (SwiGLU, GQA, QK-Norm, bias=False)
// LEARNING RATE: GPT-3 power law fit (Brown et al. 2020)
// BATCH SIZE: GPT-3/LLaMA empirical table, log-linear interpolation
// CHINCHILLA: D_opt ≈ 20 × N (Hoffmann et al. 2022)
// FLOPS: 6ND approximation (Kaplan et al. 2020)
// ACTIVATION MEMORY: 34 × T × d per layer (Korthikanti et al. 2022)
// MFU: calibrated on RTX4090 measured throughput (config.py anchors)
// SFT EPOCHS: arxiv:2602.11149 (multi-epoch on small > single-epoch on large)

const PRESETS = {
  test:       { d:128,   h:4,   kv:4,  dh:null, L:4,   ff:256,    ctx:4096,    tie:true,  qk:true, label:"Test (~6M)" },
  small:      { d:512,   h:8,   kv:4,  dh:null, L:8,   ff:1024,   ctx:8192,    tie:true,  qk:true, label:"Small (~40M)" },
  small_plus: { d:640,   h:10,  kv:5,  dh:null, L:10,  ff:1792,   ctx:16384,   tie:true,  qk:true, label:"Small+ (~70M)" },
  medium:     { d:768,   h:12,  kv:4,  dh:null, L:12,  ff:2048,   ctx:16384,   tie:true,  qk:true, label:"Medium (~107M)" },
  large:      { d:1024,  h:16,  kv:4,  dh:null, L:28,  ff:2816,   ctx:16384,   tie:true,  qk:true, label:"Large (~358M)" },
  "1B":       { d:1536,  h:16,  kv:4,  dh:null, L:32,  ff:5120,   ctx:32768,   tie:true,  qk:true, label:"1B (~1.0B)" },
  "4b":       { d:2560,  h:32,  kv:8,  dh:128,  L:36,  ff:9728,   ctx:32768,   tie:true,  qk:true, label:"4B (Qwen3-4B)" },
  "8b":       { d:4096,  h:32,  kv:8,  dh:128,  L:36,  ff:12288,  ctx:32768,   tie:false, qk:true, label:"8B (Qwen3-8B)" },
  "14b":      { d:5120,  h:40,  kv:8,  dh:128,  L:40,  ff:17408,  ctx:32768,   tie:false, qk:true, label:"14B (Qwen3-14B)" },
  "32b":      { d:5120,  h:64,  kv:8,  dh:128,  L:64,  ff:25600,  ctx:131072,  tie:false, qk:true, label:"32B (Qwen3-32B)" },
  "64b":      { d:8192,  h:64,  kv:8,  dh:128,  L:72,  ff:28672,  ctx:131072,  tie:false, qk:true, label:"64B (~61B)" },
  "96b":      { d:10240, h:80,  kv:8,  dh:128,  L:80,  ff:32768,  ctx:131072,  tie:false, qk:true, label:"96B (~99B)" },
  "128b":     { d:11264, h:88,  kv:8,  dh:128,  L:92,  ff:32768,  ctx:131072,  tie:false, qk:true, label:"128B (~128B)" },
};
const V = 40960;

// ═══════════════════════════════════════════════════════════════
// GPU DATABASE — bf16 dense peak, official specs
// ═══════════════════════════════════════════════════════════════
const GPUS = [
  { id:"4090",  name:"RTX 4090",     vram:24,  tf:83,   cls:"c", phr:null, buy:"$1,600" },
  { id:"5090",  name:"RTX 5090",     vram:32,  tf:210,  cls:"c", phr:null, buy:"$2,000" },
  { id:"pro6k", name:"RTX PRO 6000", vram:96,  tf:170,  cls:"w", phr:null, buy:"$6,800" },
  { id:"a100",  name:"A100 80GB",    vram:80,  tf:312,  cls:"d", phr:1.5,  buy:null },
  { id:"h100",  name:"H100 SXM",     vram:80,  tf:990,  cls:"d", phr:2.5,  buy:null },
  { id:"h200",  name:"H200 SXM",     vram:141, tf:990,  cls:"d", phr:3.5,  buy:null },
  { id:"b200",  name:"B200 SXM",     vram:192, tf:2250, cls:"d", phr:5.0,  buy:null },
];

// ═══════════════════════════════════════════════════════════════
// EXACT PARAMETER COUNT — from model_explain.py
// ═══════════════════════════════════════════════════════════════
// Per layer: Wq(d→H*dh) + Wk(d→kv*dh) + Wv(d→kv*dh) + Wo(H*dh→d)
//          + SwiGLU: w1(d→ff) + w2(ff→d) + w3(d→ff)
//          + 2×RMSNorm(d) + 2×QKNorm(dh) if qk_norm
// Global: Embedding(V×d) + finalNorm(d) + lmHead(d×V) if !tie
function countN(p) {
  const pr = PRESETS[p];
  const dh = pr.dh ?? Math.floor(pr.d / pr.h);
  const dA = pr.h * dh, dK = pr.kv * dh;
  const attn = pr.d * dA + pr.d * dK * 2 + dA * pr.d;
  const ffn = 3 * pr.d * pr.ff;
  const nrm = 2 * pr.d + (pr.qk ? 2 * dh : 0);
  return V * pr.d + pr.L * (attn + ffn + nrm) + pr.d + (pr.tie ? 0 : V * pr.d);
}

// ═══════════════════════════════════════════════════════════════
// MFU — calibrated on measured throughput
// ═══════════════════════════════════════════════════════════════
// Anchors from config.py:
//   RTX4090 + small(40M):  100K tok/s → eff 24.0 TF → 24/83 = 28.9%
//   RTX4090 + medium(107M): 51K tok/s → eff 32.7 TF → 32.7/83 = 39.4%
const MFU_T = [[1e6,.10],[10e6,.22],[50e6,.30],[100e6,.38],[500e6,.42],[1e9,.43],[10e9,.45],[100e9,.48]];
function getMFU(N, cls) {
  let m;
  if (N <= MFU_T[0][0]) m = MFU_T[0][1];
  else if (N >= MFU_T[MFU_T.length-1][0]) m = MFU_T[MFU_T.length-1][1];
  else { for (let i=0;i<MFU_T.length-1;i++) if (N>=MFU_T[i][0]&&N<MFU_T[i+1][0]) {
    const t=(Math.log(N)-Math.log(MFU_T[i][0]))/(Math.log(MFU_T[i+1][0])-Math.log(MFU_T[i][0]));
    m=MFU_T[i][1]+t*(MFU_T[i+1][1]-MFU_T[i][1]); break;
  }}
  return m + (cls==="d" ? 0.05 : 0);
}

// ═══════════════════════════════════════════════════════════════
// VRAM MODEL — Korthikanti et al. 2022 (Megatron-LM)
// ═══════════════════════════════════════════════════════════════
// Fixed: 16N bytes (2 bf16w + 4 fp32master + 4 fp32m + 4 fp32v + 2 bf16grad)
// Act/layer/sample: 34 × T × d bytes (Flash Attention, no T² attn scores)
// GC: inputs only (2 × T × d / layer) + 1 layer recompute (34 × T × d)
const OH = 1.5; // GB framework overhead
function vramGB(p, bs, sl, gc) {
  const pr=PRESETS[p], N=countN(p), f=16*N;
  const a = gc
    ? bs*(pr.L*sl*pr.d*2 + sl*pr.d*34)
    : bs*pr.L*sl*pr.d*34;
  return { tot:(f+a)/1e9+OH, fix:f/1e9+OH, act:a/1e9 };
}
function maxBS(p, sl, vram, gc) {
  const pr=PRESETS[p], avail=vram-(16*countN(p)/1e9+OH);
  if(avail<=0)return 0;
  const per = gc ? (pr.L*sl*pr.d*2+sl*pr.d*34)/1e9 : (pr.L*sl*pr.d*34)/1e9;
  return per<=0?64:Math.max(0,Math.floor(avail/per));
}

// ═══════════════════════════════════════════════════════════════
// TRAINING TIME — T = 6ND / (peak × MFU × nGPU × eff × 1e12)
// ═══════════════════════════════════════════════════════════════
const SE=[[1,1],[2,.92],[4,.90],[8,.85],[16,.78],[32,.70],[64,.65],[128,.60],[256,.55],[512,.50]];
function sEff(n){for(let i=SE.length-1;i>=0;i--)if(n>=SE[i][0])return SE[i][1];return 1;}
function tTime(N,D,gpu,ng){
  const m=getMFU(N,gpu.cls),e=sEff(ng),tf=gpu.tf*m*ng*e;
  return{sec:(6*N*D)/(tf*1e12),tok:D/((6*N*D)/(tf*1e12)),mfu:m,eff:e};
}

// ═══════════════════════════════════════════════════════════════
// GPU RECOMMENDATIONS — target < 30 days
// ═══════════════════════════════════════════════════════════════
function gpuRec(p, D, sl) {
  const N=countN(p);
  return GPUS.map(gpu=>{
    const mb0=maxBS(p,sl,gpu.vram,false),mb1=maxBS(p,sl,gpu.vram,true);
    if(mb0<1&&mb1<1) return{gpu,fits:false};
    const gc=mb0<1, mb=Math.min(gc?mb1:mb0,64);
    const t1=tTime(N,D,gpu,1);
    let ng=1;
    if(t1.sec>30*86400) for(let n=2;n<=512;n*=2){
      if(tTime(N,D,gpu,n).sec<=30*86400){ng=n;break;} ng=n;
    }
    const tO=ng>1?tTime(N,D,gpu,ng):t1;
    return{gpu,fits:true,gc,mb,t1s:t1.sec,tOs:tO.sec,tok1:t1.tok,tokO:tO.tok,mfu:t1.mfu,ng,cost:gpu.phr?(tO.sec/3600)*gpu.phr*ng:null};
  });
}

// ═══════════════════════════════════════════════════════════════
// SCALING LAWS
// ═══════════════════════════════════════════════════════════════
// LR: 0.8 × N^(-0.386) — fit on GPT-3 table (Brown et al. 2020)
// Verified: 125M→6e-4, 350M→3e-4, 1.3B→2e-4, 6.7B→1.2e-4, 13B→1e-4
function optLR(N){return Math.min(1e-3,Math.max(3e-5,0.8*Math.pow(N,-0.386)));}

// Batch tokens: log-linear interpolation on GPT-3/LLaMA table
function optBT(N){
  const T=[[5e6,16384],[30e6,32768],[100e6,131072],[300e6,262144],[1e9,524288],[3e9,1048576],[8e9,2097152],[15e9,2097152],[30e9,4194304],[60e9,4194304],[130e9,4194304]];
  if(N<=T[0][0])return T[0][1];if(N>=T[T.length-1][0])return T[T.length-1][1];
  for(let i=0;i<T.length-1;i++) if(N>=T[i][0]&&N<T[i+1][0]){
    const t=(Math.log(N)-Math.log(T[i][0]))/(Math.log(T[i+1][0])-Math.log(T[i][0]));
    return Math.pow(2,Math.round(Math.log2(T[i][1])+t*(Math.log2(T[i+1][1])-Math.log2(T[i][1]))));
  }
  return 262144;
}

// ═══════════════════════════════════════════════════════════════
// BATCH DECOMPOSITION — VRAM-aware
// ═══════════════════════════════════════════════════════════════
// FIX: The mathematical optimal effective batch is decomposed into
// batch_size × grad_accum × seq_len. batch_size must be conservative
// to avoid OOM on any reasonable GPU. grad_accum absorbs the rest.
//
// Conservative batch_size by model size (ensures fits in 24-80GB):
//   N < 100M:  bs ≤ 16 (any GPU)
//   N < 500M:  bs ≤ 8  (any GPU with Flash Attn)
//   N < 2B:    bs ≤ 2  (needs ≥32GB, or GC on 24GB)
//   N < 10B:   bs = 1  (needs ≥80GB)
//   N ≥ 10B:   bs = 1  (needs GC + ≥80GB)
function decompose(targetTok, sl, N) {
  const samplesPerStep = Math.max(1, Math.round(targetTok / sl));
  // Conservative bs ceiling by model size
  const maxBs = N < 100e6 ? 16 : N < 500e6 ? 8 : N < 2e9 ? 2 : 1;
  let bs = Math.min(maxBs, samplesPerStep);
  // Find clean divisor
  for (let b = bs; b >= 1; b--) {
    if (samplesPerStep % b === 0) { bs = b; break; }
  }
  const ga = Math.max(1, Math.round(samplesPerStep / bs));
  return { bs, ga };
}

// ═══════════════════════════════════════════════════════════════
// PRE-TRAIN PARAMS
// ═══════════════════════════════════════════════════════════════
function preTrain(preset, D) {
  const pr = PRESETS[preset], N = countN(preset);
  const cr = D / N, sl = pr.ctx;
  const lr = optLR(N);
  const tbt = optBT(N);
  const { bs, ga } = decompose(tbt, sl, N);
  const ebt = bs * ga * sl;
  const ms = Math.max(100, Math.round(D / ebt));
  // FIX: warmup clamped to [100, min(2000, 10% of steps)]
  const ws = Math.max(100, Math.min(2000, Math.min(Math.round(ms * 0.1), Math.round(ms * 0.01) + 100)));
  // Re-derive: 1% but never > 10% of total
  const wsFixed = Math.min(Math.max(100, Math.round(ms * 0.01)), Math.round(ms * 0.1));
  const dropout = N >= 1e9 ? 0.0 : 0.1;
  const ee = Math.max(50, Math.round(ms / 20));
  const ds = cr >= 18 ? "optimal" : cr >= 10 ? "acceptable" : cr >= 5 ? "undertrained" : "severely_undertrained";
  return {
    N, sl, lr, bs, ga, ebt, ms, ws: wsFixed,
    dropout, ee, se: Math.max(100, Math.round(ms / 10)), sae: ee,
    cr, ds, wd: 0.1, gc: 1.0, mlr: lr * 0.1,
    sch: "cosine", bt: [0.9, 0.95], D,
  };
}

// ═══════════════════════════════════════════════════════════════
// SFT PARAMS
// ═══════════════════════════════════════════════════════════════
// Key differences from pre-train:
//   - LR: 10× lower (fine-tuning, not learning from scratch)
//   - Epochs: multi-epoch for small datasets (arxiv:2602.11149)
//   - Dropout: 0.05 (NOT configurable in train_sft.py CLI — inherits from model)
//   - Batch: smaller (variable-length padding, less efficient packing)
//   - Warmup: shorter (model already has learned representations)
function sftTrain(preset, nC, avgT) {
  const pr = PRESETS[preset], N = countN(preset);
  const tst = nC * avgT;
  const sl = Math.pow(2, Math.round(Math.log2(Math.min(pr.ctx, Math.max(512, avgT * 2)))));
  const lr = optLR(N) / 10;
  const ep = nC < 1000 ? 5 : nC < 5000 ? 3 : nC < 50000 ? 2 : 1;
  const bs = N >= 1e9 ? 1 : N >= 100e6 ? 4 : 8;
  const ga = N >= 1e9 ? 8 : 4;
  const eb = bs * ga;
  const spe = Math.max(1, Math.ceil(nC / eb));
  const ms = spe * ep;
  // FIX: warmup MUST be ≤ 10% of total steps (was broken with tiny datasets)
  const ws = Math.min(Math.max(10, Math.round(ms * 0.05)), Math.round(ms * 0.1));
  const ee = Math.max(10, Math.round(ms / 20));
  const mn = N < 50e6 ? 500 : N < 200e6 ? 2000 : N < 1e9 ? 5000 : N < 10e9 ? 20000 : 50000;
  const ds = nC >= mn * 2 ? "optimal" : nC >= mn ? "acceptable" : nC >= mn / 2 ? "marginal" : "insufficient";
  return {
    N, sl, lr, bs, ga, eb, ms, ws, ep, spe, ee,
    se: Math.max(20, Math.round(ms / 10)),
    sae: Math.max(10, Math.round(ms / 20)),
    tst, mn, ds, wd: 0.1, gc: 1.0, mlr: lr * 0.1,
    sch: "cosine", bt: [0.9, 0.95], D: tst * ep,
  };
}

// ═══════════════════════════════════════════════════════════════
// FORMATTING
// ═══════════════════════════════════════════════════════════════
const fmt=n=>{if(n>=1e12)return(n/1e12).toFixed(2)+"T";if(n>=1e9)return(n/1e9).toFixed(2)+"B";if(n>=1e6)return(n/1e6).toFixed(1)+"M";if(n>=1e3)return(n/1e3).toFixed(1)+"K";return""+n};
const fN=n=>n.toLocaleString("en-US");
const fT=s=>{if(s<60)return Math.round(s)+"s";if(s<3600)return Math.round(s/60)+"min";if(s<86400)return(s/3600).toFixed(1)+"h";if(s<86400*30)return(s/86400).toFixed(1)+"d";if(s<86400*365)return(s/(86400*30)).toFixed(1)+"mo";return(s/(86400*365)).toFixed(1)+"yr"};
const fTV=s=>{if(s<3600)return Math.round(s/60)+" minuti";if(s<86400)return(s/3600).toFixed(1)+" ore";if(s<86400*30)return(s/86400).toFixed(1)+" giorni";if(s<86400*365)return(s/(86400*30)).toFixed(1)+" mesi";return(s/(86400*365)).toFixed(1)+" anni"};

// ═══════════════════════════════════════════════════════════════
// COLORS & STYLES
// ═══════════════════════════════════════════════════════════════
const SC={optimal:"#22c55e",acceptable:"#eab308",undertrained:"#f97316",severely_undertrained:"#ef4444",marginal:"#f97316",insufficient:"#ef4444"};
const SL={optimal:"Ottimale — convergenza Chinchilla",acceptable:"Accettabile — sotto l'ottimo ma funzionale",undertrained:"Sotto-addestrato — non raggiunge il potenziale",severely_undertrained:"Gravemente sotto-addestrato — modello troppo grande per questi dati",marginal:"Marginale — qualità SFT limitata",insufficient:"Insufficiente — servono più dati o un modello più piccolo"};
const C={bg:"#0a0e17",card:"#0f1520",cb:"#1a2030",ib:"#131825",ibr:"#1e2a3e",tx:"#c8ccd4",td:"#5a6270",tdd:"#3a4250",ac:"#7eb8ff",hd:"#e2e6ed"};
const FN="'JetBrains Mono','Fira Code','SF Mono','Cascadia Code',monospace";
const IS={width:"100%",padding:"9px 12px",borderRadius:6,boxSizing:"border-box",background:C.ib,border:`1px solid ${C.ibr}`,color:C.hd,fontSize:13,fontFamily:FN,outline:"none"};

// ═══════════════════════════════════════════════════════════════
// COMPONENT
// ═══════════════════════════════════════════════════════════════
export default function TrainingCalculator() {
  const [mode, setMode] = useState("pretrain");
  const [preset, setPreset] = useState("medium");
  const [tokIn, setTokIn] = useState("2000000000");
  const [nCIn, setNCIn] = useState("10000");
  const [avgIn, setAvgIn] = useState("256");

  const toks = parseInt(tokIn.replace(/\D/g,""))||0;
  const nC = parseInt(nCIn.replace(/\D/g,""))||0;
  const avg = parseInt(avgIn.replace(/\D/g,""))||256;

  const r = useMemo(()=>{
    if(mode==="pretrain"&&toks>0) return preTrain(preset,toks);
    if(mode==="sft"&&nC>0) return sftTrain(preset,nC,avg);
    return null;
  },[mode,preset,toks,nC,avg]);

  const gr = useMemo(()=>r?gpuRec(preset,r.D,r.sl):[],[r,preset]);
  const N = countN(preset);
  const pr = PRESETS[preset];

  return (
    <div style={{fontFamily:FN,background:C.bg,color:C.tx,minHeight:"100vh",padding:"20px 16px",boxSizing:"border-box"}}>
      <div style={{maxWidth:840,margin:"0 auto"}}>

        {/* Header */}
        <div style={{marginBottom:24}}>
          <h1 style={{fontSize:18,fontWeight:700,color:C.hd,margin:0,letterSpacing:"-0.02em"}}>⚙ Training Hyperparameter Calculator</h1>
          <div style={{fontSize:10,color:C.tdd,marginTop:3}}>Chinchilla · GPT-3 · Kaplan · Megatron-LM · Korthikanti · DeepSeek-V3</div>
        </div>

        {/* Mode */}
        <div style={{display:"flex",gap:2,marginBottom:14,background:C.ib,borderRadius:8,padding:3}}>
          {[["pretrain","Pre-Training"],["sft","SFT (Fine-Tuning)"]].map(([k,l])=>(
            <button key={k} onClick={()=>setMode(k)} style={{flex:1,padding:"9px 0",borderRadius:6,border:"none",cursor:"pointer",fontSize:12,fontWeight:600,fontFamily:"inherit",transition:"all .15s",background:mode===k?"#1e2a3e":"transparent",color:mode===k?C.ac:C.td}}>{l}</button>
          ))}
        </div>

        {/* Preset */}
        <Fl l="Model Size">
          <select value={preset} onChange={e=>setPreset(e.target.value)} style={{...IS,cursor:"pointer"}}>
            {Object.entries(PRESETS).map(([k,v])=><option key={k} value={k}>{v.label}</option>)}
          </select>
        </Fl>

        {/* Model strip */}
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(90px,1fr))",gap:5,marginBottom:14,padding:9,borderRadius:8,background:C.card,border:`1px solid ${C.cb}`}}>
          {[["Params",fmt(N)],["d_model",pr.d],["heads",pr.h],["kv_heads",pr.kv],["layers",pr.L],["d_ff",pr.ff],["d_head",pr.dh??Math.floor(pr.d/pr.h)],["ctx",fmt(pr.ctx)],["tie",pr.tie?"✓":"✗"]].map(([a,b])=>(
            <div key={a}><div style={{fontSize:9,color:C.tdd,textTransform:"uppercase",letterSpacing:".04em"}}>{a}</div><div style={{fontSize:12,color:C.hd,fontWeight:600}}>{b}</div></div>
          ))}
        </div>

        {/* Data input */}
        {mode==="pretrain" ? (
          <Fl l="Token Totali" h={`= ${fmt(toks)} · Chinchilla ottimale: ${fmt(N*20)}`}>
            <input type="text" value={tokIn} onChange={e=>setTokIn(e.target.value)} style={IS}/>
          </Fl>
        ) : (
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10,marginBottom:14}}>
            <Fl l="N° Conversazioni"><input type="text" value={nCIn} onChange={e=>setNCIn(e.target.value)} style={IS}/></Fl>
            <Fl l="Token Medi / Conv"><input type="text" value={avgIn} onChange={e=>setAvgIn(e.target.value)} style={IS}/></Fl>
          </div>
        )}

        {r && (<>
          {/* Status */}
          <div style={{padding:"10px 14px",borderRadius:8,marginBottom:14,background:`${SC[r.ds]}11`,border:`1px solid ${SC[r.ds]}33`,display:"flex",alignItems:"center",gap:10}}>
            <div style={{width:9,height:9,borderRadius:"50%",background:SC[r.ds],boxShadow:`0 0 8px ${SC[r.ds]}66`,flexShrink:0}}/>
            <div>
              <div style={{fontSize:12,fontWeight:600,color:SC[r.ds]}}>{r.ds.replace(/_/g," ").toUpperCase()}</div>
              <div style={{fontSize:10,color:"#8a8f9a",marginTop:1}}>{SL[r.ds]}</div>
            </div>
          </div>

          {/* Chinchilla / SFT summary */}
          {mode==="pretrain" && (
            <div style={{padding:"10px 14px",borderRadius:8,marginBottom:14,background:C.card,border:`1px solid ${C.cb}`}}>
              <div style={{fontSize:10,color:C.td,marginBottom:2}}>CHINCHILLA D/N</div>
              <span style={{fontSize:24,fontWeight:700,color:SC[r.ds]}}>{r.cr.toFixed(1)}×</span>
              <span style={{fontSize:11,color:C.td,marginLeft:8}}>(target: 20×)</span>
            </div>
          )}
          {mode==="sft" && (
            <div style={{padding:"10px 14px",borderRadius:8,marginBottom:14,background:C.card,border:`1px solid ${C.cb}`,display:"flex",gap:16,flexWrap:"wrap",alignItems:"center"}}>
              <div><span style={{fontSize:18,fontWeight:700,color:C.hd}}>{fN(nC)}</span><span style={{fontSize:11,color:C.td,marginLeft:4}}>conv</span></div>
              <div><span style={{fontSize:18,fontWeight:700,color:C.ac}}>{r.ep}</span><span style={{fontSize:11,color:C.td,marginLeft:4}}>epoche</span></div>
              <div style={{fontSize:11,color:C.td}}>minimo: {fN(r.mn)} conv per {pr.label}</div>
            </div>
          )}

          {/* Training params */}
          <Sec title="Parametri di Training">
            <PS t="ADDESTRAMENTO">
              <PR l="max_steps" v={fN(r.ms)} n={mode==="sft"?`${r.ep}ep × ${fN(r.spe)}/ep`:`${fmt(r.D)} / ${fmt(r.ebt)} tok/step`}/>
              <PR l="warmup_steps" v={r.ws} n={`${(r.ws/r.ms*100).toFixed(1)}% del totale`}/>
              <PR l="lr_schedule" v={r.sch}/>
            </PS>
            <PS t="OTTIMIZZATORE (AdamW)">
              <PR l="lr" v={r.lr.toExponential(2)} n={mode==="sft"?`pretrain(${optLR(N).toExponential(1)})/10`:"0.8×N^(-0.386) GPT-3"} hi/>
              <PR l="min_lr" v={r.mlr.toExponential(2)} n="lr × 0.1"/>
              <PR l="weight_decay" v={r.wd}/>
              <PR l="betas" v={`(${r.bt.join(", ")})`}/>
              <PR l="grad_clip" v={r.gc}/>
            </PS>
            <PS t="BATCH">
              <PR l="batch_size" v={r.bs} n={mode==="pretrain"?`conservativo per VRAM (N=${fmt(N)})`:""} hi/>
              <PR l="grad_accum" v={r.ga} hi/>
              <PR l="effective" v={mode==="pretrain"?`${r.bs}×${r.ga}×${fmt(r.sl)} = ${fmt(r.ebt)} tok/step`:`${r.bs}×${r.ga} = ${r.eb} conv/step`}/>
            </PS>
            <PS t="SEQUENZA">
              <PR l="seq_len" v={fN(r.sl)} n={mode==="sft"?`min(ctx ${fmt(pr.ctx)}, 2×avg ${avg})`:`= max_seq_len del preset`} hi/>
              {mode==="pretrain" && <PR l="dropout" v={r.dropout} n={N>=1e9?"0.0 per ≥1B (dati regolarizzano)":"0.1 per <1B"}/>}
            </PS>
            <PS t="LOGGING & CHECKPOINT">
              <PR l="eval_every" v={fN(r.ee)} n={`~${Math.round(r.ms/r.ee)} valutazioni`}/>
              <PR l="save_every" v={fN(r.se)} n={`~${Math.round(r.ms/r.se)} checkpoint`}/>
              <PR l="sample_every" v={fN(r.sae)}/>
            </PS>
          </Sec>

          {/* GPU TABLE */}
          <Sec title="GPU — Tempo Stimato & VRAM">
            <div style={{fontSize:9,color:C.tdd,padding:"5px 14px",borderBottom:"1px solid #111722"}}>
              FLOPs = 6 × {fmt(N)} × {fmt(r.D)} = <strong style={{color:C.td}}>{fmt(6*N*r.D)}</strong> FLOP · MFU calibrata su throughput misurato
            </div>
            <div style={{overflowX:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:11,minWidth:700}}>
                <thead>
                  <tr style={{borderBottom:`1px solid ${C.cb}`}}>
                    {["GPU","VRAM","","GC","MaxBS","1× GPU","Consiglio","GPU","Costo"].map(h=>(
                      <th key={h} style={{padding:"6px 7px",textAlign:"left",fontSize:9,color:C.tdd,textTransform:"uppercase",letterSpacing:".04em",fontWeight:600,whiteSpace:"nowrap"}}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {gr.map(x=>(
                    <tr key={x.gpu.id} style={{borderBottom:"1px solid #111722",opacity:x.fits?1:0.30}}>
                      <td style={{padding:"6px 7px",fontWeight:600,color:x.fits?C.hd:C.td,whiteSpace:"nowrap"}}>{x.gpu.name}</td>
                      <td style={{padding:"6px 7px",color:C.td}}>{x.gpu.vram}GB</td>
                      <td style={{padding:"6px 7px"}}><span style={{color:x.fits?"#22c55e":"#ef4444",fontWeight:700}}>{x.fits?"✓":"✗"}</span></td>
                      <td style={{padding:"6px 7px",color:x.fits?(x.gc?"#eab308":C.tdd):C.tdd}}>{x.fits?(x.gc?"sì":"no"):"—"}</td>
                      <td style={{padding:"6px 7px",color:C.tx}}>{x.fits?x.mb:"—"}</td>
                      <td style={{padding:"6px 7px",color:C.tx,whiteSpace:"nowrap"}}>{x.fits?fT(x.t1s):"—"}</td>
                      <td style={{padding:"6px 7px",color:C.ac,fontWeight:600,whiteSpace:"nowrap"}}>{x.fits?fT(x.tOs):"—"}</td>
                      <td style={{padding:"6px 7px",color:x.fits&&x.ng>1?"#eab308":C.tx}}>{x.fits?(x.ng>1?x.ng+"×":"1×"):"—"}</td>
                      <td style={{padding:"6px 7px",color:C.td,whiteSpace:"nowrap"}}>{x.fits&&x.cost!=null?`$${Math.round(x.cost).toLocaleString()}`:(x.fits&&x.gpu.buy?x.gpu.buy:"—")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {(()=>{
              const fc=gr.filter(x=>x.fits&&x.gpu.cls!=="d").sort((a,b)=>a.tOs-b.tOs)[0];
              const ch=gr.filter(x=>x.fits&&x.cost!=null).sort((a,b)=>a.cost-b.cost)[0];
              if(!fc&&!ch) return(
                <div style={{padding:"10px 14px",background:"#1a0a0a",borderTop:"1px solid #3a1515"}}>
                  <div style={{color:"#ef4444",fontWeight:700,fontSize:12}}>⚠ Nessuna GPU singola può addestrare questo modello</div>
                  <div style={{color:C.td,fontSize:10,marginTop:3}}>Serve tensor parallelism (Megatron-LM) o ZeRO Stage 3 (DeepSpeed).</div>
                </div>
              );
              return(
                <div style={{padding:"10px 14px",background:"#0a1520",borderTop:`1px solid ${C.cb}`}}>
                  <div style={{fontSize:9,color:C.td,marginBottom:5,textTransform:"uppercase",letterSpacing:".06em"}}>Raccomandazione</div>
                  {fc&&<div style={{fontSize:11,color:C.hd,marginBottom:3}}>
                    <strong style={{color:"#22c55e"}}>Local:</strong> {fc.gpu.name}{fc.ng>1?` × ${fc.ng}`:""}{fc.gc?" + gradient_checkpointing":""}
                    {" → "}<strong style={{color:C.ac}}>{fTV(fc.tOs)}</strong>
                    {" · "}{fmt(fc.tokO)} tok/s · MFU {(fc.mfu*100).toFixed(0)}%
                  </div>}
                  {ch&&<div style={{fontSize:11,color:C.hd}}>
                    <strong style={{color:"#eab308"}}>Cloud:</strong> {ch.gpu.name}{ch.ng>1?` × ${ch.ng}`:""}
                    {" → "}<strong style={{color:C.ac}}>{fTV(ch.tOs)}</strong>
                    {" · ~$"}{Math.round(ch.cost).toLocaleString()}
                  </div>}
                </div>
              );
            })()}
          </Sec>

          {/* VRAM */}
          <Sec title="VRAM Breakdown (batch=1)">
            {(()=>{
              const v=vramGB(preset,1,r.sl,false),vg=vramGB(preset,1,r.sl,true);
              return(
                <div style={{padding:"10px 14px",display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,fontSize:11}}>
                  <div>
                    <div style={{color:C.tdd,fontSize:9,textTransform:"uppercase"}}>Params+Optim+Grad</div>
                    <div style={{color:C.hd,fontWeight:600}}>{(16*N/1e9).toFixed(2)} GB</div>
                    <div style={{color:C.tdd,fontSize:9}}>16 × {fmt(N)}</div>
                  </div>
                  <div>
                    <div style={{color:C.tdd,fontSize:9,textTransform:"uppercase"}}>Attivazioni (1 sample)</div>
                    <div style={{color:C.hd,fontWeight:600}}>{v.act.toFixed(2)} GB</div>
                    <div style={{color:C.tdd,fontSize:9}}>{pr.L}×34×{fmt(r.sl)}×{pr.d}</div>
                  </div>
                  <div>
                    <div style={{color:C.tdd,fontSize:9,textTransform:"uppercase"}}>Totale senza GC</div>
                    <div style={{color:C.ac,fontWeight:700,fontSize:14}}>{v.tot.toFixed(1)} GB</div>
                  </div>
                  <div>
                    <div style={{color:C.tdd,fontSize:9,textTransform:"uppercase"}}>Totale con GC</div>
                    <div style={{color:"#22c55e",fontWeight:700,fontSize:14}}>{vg.tot.toFixed(1)} GB</div>
                  </div>
                </div>
              );
            })()}
          </Sec>

          {/* CLI — FIX: SFT command does NOT include --dropout (flag doesn't exist in train_sft.py) */}
          <div style={{marginBottom:14}}>
            <div style={{fontSize:10,color:C.td,marginBottom:4,textTransform:"uppercase",letterSpacing:".06em"}}>Comando</div>
            <pre style={{background:C.ib,border:`1px solid ${C.cb}`,borderRadius:8,padding:12,fontSize:11,color:C.ac,overflowX:"auto",lineHeight:1.7,margin:0,whiteSpace:"pre-wrap",wordBreak:"break-all"}}>
{mode==="pretrain"
?`python train.py \\
  --data data/tokenized_corpus \\
  --preset ${preset} \\
  --seq_len ${r.sl} \\
  --batch_size ${r.bs} \\
  --grad_accum ${r.ga} \\
  --max_steps ${r.ms} \\
  --lr ${r.lr.toExponential(2)} \\
  --warmup_steps ${r.ws} \\
  --weight_decay ${r.wd} \\
  --grad_clip ${r.gc} \\
  --dropout ${r.dropout} \\
  --lr_schedule ${r.sch} \\
  --eval_every ${r.ee} \\
  --save_every ${r.se} \\
  --sample_every ${r.sae} \\
  --bf16`
// SFT: NO --dropout (inherits from model), NO --preset (model from base_model)
// Uses --epochs instead of --max_steps (overrides internally)
:`python train_sft.py \\
  --data data/sft_train.jsonl \\
  --base_model checkpoints/final \\
  --seq_len ${r.sl} \\
  --batch_size ${r.bs} \\
  --grad_accum ${r.ga} \\
  --epochs ${r.ep} \\
  --lr ${r.lr.toExponential(2)} \\
  --warmup_steps ${r.ws} \\
  --weight_decay ${r.wd} \\
  --grad_clip ${r.gc} \\
  --lr_schedule ${r.sch} \\
  --eval_every ${r.ee} \\
  --save_every ${r.se} \\
  --sample_every ${r.sae} \\
  --bf16`}
            </pre>
          </div>

          {/* Derivations */}
          <details>
            <summary style={{fontSize:11,color:C.td,cursor:"pointer",padding:"5px 0",userSelect:"none"}}>Derivazioni matematiche</summary>
            <div style={{background:C.card,border:`1px solid ${C.cb}`,borderRadius:8,padding:12,fontSize:10,color:"#6a7080",lineHeight:1.8,marginTop:5}}>
              <B>Parametri (N) — esatto da model_explain.py</B>
              <div>Embedding(V·d) + L×[Attn(d·H·dh + 2·d·Kv·dh + H·dh·d) + SwiGLU(3·d·dff) + 2·Norm(d) + 2·QKNorm(dh)] + Norm(d) {!pr.tie&&"+ LMHead(V·d)"}</div>
              <div>= <B>{fN(N)}</B></div>
              <br/><B>Learning rate — GPT-3 power law (Brown et al. 2020)</B>
              <div>lr = 0.8 × N^(-0.386) = 0.8 × {fN(N)}^(-0.386) = <B>{r.lr.toExponential(4)}</B></div>
              <div>Anchor: 125M→6e-4 ✓ 350M→3e-4 ✓ 1.3B→2e-4 ✓ 6.7B→1.2e-4 ✓ 13B→1e-4 ✓</div>
              {mode==="sft"&&<div>SFT: pretrain_lr / 10 (fine-tuning, non from scratch)</div>}
              <br/><B>Batch size — GPT-3/LLaMA interpolazione empirica</B>
              <div>Target effective: {fmt(mode==="pretrain"?r.ebt:r.eb)} {mode==="pretrain"?"tok/step":"conv/step"}</div>
              <div>batch_size conservativo per VRAM (N={fmt(N)} → bs≤{r.bs}), grad_accum={r.ga} compensa</div>
              <br/><B>FLOPs (Kaplan et al. 2020)</B>
              <div>F = 6·N·D = 6×{fmt(N)}×{fmt(r.D)} = <B>{fmt(6*N*r.D)}</B></div>
              <br/><B>MFU — calibrata su misurazioni reali</B>
              <div>RTX4090+40M: 100K tok/s → MFU=29% ✓ | RTX4090+107M: 51K tok/s → MFU=39% ✓</div>
              <div>Interpolazione log-lineare: MFU({fmt(N)}) = {(getMFU(N,"c")*100).toFixed(1)}% consumer / {(getMFU(N,"d")*100).toFixed(1)}% datacenter</div>
              <br/><B>Tempo = F / (peak_TFLOPS × MFU × nGPU × scaling_eff × 10¹²)</B>
              <div>Multi-GPU scaling: 2→92% · 4→90% · 8→85% · 16→78% · 32→70% · 64→65%</div>
              <br/><B>VRAM (Korthikanti et al. 2022, Megatron-LM)</B>
              <div>Fisso: 16N = 2(bf16w) + 4(fp32master) + 4(fp32m) + 4(fp32v) + 2(bf16grad) = {(16*N/1e9).toFixed(2)} GB</div>
              <div>Attivazioni: L × 34 × T × d bytes/sample (Flash Attention elimina T² attn scores)</div>
              <div>Con GC: (2L + 34) × T × d bytes/sample (solo input/layer + 1 layer recompute)</div>
              {mode==="pretrain"&&<><br/><B>Chinchilla (Hoffmann et al. 2022)</B><div>D_opt ≈ 20·N = {fmt(N*20)}. Attuale: {fmt(toks)} → ratio = {r.cr.toFixed(2)}×</div></>}
              {mode==="sft"&&<><br/><B>SFT Epochs (arxiv:2602.11149)</B><div>Multi-epoch su piccolo {">"} single-epoch su grande. {'<'}1K→5ep · 1K-5K→3ep · 5K-50K→2ep · {'>'}50K→1ep</div></>}
              <br/><B>Warmup — clamped a ≤10% degli step totali</B>
              <div>{mode==="pretrain"?"1% target, clamp [100, 10%]":"5% target, clamp [10, 10%]"} → {r.ws} step ({(r.ws/r.ms*100).toFixed(1)}%)</div>
            </div>
          </details>
        </>)}
      </div>
    </div>
  );
}

// ═══ Sub-components ═══
function Fl({l,h,children}){return(<div style={{marginBottom:12}}><label style={{fontSize:10,color:C.td,display:"block",marginBottom:4,textTransform:"uppercase",letterSpacing:".06em"}}>{l}</label>{children}{h&&<div style={{fontSize:10,color:C.tdd,marginTop:3}}>{h}</div>}</div>);}
function Sec({title,children}){return(<div style={{borderRadius:8,overflow:"hidden",border:`1px solid ${C.cb}`,marginBottom:14}}><div style={{padding:"7px 14px",background:C.ib,fontSize:10,color:C.td,textTransform:"uppercase",letterSpacing:".06em",fontWeight:700,borderBottom:`1px solid ${C.cb}`}}>{title}</div><div style={{background:"#0c1018"}}>{children}</div></div>);}
function PS({t,children}){return(<div style={{borderBottom:"1px solid #141a26"}}><div style={{padding:"5px 14px",background:"#0e1320",fontSize:9,color:C.tdd,textTransform:"uppercase",letterSpacing:".08em",fontWeight:700}}>{t}</div>{children}</div>);}
function PR({l,v,n,hi}){return(<div style={{display:"flex",alignItems:"baseline",justifyContent:"space-between",padding:"5px 14px",borderTop:"1px solid #111722",gap:6}}><div style={{display:"flex",alignItems:"baseline",gap:5,minWidth:0,overflow:"hidden"}}><span style={{fontSize:11,color:"#5a6a7a",flexShrink:0}}>{l}</span>{n&&<span style={{fontSize:9,color:C.tdd,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{n}</span>}</div><span style={{fontSize:12,fontWeight:600,flexShrink:0,color:hi?C.ac:C.tx}}>{v}</span></div>);}
function B({children}){return<div style={{color:"#8a9ab0",fontWeight:700}}>{children}</div>;}