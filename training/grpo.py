# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Generic GRPO (Group Relative Policy Optimization) — RLVR with a PLUGGABLE verifiable reward,
no reward model, no value model. This is the framework's RL step; a project supplies only a
reward function and a prompt builder (e.g. COBOL: cobol_oracle.reward + a stub prompt).

For each record we sample G completions, score each with reward_fn, turn the group rewards into
GROUP-RELATIVE advantages  A_i = (r_i - mean)/(std + eps),  and take a policy-gradient step
  L = - mean_i [ A_i · mean_t logπ_θ(o_{i,t}) ]   (+ β·KL to a frozen reference, optional).
On-policy, single inner update per generation -> importance ratio ≈ 1 -> clean REINFORCE-with-
group-baseline form of GRPO (no PPO clip needed at 1 inner epoch).

Callbacks (project-supplied, keep this module domain-agnostic):
  reward_fn(record: dict, completion_text: str) -> (float, dict)
  build_prompt(record: dict, tok) -> list[int]        # prompt token ids incl. generation prompt

  from training.grpo import grpo_train, GRPOConfig
  grpo_train(model, tok, records, reward_fn, build_prompt, GRPOConfig(steps=500, group=8, kl_coef=0.02))

Less memory: grad_ckpt=True recomputes the policy's forward in the backward (needs dropout 0), and with bf16=True on CUDA
the reference model is kept in bf16. Long runs: save_every + out_dir write <out_dir>/last (policy, optimizer, step,
random state); resume_dir continues from it, with the reference still built from the starting model.
"""
import copy
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from training.optim import build_optimizer


@dataclass
class GRPOConfig:
    steps: int = 500
    group: int = 8                 # completions sampled per record (G)
    prompts_per_step: int = 4
    max_new: int = 320
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    lr: float = 1e-6
    optimizer: str = "adamw"           # "muon" for a base pretrained with Muon (training/optim.py)
    kl_coef: float = 0.02
    seed: int = 0
    bf16: bool = False
    eos_token_id: int | None = None     # None -> resolved from tok ("<|im_end|>")
    adv_eps: float = 1e-4
    gen_batch: int = 0                  # completions decoded together (generate_group); 0 = the whole group
    grad_ckpt: bool = False             # recompute the policy's forward in the backward (needs dropout 0)
    pad_multiple: int = 1               # scored sequences padded to a multiple: 128 on the hybrid, whose KDA kernels
                                        # are re-tuned for every new length
    save_every: int = 0                 # steps between saves of <out_dir>/last; 0 = none (the caller saves the end)
    out_dir: str | None = None


def seq_logp(model, ids_1xT, prompt_len, pad_multiple=1):
    """Mean log-prob of the completion tokens (positions >= prompt_len) under the policy. Only the completion's
    rows of the logits go to fp32. Right padding to `pad_multiple` leaves them unchanged (causal model)."""
    T = ids_1xT.shape[1]
    Tp = -(-T // pad_multiple) * pad_multiple
    out = model(F.pad(ids_1xT, (0, Tp - T)) if Tp > T else ids_1xT)
    logits = (out["logits"] if isinstance(out, dict) else out.logits)[0, prompt_len - 1:T - 1, :].float()
    tgt = ids_1xT[0, prompt_len:]                 # the rows above predict positions prompt_len..T-1
    if tgt.numel() == 0:
        return logits.sum() * 0.0
    return -F.cross_entropy(logits, tgt)


def save_state(model, tok, opt, step, rng, out_dir):
    """<out_dir>/last: the policy (save_pretrained), its tokenizer and grpo_state.pt; written to last.tmp first."""
    d = Path(out_dir) / "last"
    tmp = d.with_name("last.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    tok.save(str(tmp / "tokenizer.json"))
    torch.save({"step": step, "optimizer": opt.state_dict(), "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               tmp / "grpo_state.pt")
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)


def grpo_train(model, tok, records, reward_fn, build_prompt, cfg=None, *, device=None, log=print, resume_dir=None):
    """Run GRPO in-place on `model`. records: list of dicts (opaque to this module). Returns model.
    reward_fn(record, completion_text)->(float,dict); build_prompt(record, tok)->list[int].
    resume_dir: a <out_dir>/last of an earlier run; `model` must be the starting model (it gives the reference)."""
    cfg = cfg or GRPOConfig()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    amp = (cfg.bf16 and dev == "cuda")
    model = model.to(dev).eval()   # eval() = dropout OFF for logp/backward; eval() doesn't no_grad,
                                   # so policy-gradient backward still flows. RMSNorm is train/eval-identical.
    if cfg.grad_ckpt:
        # the model's checkpointing acts only in train(), which is the same function as eval() when dropout is 0
        if getattr(getattr(model, "config", None), "dropout", 0):
            raise ValueError("grad_ckpt runs the policy in train(): it needs a model with dropout 0")
        model.gradient_checkpointing_enable()
    eos = cfg.eos_token_id if cfg.eos_token_id is not None else tok.token_to_id("<|im_end|>")
    ref = None
    if cfg.kl_coef > 0:
        ref = copy.deepcopy(model).to(dev, dtype=torch.bfloat16 if amp else None).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    opt, _ = build_optimizer(model, cfg.lr, 0.0, cfg.optimizer)
    start = 0
    if resume_dir:
        from safetensors.torch import load_file
        missing, unexpected = model.load_state_dict(load_file(str(Path(resume_dir) / "model.safetensors"),
                                                              device=str(dev)), strict=False)
        tied = {k for k in missing if k.endswith("lm_head.weight")}
        if set(missing) - tied or unexpected:
            raise ValueError(f"resume: weights that do not match ({sorted(set(missing) - tied)[:3]}, {unexpected[:3]})")
        st = torch.load(Path(resume_dir) / "grpo_state.pt", map_location="cpu", weights_only=False)  # optimizer: moved by load_state_dict
        opt.load_state_dict(st["optimizer"])
        rng.setstate(st["rng"])
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        start = st["step"]
        log(f"resumed from {resume_dir} at step {start}")

    for step in range(start, cfg.steps):
        t0 = time.time()
        batch = [rng.choice(records) for _ in range(cfg.prompts_per_step)]
        losses, all_r, all_pass = [], [], 0
        t_gen = t_rew = 0.0
        n_tok = 0
        opt.zero_grad(set_to_none=True)
        for rec in batch:
            pids = build_prompt(rec, tok)
            p_t = torch.tensor([pids], device=dev)
            plen = p_t.shape[1]
            comps, rewards = [], []
            # the group of one prompt is decoded in batches: the prompt is read once per batch
            gb = cfg.gen_batch or cfg.group
            tg = time.time()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                for k in range(0, cfg.group, gb):
                    comps += model.generate_group(p_t, min(gb, cfg.group - k), max_new_tokens=cfg.max_new,
                                                  temperature=cfg.temperature, top_p=cfg.top_p,
                                                  top_k=cfg.top_k, eos_token_id=eos)
            t_gen += time.time() - tg
            tr = time.time()
            for full in comps:
                r, _ = reward_fn(rec, tok.decode(full[0, plen:].tolist()))
                rewards.append(r)
            t_rew += time.time() - tr
            rw = torch.tensor(rewards, device=dev)
            all_r += rewards
            all_pass += sum(1 for x in rewards if x >= 1.0)
            adv = (rw - rw.mean()) / (rw.std() + cfg.adv_eps)
            if torch.allclose(adv, torch.zeros_like(adv)):
                continue                          # whole group equal reward -> no signal
            n_tok += sum(c.shape[1] - plen for c in comps)
            if cfg.grad_ckpt:
                model.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                for full, A in zip(comps, adv):
                    lp = seq_logp(model, full, plen, cfg.pad_multiple)
                    loss = -(A.detach() * lp)
                    if ref is not None:
                        with torch.no_grad():
                            lp_ref = seq_logp(ref, full, plen, cfg.pad_multiple)
                        loss = loss + cfg.kl_coef * (lp - lp_ref) ** 2
                    (loss / (cfg.prompts_per_step * cfg.group)).backward()
                    losses.append(loss.item())
            model.eval()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        mr = sum(all_r) / max(1, len(all_r))
        pass_rate = all_pass / max(1, len(all_r))
        ml = sum(losses) / max(1, len(losses))
        log(f"step {step:4d}/{cfg.steps} | mean_reward {mr:.3f} | pass@1 {pass_rate:.3f} | "
            f"loss {ml:+.4f} | rollouts {len(all_r)} | {time.time() - t0:.0f}s (generation {t_gen:.0f}s, "
            f"reward {t_rew:.0f}s, {n_tok} tokens)")
        if cfg.save_every and cfg.out_dir and (step + 1) % cfg.save_every == 0:
            save_state(model, tok, opt, step + 1, rng, cfg.out_dir)
    return model
