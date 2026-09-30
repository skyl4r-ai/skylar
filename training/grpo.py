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
"""
import copy
import random
from dataclasses import dataclass

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


def seq_logp(model, ids_1xT, prompt_len):
    """Mean log-prob of the completion tokens (positions >= prompt_len) under the policy."""
    out = model(ids_1xT)
    logits = (out["logits"] if isinstance(out, dict) else out.logits)[:, :-1, :].float()
    tgt = ids_1xT[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]  # (T-1,)
    comp = logp[prompt_len - 1:]                  # tokens predicting positions prompt_len..T-1
    return comp.mean() if comp.numel() else logp.sum() * 0.0


def grpo_train(model, tok, records, reward_fn, build_prompt, cfg=None, *, device=None, log=print):
    """Run GRPO in-place on `model`. records: list of dicts (opaque to this module). Returns model.
    reward_fn(record, completion_text)->(float,dict); build_prompt(record, tok)->list[int]."""
    cfg = cfg or GRPOConfig()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = model.to(dev).eval()   # eval() = dropout OFF for logp/backward; eval() doesn't no_grad,
                                   # so policy-gradient backward still flows. RMSNorm is train/eval-identical.
    eos = cfg.eos_token_id if cfg.eos_token_id is not None else tok.token_to_id("<|im_end|>")
    ref = None
    if cfg.kl_coef > 0:
        ref = copy.deepcopy(model).to(dev).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    opt, _ = build_optimizer(model, cfg.lr, 0.0, cfg.optimizer)
    amp = (cfg.bf16 and dev == "cuda")

    for step in range(cfg.steps):
        batch = [rng.choice(records) for _ in range(cfg.prompts_per_step)]
        losses, all_r, all_pass = [], [], 0
        opt.zero_grad(set_to_none=True)
        for rec in batch:
            pids = build_prompt(rec, tok)
            p_t = torch.tensor([pids], device=dev)
            plen = p_t.shape[1]
            comps, rewards = [], []
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in range(cfg.group):
                    full = model.generate(p_t, max_new_tokens=cfg.max_new,
                                          temperature=cfg.temperature, top_p=cfg.top_p,
                                          top_k=cfg.top_k, eos_token_id=eos)
                    text = tok.decode(full[0, plen:].tolist())
                    comps.append(full)
                    r, _ = reward_fn(rec, text)
                    rewards.append(r)
            rw = torch.tensor(rewards, device=dev)
            all_r += rewards
            all_pass += sum(1 for x in rewards if x >= 1.0)
            adv = (rw - rw.mean()) / (rw.std() + cfg.adv_eps)
            if torch.allclose(adv, torch.zeros_like(adv)):
                continue                          # whole group equal reward -> no signal
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                for full, A in zip(comps, adv):
                    lp = seq_logp(model, full, plen)
                    loss = -(A.detach() * lp)
                    if ref is not None:
                        with torch.no_grad():
                            lp_ref = seq_logp(ref, full, plen)
                        loss = loss + cfg.kl_coef * (lp - lp_ref) ** 2
                    (loss / (cfg.prompts_per_step * cfg.group)).backward()
                    losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        mr = sum(all_r) / max(1, len(all_r))
        pass_rate = all_pass / max(1, len(all_r))
        ml = sum(losses) / max(1, len(losses))
        log(f"step {step:4d}/{cfg.steps} | mean_reward {mr:.3f} | pass@1 {pass_rate:.3f} | "
            f"loss {ml:+.4f} | rollouts {len(all_r)}")
    return model
