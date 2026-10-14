"""Train an ensemble world model with a 1-step or multi-step rollout loss.

Loss (identical formula for every horizon H):
    L = 1/H * sum_{k=1..H} mean( ((s_hat_k - s_k) / sigma_delta)^2 )   over dynamic dims
where s_hat_k is produced by feeding the model its OWN previous prediction.
For H=1 this reduces exactly to normalized next-state-delta MSE, the standard
teacher-forced objective. For H>1 gradients flow through the rollout, so the
model is penalised for errors that compound.

Data sampling is fully vectorized: episodes live as one tensor on the target
device and each step draws (episode, t) indices with torch.randint. Each
ensemble member draws from its own bootstrap resample of the episodes.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .data import EpisodeSet
from .model import EnsembleDynamics
from .utils import resolve_device, set_seed


@dataclass
class TrainConfig:
    ensemble_size: int = 5
    hidden: int = 200
    n_layers: int = 3
    steps: int = 6000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    horizon: int = 1
    grad_clip: float = 10.0
    bootstrap: bool = True
    seed: int = 0
    eval_every: int = 1000
    device: str = "auto"


class WindowSampler:
    """Samples (E, B, H+1) state windows and (E, B, H) action windows."""

    def __init__(self, es: EpisodeSet, ensemble_size: int, horizon: int, device, bootstrap: bool, seed: int):
        self.device = device
        self.S = torch.as_tensor(es.obs, device=device)
        self.A = torch.as_tensor(es.act, device=device)
        self.N, T1, _ = self.S.shape
        self.T = T1 - 1
        self.E, self.H = ensemble_size, horizon
        if horizon > self.T:
            raise ValueError(f"horizon {horizon} exceeds episode length {self.T}")
        self.gen = torch.Generator().manual_seed(seed)
        if bootstrap:
            self.member_eps = torch.randint(self.N, (self.E, self.N), generator=self.gen)
        else:
            self.member_eps = torch.arange(self.N).expand(self.E, self.N).clone()
        self.member_eps = self.member_eps.to(device)
        self.offsets = torch.arange(self.H + 1, device=device)

    def sample(self, batch_size: int):
        j = torch.randint(self.N, (self.E, batch_size), generator=self.gen).to(self.device)
        t = torch.randint(self.T - self.H + 1, (self.E, batch_size), generator=self.gen).to(self.device)
        ep = self.member_eps.gather(1, j).unsqueeze(-1)       # (E, B, 1)
        steps = t.unsqueeze(-1) + self.offsets                # (E, B, H+1)
        s_seq = self.S[ep, steps]                             # (E, B, H+1, D)
        a_seq = self.A[ep, steps[..., : self.H]]              # (E, B, H, A)
        return s_seq, a_seq


def rollout_loss(model: EnsembleDynamics, s_seq: torch.Tensor, a_seq: torch.Tensor, fn=None) -> torch.Tensor:
    """fn: optional callable to use for the forward pass (e.g. a torch.compile'd model)."""
    fn = fn or model
    H = a_seq.shape[2]
    idx = model.dyn_idx
    scale = model.d_std[idx]
    s = s_seq[:, :, 0]
    loss = s.new_zeros(())
    for k in range(H):
        s, _ = fn(s, a_seq[:, :, k])
        err = (s - s_seq[:, :, k + 1])[..., idx] / scale
        loss = loss + err.pow(2).mean()
    return loss / H


@torch.no_grad()
def one_step_nmse(model: EnsembleDynamics, es: EpisodeSet, device, chunk: int = 8192) -> float:
    """Ensemble-mean 1-step error in normalized-delta units (teacher forced)."""
    E = model.ensemble_size
    idx = model.dyn_idx
    S = torch.as_tensor(es.obs, device=device)
    A = torch.as_tensor(es.act, device=device)
    s = S[:, :-1].reshape(-1, S.shape[-1])
    a = A.reshape(-1, A.shape[-1])
    tgt = S[:, 1:].reshape(-1, S.shape[-1])
    total, n = 0.0, 0
    for i in range(0, len(s), chunk):
        sc, ac, tc = s[i : i + chunk], a[i : i + chunk], tgt[i : i + chunk]
        pred, _ = model(sc.unsqueeze(0).expand(E, -1, -1), ac.unsqueeze(0).expand(E, -1, -1))
        err = (pred.mean(0) - tc)[:, idx] / model.d_std[idx]
        total += float(err.pow(2).sum())
        n += err.numel()
    return total / n


def train_ensemble(train: EpisodeSet, val: EpisodeSet, norm: dict, cfg: TrainConfig, log=print):
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    model = EnsembleDynamics(norm, cfg.ensemble_size, cfg.hidden, cfg.n_layers).to(device)
    sampler = WindowSampler(train, cfg.ensemble_size, cfg.horizon, device, cfg.bootstrap, cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    history = []
    running, n_running = 0.0, 0
    t_start = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        s_seq, a_seq = sampler.sample(cfg.batch_size)
        loss = rollout_loss(model, s_seq, a_seq)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        running += float(loss.detach())
        n_running += 1

        if step % cfg.eval_every == 0 or step == cfg.steps:
            val_nmse = one_step_nmse(model, val, device)
            elapsed = time.perf_counter() - t_start
            history.append({"step": step, "train_loss": running / n_running, "val_1step_nmse": val_nmse, "elapsed_s": elapsed})
            log(f"    step {step:6d}  train_loss {running / n_running:.4f}  val_1step_nmse {val_nmse:.4f}  ({elapsed:.0f}s)")
            running, n_running = 0.0, 0

    train_time = time.perf_counter() - t_start
    info = {
        "config": asdict(cfg),
        "device": str(device),
        "history": history,
        "train_time_s": train_time,
        "ms_per_step": 1000.0 * train_time / cfg.steps,
        "final_val_1step_nmse": history[-1]["val_1step_nmse"] if history else float("nan"),
        "n_train_episodes": len(train),
        "n_train_transitions": train.n_transitions,
        "n_val_episodes": len(val),
        "dyn_dims": int(model.dyn_idx.numel()),
    }
    return model, info
