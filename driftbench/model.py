"""Ensemble MLP world model predicting next-state deltas.

All ensemble members are evaluated in one batched matmul (EnsembleLinear), so
an ensemble of 5 costs roughly one forward pass of a 5x wider network instead
of 5 separate Python calls.

Shapes: states s (E, B, OBS_DIM), actions a (E, B, ACT_DIM), where E is the
ensemble size. Each member can see different inputs (bootstrapped batches in
training, its own rollout in evaluation).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

INPUT_CLIP = 20.0  # normalised-input clamp so diverged rollouts stay finite


class EnsembleLinear(nn.Module):
    def __init__(self, ensemble_size: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, out_features))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (E, B, in)
        return torch.baddbmm(self.bias, x, self.weight)


class EnsembleDynamics(nn.Module):
    NORM_KEYS = ("s_mean", "s_std", "a_mean", "a_std", "d_mean", "d_std")

    def __init__(self, norm: dict, ensemble_size: int = 5, hidden: int = 200, n_layers: int = 3):
        super().__init__()
        for k in self.NORM_KEYS:
            self.register_buffer(k, torch.as_tensor(np.asarray(norm[k]), dtype=torch.float32))
        dyn_mask = torch.as_tensor(np.asarray(norm["dyn_mask"]), dtype=torch.bool)
        self.register_buffer("dyn_mask", dyn_mask)
        self.register_buffer("dyn_idx", torch.nonzero(dyn_mask).squeeze(1))

        self.ensemble_size = ensemble_size
        self.obs_dim = self.s_mean.numel()
        self.act_dim = self.a_mean.numel()
        n_dyn = int(self.dyn_idx.numel())

        layers, d_in = [], self.obs_dim + self.act_dim
        for _ in range(n_layers):
            layers += [EnsembleLinear(ensemble_size, d_in, hidden), nn.SiLU()]
            d_in = hidden
        layers.append(EnsembleLinear(ensemble_size, d_in, n_dyn))
        self.net = nn.Sequential(*layers)
        self.config = {"ensemble_size": ensemble_size, "hidden": hidden, "n_layers": n_layers}

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        """Returns (next_state, normalised_delta_prediction)."""
        sn = ((s - self.s_mean) / self.s_std).clamp(-INPUT_CLIP, INPUT_CLIP)
        an = (a - self.a_mean) / self.a_std
        d_norm = self.net(torch.cat([sn, an], dim=-1))
        idx = self.dyn_idx
        d_dyn = d_norm * self.d_std[idx] + self.d_mean[idx]
        next_s = s.index_add(-1, idx, d_dyn)
        return next_s, d_norm

    # ---------------------------------------------------------------- I/O
    def checkpoint(self, extra: dict | None = None) -> dict:
        return {"config": dict(self.config), "state_dict": self.state_dict(), "extra": extra or {}}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "EnsembleDynamics":
        sd = ckpt["state_dict"]
        norm = {k: sd[k].cpu().numpy() for k in cls.NORM_KEYS}
        norm["dyn_mask"] = sd["dyn_mask"].cpu().numpy()
        model = cls(norm, **ckpt["config"])
        model.load_state_dict(sd)
        return model


def load_model(path, map_location="cpu") -> EnsembleDynamics:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    return EnsembleDynamics.from_checkpoint(ckpt)
