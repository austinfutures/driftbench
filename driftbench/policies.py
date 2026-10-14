"""Data-collection policies for Pusher-v5.

Three behaviour sources give the dataset a diversity axis:
  random    i.i.d. uniform torques: broad but unstructured coverage.
  scripted  Jacobian-transpose controller that lines the fingertip up behind
            the object and pushes it toward the goal. Deterministic given the
            start state, so it produces many near-identical trajectories.
  noisy     The scripted controller with per-episode gain randomisation,
            Gaussian torque noise and epsilon-random actions.

The scripted controller does not need to solve the task; it needs to produce
purposeful, contact-rich motion that differs from random flailing.
collect.py reports its success rate so you can see how well it does.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .config import OBS_SLICES


class RandomPolicy:
    name = "random"

    def __init__(self, env):
        self.low = env.action_space.low.astype(np.float32)
        self.high = env.action_space.high.astype(np.float32)
        self.rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(self.low, self.high).astype(np.float32)


class ScriptedPusher:
    name = "scripted"

    KP = 20.0           # Cartesian stiffness mapped through J^T
    KD = 0.5            # joint damping
    STANDOFF = 0.07     # metres behind the object (relative to goal direction)
    ALIGN_TOL = 0.03    # switch approach -> push when this close to the standoff point
    LOST_TOL = 0.15     # switch push -> approach when the tip loses the object
    PUSH_THROUGH = 0.05 # aim this far past the object centre while pushing

    def __init__(self, env, noise_std: float = 0.0, eps: float = 0.0, randomize: bool = False):
        u = env.unwrapped
        self.model, self.data = u.model, u.data
        self.tip_id = self.model.body("tips_arm").id
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.low = env.action_space.low.astype(np.float64)
        self.high = env.action_space.high.astype(np.float64)
        self.noise_std = noise_std
        self.eps = eps
        self.randomize = randomize
        self.rng = np.random.default_rng(0)
        self.reset(self.rng)

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.phase = 0
        if self.randomize:
            self.kp = self.KP * rng.uniform(0.6, 1.4)
            self.kd = self.KD * rng.uniform(0.5, 2.0)
            self.standoff = self.STANDOFF * rng.uniform(0.7, 1.3)
        else:
            self.kp, self.kd, self.standoff = self.KP, self.KD, self.STANDOFF

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        tip = obs[OBS_SLICES["tip"]].astype(np.float64)
        obj = obs[OBS_SLICES["object"]].astype(np.float64)
        goal = obs[OBS_SLICES["goal"]].astype(np.float64)

        push = goal[:2] - obj[:2]
        push /= np.linalg.norm(push) + 1e-8
        behind = obj[:2] - push * self.standoff

        if self.phase == 0 and np.linalg.norm(tip[:2] - behind) < self.ALIGN_TOL:
            self.phase = 1
        elif self.phase == 1 and np.linalg.norm(tip[:2] - obj[:2]) > self.LOST_TOL:
            self.phase = 0

        target_xy = behind if self.phase == 0 else obj[:2] + push * self.PUSH_THROUGH
        target = np.array([target_xy[0], target_xy[1], obj[2]])

        # obs is computed from the current MjData, so data and obs are in sync here.
        mujoco.mj_jacBodyCom(self.model, self.data, self.jacp, self.jacr, self.tip_id)
        J = self.jacp[:, :7]
        tau = self.kp * (J.T @ (target - tip)) - self.kd * self.data.qvel[:7]

        if self.noise_std > 0:
            tau = tau + self.rng.normal(0.0, self.noise_std, size=7)
        if self.eps > 0 and self.rng.random() < self.eps:
            tau = self.rng.uniform(self.low, self.high)
        return np.clip(tau, self.low, self.high).astype(np.float32)


def make_policy(name: str, env):
    if name == "random":
        return RandomPolicy(env)
    if name == "scripted":
        return ScriptedPusher(env)
    if name == "noisy":
        p = ScriptedPusher(env, noise_std=0.6, eps=0.15, randomize=True)
        p.name = "noisy"
        return p
    raise ValueError(f"unknown policy {name!r}")


def apply_variant(env, variant: dict) -> None:
    """Scale the pushed object's mass/inertia and sliding friction in place."""
    m = env.unwrapped.model
    bid = m.body("object").id
    ms = float(variant.get("mass_scale", 1.0))
    fs = float(variant.get("friction_scale", 1.0))
    m.body_mass[bid] *= ms
    m.body_inertia[bid] *= ms
    for g in np.where(m.geom_bodyid == bid)[0]:
        m.geom_friction[g, 0] *= fs
