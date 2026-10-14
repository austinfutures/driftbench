"""Load processed episodes into dense arrays and compute normalization stats."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq

from .config import ACT_DIM, EP_LEN, OBS_DIM, PROCESSED_DIR

# A state dimension counts as "dynamic" (modelled and scored) only if it varies
# meaningfully across the dataset. Pusher's goal position is fixed and the
# object's height barely moves; predicting those as unchanged is exact enough
# and keeps tiny-variance dims from dominating normalized errors.
MIN_STATE_STD = 1e-3
MIN_DELTA_STD = 1e-6


@dataclass
class EpisodeSet:
    obs: np.ndarray          # (N, T+1, OBS_DIM) float32
    act: np.ndarray          # (N, T, ACT_DIM) float32
    policy: np.ndarray       # (N,) str
    episode_id: np.ndarray   # (N,) str
    name: str = ""

    def __len__(self):
        return len(self.obs)

    @property
    def n_transitions(self) -> int:
        return int(self.act.shape[0] * self.act.shape[1])

    def subset(self, idx, name=None) -> "EpisodeSet":
        idx = np.asarray(idx)
        return EpisodeSet(self.obs[idx], self.act[idx], self.policy[idx], self.episode_id[idx], name or self.name)


def _list_column_to_array(table, col, per_row):
    arr = table.column(col).combine_chunks()
    flat = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    n = table.num_rows
    if flat.size != n * per_row:
        raise ValueError(f"{col}: expected {n * per_row} values, got {flat.size}. Did validation run?")
    return flat


def load_split(name: str) -> EpisodeSet:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m driftbench.pipeline` first.")
    table = pq.read_table(path, columns=["episode_id", "policy", "obs", "act"])
    table = table.sort_by("episode_id")  # file order from Ray is not deterministic
    n = table.num_rows
    obs = _list_column_to_array(table, "obs", (EP_LEN + 1) * OBS_DIM).reshape(n, EP_LEN + 1, OBS_DIM)
    act = _list_column_to_array(table, "act", EP_LEN * ACT_DIM).reshape(n, EP_LEN, ACT_DIM)
    return EpisodeSet(
        obs=obs,
        act=act,
        policy=np.asarray(table.column("policy").to_pylist()),
        episode_id=np.asarray(table.column("episode_id").to_pylist()),
        name=name,
    )


def split_train_val(es: EpisodeSet, val_frac: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(es))
    n_val = max(1, int(round(val_frac * len(es))))
    return es.subset(idx[n_val:], es.name + "/train"), es.subset(idx[:n_val], es.name + "/val")


def compute_normalizer(es: EpisodeSet) -> dict:
    s = es.obs[:, :-1].reshape(-1, OBS_DIM).astype(np.float64)
    s_next = es.obs[:, 1:].reshape(-1, OBS_DIM).astype(np.float64)
    a = es.act.reshape(-1, ACT_DIM).astype(np.float64)
    d = s_next - s
    s_std_raw, d_std_raw = s.std(0), d.std(0)
    dyn_mask = (s_std_raw > MIN_STATE_STD) & (d_std_raw > MIN_DELTA_STD)
    return {
        "s_mean": s.mean(0).astype(np.float32),
        "s_std": np.maximum(s_std_raw, MIN_STATE_STD).astype(np.float32),
        "a_mean": a.mean(0).astype(np.float32),
        "a_std": np.maximum(a.std(0), 1e-6).astype(np.float32),
        "d_mean": d.mean(0).astype(np.float32),
        "d_std": np.maximum(d_std_raw, MIN_DELTA_STD).astype(np.float32),
        "dyn_mask": dyn_mask,
    }
