"""Drift evaluation: open-loop rollouts from held-out start states.

For each (episode, t0) start, every ensemble member rolls out its own
trajectory for H steps, replaying the episode's recorded actions and never
seeing the true state again. We then measure:

  nRMSE(h)      RMSE of the ensemble-mean prediction vs. ground truth at step h,
                in units of the per-dimension state std of the training data,
                over the dynamic dims. nRMSE = 1 is roughly what you get by
                always predicting the dataset mean, so it is the default
                "prediction is no longer useful" threshold.
  obj_err(h)    Euclidean error of the predicted object position, in metres.
  divergence horizon
                first step where the mean nRMSE curve exceeds the threshold
                (None = never within H). Also per-rollout, censored at H+1.
  disagreement  std across ensemble members (same units as nRMSE).
  Does disagreement predict error?
                Spearman/Pearson over all (rollout, step) pairs, AND within
                each horizon. Both quantities grow with h, so the pooled
                correlation is inflated by horizon alone; the within-horizon
                number is the honest test of "is the model unsure about the
                rollouts that are actually wrong". AUROC for flagging
                rollouts whose error exceeds the threshold is reported too.
  copy baseline predict s_{t0} for every future step. A world model that does
                not beat this is not modelling the dynamics.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, rankdata, spearmanr

from .config import OBS_SLICES
from .data import EpisodeSet
from .model import EnsembleDynamics

REPORT_HORIZONS = (1, 5, 10, 25, 50)


def make_starts(n_eps: int, T: int, horizon: int, starts_per_ep: int, seed: int):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_eps), starts_per_ep)
    t0 = rng.integers(0, T - horizon + 1, size=len(ep))
    return ep, t0


def auroc(scores: np.ndarray, labels: np.ndarray):
    labels = labels.astype(bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _safe_spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return None
    return float(spearmanr(x[m], y[m]).statistic)


def _first_crossing(curve: np.ndarray, thr: float):
    idx = np.nonzero(curve > thr)[0]
    return int(idx[0]) + 1 if len(idx) else None


@torch.no_grad()
def rollout(model: EnsembleDynamics, es: EpisodeSet, ep, t0, horizon: int, device):
    E = model.ensemble_size
    S = torch.as_tensor(es.obs, device=device)
    A = torch.as_tensor(es.act, device=device)
    ep_t = torch.as_tensor(ep, device=device)
    t0_t = torch.as_tensor(t0, device=device)
    s = S[ep_t, t0_t].unsqueeze(0).expand(E, -1, -1).contiguous()
    traj = []
    for k in range(horizon):
        a = A[ep_t, t0_t + k].unsqueeze(0).expand(E, -1, -1)
        s, _ = model(s, a)
        traj.append(s)
    traj = torch.stack(traj, dim=2)                                      # (E, M, H, D)
    steps = t0_t.unsqueeze(1) + 1 + torch.arange(horizon, device=device)  # (M, H)
    true = S[ep_t.unsqueeze(1), steps]                                   # (M, H, D)
    start = S[ep_t, t0_t]                                                # (M, D)
    return traj, true, start


def drift_metrics(
    model: EnsembleDynamics,
    es: EpisodeSet,
    horizon: int = 50,
    starts_per_ep: int = 4,
    threshold: float = 1.0,
    seed: int = 1234,
    scatter_points: int = 6000,
):
    device = next(model.parameters()).device
    model.eval()
    T = es.act.shape[1]
    ep, t0 = make_starts(len(es), T, horizon, starts_per_ep, seed)
    traj, true, start = rollout(model, es, ep, t0, horizon, device)

    idx = model.dyn_idx
    sstd = model.s_std[idx]
    mean_pred = traj.mean(0)

    nrmse = (((mean_pred - true)[..., idx] / sstd).pow(2).mean(-1)).sqrt()                 # (M, H)
    member_nrmse = (((traj - true.unsqueeze(0))[..., idx] / sstd).pow(2).mean(-1)).sqrt()  # (E, M, H)
    disagreement = ((traj[..., idx].std(0, unbiased=False) / sstd).pow(2).mean(-1)).sqrt()  # (M, H)
    copy_nrmse = (((start.unsqueeze(1) - true)[..., idx] / sstd).pow(2).mean(-1)).sqrt()   # (M, H)
    osl = OBS_SLICES["object"]
    obj_err = (mean_pred[..., osl] - true[..., osl]).norm(dim=-1)                          # (M, H)
    copy_obj_err = (start.unsqueeze(1)[..., osl] - true[..., osl]).norm(dim=-1)

    nrmse, member_nrmse, disagreement, copy_nrmse, obj_err, copy_obj_err = (
        x.float().cpu().numpy() for x in (nrmse, member_nrmse, disagreement, copy_nrmse, obj_err, copy_obj_err)
    )
    n_nonfinite = int((~np.isfinite(nrmse)).sum())
    nrmse_f = np.where(np.isfinite(nrmse), nrmse, np.inf)

    mean_curve = np.nanmean(np.where(np.isfinite(nrmse), nrmse, np.nan), axis=0)
    median_curve = np.median(nrmse_f, axis=0)
    p90_curve = np.quantile(nrmse_f, 0.9, axis=0)
    dis_curve = np.nanmean(disagreement, axis=0)
    copy_curve = copy_nrmse.mean(0)
    obj_curve = obj_err.mean(0)
    copy_obj_curve = copy_obj_err.mean(0)
    member_curve = member_nrmse.mean(axis=(0, 1))

    per_rollout_div = np.array([(_first_crossing(r, threshold) or horizon + 1) for r in nrmse_f])

    hs = np.broadcast_to(np.arange(1, horizon + 1), nrmse.shape)
    flat_err, flat_dis, flat_h = nrmse.ravel(), disagreement.ravel(), hs.ravel()
    m = np.isfinite(flat_err) & np.isfinite(flat_dis) & (flat_dis > 0) & (flat_err > 0)

    spearman_by_h = np.array([
        np.nan if (r := _safe_spearman(disagreement[:, h], nrmse[:, h])) is None else r for h in range(horizon)
    ])
    auroc_by_h = []
    for h in range(horizon):
        lab = nrmse_f[:, h] > threshold
        if 5 <= lab.sum() <= len(lab) - 5:
            auroc_by_h.append(auroc(disagreement[:, h], lab))
    pearson_log = None
    if m.sum() >= 10:
        pearson_log = float(pearsonr(np.log(flat_dis[m]), np.log(flat_err[m])).statistic)

    def at(curve, h):
        return float(curve[h - 1]) if h <= horizon else None

    metrics = {
        "n_rollouts": int(len(ep)),
        "horizon": horizon,
        "threshold_nrmse": threshold,
        "n_nonfinite": n_nonfinite,
        "nrmse_at": {str(h): at(mean_curve, h) for h in REPORT_HORIZONS},
        "nrmse_median_at": {str(h): at(median_curve, h) for h in REPORT_HORIZONS},
        "single_member_nrmse_at": {str(h): at(member_curve, h) for h in REPORT_HORIZONS},
        "copy_baseline_nrmse_at": {str(h): at(copy_curve, h) for h in REPORT_HORIZONS},
        "obj_err_m_at": {str(h): at(obj_curve, h) for h in REPORT_HORIZONS},
        "copy_baseline_obj_err_m_at": {str(h): at(copy_obj_curve, h) for h in REPORT_HORIZONS},
        "disagreement_at": {str(h): at(dis_curve, h) for h in REPORT_HORIZONS},
        "divergence_horizon_mean_curve": _first_crossing(mean_curve, threshold),
        "divergence_horizon_median_rollout": float(np.median(per_rollout_div)),
        "frac_rollouts_diverged_by_H": float((per_rollout_div <= horizon).mean()),
        # first step where the model is WORSE than "nothing moves" (None = never)
        "model_worse_than_copy_from": _first_crossing(mean_curve - copy_curve, 0.0),
        "disagreement_vs_error": {
            "spearman_pooled": _safe_spearman(flat_dis, flat_err),
            "pearson_log_pooled": pearson_log,
            "spearman_within_horizon_mean": float(np.nanmean(spearman_by_h)) if np.isfinite(spearman_by_h).any() else None,
            "spearman_within_horizon_at": {str(h): (None if not np.isfinite(spearman_by_h[h - 1]) else float(spearman_by_h[h - 1])) for h in REPORT_HORIZONS if h <= horizon},
            "auroc_pooled": auroc(flat_dis[np.isfinite(flat_dis)], (nrmse_f.ravel() > threshold)[np.isfinite(flat_dis)]),
            "auroc_within_horizon_mean": float(np.mean(auroc_by_h)) if auroc_by_h else None,
            "n_horizons_with_auroc": len(auroc_by_h),
        },
    }

    rng = np.random.default_rng(seed)
    sel = np.nonzero(m)[0]
    if len(sel) > scatter_points:
        sel = rng.choice(sel, scatter_points, replace=False)
    arrays = {
        "nrmse_mean_curve": mean_curve,
        "nrmse_median_curve": median_curve,
        "nrmse_p90_curve": p90_curve,
        "member_curve": member_curve,
        "copy_curve": copy_curve,
        "obj_curve": obj_curve,
        "copy_obj_curve": copy_obj_curve,
        "dis_curve": dis_curve,
        "spearman_by_h": spearman_by_h,
        "scatter_h": flat_h[sel].astype(np.int16),
        "scatter_err": flat_err[sel].astype(np.float32),
        "scatter_dis": flat_dis[sel].astype(np.float32),
    }
    return metrics, arrays
