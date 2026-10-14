"""Stage 3: train ensembles and run drift evaluation for every experiment.

Experiments (each run for every seed):
    h1_dedup      1-step loss, deduplicated data              (the baseline)
    h10_dedup     10-step rollout loss, deduplicated data     (loss mitigation)
    h1_raw        1-step loss, raw data with duplicates       (data mitigation)
    h1_balanced   1-step loss, policy-balanced data           (data mitigation)
    h1_matched    1-step loss, dedup subsampled to balanced's size
                  (control: is any balanced effect from balance or from size?)

Every model is evaluated on the same held-out start states from eval_id
(nominal physics) and eval_ood (heavier, slipperier object).

Runs are cached under results/runs/<exp>/seed<k>/; rerunning skips finished
runs unless --force. All numbers in the report come from these files.

Usage:
    python -m driftbench.experiments                      # everything, 3 seeds
    python -m driftbench.experiments --exps h1_dedup --seeds 0 --steps 500
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import torch

from .config import RUNS_DIR
from .data import compute_normalizer, load_split, split_train_val
from .evaluate import drift_metrics
from .train import TrainConfig, train_ensemble
from .utils import environment_info, read_json, resolve_device, write_json

EXPERIMENTS = {
    "h1_dedup": {"data": "dedup", "horizon": 1},
    "h10_dedup": {"data": "dedup", "horizon": 10},
    "h1_raw": {"data": "raw", "horizon": 1},
    "h1_balanced": {"data": "balanced", "horizon": 1},
    "h1_matched": {"data": "dedup_matched", "horizon": 1},
}
EVAL_SPLITS = ("eval_id", "eval_ood")


CACHE_KEYS = ("steps", "batch_size", "ensemble_size", "hidden", "n_layers", "lr", "horizon")


def _cache_matches(path, cfg, n_train, eval_kwargs) -> bool:
    """A cached run counts only if it used the same settings and the same data size."""
    try:
        old = read_json(path)
    except Exception:
        return False
    old_cfg = old.get("train", {}).get("config", {})
    if any(old_cfg.get(k) != getattr(cfg, k) for k in CACHE_KEYS):
        return False
    if old.get("train", {}).get("n_train_episodes") != n_train:
        return False
    old_eval = old.get("eval", {}).get("eval_id", {})
    return old_eval.get("horizon") == eval_kwargs["horizon"] and old_eval.get("n_rollouts") is not None


def run_one(name, spec, seed, base_cfg, datasets, eval_sets, eval_kwargs, force):
    out = RUNS_DIR / name / f"seed{seed}"
    full = datasets[spec["data"]]
    train, val = split_train_val(full, val_frac=0.1, seed=0)
    cfg = replace(base_cfg, horizon=spec["horizon"], seed=seed)

    if (out / "metrics.json").exists() and not force:
        if _cache_matches(out / "metrics.json", cfg, len(train), eval_kwargs):
            print(f"  [skip] {name} seed{seed} (cached, same settings)")
            return
        print(f"  [retrain] {name} seed{seed}: cached run used different settings or data")
    out.mkdir(parents=True, exist_ok=True)

    norm = compute_normalizer(train)
    print(f"  [train] {name} seed{seed}: {len(train)} episodes ({train.n_transitions} transitions), H={cfg.horizon}")
    model, info = train_ensemble(train, val, norm, cfg)

    metrics = {
        "experiment": name,
        "seed": seed,
        "spec": spec,
        "train": info,
        "train_policy_counts": {p: int((train.policy == p).sum()) for p in np.unique(train.policy)},
        "environment": environment_info(info["device"]),
        "eval": {},
    }
    arrays = {}
    for split, es in eval_sets.items():
        t0 = time.perf_counter()
        m, a = drift_metrics(model, es, **eval_kwargs)
        m["eval_time_s"] = time.perf_counter() - t0
        metrics["eval"][split] = m
        arrays.update({f"{split}__{k}": v for k, v in a.items()})
        div = m["divergence_horizon_mean_curve"]
        print(f"    {split}: nRMSE@10={m['nrmse_at']['10']:.3f}  @50={m['nrmse_at'].get('50') or float('nan'):.3f}  "
              f"divergence horizon={div if div is not None else '>' + str(m['horizon'])}  "
              f"spearman(within-h)={m['disagreement_vs_error']['spearman_within_horizon_mean']}")

    write_json(out / "metrics.json", metrics)
    np.savez_compressed(out / "curves.npz", **arrays)
    torch.save(model.checkpoint({"experiment": name, "seed": seed}), out / "model.pt")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exps", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=TrainConfig.steps)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--ensemble-size", type=int, default=TrainConfig.ensemble_size)
    p.add_argument("--hidden", type=int, default=TrainConfig.hidden)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--multi-horizon", type=int, default=10, help="training horizon for h10_dedup")
    p.add_argument("--eval-horizon", type=int, default=50)
    p.add_argument("--starts-per-ep", type=int, default=4)
    p.add_argument("--threshold", type=float, default=1.0, help="nRMSE divergence threshold")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    print(f"Device: {device} ({'GPU' if device.type in ('cuda', 'mps') else 'CPU'})")

    base_cfg = TrainConfig(
        ensemble_size=args.ensemble_size, hidden=args.hidden, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, device=str(device),
        eval_every=max(1, args.steps // 6),
    )
    specs = {k: dict(v) for k, v in EXPERIMENTS.items()}
    specs["h10_dedup"]["horizon"] = args.multi_horizon

    needed = sorted({specs[e]["data"] for e in args.exps})
    datasets = {name: load_split(name) for name in needed}
    eval_sets = {s: load_split(s) for s in EVAL_SPLITS}
    eval_kwargs = {"horizon": args.eval_horizon, "starts_per_ep": args.starts_per_ep, "threshold": args.threshold}

    for name in args.exps:
        for seed in args.seeds:
            run_one(name, specs[name], seed, base_cfg, datasets, eval_sets, eval_kwargs, args.force)
    print(f"Done. Runs in {RUNS_DIR}. Next: python -m driftbench.report")


if __name__ == "__main__":
    main()
