"""Stage 5: build plots and results/RESULTS.md from saved runs.

Nothing here is typed in by hand: every number is read from
results/runs/*/seed*/metrics.json, results/data_stats.json,
results/collection_stats.json and results/profile.json.
Values are mean ± std across seeds (n shown per row).

Usage:
    python -m driftbench.report
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import PLOTS_DIR, RESULTS_DIR, RUNS_DIR  # noqa: E402
from .utils import read_json  # noqa: E402

COLORS = {
    "h1_dedup": "#2f6db5",
    "h10_dedup": "#d9622b",
    "h1_raw": "#7a7a7a",
    "h1_balanced": "#2a9d6f",
    "h1_matched": "#8a5cc2",
}
LABELS = {
    "h1_dedup": "1-step loss (dedup)",
    "h10_dedup": "multi-step loss (dedup)",
    "h1_raw": "1-step, raw (with dups)",
    "h1_balanced": "1-step, policy-balanced",
    "h1_matched": "1-step, dedup size-matched",
}
SPLIT_LABEL = {"eval_id": "in-distribution", "eval_ood": "OOD (heavy, slippery object)"}


# ---------------------------------------------------------------- loading
def load_runs():
    runs = defaultdict(list)
    if not RUNS_DIR.exists():
        return runs
    for exp_dir in sorted(RUNS_DIR.iterdir()):
        for seed_dir in sorted(exp_dir.glob("seed*")):
            mpath, cpath = seed_dir / "metrics.json", seed_dir / "curves.npz"
            if mpath.exists() and cpath.exists():
                runs[exp_dir.name].append((read_json(mpath), dict(np.load(cpath))))
    return runs


def _maybe(path):
    return read_json(path) if path.exists() else None


def get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


# ---------------------------------------------------------------- formatting
def fmt(values, digits=3, scale=1.0):
    vals = [v * scale for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals):.{digits}f}"


def fmt_div(values, horizon):
    """Divergence horizon: None means it never crossed the threshold."""
    if not values:
        return "n/a"
    if all(v is not None for v in values):
        return fmt(values, 1) if len(values) > 1 else str(values[0])
    return ", ".join(str(v) if v is not None else f">{horizon}" for v in values)


def collect(runs, exp, split, *keys):
    return [get(m, "eval", split, *keys) for m, _ in runs.get(exp, [])]


def mean_value(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------- plots
def _curve(runs, exp, split, key):
    cs = [c[f"{split}__{key}"] for _, c in runs.get(exp, []) if f"{split}__{key}" in c]
    if not cs:
        return None, None
    a = np.stack(cs)
    return a.mean(0), a.std(0)


def _style(ax, title, ylabel, threshold=None):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("rollout step h (open loop)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if threshold is not None:
        ax.axhline(threshold, color="#b33", linewidth=0.8, linestyle=":", label=f"divergence threshold ({threshold:g})")


def plot_curves(runs, exps, split, path, title, threshold, log=True):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    copy_drawn = False
    for exp in exps:
        mean, std = _curve(runs, exp, split, "nrmse_mean_curve")
        if mean is None:
            continue
        h = np.arange(1, len(mean) + 1)
        n = len(runs[exp])
        ax.plot(h, mean, color=COLORS.get(exp), label=f"{LABELS.get(exp, exp)} (n={n})", linewidth=1.8)
        if n > 1:
            ax.fill_between(h, np.maximum(mean - std, 1e-6), mean + std, color=COLORS.get(exp), alpha=0.15, linewidth=0)
        if not copy_drawn:
            cmean, _ = _curve(runs, exp, split, "copy_curve")
            ax.plot(h, cmean, color="black", linestyle="--", linewidth=1, label="copy baseline (predict s_t0)")
            copy_drawn = True
    if not copy_drawn:
        plt.close(fig)
        return False
    if log:
        ax.set_yscale("log")
    _style(ax, title, "state error, nRMSE (std units)", threshold)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_id_vs_ood(runs, exps, path, threshold):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    drawn = False
    for exp in exps:
        for split, ls in (("eval_id", "-"), ("eval_ood", "--")):
            mean, _ = _curve(runs, exp, split, "nrmse_mean_curve")
            if mean is None:
                continue
            ax.plot(np.arange(1, len(mean) + 1), mean, color=COLORS.get(exp), linestyle=ls,
                    label=f"{LABELS.get(exp, exp)}, {SPLIT_LABEL[split]}", linewidth=1.6)
            drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax.set_yscale("log")
    _style(ax, "In-distribution vs. OOD physics", "state error, nRMSE (std units)", threshold)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_disagreement(runs, exp, split, path):
    if not runs.get(exp):
        return False
    _, c = runs[exp][0]
    h, err, dis = c[f"{split}__scatter_h"], c[f"{split}__scatter_err"], c[f"{split}__scatter_dis"]
    sp_mean, sp_std = _curve(runs, exp, split, "spearman_by_h")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    sc = a1.scatter(dis, err, c=h, s=4, cmap="viridis", alpha=0.6, linewidths=0)
    a1.set_xscale("log")
    a1.set_yscale("log")
    lo = max(min(dis.min(), err.min()), 1e-4)
    hi = max(dis.max(), err.max())
    a1.plot([lo, hi], [lo, hi], color="black", linewidth=0.7, linestyle=":")
    a1.set_xlabel("ensemble disagreement (std units)")
    a1.set_ylabel("actual error, nRMSE")
    a1.set_title(f"{LABELS.get(exp, exp)}, seed {runs[exp][0][0]['seed']}", fontsize=11, loc="left")
    a1.grid(alpha=0.25, linewidth=0.6)
    fig.colorbar(sc, ax=a1, label="rollout step h")
    hh = np.arange(1, len(sp_mean) + 1)
    a2.plot(hh, sp_mean, color=COLORS.get(exp), linewidth=1.8)
    if len(runs[exp]) > 1:
        a2.fill_between(hh, sp_mean - sp_std, sp_mean + sp_std, color=COLORS.get(exp), alpha=0.15, linewidth=0)
    a2.axhline(0, color="black", linewidth=0.6)
    a2.set_ylim(-1, 1)
    _style(a2, "Within-horizon Spearman(disagreement, error)", "Spearman ρ")
    a2.set_title("Within-horizon Spearman(disagreement, error)", fontsize=11, loc="left")
    for ax in (a1,):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_object_error(runs, exps, split, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    drawn = False
    for exp in exps:
        mean, _ = _curve(runs, exp, split, "obj_curve")
        if mean is None:
            continue
        h = np.arange(1, len(mean) + 1)
        ax.plot(h, 100 * mean, color=COLORS.get(exp), label=LABELS.get(exp, exp), linewidth=1.8)
        if not drawn:
            cm, _ = _curve(runs, exp, split, "copy_obj_curve")
            ax.plot(h, 100 * cm, color="black", linestyle="--", linewidth=1, label="copy baseline")
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    _style(ax, f"Object position error ({SPLIT_LABEL[split]})", "object error (cm)")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- markdown
def drift_table(runs, exps, split, horizon):
    rows = [
        "| Experiment | n | train episodes | val 1-step nMSE | nRMSE@1 | nRMSE@10 | nRMSE@25 | nRMSE@50 | divergence horizon | obj err @50 (cm) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for exp in exps:
        rs = runs.get(exp, [])
        if not rs:
            continue
        rows.append(
            f"| {LABELS.get(exp, exp)} | {len(rs)} | {rs[0][0]['train']['n_train_episodes']} | "
            f"{fmt([m['train']['final_val_1step_nmse'] for m, _ in rs], 4)} | "
            + " | ".join(fmt(collect(runs, exp, split, "nrmse_at", str(h))) for h in (1, 10, 25, 50))
            + f" | {fmt_div(collect(runs, exp, split, 'divergence_horizon_mean_curve'), horizon)}"
            + f" | {fmt(collect(runs, exp, split, 'obj_err_m_at', '50'), 2, 100)} |"
        )
    any_exp = next((e for e in exps if runs.get(e)), None)
    if any_exp:
        rows.append(
            "| *copy baseline* | | | | "
            + " | ".join(fmt(collect(runs, any_exp, split, "copy_baseline_nrmse_at", str(h))) for h in (1, 10, 25, 50))
            + f" | | {fmt(collect(runs, any_exp, split, 'copy_baseline_obj_err_m_at', '50'), 2, 100)} |"
        )
    return "\n".join(rows)


def disagreement_table(runs, exps, split):
    rows = [
        "| Experiment | Spearman (pooled) | Spearman (within-horizon mean) | Spearman @h=10 | AUROC (pooled) | AUROC (within-horizon) |",
        "|---|---|---|---|---|---|",
    ]
    for exp in exps:
        if not runs.get(exp):
            continue
        dv = lambda *k: collect(runs, exp, split, "disagreement_vs_error", *k)  # noqa: E731
        rows.append(
            f"| {LABELS.get(exp, exp)} | {fmt(dv('spearman_pooled'))} | {fmt(dv('spearman_within_horizon_mean'))} | "
            f"{fmt(dv('spearman_within_horizon_at', '10'))} | {fmt(dv('auroc_pooled'))} | {fmt(dv('auroc_within_horizon_mean'))} |"
        )
    return "\n".join(rows)


def ood_table(runs, exps, horizon):
    rows = [
        "| Experiment | nRMSE@10 ID | nRMSE@10 OOD | divergence ID | divergence OOD | disagreement@10 ID | disagreement@10 OOD |",
        "|---|---|---|---|---|---|---|",
    ]
    for exp in exps:
        if not runs.get(exp):
            continue
        rows.append(
            f"| {LABELS.get(exp, exp)} | {fmt(collect(runs, exp, 'eval_id', 'nrmse_at', '10'))} | "
            f"{fmt(collect(runs, exp, 'eval_ood', 'nrmse_at', '10'))} | "
            f"{fmt_div(collect(runs, exp, 'eval_id', 'divergence_horizon_mean_curve'), horizon)} | "
            f"{fmt_div(collect(runs, exp, 'eval_ood', 'divergence_horizon_mean_curve'), horizon)} | "
            f"{fmt(collect(runs, exp, 'eval_id', 'disagreement_at', '10'))} | "
            f"{fmt(collect(runs, exp, 'eval_ood', 'disagreement_at', '10'))} |"
        )
    return "\n".join(rows)


def data_section(ds, cs):
    if not ds:
        return "_Run `python -m driftbench.pipeline` to populate._"
    tr = ds["train"]
    lines = [
        f"Engine: `{ds.get('engine')}`. Pipeline wall time: {ds.get('wall_time_s', 0):.1f} s.",
        "",
        "| Stage | Count |",
        "|---|---|",
        f"| raw train episodes read | {tr['raw_episodes_read']} |",
        f"| synthetic faults injected (stage 1) | {sum(tr['injected_faults'].values()) if tr['injected_faults'] else 0} {tr['injected_faults'] or ''} |",
        f"| rejected by validation | {sum(tr['rejected_by_reason'].values()) if tr['rejected_by_reason'] else 0} {tr['rejected_by_reason'] or ''} |",
        f"| valid episodes | {tr['valid_episodes']} |",
        f"| retry copies injected among valid | {tr['valid_retry_copies_injected']} |",
        f"| exact duplicates found | {tr['exact_duplicates']} |",
        f"| near-duplicates removed | {tr['near_duplicates_removed']} ({100 * tr['near_duplicates_removed_frac']:.1f}% of valid) |",
        f"| near-dups removed by policy | {tr['near_dup_removed_by_policy']} |",
        "",
        "| Dataset | Episodes | Transitions |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {v['episodes']} | {v['transitions']} |" for k, v in tr["datasets"].items()]
    for split in ("eval_id", "eval_ood"):
        if split in ds:
            lines.append(f"| {split} | {ds[split]['episodes']} | {ds[split]['transitions']} |")
    if cs:
        lines += ["", "Behavior policies (train split, from collection):", "",
                  "| Policy | Episodes | mean final object-goal distance (m) | success rate (< 5 cm) |", "|---|---|---|---|"]
        for pol, s in cs.get("train", {}).get("by_policy", {}).items():
            succ = next((v for k, v in s.items() if k.startswith("success_rate")), None)
            succ_s = f"{succ:.2f}" if succ is not None else "n/a"
            lines.append(f"| {pol} | {s['episodes']} | {s['mean_final_obj_goal_dist_m']:.3f} | {succ_s} |")
    return "\n".join(lines)


def profile_section(pf):
    if not pf:
        return "_Run `python -m driftbench.profile_train` to populate._"
    env = pf["environment"]
    lines = [
        f"Device: **{env['device_type']}** (`{env['device']}`{', ' + env['gpu_name'] if env.get('gpu_name') else ''}), "
        f"torch {env['torch']}, {env['torch_threads']} threads, batch {pf['config']['batch_size']}, "
        f"ensemble {pf['config']['ensemble_size']}, {pf['config']['steps']} timed steps.",
        "",
        "| Variant | step (ms, median) | p90 | data (ms) | compute (ms) | data share | speedup vs DataLoader |",
        "|---|---|---|---|---|---|---|",
    ]
    for v, r in pf["variants"].items():
        if "error" in r:
            lines.append(f"| {v} | failed: `{r['error'][:80]}` | | | | | |")
            continue
        lines.append(
            f"| {v} | {r['step_ms_median']:.2f} | {r['step_ms_p90']:.2f} | {r['data_ms_median']:.2f} | "
            f"{r['compute_ms_median']:.2f} | {100 * r['data_frac_of_step']:.0f}% | {r.get('speedup_vs_dataloader', float('nan')):.2f}x |"
        )
    return "\n".join(lines)


def bullets(runs, ds, pf, horizon):
    """Draft resume bullets filled ONLY from measured numbers. Read before using."""
    out = []
    d1 = collect(runs, "h1_dedup", "eval_id", "divergence_horizon_mean_curve")
    d10 = collect(runs, "h10_dedup", "eval_id", "divergence_horizon_mean_curve")
    if d1 and d10:
        if all(v is not None for v in d1 + d10):
            a, b = np.mean(d1), np.mean(d10)
            verb = "extended" if b > a else "did not extend" if b == a else "shortened"
            out.append(f"Measured open-loop rollout drift in a MuJoCo world model; multi-step loss {verb} the divergence "
                       f"horizon from {a:.1f} to {b:.1f} steps (mean of {len(d1)} seeds).")
        else:
            out.append(f"Divergence horizons (1-step vs multi-step): {fmt_div(d1, horizon)} vs {fmt_div(d10, horizon)}. "
                       "At least one run never crossed the threshold; report nRMSE@50 instead.")
        n1 = mean_value(collect(runs, "h1_dedup", "eval_id", "nrmse_at", "50"))
        n10 = mean_value(collect(runs, "h10_dedup", "eval_id", "nrmse_at", "50"))
        if n1 and n10:
            out.append(f"50-step error: {n1:.3f} (1-step loss) vs {n10:.3f} (multi-step loss), {n1 / n10:.2f}x ratio.")
    w = mean_value(collect(runs, "h1_dedup", "eval_id", "disagreement_vs_error", "spearman_within_horizon_mean"))
    pooled = mean_value(collect(runs, "h1_dedup", "eval_id", "disagreement_vs_error", "spearman_pooled"))
    if w is not None and pooled is not None:
        out.append(f"Ensemble disagreement vs rollout error: Spearman ρ = {pooled:.2f} pooled over horizons, "
                   f"{w:.2f} within a fixed horizon (the within-horizon number is the one to quote).")
    if ds:
        tr = ds["train"]
        out.append(f"Ray Data pipeline over {tr['raw_episodes_read']} trajectories: rejected "
                   f"{sum(tr['rejected_by_reason'].values()) if tr['rejected_by_reason'] else 0} corrupt episodes and removed "
                   f"{100 * tr['near_duplicates_removed_frac']:.1f}% near-duplicates "
                   f"({tr['exact_duplicates']} exact).")
    if pf:
        best = max((r for r in pf["variants"].values() if "speedup_vs_dataloader" in r),
                   key=lambda r: r["speedup_vs_dataloader"], default=None)
        if best:
            out.append(f"Profiled training on {pf['environment']['device_type']}; replacing per-sample DataLoader batching "
                       f"cut step time {best['speedup_vs_dataloader']:.1f}x.")
    return "\n".join(f"- {b}" for b in out) if out else "_Not enough runs yet._"


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    ds = _maybe(RESULTS_DIR / "data_stats.json")
    cs = _maybe(RESULTS_DIR / "collection_stats.json")
    pf = _maybe(RESULTS_DIR / "profile.json")

    any_m = next((rs[0][0] for rs in runs.values() if rs), None)
    horizon = get(any_m, "eval", "eval_id", "horizon") or 50
    thr = get(any_m, "eval", "eval_id", "threshold_nrmse") or 1.0

    loss_exps = ["h1_dedup", "h10_dedup"]
    data_exps = ["h1_raw", "h1_dedup", "h1_matched", "h1_balanced"]
    all_exps = [e for e in COLORS if runs.get(e)]

    made = []
    if plot_curves(runs, loss_exps, "eval_id", PLOTS_DIR / "drift_loss.png", "Drift vs. horizon: training loss", thr):
        made.append(("drift_loss.png", "1-step vs multi-step loss"))
    if plot_curves(runs, data_exps, "eval_id", PLOTS_DIR / "drift_data.png", "Drift vs. horizon: training data", thr):
        made.append(("drift_data.png", "raw vs dedup vs balanced vs size-matched"))
    if plot_id_vs_ood(runs, loss_exps, PLOTS_DIR / "drift_id_vs_ood.png", thr):
        made.append(("drift_id_vs_ood.png", "in-distribution vs OOD physics"))
    if plot_disagreement(runs, "h1_dedup", "eval_id", PLOTS_DIR / "disagreement_vs_error.png"):
        made.append(("disagreement_vs_error.png", "does ensemble disagreement predict error?"))
    if plot_object_error(runs, loss_exps, "eval_id", PLOTS_DIR / "object_error.png"):
        made.append(("object_error.png", "object position error in cm"))

    env = get(any_m, "environment") or {}
    md = [
        "# DriftBench results",
        "",
        "_Auto-generated by `python -m driftbench.report`. Every number below is read from files written by actual runs; "
        "do not edit by hand. Values are mean ± std across seeds; n = number of seeds._",
        "",
        f"**Hardware:** trained on **{env.get('device_type', '?')}** (`{env.get('device', '?')}`"
        f"{', ' + env['gpu_name'] if env.get('gpu_name') else ''}), {env.get('cpu', '?')}, {env.get('cpu_count', '?')} logical CPUs. "
        f"torch {env.get('torch', '?')}, mujoco {env.get('mujoco', '?')}, gymnasium {env.get('gymnasium', '?')}, "
        f"ray {env.get('ray', '?')}. Report generated from runs dated {env.get('timestamp', '?')}.",
        "",
        f"**Metric:** nRMSE = RMSE of the ensemble-mean open-loop prediction in units of the training-set state std "
        f"(dynamic dims only). Divergence horizon = first step where mean nRMSE > {thr:g} (≈ no better than predicting the "
        f"dataset mean). `>{horizon}` means it never crossed within {horizon} steps.",
        "",
        "## Data pipeline",
        "",
        data_section(ds, cs),
        "",
        "## Drift: training loss (in-distribution)",
        "",
        drift_table(runs, loss_exps, "eval_id", horizon),
        "",
        "## Drift: training data (in-distribution)",
        "",
        drift_table(runs, data_exps, "eval_id", horizon),
        "",
        "`h1_matched` has the same number of episodes as `h1_balanced` but keeps the dedup policy mix, so "
        "balanced − matched isolates the effect of balance from the effect of having less data.",
        "",
        "## Does ensemble disagreement predict error? (in-distribution)",
        "",
        disagreement_table(runs, all_exps, "eval_id"),
        "",
        "Pooled correlation is inflated because both disagreement and error grow with horizon. "
        "The within-horizon number asks the useful question: at a fixed step, are the rollouts the ensemble is unsure "
        "about the ones that are actually wrong?",
        "",
        "## Out-of-distribution physics",
        "",
        ood_table(runs, loss_exps + ["h1_balanced"], horizon),
        "",
        "## OOD drift table",
        "",
        drift_table(runs, loss_exps, "eval_ood", horizon),
        "",
        "## Training-step profile",
        "",
        profile_section(pf),
        "",
        "## Plots",
        "",
    ]
    md += [f"![{desc}](plots/{name})" for name, desc in made] or ["_No runs found._"]
    md += [
        "",
        "## Draft resume bullets (auto-filled from the numbers above)",
        "",
        "Check each against the tables before using it. If a number goes the 'wrong' way, that is a finding, not a bug to hide.",
        "",
        bullets(runs, ds, pf, horizon),
        "",
    ]
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'RESULTS.md'} and {len(made)} plots in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
