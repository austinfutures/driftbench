"""Stage 1: collect Pusher-v5 trajectories from a mix of policies.

Writes one Parquet shard per worker chunk to data/raw/<split>/, one row per
episode. Obs/actions are stored as flattened float32 lists.

Two realistic data problems are injected into the TRAIN split on purpose so
the pipeline in stage 2 has something real to catch (both are disclosed in the
README and counted in results/data_stats.json):
  * retry duplicates: a fraction of scripted jobs are re-run with the same seed,
    mimicking a worker that crashed after writing and got retried.
  * faults: a small fraction of episodes are truncated or get a NaN written in.

Usage:
    python -m driftbench.collect                  # full plan
    python -m driftbench.collect --scale 0.1      # 10% of the episodes (smoke test)
"""
from __future__ import annotations

import argparse
import shutil
import time
import zlib
from multiprocessing import get_context

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import ACT_DIM, ENV_ID, EP_LEN, OBS_DIM, OBS_SLICES, RAW_DIR, RESULTS_DIR, VARIANTS
from .utils import write_json

PLAN = {
    "train": {
        "variant": "nominal",
        "counts": {"scripted": 600, "noisy": 150, "random": 100},
        "seed_base": 0,
        "dup_frac": 0.25,    # fraction of scripted jobs re-run with the same seed
        "fault_frac": 0.02,  # fraction of episodes truncated / NaN-corrupted
    },
    "eval_id": {
        "variant": "nominal",
        "counts": {"scripted": 50, "noisy": 50, "random": 50},
        "seed_base": 1_000_000,
        "dup_frac": 0.0,
        "fault_frac": 0.0,
    },
    "eval_ood": {
        "variant": "heavy_slippery",
        "counts": {"scripted": 50, "noisy": 50, "random": 50},
        "seed_base": 2_000_000,
        "dup_frac": 0.0,
        "fault_frac": 0.0,
    },
}

SCHEMA = pa.schema([
    ("episode_id", pa.string()),
    ("split", pa.string()),
    ("policy", pa.string()),
    ("variant", pa.string()),
    ("seed", pa.int64()),
    ("is_retry", pa.bool_()),
    ("injected_fault", pa.string()),
    ("length", pa.int32()),
    ("obs_dim", pa.int32()),
    ("act_dim", pa.int32()),
    ("obs", pa.list_(pa.float32())),
    ("act", pa.list_(pa.float32())),
    ("init_obj_goal_dist", pa.float32()),
    ("final_obj_goal_dist", pa.float32()),
])

SUCCESS_DIST = 0.05  # meters; only used for reporting how good the scripted policy is


def plan_jobs(split: str, spec: dict, scale: float, rng: np.random.Generator) -> list[dict]:
    jobs: list[dict] = []
    seed = spec["seed_base"]
    for policy, n in spec["counts"].items():
        n = max(4, int(round(n * scale)))
        for _ in range(n):
            jobs.append({"policy": policy, "seed": seed, "is_retry": False})
            seed += 1

    scripted = [j for j in jobs if j["policy"] == "scripted"]
    n_dup = int(round(spec["dup_frac"] * len(scripted)))
    if n_dup:
        picks = rng.choice(len(scripted), size=n_dup, replace=False)
        jobs += [dict(scripted[i], is_retry=True) for i in picks]

    order = rng.permutation(len(jobs))
    jobs = [jobs[i] for i in order]

    n_fault = int(round(spec["fault_frac"] * len(jobs)))
    fault_idx = set(rng.choice(len(jobs), size=n_fault, replace=False).tolist()) if n_fault else set()
    for i, j in enumerate(jobs):
        j["episode_id"] = f"{split}-{i:06d}"
        j["fault"] = str(rng.choice(["truncate", "nan"])) if i in fault_idx else "none"
        j["fault_seed"] = int(rng.integers(0, 2**31 - 1))
    return jobs


def _obj_goal_dist(o: np.ndarray) -> float:
    obj, goal = o[OBS_SLICES["object"]], o[OBS_SLICES["goal"]]
    return float(np.linalg.norm(obj[:2] - goal[:2]))


def run_episode(env, policy, job: dict):
    obs, _ = env.reset(seed=int(job["seed"]))
    policy.reset(np.random.default_rng(int(job["seed"])))
    obs_buf = [np.asarray(obs, dtype=np.float32)]
    act_buf = []
    for _ in range(EP_LEN):
        a = policy(obs)
        obs, _, terminated, truncated, _ = env.step(a)
        obs_buf.append(np.asarray(obs, dtype=np.float32))
        act_buf.append(np.asarray(a, dtype=np.float32))
        if terminated or truncated:
            break
    o = np.stack(obs_buf)
    a = np.stack(act_buf)
    init_d, final_d = _obj_goal_dist(o[0]), _obj_goal_dist(o[-1])

    frng = np.random.default_rng(job["fault_seed"])
    if job["fault"] == "truncate":
        cut = int(frng.integers(20, 80))
        o, a = o[: cut + 1], a[:cut]
    elif job["fault"] == "nan":
        o = o.copy()
        o[int(frng.integers(1, len(o))), int(frng.integers(0, OBS_DIM))] = np.nan
    return o, a, init_d, final_d


def _collect_shard(task):
    split, variant_name, jobs, path = task
    import gymnasium as gym

    from .policies import apply_variant, make_policy

    env = gym.make(ENV_ID)
    apply_variant(env, VARIANTS[variant_name])
    policies = {name: make_policy(name, env) for name in sorted({j["policy"] for j in jobs})}

    rows = []
    for job in jobs:
        o, a, init_d, final_d = run_episode(env, policies[job["policy"]], job)
        rows.append({
            "episode_id": job["episode_id"],
            "split": split,
            "policy": job["policy"],
            "variant": variant_name,
            "seed": int(job["seed"]),
            "is_retry": bool(job["is_retry"]),
            "injected_fault": job["fault"],
            "length": int(len(a)),
            "obs_dim": OBS_DIM,
            "act_dim": ACT_DIM,
            "obs": o.ravel().tolist(),
            "act": a.ravel().tolist(),
            "init_obj_goal_dist": init_d,
            "final_obj_goal_dist": final_d,
        })
    env.close()
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    return len(rows)


def collect_split(split: str, scale: float, workers: int, shard_size: int, seed: int) -> dict:
    spec = PLAN[split]
    rng = np.random.default_rng([seed, zlib.crc32(split.encode())])  # str hash() is salted per process
    jobs = plan_jobs(split, spec, scale, rng)

    out_dir = RAW_DIR / split
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (split, spec["variant"], jobs[i : i + shard_size], str(out_dir / f"shard-{k:04d}.parquet"))
        for k, i in enumerate(range(0, len(jobs), shard_size))
    ]
    t0 = time.perf_counter()
    done = 0
    with get_context("spawn").Pool(processes=max(1, workers)) as pool:
        for n in pool.imap_unordered(_collect_shard, tasks):
            done += n
            print(f"  [{split}] {done}/{len(jobs)} episodes", flush=True)
    wall = time.perf_counter() - t0

    t = pq.read_table(out_dir, columns=["policy", "is_retry", "injected_fault", "final_obj_goal_dist", "init_obj_goal_dist"]).to_pandas()
    by_policy = {}
    for policy, g in t.groupby("policy"):
        by_policy[policy] = {
            "episodes": int(len(g)),
            "mean_init_obj_goal_dist_m": float(g.init_obj_goal_dist.mean()),
            "mean_final_obj_goal_dist_m": float(g.final_obj_goal_dist.mean()),
            f"success_rate_final_dist_lt_{SUCCESS_DIST}m": float((g.final_obj_goal_dist < SUCCESS_DIST).mean()),
        }
    return {
        "variant": spec["variant"],
        "physics": VARIANTS[spec["variant"]],
        "episodes": int(len(t)),
        "retry_copies_injected": int(t.is_retry.sum()),
        "faults_injected": t.injected_fault.value_counts().to_dict(),
        "by_policy": by_policy,
        "wall_time_s": wall,
        "workers": workers,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=list(PLAN))
    p.add_argument("--scale", type=float, default=1.0, help="multiply episode counts (e.g. 0.1 for a smoke test)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    stats = {"scale": args.scale}
    for split in args.splits:
        print(f"Collecting {split} ...", flush=True)
        stats[split] = collect_split(split, args.scale, args.workers, args.shard_size, args.seed)
        for pol, s in stats[split]["by_policy"].items():
            print(f"  {pol:9s} n={s['episodes']:4d}  final obj-goal dist={s['mean_final_obj_goal_dist_m']:.3f} m")
    write_json(RESULTS_DIR / "collection_stats.json", stats)
    print(f"Wrote {RESULTS_DIR / 'collection_stats.json'}")


if __name__ == "__main__":
    main()
