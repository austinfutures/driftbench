"""Stage 2: data pipeline. validate -> hash -> deduplicate -> balance -> sharded Parquet.

Default engine is Ray Data. `--engine local` runs the exact same batch
functions on in-memory Arrow tables, for debugging or if Ray gives you trouble on
your OS. Both write the same datasets to data/processed/:

    raw            validated train episodes, duplicates kept
    dedup          validated, near-duplicates removed
    balanced       dedup, downsampled to equal episodes per policy
    dedup_matched  dedup, random subsample with the SAME size as `balanced`
                   (control: separates "more balanced" from "smaller")
    eval_id        validated in-distribution eval episodes
    eval_ood       validated out-of-distribution eval episodes

Deduplication hashes a quantized, temporally downsampled copy of each
trajectory's fingertip and object positions. Two episodes that land in the
same bins at every sampled step are treated as duplicates. Exact-duplicate
counts (hash of the full-precision trajectory) are reported alongside so you
can check the pipeline against the retry copies injected in stage 1.

Usage:
    python -m driftbench.pipeline
    python -m driftbench.pipeline --engine local
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import ACT_DIM, EP_LEN, OBS_DIM, OBS_SLICES, PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from .utils import write_json

OBS_ABS_LIMIT = 1e3
NEAR_DUP_DIMS = np.r_[OBS_SLICES["tip"], 17, 18]  # fingertip xyz + object xy

# All batch functions take and return pyarrow Tables. Going through pandas
# breaks list<float> columns with newer pandas (3.x) + Ray, so we never do.


# ---------------------------------------------------------------- batch functions
def _list_rows(table: pa.Table, col: str):
    """Per-row numpy views of a list<float> column."""
    arr = table.column(col).combine_chunks()
    if len(arr) == 0:
        return []
    offsets = arr.offsets.to_numpy()
    values = arr.values.to_numpy(zero_copy_only=False)
    return [values[offsets[i] : offsets[i + 1]] for i in range(len(arr))]


def validate_batch(t: pa.Table) -> pa.Table:
    lengths = t.column("length").to_pylist()
    reasons = []
    for length, obs, act in zip(lengths, _list_rows(t, "obs"), _list_rows(t, "act")):
        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        if int(length) != EP_LEN:
            r = "truncated"
        elif obs.size != (EP_LEN + 1) * OBS_DIM or act.size != EP_LEN * ACT_DIM:
            r = "bad_shape"
        elif not (np.isfinite(obs).all() and np.isfinite(act).all()):
            r = "non_finite"
        elif np.abs(obs).max() > OBS_ABS_LIMIT:
            r = "out_of_range"
        else:
            r = "ok"
        reasons.append(r)
    return t.append_column("reject_reason", pa.array(reasons, type=pa.string()))


def keep_valid(t: pa.Table) -> pa.Table:
    return t.filter(pc.equal(t["reject_reason"], "ok"))


def make_hasher(quantum: float, stride: int):
    def hash_batch(t: pa.Table) -> pa.Table:
        exact, near = [], []
        for obs in _list_rows(t, "obs"):
            o = np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM)
            exact.append(hashlib.blake2b(np.round(o, 6).tobytes(), digest_size=16).hexdigest())
            q = np.round(o[::stride][:, NEAR_DUP_DIMS] / quantum).astype(np.int32)
            near.append(hashlib.blake2b(q.tobytes(), digest_size=16).hexdigest())
        t = t.append_column("exact_hash", pa.array(exact, type=pa.string()))
        return t.append_column("near_hash", pa.array(near, type=pa.string()))

    return hash_batch


def keep_first(t: pa.Table) -> pa.Table:
    """Deterministic representative per duplicate group."""
    return t.sort_by("episode_id").slice(0, 1)


def filter_policy(policy: str):
    def f(t: pa.Table) -> pa.Table:
        return t.filter(pc.equal(t["policy"], policy))

    return f


def _small_table_to_pandas(t: pa.Table, cols):
    # drop schema metadata so stale pandas metadata can never be re-interpreted
    return t.select(cols).replace_schema_metadata(None).to_pandas()


# ---------------------------------------------------------------- engines
class RayEngine:
    def __init__(self, num_cpus, progress):
        import ray

        self.ray = ray
        ray.init(num_cpus=num_cpus, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
        ctx = ray.data.DataContext.get_current()
        ctx.enable_progress_bars = progress

    def read(self, path):
        return self.ray.data.read_parquet(str(path))

    def map(self, ds, fn):
        return ds.map_batches(fn, batch_format="pyarrow", batch_size=64)

    def materialize(self, ds):
        return ds.materialize()

    def columns(self, ds, cols):
        tables = [self.ray.get(ref) for ref in ds.select_columns(cols).to_arrow_refs()]
        tables = [t.replace_schema_metadata(None) for t in tables if t.num_rows]
        if not tables:
            import pandas as pd

            return pd.DataFrame(columns=cols)
        return _small_table_to_pandas(pa.concat_tables(tables), cols)

    def dedup(self, ds, key):
        return ds.groupby(key).map_groups(keep_first, batch_format="pyarrow")

    def sample(self, ds, n, seed):
        return ds.random_shuffle(seed=seed).limit(n)

    def union(self, parts):
        return parts[0].union(*parts[1:]) if len(parts) > 1 else parts[0]

    def count(self, ds):
        return ds.count()

    def write(self, ds, out, shards):
        ds.repartition(shards).write_parquet(str(out))

    def shutdown(self):
        self.ray.shutdown()


class LocalEngine:
    """Same batch functions on in-memory Arrow tables. Useful for debugging."""

    def read(self, path):
        return pq.read_table(path).replace_schema_metadata(None)

    def map(self, t, fn):
        return fn(t)

    def materialize(self, t):
        return t

    def columns(self, t, cols):
        return _small_table_to_pandas(t, cols)

    def dedup(self, t, key):
        t = t.sort_by("episode_id")
        seen, keep = set(), []
        for i, k in enumerate(t.column(key).to_pylist()):
            if k not in seen:
                seen.add(k)
                keep.append(i)
        return t.take(pa.array(keep, type=pa.int64()))

    def sample(self, t, n, seed):
        idx = np.random.default_rng(seed).permutation(t.num_rows)[: min(n, t.num_rows)]
        return t.take(pa.array(idx, type=pa.int64()))

    def union(self, parts):
        return pa.concat_tables(parts)

    def count(self, t):
        return t.num_rows

    def write(self, t, out, shards):
        out.mkdir(parents=True, exist_ok=True)
        for k, idx in enumerate(np.array_split(np.arange(t.num_rows), shards)):
            if len(idx):
                pq.write_table(t.take(pa.array(idx, type=pa.int64())), out / f"part-{k:04d}.parquet")

    def shutdown(self):
        pass


# ---------------------------------------------------------------- pipeline
def _write(engine, ds, name, shards):
    out = PROCESSED_DIR / name
    shutil.rmtree(out, ignore_errors=True)
    engine.write(ds, out, shards)


def _size(n_episodes):
    return {"episodes": int(n_episodes), "transitions": int(n_episodes) * EP_LEN}


def process_train(engine, quantum, stride, shards, seed) -> dict:
    ds = engine.materialize(engine.map(engine.read(RAW_DIR / "train"), validate_batch))
    meta = engine.columns(ds, ["policy", "reject_reason", "is_retry", "injected_fault"])
    valid = engine.map(ds, keep_valid)
    hashed = engine.materialize(engine.map(valid, make_hasher(quantum, stride)))
    hcols = engine.columns(hashed, ["episode_id", "policy", "exact_hash", "near_hash", "is_retry"])

    n_valid = len(hcols)
    n_exact_unique = hcols["exact_hash"].nunique()
    n_near_unique = hcols["near_hash"].nunique()

    dedup = engine.materialize(engine.dedup(hashed, "near_hash"))
    dcols = engine.columns(dedup, ["policy"])
    per_policy = dcols["policy"].value_counts().to_dict()
    n_min = int(min(per_policy.values()))

    balanced = engine.materialize(engine.union([
        engine.sample(engine.map(dedup, filter_policy(p)), n_min, seed) for p in sorted(per_policy)
    ]))
    matched = engine.materialize(engine.sample(dedup, n_min * len(per_policy), seed))

    _write(engine, hashed, "raw", shards)
    _write(engine, dedup, "dedup", shards)
    _write(engine, balanced, "balanced", shards)
    _write(engine, matched, "dedup_matched", shards)

    n_dedup = engine.count(dedup)
    return {
        "raw_episodes_read": int(len(meta)),
        "raw_by_policy": meta["policy"].value_counts().to_dict(),
        "injected_faults": meta.loc[meta.injected_fault != "none", "injected_fault"].value_counts().to_dict(),
        "rejected_by_reason": meta.loc[meta.reject_reason != "ok", "reject_reason"].value_counts().to_dict(),
        "valid_episodes": int(n_valid),
        "valid_retry_copies_injected": int(hcols["is_retry"].sum()),
        "exact_duplicates": int(n_valid - n_exact_unique),
        "near_duplicates_removed": int(n_valid - n_near_unique),
        "near_duplicates_removed_frac": float((n_valid - n_near_unique) / max(n_valid, 1)),
        "dedup_by_policy": {k: int(v) for k, v in per_policy.items()},
        "near_dup_removed_by_policy": {
            p: int((hcols.policy == p).sum() - per_policy.get(p, 0)) for p in sorted(hcols.policy.unique())
        },
        "balanced_per_policy": n_min,
        "hash_params": {"quantum_m": quantum, "stride_steps": stride, "dims": "fingertip xyz + object xy"},
        "datasets": {
            "raw": _size(n_valid),
            "dedup": _size(n_dedup),
            "balanced": _size(engine.count(balanced)),
            "dedup_matched": _size(engine.count(matched)),
        },
    }


def process_eval(engine, split, shards) -> dict:
    ds = engine.materialize(engine.map(engine.read(RAW_DIR / split), validate_batch))
    meta = engine.columns(ds, ["policy", "reject_reason"])
    valid = engine.materialize(engine.map(ds, keep_valid))
    _write(engine, valid, split, shards)
    n = engine.count(valid)
    return {
        "raw_episodes_read": int(len(meta)),
        "rejected_by_reason": meta.loc[meta.reject_reason != "ok", "reject_reason"].value_counts().to_dict(),
        "by_policy": meta.loc[meta.reject_reason == "ok", "policy"].value_counts().to_dict(),
        **_size(n),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=["ray", "local"], default="ray")
    p.add_argument("--num-cpus", type=int, default=None)
    p.add_argument("--quantum", type=float, default=0.02, help="position bin size in metres for near-dup hashing")
    p.add_argument("--stride", type=int, default=10, help="hash every Nth timestep")
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress", action="store_true", help="show Ray progress bars")
    args = p.parse_args(argv)

    engine = RayEngine(args.num_cpus, args.progress) if args.engine == "ray" else LocalEngine()
    t0 = time.perf_counter()
    try:
        stats = {"engine": args.engine, "train": process_train(engine, args.quantum, args.stride, args.shards, args.seed)}
        for split in ("eval_id", "eval_ood"):
            if (RAW_DIR / split).exists():
                stats[split] = process_eval(engine, split, max(1, args.shards // 4))
    finally:
        engine.shutdown()
    stats["wall_time_s"] = time.perf_counter() - t0
    write_json(RESULTS_DIR / "data_stats.json", stats)

    tr = stats["train"]
    print(f"train: {tr['raw_episodes_read']} read, {tr['valid_episodes']} valid, "
          f"{tr['exact_duplicates']} exact dups (injected retry copies: {tr['valid_retry_copies_injected']}), "
          f"{tr['near_duplicates_removed']} near-dups removed ({100 * tr['near_duplicates_removed_frac']:.1f}%)")
    for name, s in tr["datasets"].items():
        print(f"  {name:14s} {s['episodes']:5d} episodes  {s['transitions']:7d} transitions")
    print(f"Wrote {RESULTS_DIR / 'data_stats.json'}")


if __name__ == "__main__":
    main()
