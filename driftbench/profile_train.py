"""Stage 4: profile the training step and measure the effect of fixes.

Variants (same model, same 1-step loss, same batch size):
    dataloader   textbook PyTorch: a Dataset of numpy transitions with
                 per-sample __getitem__, DataLoader collation, then copy to device
    vectorized   episodes pre-loaded as one device tensor, batches drawn with
                 torch.randint indexing (what train.py uses)
    compiled     vectorized + torch.compile on the model (only with --compile;
                 needs a working compiler toolchain, often unavailable on Windows)

For each variant we time data and compute separately for --steps steps (after
warm-up) and run torch.profiler over a shorter window. Results go to
results/profile.json and results/profile/<variant>_table.txt plus a Chrome
trace you can open at chrome://tracing or https://ui.perfetto.dev.

Usage:
    python -m driftbench.profile_train
    python -m driftbench.profile_train --compile --device cuda
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset

from .config import PROFILE_DIR, RESULTS_DIR
from .data import compute_normalizer, load_split
from .model import EnsembleDynamics
from .train import WindowSampler, rollout_loss
from .utils import environment_info, resolve_device, set_seed, write_json


class TransitionDataset(Dataset):
    def __init__(self, es):
        self.s = es.obs[:, :-1].reshape(-1, es.obs.shape[-1])
        self.a = es.act.reshape(-1, es.act.shape[-1])
        self.s2 = es.obs[:, 1:].reshape(-1, es.obs.shape[-1])

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i], self.a[i], self.s2[i]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def make_batch_fn(variant, es, E, batch_size, device, seed):
    if variant == "dataloader":
        loader = DataLoader(TransitionDataset(es), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

        def gen():
            while True:
                for s, a, s2 in loader:
                    s, a, s2 = s.to(device), a.to(device), s2.to(device)
                    # all members see the same batch (no bootstrap) - a common simplification
                    s_seq = torch.stack([s, s2], dim=1).unsqueeze(0).expand(E, -1, -1, -1)
                    a_seq = a.unsqueeze(1).unsqueeze(0).expand(E, -1, -1, -1)
                    yield s_seq, a_seq

        it = gen()
        return lambda: next(it)
    sampler = WindowSampler(es, E, 1, device, True, seed)
    return lambda: sampler.sample(batch_size)


def profile_variant(variant, es, norm, args, device):
    set_seed(0)
    model = EnsembleDynamics(norm, args.ensemble_size, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fwd_model = model
    if variant == "compiled":
        fwd_model = torch.compile(model)
    next_batch = make_batch_fn(variant, es, args.ensemble_size, args.batch_size, device, 0)

    def step(timed=None):
        t0 = time.perf_counter()
        with record_function("data"):
            s_seq, a_seq = next_batch()
            _sync(device)
        t1 = time.perf_counter()
        with record_function("compute"):
            loss = rollout_loss(model, s_seq, a_seq, fn=fwd_model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            _sync(device)
        t2 = time.perf_counter()
        if timed is not None:
            timed.append((1000 * (t1 - t0), 1000 * (t2 - t1)))

    for _ in range(args.warmup):
        step()
    timed = []
    for _ in range(args.steps):
        step(timed)
    data_ms, compute_ms = np.array(timed).T
    total = data_ms + compute_ms

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=activities) as prof:
        for _ in range(args.profile_steps):
            step()
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    (PROFILE_DIR / f"{variant}_table.txt").write_text(prof.key_averages().table(sort_by=sort_key, row_limit=30))
    prof.export_chrome_trace(str(PROFILE_DIR / f"{variant}_trace.json"))

    return {
        "steps_timed": args.steps,
        "step_ms_median": float(np.median(total)),
        "step_ms_p90": float(np.quantile(total, 0.9)),
        "data_ms_median": float(np.median(data_ms)),
        "compute_ms_median": float(np.median(compute_ms)),
        "data_frac_of_step": float(np.median(data_ms / total)),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="dedup")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--profile-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--hidden", type=int, default=200)
    p.add_argument("--compile", action="store_true", help="also profile torch.compile")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    es = load_split(args.data)
    norm = compute_normalizer(es)
    variants = ["dataloader", "vectorized"] + (["compiled"] if args.compile else [])

    results = {}
    for v in variants:
        print(f"Profiling {v} on {device} ...", flush=True)
        try:
            results[v] = profile_variant(v, es, norm, args, device)
        except Exception as e:  # torch.compile commonly fails without a toolchain
            results[v] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {v} failed: {results[v]['error']}")
            continue
        r = results[v]
        print(f"  step {r['step_ms_median']:.2f} ms (data {r['data_ms_median']:.2f} / compute {r['compute_ms_median']:.2f}), "
              f"data = {100 * r['data_frac_of_step']:.0f}% of step")

    base = results.get("dataloader", {}).get("step_ms_median")
    for v, r in results.items():
        if base and "step_ms_median" in r:
            r["speedup_vs_dataloader"] = base / r["step_ms_median"]

    out = {
        "environment": environment_info(device),
        "config": vars(args),
        "n_transitions": es.n_transitions,
        "variants": results,
    }
    write_json(RESULTS_DIR / "profile.json", out)
    print(f"Wrote {RESULTS_DIR / 'profile.json'} and tables/traces in {PROFILE_DIR}")


if __name__ == "__main__":
    main()
