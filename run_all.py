"""Run the whole DriftBench pipeline end to end.

    python run_all.py --quick      # smoke test: tiny data, short training, 1 seed
    python run_all.py              # full run
    python run_all.py --stages experiments report   # resume from a stage

Stages: collect -> pipeline -> experiments -> profile -> report
"""
import argparse
import time

STAGES = ["collect", "pipeline", "experiments", "profile", "report"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="smoke-test settings (numbers are NOT meaningful)")
    p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    p.add_argument("--engine", choices=["ray", "local"], default="ray", help="data pipeline engine")
    p.add_argument("--workers", type=int, default=4, help="collection worker processes")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--compile", action="store_true", help="also profile torch.compile")
    args = p.parse_args()

    seeds = args.seeds or ([0] if args.quick else [0, 1, 2])
    t0 = time.perf_counter()

    if "collect" in args.stages:
        from driftbench import collect
        collect.main(["--workers", str(args.workers)] + (["--scale", "0.08"] if args.quick else []))

    if "pipeline" in args.stages:
        from driftbench import pipeline
        pipeline.main(["--engine", args.engine] + (["--shards", "2"] if args.quick else []))

    if "experiments" in args.stages:
        from driftbench import experiments
        exp_args = ["--seeds", *map(str, seeds), "--device", args.device]
        if args.quick:
            exp_args += ["--steps", "300", "--starts-per-ep", "2", "--force"]
        experiments.main(exp_args)

    if "profile" in args.stages:
        from driftbench import profile_train
        prof_args = ["--device", args.device] + (["--compile"] if args.compile else [])
        if args.quick:
            prof_args += ["--steps", "30", "--warmup", "5", "--profile-steps", "5"]
        profile_train.main(prof_args)

    if "report" in args.stages:
        from driftbench import report
        report.main([])

    print(f"\nAll requested stages finished in {time.perf_counter() - t0:.0f} s. See results/RESULTS.md")
    if args.quick:
        print("NOTE: --quick numbers are a smoke test only. Run without --quick before quoting anything.")


if __name__ == "__main__":
    main()
