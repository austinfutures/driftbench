"""Shared constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
PLOTS_DIR = RESULTS_DIR / "plots"
PROFILE_DIR = RESULTS_DIR / "profile"

ENV_ID = "Pusher-v5"
OBS_DIM = 23
ACT_DIM = 7
EP_LEN = 100  # Pusher-v5 TimeLimit

# Pusher-v5 observation layout:
#   qpos[:7] (arm joints), qvel[:7], tips_arm xyz, object xyz, goal xyz
OBS_SLICES = {
    "qpos": slice(0, 7),
    "qvel": slice(7, 14),
    "tip": slice(14, 17),
    "object": slice(17, 20),
    "goal": slice(20, 23),
}

# Physics variants. "heavy_slippery" is the out-of-distribution test task.
VARIANTS = {
    "nominal": {"mass_scale": 1.0, "friction_scale": 1.0},
    "heavy_slippery": {"mass_scale": 4.0, "friction_scale": 0.25},
}

POLICIES = ("scripted", "noisy", "random")
