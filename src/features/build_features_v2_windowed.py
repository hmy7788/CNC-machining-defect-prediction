"""Build the v2 windowed feature table (experiment 004).

Same recipe as v1 (src/features/build_features_v1.py, experiment 002) with exactly
one variable changed: instead of collapsing a whole experiment into a single row,
each experiment's time series is cut into non-overlapping windows of
`WINDOW_SIZE` rows and every window becomes one row.

Everything else is deliberately identical to 002/003 so that 004 can be compared
against the 003 baseline without confounds:
  * labels come from `train.csv`, never from the integrated dataset's `Y_*`
    (docs/decisions/001-data-source-strategy.md);
  * experiments 14, 19, 24, 25 are dropped (byte-identical pairs with conflicting
    labels, docs/failures/002-duplicate-experiment-conflicting-labels.md);
  * the train/test experiment lists are copied verbatim from 002 -- they are NOT
    re-drawn here, so the split is bit-for-bit the one 003 was evaluated on;
  * the same columns are excluded (`material`, the four global-constant Z channels).

The experiment-level label is broadcast to every window of that experiment: a
window does not get its own label.

GROUP LEAKAGE WARNING: windows cut from one experiment are slices of a single
time series and are NOT independent samples. Never shuffle/split at window level.
Split and cross-validate on the `No` column (GroupKFold / LeaveOneGroupOut).

Reads `data/raw/` read-only, writes `data/processed/features_v2_*`.

Run: python src/features/build_features_v2_windowed.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "CNC 비식별화 원본데이터_1209"
EXP_DIR = RAW_DIR / "CNC Virtual Data set _v2"
META_CSV = RAW_DIR / "train.csv"
OUT_DIR = ROOT / "data" / "processed"

# 100 rows ~= 10 s of telemetry (100 Hz sampling downsampled to 10 Hz in the source
# dataset). Non-overlapping: overlapping windows would put near-duplicate rows in the
# table, inflating the effective sample count without adding information -- and if a
# later step ever split at window level, overlap would leak outright.
WINDOW_SIZE = 100

# Verbatim from experiment 002 -- do not re-draw. Changing these would break the
# comparison against the 003 baseline.
TRAIN_NOS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 18, 20, 21, 22, 23]
TEST_NOS = [4, 8, 12, 15, 16]

# (14, 24) and (19, 25) are byte-identical telemetry files with conflicting labels.
DUPLICATE_CONFLICT_NOS = [14, 19, 24, 25]

# Constant across the whole dataset (EDA 001) -- zero information.
GLOBAL_CONSTANT_COLS = [
    "Z_CurrentFeedback",
    "Z_DCBusVoltage",
    "Z_OutputCurrent",
    "Z_OutputVoltage",
]

PROCESS_COL = "Machining_Process"
TARGET = "tool_condition_worn"
SECONDARY_TARGET = "passed_visual_inspection_yes"
# `material` is excluded: constant (aluminum) across all 25 experiments.
SETTING_COLS = ["feedrate", "clamp_pressure"]
AGGS = ["mean", "std", "min", "max"]

GROUP_COL = "No"
WINDOW_COL = "window_idx"
ID_COLS = [GROUP_COL, WINDOW_COL]
LABEL_COLS = [TARGET, SECONDARY_TARGET, "machining_finalized_yes"]


def load_metadata() -> pd.DataFrame:
    """train.csv pads values with spaces for alignment -- strip them (see glossary)."""
    meta = pd.read_csv(META_CSV, skipinitialspace=True)
    meta.columns = [c.strip() for c in meta.columns]
    for col in meta.columns:
        if meta[col].dtype == object:
            meta[col] = meta[col].str.strip().replace("", np.nan)
    return meta


def experiment_files() -> dict[int, Path]:
    files = {}
    for path in sorted(EXP_DIR.glob("experiment_*.csv")):
        match = re.search(r"experiment_(\d+)\.csv$", path.name)
        if match:
            files[int(match.group(1))] = path
    return files


def summarize_window(no: int, window_idx: int, window: pd.DataFrame) -> dict:
    """Collapse one time window into a single row of summary features (as in 002)."""
    numeric = window.drop(columns=[PROCESS_COL]).drop(columns=GLOBAL_CONSTANT_COLS)
    numeric = numeric.apply(pd.to_numeric, errors="coerce")

    stats = numeric.agg(AGGS)
    feats = {f"{col}_{agg}": stats.loc[agg, col] for col in numeric.columns for agg in AGGS}

    feats["n_rows"] = float(len(window))
    # Time share per machining stage inside this window (fractions, as in 002).
    shares = window[PROCESS_COL].value_counts(normalize=True)
    for stage, frac in shares.items():
        feats[f"proc_frac_{stage.replace(' ', '_')}"] = float(frac)

    feats[GROUP_COL] = no
    feats[WINDOW_COL] = window_idx
    return feats


def window_experiment(no: int, path: Path) -> list[dict]:
    """Cut one experiment into non-overlapping WINDOW_SIZE-row windows.

    The trailing remainder (< WINDOW_SIZE rows) is dropped so every window is
    summarized over the same number of samples -- otherwise a short final window
    would have noisier std/min/max than the rest and `n_rows` would silently encode
    "last window of the run".
    """
    df = pd.read_csv(path)
    df[PROCESS_COL] = df[PROCESS_COL].str.strip().str.title()  # "end" == "End"

    n_windows = len(df) // WINDOW_SIZE
    rows = []
    for w in range(n_windows):
        window = df.iloc[w * WINDOW_SIZE : (w + 1) * WINDOW_SIZE]
        rows.append(summarize_window(no, w, window))
    return rows


def build_feature_table() -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    meta = load_metadata()
    files = experiment_files()

    kept_nos = [no for no in sorted(files) if no not in DUPLICATE_CONFLICT_NOS]
    assert sorted(TRAIN_NOS + TEST_NOS) == kept_nos, (
        "the 002 train/test experiment lists no longer cover exactly the kept experiments"
    )

    raw_lengths: dict[int, int] = {}
    rows: list[dict] = []
    for no in kept_nos:
        raw_lengths[no] = len(pd.read_csv(files[no], usecols=[PROCESS_COL]))
        rows.extend(window_experiment(no, files[no]))

    feats = pd.DataFrame(rows)
    window_counts = feats.groupby(GROUP_COL).size().to_dict()

    # Stages absent from a window get share 0, not NaN.
    proc_cols = [c for c in feats.columns if c.startswith("proc_frac_")]
    feats[proc_cols] = feats[proc_cols].fillna(0.0)

    meta = meta[meta[GROUP_COL].isin(kept_nos)].copy()
    meta[TARGET] = (meta["tool_condition"] == "worn").astype(int)
    meta["machining_finalized_yes"] = (meta["machining_finalized"] == "yes").astype(int)
    # NaN stays NaN: unfinished parts were never visually inspected (glossary).
    meta[SECONDARY_TARGET] = meta["passed_visual_inspection"].map({"yes": 1.0, "no": 0.0})

    # Experiment-level labels and machining settings are broadcast to every window
    # of that experiment by this merge.
    table = meta[[GROUP_COL, *SETTING_COLS, *LABEL_COLS]].merge(feats, on=GROUP_COL, how="inner")
    assert len(table) == len(feats), "metadata/telemetry join lost or duplicated windows"

    front = [*ID_COLS, *LABEL_COLS]
    table = table[front + [c for c in table.columns if c not in front]]
    return table, window_counts, raw_lengths


def main() -> int:
    if not EXP_DIR.exists():
        print(f"원본 데이터 없음: {EXP_DIR} -- KAMP에서 내려받아야 합니다.")
        return 1

    table, window_counts, raw_lengths = build_feature_table()

    train = table[table[GROUP_COL].isin(TRAIN_NOS)].sort_values(ID_COLS).reset_index(drop=True)
    test = table[table[GROUP_COL].isin(TEST_NOS)].sort_values(ID_COLS).reset_index(drop=True)
    assert len(train) + len(test) == len(table), "an experiment fell outside both splits"
    assert not set(train[GROUP_COL]) & set(test[GROUP_COL]), "an experiment appears in both splits"

    feature_cols = [c for c in table.columns if c not in ID_COLS + LABEL_COLS]

    # Zero-variance-in-train columns carry no signal and would divide by zero when
    # scaled. Detected on the TRAIN split only -- looking at test here would leak.
    train_std = train[feature_cols].std(ddof=0)
    dead_cols = sorted(train_std.fillna(0.0)[train_std.fillna(0.0) == 0.0].index)
    feature_cols = [c for c in feature_cols if c not in dead_cols]
    train = train.drop(columns=dead_cols)
    test = test.drop(columns=dead_cols)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "features_v2_train.csv", index=False)
    test.to_csv(OUT_DIR / "features_v2_test.csv", index=False)

    per_experiment = {
        str(no): {
            "raw_rows": int(raw_lengths[no]),
            "n_windows": int(window_counts[no]),
            "dropped_tail_rows": int(raw_lengths[no] % WINDOW_SIZE),
            "split": "train" if no in TRAIN_NOS else "test",
        }
        for no in sorted(window_counts)
    }

    summary = {
        "window_size": WINDOW_SIZE,
        "overlap": "none (non-overlapping)",
        "excluded_experiments": DUPLICATE_CONFLICT_NOS,
        "n_experiments": len(window_counts),
        "n_windows_total": int(len(table)),
        "n_windows_train": int(len(train)),
        "n_windows_test": int(len(test)),
        "windows_per_experiment_min": int(min(window_counts.values())),
        "windows_per_experiment_max": int(max(window_counts.values())),
        "n_features": len(feature_cols),
        "n_process_share_features": len([c for c in feature_cols if c.startswith("proc_frac_")]),
        "dropped_zero_variance_in_train": dead_cols,
        "train_No": TRAIN_NOS,
        "test_No": TEST_NOS,
        "train_target_counts_windows": train[TARGET].value_counts().sort_index().to_dict(),
        "test_target_counts_windows": test[TARGET].value_counts().sort_index().to_dict(),
        "train_target_counts_experiments": (
            train.drop_duplicates(GROUP_COL)[TARGET].value_counts().sort_index().to_dict()
        ),
        "test_target_counts_experiments": (
            test.drop_duplicates(GROUP_COL)[TARGET].value_counts().sort_index().to_dict()
        ),
        "train_secondary_target_missing_windows": int(train[SECONDARY_TARGET].isna().sum()),
        "test_secondary_target_missing_windows": int(test[SECONDARY_TARGET].isna().sum()),
        "feature_nan_count": int(train[feature_cols].isna().sum().sum())
        + int(test[feature_cols].isna().sum().sum()),
        "total_dropped_tail_rows": sum(v["dropped_tail_rows"] for v in per_experiment.values()),
        "per_experiment": per_experiment,
        "group_column": GROUP_COL,
        "cv_requirement": "GroupKFold / LeaveOneGroupOut on `No` -- never shuffle windows",
        "outputs": [
            str((OUT_DIR / "features_v2_train.csv").relative_to(ROOT)),
            str((OUT_DIR / "features_v2_test.csv").relative_to(ROOT)),
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print("\nfeature columns:")
    print(json.dumps(feature_cols, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
