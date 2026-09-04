"""Build the v1 baseline feature table (experiment 002).

One row per machining experiment (not per telemetry row): every sensor channel is
collapsed into mean/std/min/max over the whole run, plus the time share spent in
each `Machining_Process` stage.

Key decisions (rationale in docs/experiments/002-preprocess-baseline-features.md):
  * labels come from `train.csv`, never from the integrated dataset's `Y_*`
    (docs/decisions/001-data-source-strategy.md).
  * experiments 14, 19, 24, 25 are dropped: (14, 24) and (19, 25) are byte-identical
    files carrying conflicting labels (docs/failures/002-...md). N: 25 -> 21.
  * the split is stratified over *experiments* (`No`), never over rows.
  * scaling statistics are fit on the train split only and written out separately,
    so nothing from the test split can leak into training.

Reads `data/raw/` read-only, writes `data/processed/features_v1_*`.

Run: python src/features/build_features_v1.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "CNC 비식별화 원본데이터_1209"
EXP_DIR = RAW_DIR / "CNC Virtual Data set _v2"
META_CSV = RAW_DIR / "train.csv"
OUT_DIR = ROOT / "data" / "processed"

RANDOM_SEED = 42
TEST_SIZE = 5  # 5 of 21 experiments ~= 24%

# (14, 24) and (19, 25) are byte-identical telemetry files with conflicting labels.
# Neither label can be verified from the data, so both pairs are excluded rather
# than arbitrarily picking a "correct" one. See docs/failures/002-*.md.
DUPLICATE_CONFLICT_NOS = [14, 19, 24, 25]

# Constant across the whole dataset (EDA 001) -- zero information, and their std
# would be 0, which breaks scaling.
GLOBAL_CONSTANT_COLS = [
    "Z_CurrentFeedback",
    "Z_DCBusVoltage",
    "Z_OutputCurrent",
    "Z_OutputVoltage",
]

PROCESS_COL = "Machining_Process"
TARGET = "tool_condition_worn"
SECONDARY_TARGET = "passed_visual_inspection_yes"
# Machining parameters that are known before the cut starts -- legitimate features.
# `material` is excluded: constant (aluminum) across all 25 experiments.
SETTING_COLS = ["feedrate", "clamp_pressure"]
AGGS = ["mean", "std", "min", "max"]

ID_COLS = ["No"]
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


def summarize_experiment(no: int, path: Path) -> pd.Series:
    """Collapse one experiment's time series into a single row of summary features."""
    df = pd.read_csv(path)
    df[PROCESS_COL] = df[PROCESS_COL].str.strip().str.title()  # "end" == "End"

    numeric = df.drop(columns=[PROCESS_COL]).drop(columns=GLOBAL_CONSTANT_COLS)
    numeric = numeric.apply(pd.to_numeric, errors="coerce")

    stats = numeric.agg(AGGS)
    feats = {f"{col}_{agg}": stats.loc[agg, col] for col in numeric.columns for agg in AGGS}

    feats["n_rows"] = float(len(df))
    # Time share per machining stage (fractions, so the differing run lengths
    # do not turn this into a duplicate of n_rows).
    shares = df[PROCESS_COL].value_counts(normalize=True)
    for stage, frac in shares.items():
        feats[f"proc_frac_{stage.replace(' ', '_')}"] = float(frac)

    feats["No"] = no
    return pd.Series(feats)


def build_feature_table() -> pd.DataFrame:
    meta = load_metadata()
    files = experiment_files()

    kept_nos = [no for no in sorted(files) if no not in DUPLICATE_CONFLICT_NOS]
    rows = [summarize_experiment(no, files[no]) for no in kept_nos]
    feats = pd.DataFrame(rows)

    # Stages absent from a run get share 0, not NaN.
    proc_cols = [c for c in feats.columns if c.startswith("proc_frac_")]
    feats[proc_cols] = feats[proc_cols].fillna(0.0)

    meta = meta[meta["No"].isin(kept_nos)].copy()
    meta[TARGET] = (meta["tool_condition"] == "worn").astype(int)
    meta["machining_finalized_yes"] = (meta["machining_finalized"] == "yes").astype(int)
    # NaN stays NaN: unfinished parts were never visually inspected (glossary).
    meta[SECONDARY_TARGET] = meta["passed_visual_inspection"].map({"yes": 1.0, "no": 0.0})

    table = meta[["No", *SETTING_COLS, *LABEL_COLS]].merge(feats, on="No", how="inner")
    assert len(table) == len(kept_nos), "metadata/telemetry join lost experiments"
    return table


def split_by_experiment(table: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Stratified hold-out over experiments (`No`), stratified on the primary target.

    This assigns whole experiments, never rows: every telemetry row of an experiment
    ends up on the same side of the split, so no within-run information can leak.
    """
    train_no, test_no = train_test_split(
        table["No"].to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=table[TARGET].to_numpy(),
    )
    return sorted(int(n) for n in train_no), sorted(int(n) for n in test_no)


def main() -> int:
    if not EXP_DIR.exists():
        print(f"원본 데이터 없음: {EXP_DIR} -- KAMP에서 내려받아야 합니다.")
        return 1

    table = build_feature_table()
    train_no, test_no = split_by_experiment(table)

    train = table[table["No"].isin(train_no)].sort_values("No").reset_index(drop=True)
    test = table[table["No"].isin(test_no)].sort_values("No").reset_index(drop=True)

    feature_cols = [c for c in table.columns if c not in ID_COLS + LABEL_COLS]

    # Zero-variance-in-train columns carry no signal and would divide by zero when
    # scaled. Detected on the TRAIN split only -- looking at test here would leak.
    train_std = train[feature_cols].std(ddof=0)
    dead_cols = sorted(train_std[train_std.fillna(0.0) == 0.0].index)
    feature_cols = [c for c in feature_cols if c not in dead_cols]
    train = train.drop(columns=dead_cols)
    test = test.drop(columns=dead_cols)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "features_v1_train.csv", index=False)
    test.to_csv(OUT_DIR / "features_v1_test.csv", index=False)

    # Scaling params fit on train only. Not applied to the stored CSVs (tree models
    # do not need them); a model that needs scaling applies these to BOTH splits.
    scaler = {
        "fit_on": "train split only",
        "n_train_experiments": int(len(train)),
        "mean": {c: float(train[c].mean()) for c in feature_cols},
        "std": {c: float(train[c].std(ddof=0)) for c in feature_cols},
    }
    (OUT_DIR / "features_v1_scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")

    summary = {
        "excluded_experiments": DUPLICATE_CONFLICT_NOS,
        "n_experiments": int(len(table)),
        "n_features": len(feature_cols),
        "n_process_share_features": len([c for c in feature_cols if c.startswith("proc_frac_")]),
        "dropped_zero_variance_in_train": dead_cols,
        "random_seed": RANDOM_SEED,
        "train_No": train_no,
        "test_No": test_no,
        "train_target_counts": train[TARGET].value_counts().sort_index().to_dict(),
        "test_target_counts": test[TARGET].value_counts().sort_index().to_dict(),
        "train_secondary_target_missing": int(train[SECONDARY_TARGET].isna().sum()),
        "test_secondary_target_missing": int(test[SECONDARY_TARGET].isna().sum()),
        "feature_nan_count": int(train[feature_cols].isna().sum().sum())
        + int(test[feature_cols].isna().sum().sum()),
        "outputs": [
            str((OUT_DIR / "features_v1_train.csv").relative_to(ROOT)),
            str((OUT_DIR / "features_v1_test.csv").relative_to(ROOT)),
            str((OUT_DIR / "features_v1_scaler.json").relative_to(ROOT)),
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print("\nfeature columns:")
    print(json.dumps(feature_cols, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
