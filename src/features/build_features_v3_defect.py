"""Build the v3 feature table (experiment 006) -- official KAMP defect target.

Same aggregation as v1 (one row per machining experiment: every sensor channel
collapsed into mean/std/min/max over the whole run, plus the time share spent in
each `Machining_Process` stage). The ONLY thing that changes versus v1/002 is the
target and, as a consequence, which experiments have to be dropped.

Key decisions (rationale in docs/experiments/006-preprocess-defect-target.md):
  * target `defect` follows the official KAMP guidebook (docs/03. Guidebook_CNC.pdf,
    pp. 27-40, summarised in docs/domain/glossary.md):
        machining_finalized != "yes"                        -> defect (1)
        machining_finalized == "yes" and passed != "yes"    -> defect (1)
        machining_finalized == "yes" and passed == "yes"    -> good   (0)
  * labels come from `train.csv`, never from the integrated dataset's `Y_*`
    (docs/decisions/001-data-source-strategy.md).
  * only experiments 19 and 25 are dropped. (19, 25) are byte-identical files whose
    `defect` labels conflict (19 -> 1, 25 -> 0). The other byte-identical pair
    (14, 24) does NOT conflict under this target (both -> 0), so it is kept --
    but both members must land in the SAME split, see PAIRED_NOS below.
  * the split is stratified over *experiments* (`No`), never over rows.

Reads `data/raw/` read-only, writes `data/processed/features_v3_*`.

Run: python src/features/build_features_v3_defect.py
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

# Seed 42 is used only to assign whole experiments to train/test, and only once:
# the resulting lists are frozen as TRAIN_NOS/TEST_NOS below so that a different
# sklearn version can never silently reshuffle the split (same reasoning as 004).
RANDOM_SEED = 42
TEST_SIZE = 5  # 5 of 23 experiments ~= 22%

# (19, 25) are byte-identical telemetry files whose `defect` labels disagree
# (19: finalized=yes/passed=no -> 1, 25: finalized=yes/passed=yes -> 0). Neither
# label can be verified from the data, so the pair is excluded. See
# docs/failures/002-duplicate-experiment-conflicting-labels.md. N: 25 -> 23.
DUPLICATE_CONFLICT_NOS = [19, 25]

# (14, 24) are also byte-identical, but they agree under this target (both good).
# They are kept, yet they are NOT two independent samples -- the telemetry is the
# very same bytes. Splitting them across train/test would put a literal copy of a
# training row into the test set and inflate the score. They must stay together.
PAIRED_NOS = (14, 24)

# Constant across the whole dataset (EDA 001) -- zero information, and their std
# would be 0, which breaks scaling.
GLOBAL_CONSTANT_COLS = [
    "Z_CurrentFeedback",
    "Z_DCBusVoltage",
    "Z_OutputCurrent",
    "Z_OutputVoltage",
]

PROCESS_COL = "Machining_Process"
TARGET = "defect"  # 1 = defective part, 0 = good part
AUX_LABEL = "tool_condition_worn"  # 002/003/005's target, kept for reference only
# Machining parameters that are known before the cut starts -- legitimate features.
# `material` is excluded: constant (aluminum) across all 25 experiments.
SETTING_COLS = ["feedrate", "clamp_pressure"]
AGGS = ["mean", "std", "min", "max"]

ID_COLS = ["No"]
LABEL_COLS = [TARGET, AUX_LABEL]

# Frozen split (derived once with RANDOM_SEED, see derive_split()). PAIRED_NOS are
# pinned to train first; the remaining 21 experiments are stratified on `defect`.
TRAIN_NOS = [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 17, 18, 20, 21, 22, 23, 24]
TEST_NOS = [6, 10, 11, 15, 16]


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
    # Official KAMP definition: a part is good only when the cut finished AND it
    # passed visual inspection. Unfinished runs have no inspection result (NaN),
    # and they count as defective, so the `!= "yes"` comparison handles them.
    finalized = meta["machining_finalized"] == "yes"
    passed = meta["passed_visual_inspection"] == "yes"
    meta[TARGET] = (~(finalized & passed)).astype(int)
    meta[AUX_LABEL] = (meta["tool_condition"] == "worn").astype(int)

    # `machining_finalized` / `passed_visual_inspection` are deliberately NOT stored:
    # together they ARE the target, so keeping them around only invites a trivial leak.
    table = meta[["No", *SETTING_COLS, *LABEL_COLS]].merge(feats, on="No", how="inner")
    assert len(table) == len(kept_nos), "metadata/telemetry join lost experiments"
    return table


def derive_split(table: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Reproduce the frozen split, then assert it matches TRAIN_NOS/TEST_NOS.

    The paired experiments (14, 24) are byte-identical telemetry, so they are pinned
    to the train side *before* splitting; only the remaining 21 experiments go through
    the stratified draw. This assigns whole experiments, never rows: every telemetry
    row of an experiment ends up on the same side of the split.
    """
    free = table[~table["No"].isin(PAIRED_NOS)]
    train_no, test_no = train_test_split(
        free["No"].to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=free[TARGET].to_numpy(),
    )
    train_no = sorted(int(n) for n in train_no) + list(PAIRED_NOS)
    train_no = sorted(train_no)
    test_no = sorted(int(n) for n in test_no)

    assert (train_no, test_no) == (TRAIN_NOS, TEST_NOS), (
        "seed-42 draw no longer reproduces the frozen split (sklearn version change?). "
        "TRAIN_NOS/TEST_NOS stay authoritative -- do not silently reshuffle."
    )
    assert sorted(TRAIN_NOS + TEST_NOS) == sorted(table["No"].tolist()), (
        "frozen split does not cover exactly the kept experiments"
    )
    assert not set(TRAIN_NOS) & set(TEST_NOS), "an experiment is in both splits"
    a, b = PAIRED_NOS
    assert (a in TRAIN_NOS) == (b in TRAIN_NOS), (
        f"byte-identical experiments {a}/{b} must share a split"
    )
    return TRAIN_NOS, TEST_NOS


def main() -> int:
    if not EXP_DIR.exists():
        print(f"원본 데이터 없음: {EXP_DIR} -- KAMP에서 내려받아야 합니다.")
        return 1

    table = build_feature_table()
    train_no, test_no = derive_split(table)

    train = table[table["No"].isin(train_no)].sort_values("No").reset_index(drop=True)
    test = table[table["No"].isin(test_no)].sort_values("No").reset_index(drop=True)

    feature_cols = [c for c in table.columns if c not in ID_COLS + LABEL_COLS]

    # Zero-variance-in-train columns carry no signal and would divide by zero when
    # scaled. Detected on the TRAIN split only -- looking at test here would leak.
    train_std = train[feature_cols].std(ddof=0).fillna(0.0)
    dead_cols = sorted(train_std[train_std == 0.0].index)
    feature_cols = [c for c in feature_cols if c not in dead_cols]
    train = train.drop(columns=dead_cols)
    test = test.drop(columns=dead_cols)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "features_v3_train.csv", index=False)
    test.to_csv(OUT_DIR / "features_v3_test.csv", index=False)

    # No global scaler file is written (unlike v1): 003 showed it must not be used
    # inside CV, where scaling has to be re-fit per fold. Stored CSVs are raw scale.
    summary = {
        "target": TARGET,
        "target_definition": (
            "1 (defect) if machining_finalized != 'yes' or passed_visual_inspection != 'yes'; "
            "0 (good) if both are 'yes'"
        ),
        "excluded_experiments": DUPLICATE_CONFLICT_NOS,
        "paired_experiments_same_split": list(PAIRED_NOS),
        "n_experiments": int(len(table)),
        "n_features": len(feature_cols),
        "n_process_share_features": len([c for c in feature_cols if c.startswith("proc_frac_")]),
        "dropped_zero_variance_in_train": dead_cols,
        "random_seed": RANDOM_SEED,
        "train_No": train_no,
        "test_No": test_no,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_target_counts": train[TARGET].value_counts().sort_index().to_dict(),
        "test_target_counts": test[TARGET].value_counts().sort_index().to_dict(),
        "train_aux_label_counts": train[AUX_LABEL].value_counts().sort_index().to_dict(),
        "test_aux_label_counts": test[AUX_LABEL].value_counts().sort_index().to_dict(),
        "label_nan_count": int(train[LABEL_COLS].isna().sum().sum())
        + int(test[LABEL_COLS].isna().sum().sum()),
        "feature_nan_count": int(train[feature_cols].isna().sum().sum())
        + int(test[feature_cols].isna().sum().sum()),
        "outputs": [
            str((OUT_DIR / "features_v3_train.csv").relative_to(ROOT)),
            str((OUT_DIR / "features_v3_test.csv").relative_to(ROOT)),
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print("\nper-experiment labels:")
    print(
        table[["No", TARGET, AUX_LABEL]]
        .assign(split=lambda d: np.where(d["No"].isin(train_no), "train", "test"))
        .sort_values("No")
        .to_string(index=False)
    )
    print("\nfeature columns:")
    print(json.dumps(feature_cols, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
