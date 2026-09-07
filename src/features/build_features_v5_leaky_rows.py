"""Build the v5 ROW-LEVEL feature table (experiment 010) -- intentional leak study.

**WARNING: this experiment deliberately reproduces a data-leakage pattern.** The
official KAMP guidebook (docs/03. Guidebook_CNC.pdf, pp. 33-41) does not aggregate
each experiment into one row: it feeds every raw telemetry TIMESTEP as its own
training sample, broadcasts the experiment-level label to every one of its rows,
and then splits ROWS at random (`fit(..., shuffle=True, validation_split=0.1)`).
CLAUDE.md's "필수: 검증 규율" forbids exactly this (row-level random split on
experiment-unit time series). This script builds the row-level table needed to
QUANTIFY how much that leak inflates the score -- not to adopt it. See
docs/experiments/010-leaky-official-reproduction.md for the full write-up and the
mandatory warning banner.

Two conditions are produced from the SAME pooled row-level table, so the model and
the features are held constant and only the split changes:
  * "leaky"   -- rows shuffled at random across ALL experiments (no group
                 awareness at all), matching the guidebook's method.
  * "grouped" -- whole experiments assigned to train/test, reusing 006/007's
                 frozen TRAIN_NOS/TEST_NOS so every row of an experiment lands on
                 the same side. This is the leak-controlled condition.

Label and experiment population are otherwise identical to 006/007:
  * target `defect` = NOT(machining_finalized == 'yes' AND passed_visual_inspection
    == 'yes'), from `train.csv` (docs/decisions/001-data-source-strategy.md).
  * experiments 19 and 25 are excluded (byte-identical telemetry, conflicting
    `defect` labels -- docs/failures/002-duplicate-experiment-conflicting-labels.md).
  * experiments 14 and 24 are byte-identical telemetry and are BOTH kept (as 006/007
    do). Under the "leaky" row-random split this means literal duplicate rows can
    land on opposite sides of the split -- a SECOND, independent source of leakage
    on top of the row-level shuffle itself. This is intentional (the guidebook's
    method does nothing to prevent it either) and is called out explicitly in the
    experiment doc, not hidden.

Reads `data/raw/` read-only, writes `data/processed/features_v5_*`.

Run: python src/features/build_features_v5_leaky_rows.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "CNC 비식별화 원본데이터_1209"
EXP_DIR = RAW_DIR / "CNC Virtual Data set _v2"
META_CSV = RAW_DIR / "train.csv"
OUT_DIR = ROOT / "data" / "processed"

# Identical population choices to 006 (features_v3), for the "grouped" condition to
# be a direct, apples-to-apples reuse of 007's frozen split.
DUPLICATE_CONFLICT_NOS = [19, 25]
PAIRED_NOS = (14, 24)
TRAIN_NOS = [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 17, 18, 20, 21, 22, 23, 24]
TEST_NOS = [6, 10, 11, 15, 16]

GLOBAL_CONSTANT_COLS = [
    "Z_CurrentFeedback",
    "Z_DCBusVoltage",
    "Z_OutputCurrent",
    "Z_OutputVoltage",
]
PROCESS_COL = "Machining_Process"
# Same ordinal mapping the guidebook's machining_process() function uses (p.28-30).
PROCESS_MAP = {
    "Prep": 0,
    "Layer 1 Up": 1,
    "Layer 1 Down": 2,
    "Layer 2 Up": 3,
    "Layer 2 Down": 4,
    "Layer 3 Up": 5,
    "Layer 3 Down": 6,
    "Repositioning": 7,
    "End": 8,
    "Starting": 9,
}
TARGET = "defect"
ID_COL = "No"
ROW_SPLIT_TEST_SIZE = 0.2
ROW_SPLIT_SEED = 42


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(META_CSV, skipinitialspace=True)
    meta.columns = [c.strip() for c in meta.columns]
    for col in meta.columns:
        if meta[col].dtype == object:
            meta[col] = meta[col].str.strip().replace("", np.nan)
    finalized = meta["machining_finalized"] == "yes"
    passed = meta["passed_visual_inspection"] == "yes"
    meta[TARGET] = (~(finalized & passed)).astype(int)
    return meta


def experiment_files() -> dict[int, Path]:
    files = {}
    for path in sorted(EXP_DIR.glob("experiment_*.csv")):
        match = re.search(r"experiment_(\d+)\.csv$", path.name)
        if match:
            files[int(match.group(1))] = path
    return files


def load_experiment_rows(no: int, path: Path, label: int) -> pd.DataFrame:
    """Every raw timestep of one experiment, as its own row, with the label broadcast."""
    df = pd.read_csv(path)
    df[PROCESS_COL] = df[PROCESS_COL].str.strip().str.title().replace({"End": "End"})
    unmapped = set(df[PROCESS_COL].unique()) - set(PROCESS_MAP)
    assert not unmapped, f"experiment_{no}: unmapped Machining_Process values {unmapped}"
    df[PROCESS_COL] = df[PROCESS_COL].map(PROCESS_MAP)

    df = df.drop(columns=GLOBAL_CONSTANT_COLS)
    numeric_cols = [c for c in df.columns if c != PROCESS_COL]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df[ID_COL] = no
    df[TARGET] = label
    return df


def build_row_table() -> pd.DataFrame:
    meta = load_metadata()
    files = experiment_files()
    kept_nos = [no for no in sorted(files) if no not in DUPLICATE_CONFLICT_NOS]
    label_by_no = meta.set_index(ID_COL)[TARGET].to_dict()

    frames = [load_experiment_rows(no, files[no], label_by_no[no]) for no in kept_nos]
    table = pd.concat(frames, axis=0, ignore_index=True)
    assert not table.drop(columns=[ID_COL, TARGET]).isna().any().any(), "NaN in row features"
    return table


def split_leaky(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Row-level random split, ignoring `No` entirely -- the leak under study."""
    train, test = train_test_split(
        table,
        test_size=ROW_SPLIT_TEST_SIZE,
        random_state=ROW_SPLIT_SEED,
        shuffle=True,
        stratify=table[TARGET],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_grouped(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Whole-experiment split, reusing 006/007's frozen TRAIN_NOS/TEST_NOS."""
    assert sorted(TRAIN_NOS + TEST_NOS) == sorted(table[ID_COL].unique()), (
        "TRAIN_NOS/TEST_NOS no longer cover exactly the kept experiments"
    )
    train = table[table[ID_COL].isin(TRAIN_NOS)].reset_index(drop=True)
    test = table[table[ID_COL].isin(TEST_NOS)].reset_index(drop=True)
    return train, test


def main() -> int:
    if not EXP_DIR.exists():
        print(f"원본 데이터 없음: {EXP_DIR} -- KAMP에서 내려받아야 합니다.", file=sys.stderr)
        return 1

    table = build_row_table()
    feature_cols = [c for c in table.columns if c not in (ID_COL, TARGET)]

    leaky_train, leaky_test = split_leaky(table)
    grouped_train, grouped_test = split_grouped(table)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leaky_train.to_csv(OUT_DIR / "features_v5_leaky_train.csv", index=False)
    leaky_test.to_csv(OUT_DIR / "features_v5_leaky_test.csv", index=False)
    grouped_train.to_csv(OUT_DIR / "features_v5_grouped_train.csv", index=False)
    grouped_test.to_csv(OUT_DIR / "features_v5_grouped_test.csv", index=False)

    dup_rows = table[table[ID_COL].isin(PAIRED_NOS)]
    a = dup_rows[dup_rows[ID_COL] == PAIRED_NOS[0]].drop(columns=[ID_COL]).to_numpy()
    b = dup_rows[dup_rows[ID_COL] == PAIRED_NOS[1]].drop(columns=[ID_COL]).to_numpy()
    no14_in_train = PAIRED_NOS[0] in set(leaky_train[ID_COL])
    no14_in_test = PAIRED_NOS[0] in set(leaky_test[ID_COL])
    no24_in_train = PAIRED_NOS[1] in set(leaky_train[ID_COL])
    no24_in_test = PAIRED_NOS[1] in set(leaky_test[ID_COL])
    # True iff at least one of the pair has rows on BOTH sides, or the two experiments
    # ended up on opposite sides -- either way a literal duplicate of a test row sits
    # in train (or vice versa) under the leaky split.
    duplicate_pair_leaks_across_leaky_split = bool(
        (no14_in_train and no14_in_test)
        or (no24_in_train and no24_in_test)
        or (no14_in_train and no24_in_test)
        or (no24_in_train and no14_in_test)
    )

    summary = {
        "target": TARGET,
        "warning": (
            "'leaky' split is an INTENTIONAL row-level random split -- reproduces the "
            "guidebook leak under study, not a valid methodology for this project."
        ),
        "excluded_experiments": DUPLICATE_CONFLICT_NOS,
        "paired_experiments_both_kept": list(PAIRED_NOS),
        "duplicate_pair_rows_identical": bool(np.array_equal(a, b)) if len(a) and len(b) else None,
        "duplicate_pair_leaks_across_leaky_split": duplicate_pair_leaks_across_leaky_split,
        "n_experiments": int(table[ID_COL].nunique()),
        "n_rows_total": int(len(table)),
        "n_features": len(feature_cols),
        "row_target_counts": table[TARGET].value_counts().sort_index().to_dict(),
        "leaky_split": {
            "method": "train_test_split(shuffle=True, stratify=defect) over ALL rows, "
            "ignoring No entirely",
            "test_size": ROW_SPLIT_TEST_SIZE,
            "random_state": ROW_SPLIT_SEED,
            "train_rows": int(len(leaky_train)),
            "test_rows": int(len(leaky_test)),
            "train_target_counts": leaky_train[TARGET].value_counts().sort_index().to_dict(),
            "test_target_counts": leaky_test[TARGET].value_counts().sort_index().to_dict(),
            "experiments_appearing_in_both_train_and_test": int(
                len(set(leaky_train[ID_COL]) & set(leaky_test[ID_COL]))
            ),
        },
        "grouped_split": {
            "method": "whole-experiment split, reusing 006/007's frozen TRAIN_NOS/TEST_NOS",
            "train_No": TRAIN_NOS,
            "test_No": TEST_NOS,
            "train_rows": int(len(grouped_train)),
            "test_rows": int(len(grouped_test)),
            "train_target_counts": grouped_train[TARGET].value_counts().sort_index().to_dict(),
            "test_target_counts": grouped_test[TARGET].value_counts().sort_index().to_dict(),
        },
        "outputs": [
            str((OUT_DIR / f"features_v5_{name}.csv").relative_to(ROOT)).replace("\\", "/")
            for name in ("leaky_train", "leaky_test", "grouped_train", "grouped_test")
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
