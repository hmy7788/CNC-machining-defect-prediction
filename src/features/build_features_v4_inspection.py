"""Build the v4 feature table (experiment 008) -- visual-inspection target on finished runs.

Same aggregation as v1/v3 (one row per machining experiment: every sensor channel
collapsed into mean/std/min/max over the whole run, plus the time share spent in
each `Machining_Process` stage). What changes versus v3/006 is the POPULATION and
the TARGET, and the reason is docs/failures/004:

  006/007 used `defect = NOT(machining_finalized AND passed_visual_inspection)`.
  Half of that definition is "the cut never finished", and an unfinished cut leaves
  a physical trace in the telemetry: it is short. `n_rows` alone reached the same
  balanced accuracy (0.8375) as the best 181-feature model, i.e. the model was
  reading "this run stopped early", which is a recording of the outcome, not a
  prediction of it. Dropping the progress columns did not help -- the same
  information is smeared across the ordinary sensor statistics.

v4 removes the shortcut at its source instead of removing columns: keep ONLY the
runs that went all the way (`machining_finalized == "yes"`) and predict
`inspection_fail` inside that subset. Every remaining experiment finished, so
"how far did this run get" is no longer entangled with the label and the progress
features (`n_rows`, `M_sequence_number_*`, `proc_frac_*`) are ordinary features
again -- they are deliberately NOT dropped here (this is the difference versus the
ablation arm of 007).

Key decisions (rationale in docs/experiments/008-preprocess-inspection-target.md):
  * population: machining_finalized == "yes" (19 experiments), minus the pair
    (19, 25) -- byte-identical files whose `passed_visual_inspection` disagrees
    (19 -> no, 25 -> yes), see docs/failures/002. 19 -> 17 experiments.
  * target `inspection_fail` = 1 when passed_visual_inspection == "no", 0 when "yes".
  * labels come from `train.csv`, never from the integrated dataset's `Y_*`
    (docs/decisions/001-data-source-strategy.md).
  * (14, 24) are byte-identical too but agree here (both "yes"), so they are kept
    and pinned to the SAME split, see PAIRED_NOS below.
  * the split is stratified over *experiments* (`No`), never over rows.

Reads `data/raw/` read-only, writes `data/processed/features_v4_*`.

Run: python src/features/build_features_v4_inspection.py
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
# sklearn version can never silently reshuffle the split (same reasoning as 004/006).
RANDOM_SEED = 42
TEST_SIZE = 4  # 4 of 17 experiments ~= 24%, in line with 006's 5 of 23 ~= 22%

# Only runs that finished are in scope -- that is the whole point of v4 (see module
# docstring and docs/failures/004). This drops No = 4, 5, 7, 16, 20, 23 (6 runs).
FINALIZED_VALUE = "yes"

# (19, 25) are byte-identical telemetry files whose `passed_visual_inspection`
# disagrees (19: no, 25: yes), so under THIS target they are a genuine label
# conflict, exactly as they were under `defect` in 006. Neither label can be
# verified from the data, so the pair is excluded. See
# docs/failures/002-duplicate-experiment-conflicting-labels.md. N: 19 -> 17.
DUPLICATE_CONFLICT_NOS = [19, 25]

# (14, 24) are also byte-identical, but they agree under this target (both "yes").
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
TARGET = "inspection_fail"  # 1 = failed visual inspection, 0 = passed
AUX_LABEL = "tool_condition_worn"  # 002/003/005's target, kept for reference only
# Machining parameters that are known before the cut starts -- legitimate features.
# `material` is excluded: constant (aluminum) across all 25 experiments.
SETTING_COLS = ["feedrate", "clamp_pressure"]
AGGS = ["mean", "std", "min", "max"]

ID_COLS = ["No"]
LABEL_COLS = [TARGET, AUX_LABEL]

# Frozen split (derived once with RANDOM_SEED, see derive_split()). PAIRED_NOS are
# pinned to train first; the remaining 15 experiments are stratified on the target.
TRAIN_NOS = [1, 2, 3, 6, 8, 10, 13, 14, 17, 18, 21, 22, 24]
TEST_NOS = [9, 11, 12, 15]


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

    # Kept on purpose in v4 (unlike the 007 ablation): every experiment here ran to
    # completion, so run length no longer encodes half of the target's definition.
    feats["n_rows"] = float(len(df))
    # Time share per machining stage (fractions, so the differing run lengths
    # do not turn this into a duplicate of n_rows).
    shares = df[PROCESS_COL].value_counts(normalize=True)
    for stage, frac in shares.items():
        feats[f"proc_frac_{stage.replace(' ', '_')}"] = float(frac)

    feats["No"] = no
    return pd.Series(feats)


def select_population(meta: pd.DataFrame, available_nos: list[int]) -> list[int]:
    """Finished runs only, minus the label-conflicting duplicate pair."""
    finalized = meta.loc[meta["machining_finalized"] == FINALIZED_VALUE, "No"].tolist()
    kept = [
        int(no)
        for no in sorted(available_nos)
        if no in set(finalized) and no not in DUPLICATE_CONFLICT_NOS
    ]
    # The whole premise of v4: no unfinished run survives the filter, so run length
    # cannot stand in for "was this cut aborted". Checked at runtime, not assumed.
    dropped = meta.loc[meta["machining_finalized"] != FINALIZED_VALUE, "No"].tolist()
    assert not set(kept) & set(dropped), "an unfinished run slipped into the population"
    assert not set(kept) & set(DUPLICATE_CONFLICT_NOS), "conflicting duplicate not excluded"
    return kept


def build_feature_table() -> pd.DataFrame:
    meta = load_metadata()
    files = experiment_files()
    kept_nos = select_population(meta, list(files))

    rows = [summarize_experiment(no, files[no]) for no in kept_nos]
    feats = pd.DataFrame(rows)

    # Stages absent from a run get share 0, not NaN.
    proc_cols = [c for c in feats.columns if c.startswith("proc_frac_")]
    feats[proc_cols] = feats[proc_cols].fillna(0.0)

    meta = meta[meta["No"].isin(kept_nos)].copy()
    # Inside this population `passed_visual_inspection` is never missing: it is only
    # blank for unfinished runs, and those are already filtered out.
    assert meta["passed_visual_inspection"].notna().all(), "unexpected missing inspection result"
    assert set(meta["passed_visual_inspection"]) <= {"yes", "no"}, "unexpected inspection value"
    meta[TARGET] = (meta["passed_visual_inspection"] == "no").astype(int)
    meta[AUX_LABEL] = (meta["tool_condition"] == "worn").astype(int)

    # `machining_finalized` / `passed_visual_inspection` are deliberately NOT stored.
    # `passed_visual_inspection` IS the target. `machining_finalized` is constant
    # ("yes") here, so it carries no information anyway -- not storing it is stronger
    # than relying on the zero-variance filter to notice. 006's `defect` is not stored
    # either: on this population defect == inspection_fail exactly (defect =
    # NOT(finalized AND passed) and finalized is always true here), so saving it would
    # be saving the target twice under a second name. Same reasoning as 006 section (6).
    table = meta[["No", *SETTING_COLS, *LABEL_COLS]].merge(feats, on="No", how="inner")
    assert len(table) == len(kept_nos), "metadata/telemetry join lost experiments"
    return table


def derive_split(table: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Reproduce the frozen split, then assert it matches TRAIN_NOS/TEST_NOS.

    The paired experiments (14, 24) are byte-identical telemetry, so they are pinned
    to the train side *before* splitting; only the remaining 15 experiments go through
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
    for split_nos in (TRAIN_NOS, TEST_NOS):
        labels = set(table.loc[table["No"].isin(split_nos), TARGET])
        assert labels == {0, 1}, "each split must contain both classes"
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
    train.to_csv(OUT_DIR / "features_v4_train.csv", index=False)
    test.to_csv(OUT_DIR / "features_v4_test.csv", index=False)

    # No global scaler file is written (same as v2/v3): 003 showed it must not be used
    # inside CV, where scaling has to be re-fit per fold. Stored CSVs are raw scale.
    n_rows_col = table[["No", TARGET, "n_rows"]]
    summary = {
        "target": TARGET,
        "target_definition": (
            "1 (fail) if passed_visual_inspection == 'no'; 0 (pass) if 'yes'. "
            "Population restricted to machining_finalized == 'yes'."
        ),
        "population_filter": f"machining_finalized == '{FINALIZED_VALUE}'",
        "excluded_unfinished_experiments": sorted(
            set(range(1, 26)) - set(table["No"]) - set(DUPLICATE_CONFLICT_NOS)
        ),
        "excluded_label_conflict_experiments": DUPLICATE_CONFLICT_NOS,
        "paired_experiments_same_split": list(PAIRED_NOS),
        "n_experiments": int(len(table)),
        "n_features": len(feature_cols),
        "n_process_share_features": len([c for c in feature_cols if c.startswith("proc_frac_")]),
        "progress_features_kept": True,
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
        # Evidence that the 007 shortcut is gone: with unfinished runs removed the
        # two label groups should no longer be separated by run length.
        "n_rows_by_target": {
            str(int(label)): {
                "count": int(len(grp)),
                "min": float(grp["n_rows"].min()),
                "max": float(grp["n_rows"].max()),
                "mean": float(grp["n_rows"].mean()),
            }
            for label, grp in n_rows_col.groupby(TARGET)
        },
        "outputs": [
            str((OUT_DIR / "features_v4_train.csv").relative_to(ROOT)),
            str((OUT_DIR / "features_v4_test.csv").relative_to(ROOT)),
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print("\nper-experiment labels:")
    print(
        table[["No", TARGET, AUX_LABEL, "n_rows"]]
        .assign(split=lambda d: np.where(d["No"].isin(train_no), "train", "test"))
        .sort_values("No")
        .to_string(index=False)
    )
    print("\nfeature columns:")
    print(json.dumps(feature_cols, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
