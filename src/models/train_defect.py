"""Train and evaluate the v3 defect classifiers (experiment 007).

Input is the experiment-level feature table from experiment 006
(`data/processed/features_v3_{train,test}.csv`, 18 + 5 rows, 181 features).
One row = one machining experiment.

**The target here is `defect` (1 = scrap, 0 = good part), NOT `tool_condition_worn`.**
006 established from the official KAMP guidebook that `defect` is the task the
dataset actually defines. The two labels agree on only 14/23 experiments, so
003/005 scores (tool_condition target) are NOT a baseline for this experiment.

Design constraints (rationale in docs/experiments/007-model-defect.md):
  * 14/24 PAIRING. `experiment_14.csv` and `experiment_24.csv` are byte-identical
    files, so their 181 feature values are identical too, and 006 deliberately put
    both in train. Under a plain Leave-One-Out that gives two folds where the
    held-out row is a verbatim copy of a training row -- guaranteed correct, and
    optimistic. CV therefore uses `LeaveOneGroupOut` over `group = No` with
    `24 -> 14` remapped, i.e. 17 effective groups over 18 rows: the pair is always
    trained together or held out together. The naive 18-fold LOO is ALSO computed,
    but only as evidence for how much that pairing was worth -- never as the
    reported figure.
  * p >> n (181 features, 18 training experiments / 17 groups), so hyperparameters
    are FIXED at strongly-regularised values -- no grid search, same reasoning as
    003: any CV score used to pick a setting stops being an honest estimate.
  * scaling happens inside the pipeline, so each fold refits its scaler on the
    training rows of that fold only. 006 deliberately shipped no global scaler.
  * a fold holds out 1-2 rows, so per-fold F1/AUC are degenerate. Predictions are
    *pooled* across folds and scored once, as in 003.
  * the permutation check shuffles labels ACROSS GROUPS (the 14/24 pair keeps a
    single shuffled label), so the null distribution faces the same problem shape.
  * the 5 test experiments are touched exactly ONCE, at the end, as a hold-out
    sanity check -- not as "final performance".

Reads `data/processed/` read-only, writes `models/`. Prints a JSON report to stdout.

Run: python src/models/train_defect.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_CSV = PROCESSED_DIR / "features_v3_train.csv"
TEST_CSV = PROCESSED_DIR / "features_v3_test.csv"
MODELS_DIR = ROOT / "models"

TARGET = "defect"
# Never features: the experiment id, the target itself, and the OLD target kept in
# the file as reference only. `tool_condition_worn` agrees with `defect` on 61% of
# experiments, so letting it slip in would quietly inflate every number.
NON_FEATURE_COLS = ["No", "defect", "tool_condition_worn"]

# The duplicate experiment pair: identical telemetry, therefore identical features.
# Both live in train (006 section 3). Mapping the second onto the first makes them
# one CV group.
DUPLICATE_PAIR = (14, 24)

# Run-progress features: how far the cut got before the log ended.
#
# These are NOT ordinary sensor statistics. `defect = NOT(finalized AND passed)`, and
# an unfinalised run is one that STOPPED EARLY -- so it has fewer rows, a lower final
# NC sequence number, and never reaches the late Machining_Process stages. In train,
# the 5 unfinalised runs have 462-605 rows against 565-2332 for the finalised ones, so
# `n_rows` on its own separates 16/18 experiments. That is not a model predicting an
# outcome; it is a feature recording one. Nothing here reads a label column -- 006
# deliberately did not ship `machining_finalized` -- but these columns are a proxy for
# half the target's definition, which has the same effect on the score.
#
# The full-feature run is still reported (it is the requested configuration), and an
# ablation without this block is reported beside it as the honest estimate.
PROGRESS_PREFIXES = ("M_sequence_number", "proc_frac_")
PROGRESS_EXACT = ("n_rows",)

# Fixed, strongly-regularised hyperparameters -- identical to 003/005. NOT tuned.
LOGREG_C = 0.1
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 3
RF_SEEDS = [0, 1, 2, 3, 4]
PERMUTATION_SEEDS = list(range(10))
TOP_FEATURES = 10


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Return (X, y, No, feature_names) for one split, with the leakage assert applied."""
    df = pd.read_csv(path)
    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    for banned in NON_FEATURE_COLS:
        assert banned not in features, f"leakage: {banned!r} ended up in the feature list"
    assert not df[features].isna().any().any(), f"{path.name}: NaN in features"
    return df[features], df[TARGET].astype(int), df["No"].astype(int), features


def make_groups(no: pd.Series) -> pd.Series:
    """CV group id: the experiment number, with the duplicate pair collapsed.

    `24 -> 14`, so the two byte-identical experiments share one group and can never
    end up on opposite sides of a fold.
    """
    keep, drop = DUPLICATE_PAIR
    return no.replace({drop: keep})


def progress_features(features: list[str]) -> list[str]:
    """The run-progress proxy block -- see PROGRESS_PREFIXES for why it is singled out."""
    return [
        c
        for c in features
        if c in PROGRESS_EXACT or any(c.startswith(p) for p in PROGRESS_PREFIXES)
    ]


def make_logreg() -> Pipeline:
    """Deterministic L2 logistic regression; scaler is refit inside every CV fold."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=LOGREG_C,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def make_rf(seed: int) -> RandomForestClassifier:
    """Shallow random forest. Trees are scale-invariant, so no scaler here."""
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def logo_pooled(model_factory, x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    """LeaveOneGroupOut over experiment groups; metrics on the pooled predictions.

    A fold holds out one group (1 row, or 2 rows for the 14/24 pair), so per-fold
    F1/AUC are undefined. Pooling every held-out prediction and scoring once is the
    standard way to get a usable estimate at this sample size (same as 003).
    """
    logo = LeaveOneGroupOut()
    preds = np.empty(len(y), dtype=int)
    scores = np.empty(len(y), dtype=float)
    for train_idx, test_idx in logo.split(x, y, groups):
        model = model_factory()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = model.predict(x.iloc[test_idx])
        scores[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]
    return pooled_metrics(y.to_numpy(), preds, scores)


def mean_std(runs: list[dict]) -> dict:
    """Collapse several runs of the same metric dict into mean/std per metric."""
    out = {}
    for key in runs[0]:
        values = [r[key] for r in runs]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    return out


def holdout_eval(model, x_test: pd.DataFrame, y_test: pd.Series, no_test: pd.Series) -> dict:
    pred = model.predict(x_test)
    score = model.predict_proba(x_test)[:, 1]
    result = pooled_metrics(y_test.to_numpy(), pred, score)
    result["n"] = int(len(y_test))
    result["experiments"] = [int(v) for v in no_test]
    result["y_true"] = [int(v) for v in y_test]
    result["y_pred"] = [int(v) for v in pred]
    result["y_score"] = [round(float(v), 4) for v in score]
    return result


def shuffle_labels_by_group(y: pd.Series, groups: pd.Series, seed: int) -> pd.Series:
    """Permute labels ACROSS groups, keeping one label per group.

    The 14/24 pair therefore always receives the same shuffled label, exactly as it
    always carries the same real label -- the null model faces the same problem
    shape as the real one, which is what makes the comparison meaningful.
    """
    rng = np.random.default_rng(seed)
    group_labels = y.groupby(groups.to_numpy()).first()
    shuffled = pd.Series(rng.permutation(group_labels.to_numpy()), index=group_labels.index)
    return pd.Series(groups.map(shuffled).to_numpy(), index=y.index)


def permutation_check(model_factory, x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    """Group-preserving label permutation: scores must collapse to chance level.

    Column-name asserts cannot catch a feature that accidentally encodes the label;
    a permuted-label run that still scored high would. This also supplies the null
    distribution the real score has to beat -- 006 requires a fresh one, because the
    label composition (10/8 on `defect`) differs from 003's.

    Run once per model family: 003/005 only ever built a logistic null, but a null has
    to come from the same estimator as the score it judges, and here it is the forest
    that scores highest.
    """
    runs = [
        logo_pooled(model_factory, x, shuffle_labels_by_group(y, groups, seed), groups)
        for seed in PERMUTATION_SEEDS
    ]

    def summarise(key: str) -> dict:
        values = [r[key] for r in runs]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "max": float(np.max(values)),
        }

    return {
        "n_permutations": len(PERMUTATION_SEEDS),
        "shuffle_unit": "CV group (No with 24 -> 14) -- the duplicate pair shares one label",
        "accuracy": summarise("accuracy"),
        "balanced_accuracy": summarise("balanced_accuracy"),
        "roc_auc": summarise("roc_auc"),
    }


def top_feature_importances(rf: RandomForestClassifier, features: list[str]) -> list[dict]:
    """Top RF importances -- a quick look at what the forest leaned on."""
    order = np.argsort(rf.feature_importances_)[::-1][:TOP_FEATURES]
    return [
        {"feature": features[i], "importance": float(rf.feature_importances_[i])} for i in order
    ]


def main() -> int:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        print(
            "data/processed/features_v3_*.csv 없음 -- 먼저 "
            "python src/features/build_features_v3_defect.py 를 실행하세요.",
            file=sys.stderr,
        )
        return 1

    x_train, y_train, no_train, features = load_split(TRAIN_CSV)
    x_test, y_test, no_test, test_features = load_split(TEST_CSV)
    assert features == test_features, "train/test feature columns differ"
    assert not set(no_train) & set(no_test), "an experiment appears in both train and test"

    keep, drop = DUPLICATE_PAIR
    assert keep in set(no_train) and drop in set(no_train), (
        f"006 puts both {keep} and {drop} in train; the pairing logic assumes that"
    )
    # The pairing only matters because the rows really are identical. Verify, don't assume.
    dup_rows = x_train[no_train.isin(DUPLICATE_PAIR)].to_numpy()
    duplicate_rows_identical = bool(np.array_equal(dup_rows[0], dup_rows[1]))

    groups = make_groups(no_train)
    # `groups` is what makes 14/24 one fold. If the remap silently failed we would be
    # back to the optimistic 18-fold CV without noticing.
    assert groups.nunique() == no_train.nunique() - 1, "24 -> 14 group remap did not take effect"
    assert drop not in set(groups), "the duplicate experiment still has its own group"

    leak_check = {
        "n_features": len(features),
        "excluded_columns": NON_FEATURE_COLS,
        "banned_columns_in_features": [c for c in NON_FEATURE_COLS if c in features],
        "train_test_experiment_overlap": sorted(set(no_train) & set(no_test)),
        "duplicate_pair": list(DUPLICATE_PAIR),
        "duplicate_rows_identical": duplicate_rows_identical,
        "duplicate_pair_same_label": bool(
            y_train[no_train == keep].iloc[0] == y_train[no_train == drop].iloc[0]
        ),
        "n_cv_groups": int(groups.nunique()),
        "assert_passed": True,
    }

    # Reported CV: 17 groups, the pair merged.
    logreg_logo = logo_pooled(make_logreg, x_train, y_train, groups)
    rf_logo_runs = [
        logo_pooled(lambda s=seed: make_rf(s), x_train, y_train, groups) for seed in RF_SEEDS
    ]

    # Diagnostic only: the naive 18-fold LOO that treats 24 as an independent sample.
    # Reported side by side so the size of the pairing effect is on the record --
    # this is NOT the headline number.
    naive_groups = no_train
    logreg_naive = logo_pooled(make_logreg, x_train, y_train, naive_groups)
    rf_naive_runs = [
        logo_pooled(lambda s=seed: make_rf(s), x_train, y_train, naive_groups) for seed in RF_SEEDS
    ]

    # ---- Ablation: drop the run-progress proxy block -------------------------------
    # `n_rows` and friends encode "did the cut finish", which is half of the target's
    # definition (see PROGRESS_PREFIXES). This ablation is what the sensor signal is
    # worth once that shortcut is gone, and a single-feature `n_rows` model measures
    # how much of the full-feature score the shortcut explains on its own.
    prog = progress_features(features)
    sensor_only = [c for c in features if c not in prog]
    x_sensor = x_train[sensor_only]
    logreg_sensor = logo_pooled(make_logreg, x_sensor, y_train, groups)
    rf_sensor_runs = [
        logo_pooled(lambda s=seed: make_rf(s), x_sensor, y_train, groups) for seed in RF_SEEDS
    ]
    x_nrows = x_train[["n_rows"]]
    logreg_nrows = logo_pooled(make_logreg, x_nrows, y_train, groups)
    rf_nrows = mean_std(
        [logo_pooled(lambda s=seed: make_rf(s), x_nrows, y_train, groups) for seed in RF_SEEDS]
    )

    logreg = make_logreg().fit(x_train, y_train)
    rf = make_rf(RF_SEEDS[0]).fit(x_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logreg_path = MODELS_DIR / "defect_logreg_v3.pkl"
    rf_path = MODELS_DIR / "defect_rf_v3.pkl"
    joblib.dump(logreg, logreg_path)
    joblib.dump(rf, rf_path)

    report = {
        "experiment": "007",
        "target": TARGET,
        "target_note": (
            "defect (1 = scrap) -- a DIFFERENT problem from the tool_condition target "
            "of 003/005; those scores are not a baseline here."
        ),
        "data": {
            "train_rows": int(len(x_train)),
            "train_cv_groups": int(groups.nunique()),
            "test_rows": int(len(x_test)),
            "n_features": len(features),
            "train_label_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
            "test_label_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
            "train_experiments": [int(v) for v in sorted(no_train)],
            "test_experiments": [int(v) for v in sorted(no_test)],
        },
        "cv": (
            "LeaveOneGroupOut(groups=No with 24 -> 14) over the 18 training rows = "
            "17 folds; metrics pooled over the 18 held-out predictions"
        ),
        "leak_check": leak_check,
        "logreg_l2": {
            "params": {
                "penalty": "l2",
                "C": LOGREG_C,
                "class_weight": "balanced",
                "max_iter": 1000,
                "solver": "lbfgs",
                "scaler": "StandardScaler refit inside each CV fold",
            },
            "seeds": "deterministic -- no seed repetition (see docstring)",
            "logo": logreg_logo,
            "logo_naive_18fold": logreg_naive,
            "holdout": holdout_eval(logreg, x_test, y_test, no_test),
        },
        "random_forest": {
            "params": {
                "n_estimators": RF_N_ESTIMATORS,
                "max_depth": RF_MAX_DEPTH,
                "class_weight": "balanced",
            },
            "seeds": RF_SEEDS,
            "logo_per_seed": rf_logo_runs,
            "logo": mean_std(rf_logo_runs),
            "logo_naive_18fold": mean_std(rf_naive_runs),
            "holdout": holdout_eval(rf, x_test, y_test, no_test),
            "holdout_seed": RF_SEEDS[0],
            "top_feature_importances": top_feature_importances(rf, features),
        },
        "pairing_effect": {
            "note": (
                "17-group CV (reported) minus naive 18-fold LOO (diagnostic). A positive "
                "naive-minus-merged gap is the optimism the 14/24 pairing removes."
            ),
            "logreg_balanced_accuracy_merged": logreg_logo["balanced_accuracy"],
            "logreg_balanced_accuracy_naive": logreg_naive["balanced_accuracy"],
            "rf_balanced_accuracy_merged_mean": mean_std(rf_logo_runs)["balanced_accuracy_mean"],
            "rf_balanced_accuracy_naive_mean": mean_std(rf_naive_runs)["balanced_accuracy_mean"],
        },
        "progress_feature_ablation": {
            "note": (
                "The run-progress block encodes how far the cut got, and an unfinalised "
                "run (= defective by definition) stops early. Dropping it is the honest "
                "estimate of what the SENSOR signal is worth."
            ),
            "progress_features": prog,
            "n_progress_features": len(prog),
            "n_features_after_drop": len(sensor_only),
            "logreg_sensor_only": logreg_sensor,
            "rf_sensor_only": mean_std(rf_sensor_runs),
            "logreg_n_rows_only": logreg_nrows,
            "rf_n_rows_only": rf_nrows,
            "permutation_null_logreg_sensor_only": permutation_check(
                make_logreg, x_sensor, y_train, groups
            ),
        },
        "permutation_check_logreg": permutation_check(make_logreg, x_train, y_train, groups),
        "permutation_check_rf": permutation_check(
            lambda: make_rf(RF_SEEDS[0]), x_train, y_train, groups
        ),
        "artifacts": {
            "logreg": str(logreg_path.relative_to(ROOT)).replace("\\", "/"),
            "random_forest": str(rf_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "holdout_note": (
            "5 experiments, looked at exactly once -- sanity check, NOT a final "
            "performance figure. One experiment moves accuracy by 20 percentage points."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
