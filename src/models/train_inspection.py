"""Train and evaluate the v4 visual-inspection classifiers (experiment 009).

Input is the experiment-level feature table from experiment 008
(`data/processed/features_v4_{train,test}.csv`, 13 + 4 rows, 181 features).
One row = one machining experiment, and every experiment in this table ran to
completion (`machining_finalized == "yes"`).

**The target here is `inspection_fail` (1 = failed visual inspection, 0 = passed).**
That is NOT the `defect` target of 006/007 and NOT the `tool_condition_worn`
target of 002/003/005 -- 008 narrowed the population to finished runs precisely
so that "did the cut stop early" (half of 007's `defect` definition) can no longer
be read off the telemetry. 007's 0.8375 is therefore not a baseline here; the
baselines are (a) a fresh group-preserving label-shuffle null per model,
(b) majority-class balanced accuracy 0.5, (c) the single-feature diagnostics below.

Design constraints (rationale in docs/experiments/009-model-inspection.md):
  * 14/24 PAIRING, as in 007. `experiment_14.csv` and `experiment_24.csv` are
    byte-identical files, so their 181 feature values are identical, and 008 put
    both in train. CV therefore uses `LeaveOneGroupOut` over `group = No` with
    `24 -> 14` remapped: 12 effective groups over 13 rows, and the pair is always
    trained together or held out together.
  * p >> n (181 features, 13 training experiments / 12 groups) -- worse than 007.
    Hyperparameters are FIXED at the strongly-regularised 003/006/007 values.
    NO grid search: any CV score used to pick a setting stops being honest.
  * scaling happens inside the pipeline, so every fold refits its scaler on that
    fold's training rows only. 008 deliberately shipped no global scaler.
  * a fold holds out 1-2 rows, so per-fold F1/AUC are degenerate. Predictions are
    *pooled* across the 12 folds and scored once, as in 003/007.
  * PERMUTATIONS ARE 100 PER MODEL (007 used 10, whose floor on the empirical
    p-value is 1/11 = 0.091 -- too coarse to say anything about significance).
    Labels are shuffled ACROSS GROUPS so the 14/24 pair keeps a single label.
  * RESIDUAL-RISK DIAGNOSTICS. 008 flagged two ways this table could still score
    without any sensor signal, and both are measured here rather than assumed away:
      - `n_rows` alone (spearman -0.81 with `feedrate`; 008 got train balanced
        accuracy 0.778 out of it with a post-hoc threshold),
      - `feedrate` alone, plus a 180-feature run with `feedrate` DROPPED. In 13
        experiments a feedrate value can be a near-identifier: 12 and 15 each occur
        exactly once in train and both failed inspection, and test `No`=9 carries
        feedrate 15, a value train never sees.
  * the 4 test experiments are touched exactly ONCE, at the end. With a single
    failure among them this is a sanity check, never a performance figure.

Reads `data/processed/` read-only, writes `models/`. Prints a JSON report to stdout.

Run: python src/models/train_inspection.py
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
TRAIN_CSV = PROCESSED_DIR / "features_v4_train.csv"
TEST_CSV = PROCESSED_DIR / "features_v4_test.csv"
MODELS_DIR = ROOT / "models"

TARGET = "inspection_fail"
# Never features: the experiment id, the target itself, and the OLD target that 008
# kept in the file for reference only. `tool_condition_worn` is a different label
# (it disagrees with this target on several experiments); letting it slip into the
# feature list would quietly change what is being measured.
NON_FEATURE_COLS = ["No", "inspection_fail", "tool_condition_worn"]

# The duplicate experiment pair: identical telemetry, therefore identical features.
# 008 put both in train (section 5). Mapping the second onto the first makes them one
# CV group, so no fold ever holds out a verbatim copy of a training row.
DUPLICATE_PAIR = (14, 24)

# Diagnostic feature subsets -- 008 "남아 있는 위험" 1 and 2.
NROWS_FEATURE = "n_rows"
FEEDRATE_FEATURE = "feedrate"

# Fixed, strongly-regularised hyperparameters -- identical to 003/005/007. NOT tuned.
LOGREG_C = 0.1
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 3
RF_SEEDS = [0, 1, 2, 3, 4]
# 100 permutations: the empirical p-value floor is 1/(100+1) = 0.0099, so unlike 007
# a result CAN come out below 0.05. 13 rows makes each run cheap.
PERMUTATION_SEEDS = list(range(100))
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
    """CV group id: the experiment number, with the duplicate pair collapsed (24 -> 14)."""
    keep, drop = DUPLICATE_PAIR
    return no.replace({drop: keep})


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
    only usable estimate at this sample size (same as 003/007).
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


def both_models_on(
    x: pd.DataFrame, y: pd.Series, groups: pd.Series, label: str, columns: list[str]
) -> dict:
    """Run the same two fixed models on a feature subset -- used for every diagnostic."""
    subset = x[columns]
    return {
        "features": label,
        "n_features": len(columns),
        "logreg": logo_pooled(make_logreg, subset, y, groups),
        "rf": mean_std(
            [logo_pooled(lambda s=seed: make_rf(s), subset, y, groups) for seed in RF_SEEDS]
        ),
    }


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
    always carries the same real label -- the null model faces the same problem shape
    as the real one, which is what makes the comparison meaningful.
    """
    rng = np.random.default_rng(seed)
    group_labels = y.groupby(groups.to_numpy()).first()
    shuffled = pd.Series(rng.permutation(group_labels.to_numpy()), index=group_labels.index)
    return pd.Series(groups.map(shuffled).to_numpy(), index=y.index)


def permutation_check(
    model_factory, x: pd.DataFrame, y: pd.Series, groups: pd.Series, observed: float
) -> dict:
    """Group-preserving label permutation, 100 draws, with an empirical p-value.

    A null has to come from the same estimator as the score it judges (007's
    troubleshooting note), so this is called once per model family.

    p = (1 + #{null >= observed}) / (1 + n_permutations) -- the standard
    add-one estimator, which never returns 0 and is defined for 100 draws down
    to 0.0099.
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
            "p95": float(np.percentile(values, 95)),
        }

    null_bal = np.array([r["balanced_accuracy"] for r in runs])
    n_ge = int(np.sum(null_bal >= observed - 1e-12))
    return {
        "n_permutations": len(PERMUTATION_SEEDS),
        "shuffle_unit": "CV group (No with 24 -> 14) -- the duplicate pair shares one label",
        "observed_balanced_accuracy": float(observed),
        "n_null_ge_observed": n_ge,
        "p_value_balanced_accuracy": float((1 + n_ge) / (1 + len(PERMUTATION_SEEDS))),
        "p_value_floor": float(1 / (1 + len(PERMUTATION_SEEDS))),
        "accuracy": summarise("accuracy"),
        "balanced_accuracy": summarise("balanced_accuracy"),
        "roc_auc": summarise("roc_auc"),
    }


def top_feature_importances(rf: RandomForestClassifier, features: list[str]) -> list[dict]:
    """Top RF importances -- how much of the forest leans on n_rows / feedrate."""
    order = np.argsort(rf.feature_importances_)[::-1][:TOP_FEATURES]
    return [
        {"feature": features[i], "importance": float(rf.feature_importances_[i])} for i in order
    ]


def main() -> int:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        print(
            "data/processed/features_v4_*.csv 없음 -- 먼저 "
            "python src/features/build_features_v4_inspection.py 를 실행하세요.",
            file=sys.stderr,
        )
        return 1

    x_train, y_train, no_train, features = load_split(TRAIN_CSV)
    x_test, y_test, no_test, test_features = load_split(TEST_CSV)
    assert features == test_features, "train/test feature columns differ"
    assert not set(no_train) & set(no_test), "an experiment appears in both train and test"
    assert NROWS_FEATURE in features and FEEDRATE_FEATURE in features, (
        "the two diagnostic columns 008 asked about are missing from the feature table"
    )

    keep, drop = DUPLICATE_PAIR
    assert keep in set(no_train) and drop in set(no_train), (
        f"008 puts both {keep} and {drop} in train; the pairing logic assumes that"
    )
    # The pairing only matters because the rows really are identical. Verify, don't assume.
    dup_rows = x_train[no_train.isin(DUPLICATE_PAIR)].to_numpy()
    duplicate_rows_identical = bool(np.array_equal(dup_rows[0], dup_rows[1]))

    groups = make_groups(no_train)
    # `groups` is what makes 14/24 one fold. A silently-failed remap would put us back
    # on an optimistic 13-fold LOO without any visible symptom.
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

    # ---- Reported CV: 12 groups, the pair merged ----------------------------------
    logreg_logo = logo_pooled(make_logreg, x_train, y_train, groups)
    rf_logo_runs = [
        logo_pooled(lambda s=seed: make_rf(s), x_train, y_train, groups) for seed in RF_SEEDS
    ]
    rf_logo = mean_std(rf_logo_runs)

    # ---- Residual-risk diagnostics (008 "남아 있는 위험" 1 and 2) --------------------
    # If the 181-feature models cannot beat `n_rows` alone or `feedrate` alone, nothing
    # was learned from the sensors. If dropping `feedrate` collapses the score, the
    # models memorised a per-experiment setting rather than reading telemetry.
    without_feedrate = [c for c in features if c != FEEDRATE_FEATURE]
    diagnostics = {
        "n_rows_only": both_models_on(x_train, y_train, groups, "n_rows", [NROWS_FEATURE]),
        "feedrate_only": both_models_on(x_train, y_train, groups, "feedrate", [FEEDRATE_FEATURE]),
        "feedrate_dropped": both_models_on(
            x_train, y_train, groups, "all features except feedrate", without_feedrate
        ),
        "n_rows_and_feedrate": both_models_on(
            x_train, y_train, groups, "n_rows + feedrate", [NROWS_FEATURE, FEEDRATE_FEATURE]
        ),
    }

    train_feedrates = sorted({int(v) for v in x_train[FEEDRATE_FEATURE]})
    feedrate_counts = x_train[FEEDRATE_FEATURE].value_counts().sort_index()
    diagnostics["feedrate_as_near_identifier"] = {
        "note": (
            "008 warning 2: with 13 experiments a feedrate value can single out one "
            "experiment. Values occurring once in train are listed with their label."
        ),
        "train_feedrate_counts": {str(int(k)): int(v) for k, v in feedrate_counts.items()},
        "singleton_values": [
            {
                "feedrate": int(v),
                "No": int(no_train[x_train[FEEDRATE_FEATURE] == v].iloc[0]),
                "inspection_fail": int(y_train[x_train[FEEDRATE_FEATURE] == v].iloc[0]),
            }
            for v, c in feedrate_counts.items()
            if c == 1
        ],
        "test_feedrates_unseen_in_train": [
            {"No": int(n), "feedrate": int(v)}
            for n, v in zip(no_test, x_test[FEEDRATE_FEATURE], strict=True)
            if int(v) not in train_feedrates
        ],
    }

    # ---- Final fit on all 13 training experiments, then the hold-out, once ---------
    logreg = make_logreg().fit(x_train, y_train)
    rf = make_rf(RF_SEEDS[0]).fit(x_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logreg_path = MODELS_DIR / "inspection_logreg_v4.pkl"
    rf_path = MODELS_DIR / "inspection_rf_v4.pkl"
    joblib.dump(logreg, logreg_path)
    joblib.dump(rf, rf_path)

    report = {
        "experiment": "009",
        "target": TARGET,
        "target_note": (
            "inspection_fail (1 = failed visual inspection) on the 17 FINISHED "
            "experiments only. Different problem and different population from 007 "
            "(defect, 23 experiments) and from 003/005 (tool_condition_worn) -- "
            "none of those scores is a baseline here."
        ),
        "data": {
            "train_rows": int(len(x_train)),
            "train_cv_groups": int(groups.nunique()),
            "test_rows": int(len(x_test)),
            "n_features": len(features),
            "train_label_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
            "test_label_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
            "train_positive_rate": float(y_train.mean()),
            "train_experiments": [int(v) for v in sorted(no_train)],
            "test_experiments": [int(v) for v in sorted(no_test)],
        },
        "metric_choice": (
            "train is 9 pass / 4 fail (positive rate 0.308), so accuracy alone is not "
            "reported as the headline: a constant 'pass' predictor already scores 0.692. "
            "Primary metric is balanced accuracy (0.5 = majority-class baseline, and it "
            "keeps 003-008 comparable); F1 on the failure class and threshold-free ROC "
            "AUC are reported beside it. All are computed on the pooled 12-fold "
            "predictions because a single fold holds out 1-2 rows."
        ),
        "cv": (
            "LeaveOneGroupOut(groups=No with 24 -> 14) over the 13 training rows = "
            "12 folds; metrics pooled over the 13 held-out predictions"
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
            "logo": rf_logo,
            "holdout": holdout_eval(rf, x_test, y_test, no_test),
            "holdout_seed": RF_SEEDS[0],
            "top_feature_importances": top_feature_importances(rf, features),
        },
        "residual_risk_diagnostics": diagnostics,
        "permutation_check_logreg": permutation_check(
            make_logreg, x_train, y_train, groups, logreg_logo["balanced_accuracy"]
        ),
        "permutation_check_rf": permutation_check(
            lambda: make_rf(RF_SEEDS[0]),
            x_train,
            y_train,
            groups,
            rf_logo_runs[0]["balanced_accuracy"],
        ),
        # The n_rows-only logistic outscores every 181-feature model here, which is
        # exactly the outcome 008 warned about. A score that high needs its own null:
        # a 1-feature model has a different null distribution from a 181-feature one,
        # so the full-feature null cannot judge it.
        "permutation_check_logreg_n_rows_only": permutation_check(
            make_logreg,
            x_train[[NROWS_FEATURE]],
            y_train,
            groups,
            diagnostics["n_rows_only"]["logreg"]["balanced_accuracy"],
        ),
        "artifacts": {
            "logreg": str(logreg_path.relative_to(ROOT)).replace("\\", "/"),
            "random_forest": str(rf_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "holdout_note": (
            "4 experiments (3 pass / 1 fail), looked at exactly once -- sanity check, "
            "NOT a final performance figure. The single failing experiment sets "
            "sensitivity to either 0 or 1, and one experiment moves accuracy by 25 "
            "percentage points."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
