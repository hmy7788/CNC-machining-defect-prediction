"""Train and evaluate the v2 windowed tool-wear classifiers (experiment 005).

Input is the *window-level* feature table from experiment 004
(`data/processed/features_v2_{train,test}.csv`, 201 + 52 rows, 183 features).
Each row is a 100-row time window of one machining experiment; the experiment id
`No` is the group column.

Design constraints (rationale in docs/experiments/004-preprocess-windowed-features.md
"반드시 지킬 것 -- 그룹 누수 방지", and 005-model-windowed.md):
  * Windows from the same experiment are slices of ONE time series and share the
    same label / feedrate / clamp_pressure. Any row-wise shuffled split
    (KFold, StratifiedKFold, train_test_split) is immediate leakage. CV is
    therefore `LeaveOneGroupOut(groups=No)` -- 16 training experiments = 16 folds,
    the direct counterpart of the Leave-One-Out over experiments used in 003.
  * Metrics are reported at BOTH levels: window level (pooled held-out window
    predictions) and experiment level (windows of a held-out experiment
    aggregated into one prediction). The experiment-level figure is the one
    comparable to 003. A window score far above the experiment score means the
    model memorised "which experiment is this", not tool wear.
  * The permutation check shuffles labels AT THE EXPERIMENT LEVEL, so every
    window of an experiment keeps a single (shuffled) label. Shuffling per window
    would destroy the null distribution itself.
  * Same two fixed, strongly-regularised models as 003, no hyperparameter search:
    the number of independent groups is still 16, so tuning would just be
    selection bias on a 16-group CV score.
  * Scaling happens inside the pipeline, so each fold refits its scaler on 15
    experiments only. 004 deliberately shipped no global scaler file.
  * The 52 test windows (5 experiments) are touched exactly ONCE, at the end, as a
    hold-out sanity check -- not as "final performance".

Reads `data/processed/` read-only, writes `models/`. Prints a JSON report to stdout.

Run: python src/models/train_windowed.py
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
TRAIN_CSV = PROCESSED_DIR / "features_v2_train.csv"
TEST_CSV = PROCESSED_DIR / "features_v2_test.csv"
MODELS_DIR = ROOT / "models"

TARGET = "tool_condition_worn"
GROUP = "No"
# Never features: the group id, the within-experiment window index, the target
# itself, and two post-hoc label-ish columns only knowable after the cut finishes.
NON_FEATURE_COLS = [
    "No",
    "window_idx",
    "tool_condition_worn",
    "passed_visual_inspection_yes",
    "machining_finalized_yes",
]

# Fixed, strongly-regularised hyperparameters -- identical to 003. NOT tuned.
LOGREG_C = 0.1
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 3
RF_SEEDS = [0, 1, 2, 3, 4]
PERMUTATION_SEEDS = list(range(10))
TOP_FEATURES = 10


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Return (X, y, groups, feature_names) for one split, with the leakage assert."""
    df = pd.read_csv(path)
    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    for banned in NON_FEATURE_COLS:
        assert banned not in features, f"leakage: {banned!r} ended up in the feature list"
    assert not df[features].isna().any().any(), f"{path.name}: NaN in features"
    return df[features], df[TARGET].astype(int), df[GROUP].astype(int), features


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


def aggregate_to_experiments(
    groups: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """Collapse window-level predictions into one prediction per experiment.

    Two aggregations, both required by 004:
      * mean_prob   -- mean predicted probability over the experiment's windows,
                       thresholded at 0.5 (the probability doubles as the ROC score)
      * majority_vote -- majority of the window hard predictions (ties -> 1, which
                       matches `mean >= 0.5`); the fraction voting 1 is the ROC score
    A long experiment (No=11 alone has 23/201 windows) dominates the window-level
    figure; this aggregation gives every experiment exactly one vote.
    """
    frame = pd.DataFrame(
        {"group": groups.to_numpy(), "y": y_true.to_numpy(), "pred": y_pred, "score": y_score}
    )
    per_exp = frame.groupby("group").agg(
        y=("y", "first"),
        n_labels=("y", "nunique"),
        mean_prob=("score", "mean"),
        vote_frac=("pred", "mean"),
        n_windows=("y", "size"),
    )
    assert (per_exp["n_labels"] == 1).all(), "an experiment carries more than one label"
    y_exp = per_exp["y"].to_numpy()
    return {
        "n_experiments": int(len(per_exp)),
        "mean_prob": pooled_metrics(
            y_exp,
            (per_exp["mean_prob"].to_numpy() >= 0.5).astype(int),
            per_exp["mean_prob"].to_numpy(),
        ),
        "majority_vote": pooled_metrics(
            y_exp,
            (per_exp["vote_frac"].to_numpy() >= 0.5).astype(int),
            per_exp["vote_frac"].to_numpy(),
        ),
    }


def logo_pooled(
    model_factory, x: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> tuple[dict, dict]:
    """LeaveOneGroupOut over experiments; returns (window-level, experiment-level).

    Every fold holds out one whole experiment (4-23 windows), so nothing from the
    held-out experiment -- not even its scaler statistics -- reaches the fit.
    Window-level metrics are computed on the pooled held-out predictions, as in
    003's `loo_pooled`; the experiment-level dict aggregates those same
    predictions per experiment.
    """
    logo = LeaveOneGroupOut()
    preds = np.empty(len(y), dtype=int)
    scores = np.empty(len(y), dtype=float)
    for train_idx, test_idx in logo.split(x, y, groups):
        model = model_factory()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = model.predict(x.iloc[test_idx])
        scores[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]
    window = pooled_metrics(y.to_numpy(), preds, scores)
    experiment = aggregate_to_experiments(groups, y, preds, scores)
    return window, experiment


def mean_std(runs: list[dict]) -> dict:
    """Collapse several runs of the same (flat) metric dict into mean/std per metric."""
    out = {}
    for key in runs[0]:
        values = [r[key] for r in runs]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    return out


def holdout_eval(model, x_test: pd.DataFrame, y_test: pd.Series, groups: pd.Series) -> dict:
    pred = model.predict(x_test)
    score = model.predict_proba(x_test)[:, 1]
    result = {
        "n_windows": int(len(y_test)),
        "window": pooled_metrics(y_test.to_numpy(), pred, score),
        "experiment": aggregate_to_experiments(groups, y_test, pred, score),
    }
    per_exp = pd.DataFrame({"group": groups.to_numpy(), "y": y_test.to_numpy(), "score": score})
    agg = per_exp.groupby("group").agg(y=("y", "first"), mean_prob=("score", "mean"))
    result["per_experiment"] = {
        str(int(g)): {"y_true": int(r.y), "mean_prob": float(r.mean_prob)}
        for g, r in agg.iterrows()
    }
    return result


def shuffle_labels_by_group(y: pd.Series, groups: pd.Series, seed: int) -> pd.Series:
    """Permute labels ACROSS experiments, keeping one label per experiment.

    Shuffling window-wise would break the group structure and hand the null model
    an easier problem than the real one, making the null distribution meaningless.
    """
    rng = np.random.default_rng(seed)
    exp_labels = y.groupby(groups.to_numpy()).first()
    shuffled = pd.Series(rng.permutation(exp_labels.to_numpy()), index=exp_labels.index)
    return pd.Series(groups.map(shuffled).to_numpy(), index=y.index)


def permutation_check(x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    """Group-preserving label permutation: scores must collapse to chance level.

    Column-name asserts cannot catch a feature that accidentally encodes the
    label; a permuted-label run that still scores high would.
    """
    window_acc, exp_bal_acc, exp_acc = [], [], []
    for seed in PERMUTATION_SEEDS:
        y_shuffled = shuffle_labels_by_group(y, groups, seed)
        window, experiment = logo_pooled(make_logreg, x, y_shuffled, groups)
        window_acc.append(window["accuracy"])
        exp_bal_acc.append(experiment["mean_prob"]["balanced_accuracy"])
        exp_acc.append(experiment["mean_prob"]["accuracy"])

    def summarise(values: list[float]) -> dict:
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "max": float(np.max(values)),
        }

    return {
        "n_permutations": len(PERMUTATION_SEEDS),
        "shuffle_unit": "experiment (No) -- all windows of an experiment share one shuffled label",
        "window_accuracy": summarise(window_acc),
        "experiment_accuracy": summarise(exp_acc),
        "experiment_balanced_accuracy": summarise(exp_bal_acc),
    }


def top_feature_importances(rf: RandomForestClassifier, features: list[str]) -> list[dict]:
    """Top RF importances -- 004 asked to check whether progress-position features
    (`M_sequence_number_*`, `proc_frac_*`) dominate, which would mean the model
    learned "where in the cut am I", not "is the tool worn"."""
    order = np.argsort(rf.feature_importances_)[::-1][:TOP_FEATURES]
    return [
        {"feature": features[i], "importance": float(rf.feature_importances_[i])} for i in order
    ]


def main() -> int:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        print(
            "data/processed/features_v2_*.csv 없음 -- 먼저 "
            "python src/features/build_features_v2_windowed.py 를 실행하세요.",
            file=sys.stderr,
        )
        return 1

    x_train, y_train, g_train, features = load_split(TRAIN_CSV)
    x_test, y_test, g_test, test_features = load_split(TEST_CSV)
    assert features == test_features, "train/test feature columns differ"
    assert not set(g_train) & set(g_test), "an experiment appears in both train and test"

    leak_check = {
        "n_features": len(features),
        "excluded_columns": NON_FEATURE_COLS,
        "banned_columns_in_features": [c for c in NON_FEATURE_COLS if c in features],
        "train_test_group_overlap": sorted(set(g_train) & set(g_test)),
        "assert_passed": True,
    }

    logreg_window, logreg_exp = logo_pooled(make_logreg, x_train, y_train, g_train)
    rf_runs = [
        logo_pooled(lambda s=seed: make_rf(s), x_train, y_train, g_train) for seed in RF_SEEDS
    ]
    rf_window_runs = [w for w, _ in rf_runs]
    rf_exp_mean_prob_runs = [e["mean_prob"] for _, e in rf_runs]
    rf_exp_majority_runs = [e["majority_vote"] for _, e in rf_runs]

    logreg = make_logreg().fit(x_train, y_train)
    rf = make_rf(RF_SEEDS[0]).fit(x_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logreg_path = MODELS_DIR / "windowed_logreg_v2.pkl"
    rf_path = MODELS_DIR / "windowed_rf_v2.pkl"
    joblib.dump(logreg, logreg_path)
    joblib.dump(rf, rf_path)

    report = {
        "experiment": "005",
        "data": {
            "train_windows": int(len(x_train)),
            "train_experiments": int(g_train.nunique()),
            "test_windows": int(len(x_test)),
            "test_experiments": int(g_test.nunique()),
            "n_features": len(features),
            "train_window_label_counts": y_train.value_counts().sort_index().to_dict(),
            "train_experiment_label_counts": (
                y_train.groupby(g_train.to_numpy()).first().value_counts().sort_index().to_dict()
            ),
            "test_window_label_counts": y_test.value_counts().sort_index().to_dict(),
            "windows_per_experiment": g_train.value_counts().sort_index().to_dict(),
        },
        "cv": "LeaveOneGroupOut(groups=No) over the 16 training experiments (16 folds)",
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
            "logo_window": logreg_window,
            "logo_experiment": logreg_exp,
            "holdout": holdout_eval(logreg, x_test, y_test, g_test),
        },
        "random_forest": {
            "params": {
                "n_estimators": RF_N_ESTIMATORS,
                "max_depth": RF_MAX_DEPTH,
                "class_weight": "balanced",
            },
            "seeds": RF_SEEDS,
            "logo_window_per_seed": rf_window_runs,
            "logo_window": mean_std(rf_window_runs),
            "logo_experiment_mean_prob": mean_std(rf_exp_mean_prob_runs),
            "logo_experiment_majority_vote": mean_std(rf_exp_majority_runs),
            "holdout": holdout_eval(rf, x_test, y_test, g_test),
            "holdout_seed": RF_SEEDS[0],
            "top_feature_importances": top_feature_importances(rf, features),
        },
        "permutation_check_logreg": permutation_check(x_train, y_train, g_train),
        "artifacts": {
            "logreg": str(logreg_path.relative_to(ROOT)).replace("\\", "/"),
            "random_forest": str(rf_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "holdout_note": (
            "5 experiments / 52 windows, looked at exactly once -- sanity check, "
            "NOT a final performance figure."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
