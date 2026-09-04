"""Train and evaluate the v1 baseline tool-wear classifiers (experiment 003).

Input is the experiment-level feature table from experiment 002
(`data/processed/features_v1_{train,test}.csv`, 16 + 5 rows, 181 features).

Design constraints (rationale in docs/experiments/003-model-baseline.md):
  * p >> n (181 features, 16 training experiments), so hyperparameters are FIXED
    at strongly-regularised values -- no grid search. Tuning on 16 samples is
    itself an overfitting path, and any CV score used to pick a setting stops
    being an honest estimate of that setting's performance.
  * primary CV is Leave-One-Out over the 16 *training* experiments only. The
    5 test experiments are touched exactly once, at the very end, as a
    hold-out sanity check -- not as "final performance".
  * scaling happens inside the CV pipeline, so each LOO fold fits its scaler on
    15 experiments only. Nothing from the held-out experiment leaks in.
  * LOO leaves a single sample per fold, so per-fold metrics are degenerate
    (accuracy is 0 or 1, F1 undefined). Predictions are therefore *pooled*
    across the 16 folds and the metrics computed once over the pooled vector.

Reads `data/processed/` read-only, writes `models/`. Prints a JSON report to stdout.

Run: python src/models/train_baseline.py
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
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_CSV = PROCESSED_DIR / "features_v1_train.csv"
TEST_CSV = PROCESSED_DIR / "features_v1_test.csv"
MODELS_DIR = ROOT / "models"

TARGET = "tool_condition_worn"
# Never features: the experiment id, the target itself, and two post-hoc label-ish
# columns that are only knowable after the cut finishes (leakage).
NON_FEATURE_COLS = [
    "No",
    "tool_condition_worn",
    "passed_visual_inspection_yes",
    "machining_finalized_yes",
]

# Fixed, strongly-regularised hyperparameters. NOT tuned -- see module docstring.
LOGREG_C = 0.1
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 3
RF_SEEDS = [0, 1, 2, 3, 4]
PERMUTATION_SEEDS = list(range(10))


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_names) for one split, with the leakage assert applied."""
    df = pd.read_csv(path)
    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    for banned in NON_FEATURE_COLS:
        assert banned not in features, f"leakage: {banned!r} ended up in the feature list"
    assert not df[features].isna().any().any(), f"{path.name}: NaN in features"
    return df[features], df[TARGET].astype(int), features


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


def loo_pooled(model_factory, x: pd.DataFrame, y: pd.Series) -> dict:
    """Leave-One-Out over experiments; metrics computed on the pooled predictions.

    A single LOO fold holds out one experiment, so its own accuracy is 0 or 1 and
    F1/AUC are undefined. Pooling the 16 held-out predictions and scoring once is
    the standard way to get a usable LOO estimate.
    """
    loo = LeaveOneOut()
    preds = np.empty(len(y), dtype=int)
    scores = np.empty(len(y), dtype=float)
    for train_idx, test_idx in loo.split(x):
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


def holdout_eval(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = model.predict(x_test)
    score = model.predict_proba(x_test)[:, 1]
    result = pooled_metrics(y_test.to_numpy(), pred, score)
    result["n"] = int(len(y_test))
    result["y_true"] = y_test.tolist()
    result["y_pred"] = [int(v) for v in pred]
    return result


def permutation_check(x: pd.DataFrame, y: pd.Series) -> dict:
    """Sanity check: with shuffled labels, LOO accuracy must collapse to ~chance.

    If a leaked column were doing the work, a permuted-label run would still score
    high. Cheap insurance on top of the column-name assert.
    """
    accs = []
    for seed in PERMUTATION_SEEDS:
        rng = np.random.default_rng(seed)
        y_shuffled = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        accs.append(loo_pooled(make_logreg, x, y_shuffled)["accuracy"])
    return {
        "n_permutations": len(accs),
        "loo_accuracy_mean": float(np.mean(accs)),
        "loo_accuracy_std": float(np.std(accs)),
        "loo_accuracy_max": float(np.max(accs)),
    }


def main() -> int:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        print(
            "data/processed/features_v1_*.csv 없음 -- 먼저 "
            "python src/features/build_features_v1.py 를 실행하세요.",
            file=sys.stderr,
        )
        return 1

    x_train, y_train, features = load_split(TRAIN_CSV)
    x_test, y_test, test_features = load_split(TEST_CSV)
    assert features == test_features, "train/test feature columns differ"

    leak_check = {
        "n_features": len(features),
        "excluded_columns": NON_FEATURE_COLS,
        "banned_columns_in_features": [c for c in NON_FEATURE_COLS if c in features],
        "assert_passed": True,
    }

    logreg_loo = loo_pooled(make_logreg, x_train, y_train)
    rf_loo_runs = [loo_pooled(lambda s=seed: make_rf(s), x_train, y_train) for seed in RF_SEEDS]

    logreg = make_logreg().fit(x_train, y_train)
    rf = make_rf(RF_SEEDS[0]).fit(x_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logreg_path = MODELS_DIR / "baseline_logreg_v1.pkl"
    rf_path = MODELS_DIR / "baseline_rf_v1.pkl"
    joblib.dump(logreg, logreg_path)
    joblib.dump(rf, rf_path)

    report = {
        "experiment": "003",
        "data": {
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "n_features": len(features),
            "train_label_counts": y_train.value_counts().sort_index().to_dict(),
            "test_label_counts": y_test.value_counts().sort_index().to_dict(),
        },
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
            "loo": logreg_loo,
            "holdout": holdout_eval(logreg, x_test, y_test),
        },
        "random_forest": {
            "params": {
                "n_estimators": RF_N_ESTIMATORS,
                "max_depth": RF_MAX_DEPTH,
                "class_weight": "balanced",
            },
            "seeds": RF_SEEDS,
            "loo_per_seed": rf_loo_runs,
            "loo": mean_std(rf_loo_runs),
            "holdout": holdout_eval(rf, x_test, y_test),
            "holdout_seed": RF_SEEDS[0],
        },
        "permutation_check_logreg": permutation_check(x_train, y_train),
        "artifacts": {
            "logreg": str(logreg_path.relative_to(ROOT)).replace("\\", "/"),
            "random_forest": str(rf_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
