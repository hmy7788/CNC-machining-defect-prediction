"""Quantify the guidebook's row-level-shuffle leak (experiment 010).

**WARNING: this script trains on an INTENTIONALLY LEAKY split.** It exists to
measure how much score inflation that leak is worth, NOT to report a defect-
prediction performance number. See docs/experiments/010-leaky-official-
reproduction.md and the warning banner in
src/features/build_features_v5_leaky_rows.py before reusing anything here.

Design: the SAME row-level table, the SAME fixed-hyperparameter logreg/RF, and the
SAME evaluation code are run under two conditions that differ ONLY in how rows are
split into train/test:
  * "leaky"   -- data/processed/features_v5_leaky_{train,test}.csv. Rows are
                 shuffled at random across all 23 experiments (the guidebook's
                 method, see 010's feature-building script). The same experiment's
                 timesteps are on both sides.
  * "grouped" -- data/processed/features_v5_grouped_{train,test}.csv. Whole
                 experiments are assigned to one side (007's frozen split), so no
                 experiment straddles train/test.

Holding the model and features fixed isolates the split method as the only
variable, so the gap between the two conditions IS the leak's score inflation.

A second, sharper diagnostic runs alongside the plain scores: a group-permuted
label null under EACH split condition (each experiment gets one shuffled label,
broadcast to all its rows -- same scheme as 007/009's permutation_check). Under
the grouped split this null should collapse to chance, as in every prior
experiment. Under the leaky split, if the null ALSO scores far above chance, that
is direct proof the leaky split's score is driven by the model recognising which
EXPERIMENT a row's raw sensor values came from (row-level memorisation) rather
than any label signal -- because the labels are meaningless noise in the null, yet
the row-identity leak is still there.

Reads `data/processed/` read-only, writes nothing to `models/` (diagnostic only,
not a model meant for reuse). Prints a JSON report to stdout.

Run: python src/models/train_leaky_official.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "defect"
NON_FEATURE_COLS = ["No", TARGET]

LOGREG_C = 0.1
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 3
RF_SEEDS = [0, 1, 2, 3, 4]
PERMUTATION_SEEDS = list(range(10))


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    path = PROCESSED_DIR / f"features_v5_{name}.csv"
    df = pd.read_csv(path)
    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    for banned in NON_FEATURE_COLS:
        assert banned not in features, f"leakage: {banned!r} ended up in the feature list"
    assert not df[features].isna().any().any(), f"{path.name}: NaN in features"
    return df[features], df[TARGET].astype(int), df["No"].astype(int), features


def make_logreg() -> Pipeline:
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
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def fit_eval(model_factory, x_train, y_train, x_test, y_test) -> dict:
    model = model_factory()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = model.predict_proba(x_test)[:, 1]
    return eval_metrics(y_test.to_numpy(), pred, score)


def mean_std(runs: list[dict]) -> dict:
    out = {}
    for key in runs[0]:
        values = [r[key] for r in runs]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    return out


def shuffle_labels_by_group(y: pd.Series, groups: pd.Series, seed: int) -> pd.Series:
    """Permute labels ACROSS experiments, one shuffled label per experiment.

    Every row of an experiment gets that experiment's shuffled label -- so the
    row-identity signal (if the model is exploiting one) survives untouched, but
    the label is now meaningless noise. Identical scheme to 007/009.
    """
    rng = np.random.default_rng(seed)
    group_labels = y.groupby(groups.to_numpy()).first()
    shuffled = pd.Series(rng.permutation(group_labels.to_numpy()), index=group_labels.index)
    return pd.Series(groups.map(shuffled).to_numpy(), index=y.index)


def permutation_check(model_factory, x_train, y_train, no_train, x_test, y_test, no_test) -> dict:
    runs = []
    for seed in PERMUTATION_SEEDS:
        y_train_perm = shuffle_labels_by_group(y_train, no_train, seed)
        y_test_perm = shuffle_labels_by_group(y_test, no_test, seed)
        runs.append(fit_eval(model_factory, x_train, y_train_perm, x_test, y_test_perm))

    def summarise(key: str) -> dict:
        values = [r[key] for r in runs]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "max": float(np.max(values)),
        }

    return {
        "n_permutations": len(PERMUTATION_SEEDS),
        "shuffle_unit": "experiment (No) -- every row of an experiment gets one shuffled label",
        "accuracy": summarise("accuracy"),
        "balanced_accuracy": summarise("balanced_accuracy"),
        "roc_auc": summarise("roc_auc"),
    }


def run_condition(name: str) -> dict:
    x_train, y_train, no_train, features = load_split(f"{name}_train")
    x_test, y_test, no_test, test_features = load_split(f"{name}_test")
    assert features == test_features, f"{name}: train/test feature columns differ"

    overlap = sorted(set(no_train) & set(no_test))

    logreg_real = fit_eval(make_logreg, x_train, y_train, x_test, y_test)
    rf_real_runs = [
        fit_eval(lambda s=seed: make_rf(s), x_train, y_train, x_test, y_test) for seed in RF_SEEDS
    ]

    logreg_null = permutation_check(
        make_logreg, x_train, y_train, no_train, x_test, y_test, no_test
    )
    rf_null = permutation_check(
        lambda: make_rf(RF_SEEDS[0]), x_train, y_train, no_train, x_test, y_test, no_test
    )

    return {
        "n_features": len(features),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_experiments": int(no_train.nunique()),
        "test_experiments": int(no_test.nunique()),
        "experiments_in_both_train_and_test": overlap,
        "train_label_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "test_label_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "logreg_l2": {
            "real": logreg_real,
            "group_permuted_null": logreg_null,
        },
        "random_forest": {
            "real_per_seed": rf_real_runs,
            "real": mean_std(rf_real_runs),
            "group_permuted_null": rf_null,
        },
    }


def main() -> int:
    required = [
        PROCESSED_DIR / f"features_v5_{part}.csv"
        for part in ("leaky_train", "leaky_test", "grouped_train", "grouped_test")
    ]
    if not all(p.exists() for p in required):
        print(
            "data/processed/features_v5_*.csv 없음 -- 먼저 "
            "python src/features/build_features_v5_leaky_rows.py 를 실행하세요.",
            file=sys.stderr,
        )
        return 1

    leaky = run_condition("leaky")
    grouped = run_condition("grouped")

    report = {
        "experiment": "010",
        "warning": (
            "INTENTIONAL LEAKAGE STUDY. The 'leaky' condition reproduces the official "
            "KAMP guidebook's row-level random split on purpose, to measure its score "
            "inflation. Do NOT report 'leaky' numbers as defect-prediction performance. "
            "Only 'grouped' follows this project's validation discipline."
        ),
        "target": TARGET,
        "same_across_both_conditions": [
            "row-level feature table (features_v5, 44 sensor/process columns)",
            "defect label definition and the 23-experiment population (006/007)",
            "model family and fixed hyperparameters (logreg C=0.1, RF depth=3/200 trees)",
        ],
        "only_difference": "how rows are assigned to train vs test",
        "leaky": leaky,
        "grouped": grouped,
        "comparison": {
            "note": (
                "Both conditions' 'real' scores measured with an IDENTICAL held-out "
                "evaluation (single train/test split, no CV) so the gap isolates the "
                "split method. The group_permuted_null shows whether the leaky split's "
                "score survives even when the label is meaningless noise -- if it does, "
                "the score is coming from row-identity memorisation, not label signal."
            ),
            "logreg_real_balanced_accuracy_leaky": leaky["logreg_l2"]["real"]["balanced_accuracy"],
            "logreg_real_balanced_accuracy_grouped": grouped["logreg_l2"]["real"][
                "balanced_accuracy"
            ],
            "logreg_null_balanced_accuracy_leaky": leaky["logreg_l2"]["group_permuted_null"][
                "balanced_accuracy"
            ]["mean"],
            "logreg_null_balanced_accuracy_grouped": grouped["logreg_l2"]["group_permuted_null"][
                "balanced_accuracy"
            ]["mean"],
            "rf_real_balanced_accuracy_leaky": leaky["random_forest"]["real"][
                "balanced_accuracy_mean"
            ],
            "rf_real_balanced_accuracy_grouped": grouped["random_forest"]["real"][
                "balanced_accuracy_mean"
            ],
            "rf_null_balanced_accuracy_leaky": leaky["random_forest"]["group_permuted_null"][
                "balanced_accuracy"
            ]["mean"],
            "rf_null_balanced_accuracy_grouped": grouped["random_forest"]["group_permuted_null"][
                "balanced_accuracy"
            ]["mean"],
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
