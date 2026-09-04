"""Baseline EDA for the CNC tool-wear dataset (experiment 001).

Reads `data/raw/CNC 비식별화 원본데이터_1209/` read-only, prints a text summary and
writes figures to `reports/figures/001_*.png`.

Run: python src/visualization/eda_baseline.py

Every number in docs/experiments/001-eda-cnc-baseline.md comes from this script's
stdout -- nothing is hardcoded here.
"""

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "CNC 비식별화 원본데이터_1209"
EXP_DIR = RAW_DIR / "CNC Virtual Data set _v2"
META_CSV = RAW_DIR / "train.csv"
FIG_DIR = ROOT / "reports" / "figures"

# Outlier rule: a value more than 1.5*IQR outside the quartiles of its own column,
# pooled over all experiments. Chosen because the sensor channels are heavily
# multi-modal (idle vs. cutting), so a z-score rule would flag whole phases.
IQR_MULTIPLIER = 1.5

LABEL_COLS = ["tool_condition", "machining_finalized", "passed_visual_inspection"]


def load_metadata() -> pd.DataFrame:
    """train.csv pads values with spaces for alignment -- strip them (see glossary)."""
    meta = pd.read_csv(META_CSV, skipinitialspace=True)
    for col in meta.columns:
        if meta[col].dtype == object:
            meta[col] = meta[col].str.strip()
    return meta


def experiment_files() -> dict[int, Path]:
    files = {}
    for path in sorted(EXP_DIR.glob("experiment_*.csv")):
        match = re.search(r"experiment_(\d+)\.csv$", path.name)
        if match:
            files[int(match.group(1))] = path
    return files


def load_experiments(files: dict[int, Path]) -> pd.DataFrame:
    frames = []
    for no, path in sorted(files.items()):
        frame = pd.read_csv(path)
        frame.insert(0, "No", no)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def normalize_process(series: pd.Series) -> pd.Series:
    """'End' and 'end' are the same phase written two ways -- fold the casing."""
    return series.str.strip().str.title()


def summarize_labels(meta: pd.DataFrame) -> dict:
    out = {"n_experiments": int(len(meta))}
    for col in LABEL_COLS:
        counts = meta[col].value_counts(dropna=False)
        out[col] = {str(k): int(v) for k, v in counts.items()}
    finalized = meta[meta["machining_finalized"] == "yes"]
    out["crosstab_tool_vs_visual_finalized_only"] = pd.crosstab(
        finalized["tool_condition"], finalized["passed_visual_inspection"]
    ).to_dict()
    out["constant_metadata_cols"] = sorted(
        c for c in meta.columns if meta[c].nunique(dropna=False) <= 1
    )
    return out


def summarize_series(all_rows: pd.DataFrame, files: dict[int, Path]) -> dict:
    lengths = all_rows.groupby("No").size()
    numeric = all_rows.drop(columns=["No"]).select_dtypes("number")

    global_const = sorted(numeric.columns[numeric.nunique() <= 1])
    per_exp_const_everywhere = None
    for _, group in all_rows.groupby("No"):
        nun = group.drop(columns=["No"]).nunique()
        const = set(nun[nun <= 1].index)
        per_exp_const_everywhere = (
            const if per_exp_const_everywhere is None else per_exp_const_everywhere & const
        )

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
    outlier_share = ((numeric < lo) | (numeric > hi)).mean().sort_values(ascending=False)

    return {
        "n_files": len(files),
        "total_rows": int(len(all_rows)),
        "row_count_min": int(lengths.min()),
        "row_count_max": int(lengths.max()),
        "row_count_median": float(lengths.median()),
        "row_count_mean": float(lengths.mean()),
        "row_count_std": float(lengths.std()),
        "n_columns": int(all_rows.shape[1] - 1),
        "missing_cells": int(all_rows.isna().sum().sum()),
        "constant_cols_global": global_const,
        "constant_cols_in_every_experiment": sorted(per_exp_const_everywhere or []),
        "duplicate_row_share": float(all_rows.drop(columns=["No"]).duplicated().mean()),
        "outlier_share_top5": {k: round(float(v), 4) for k, v in outlier_share.head(5).items()},
        "outlier_share_max": float(outlier_share.max()),
        "process_labels_raw": sorted(all_rows["Machining_Process"].unique()),
        "process_counts": {
            str(k): int(v)
            for k, v in normalize_process(all_rows["Machining_Process"]).value_counts().items()
        },
        "row_counts": {int(k): int(v) for k, v in lengths.items()},
    }


def find_duplicate_experiments(files: dict[int, Path], all_rows: pd.DataFrame) -> dict:
    by_hash: dict[str, list[int]] = {}
    for no, path in sorted(files.items()):
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        by_hash.setdefault(digest, []).append(no)
    identical = [v for v in by_hash.values() if len(v) > 1]

    near = []
    frames = {no: g.drop(columns=["No"]) for no, g in all_rows.groupby("No")}
    for a, b in combinations(sorted(frames), 2):
        fa, fb = frames[a], frames[b]
        if len(fa) != len(fb):
            continue
        na = fa.select_dtypes("number").to_numpy()
        nb = fb.select_dtypes("number").to_numpy()
        match = float((na == nb).mean())
        proc_match = float(
            (
                normalize_process(fa["Machining_Process"]).to_numpy()
                == normalize_process(fb["Machining_Process"]).to_numpy()
            ).mean()
        )
        near.append(
            {"pair": [a, b], "cell_match": round(match, 3), "process_match": round(proc_match, 3)}
        )
    return {"identical_file_groups": identical, "same_length_pairs": near}


def experiment_signature(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Length-independent fingerprint per experiment: z-scored mean+std of every channel."""
    numeric_cols = [
        c
        for c in all_rows.select_dtypes("number").columns
        if c != "No" and all_rows[c].nunique() > 1
    ]
    grouped = all_rows.groupby("No")[numeric_cols]
    sig = pd.concat([grouped.mean(), grouped.std()], axis=1)
    sig = (sig - sig.mean()) / sig.std().replace(0, np.nan)
    return sig.fillna(0.0)


def plot_labels(meta: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    palette = sns.color_palette("Set2")
    for ax, col in zip(axes.flat, LABEL_COLS, strict=False):
        counts = meta[col].fillna("(blank)").value_counts()
        ax.bar(counts.index.astype(str), counts.values, color=palette[: len(counts)])
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v}\n({v / len(meta):.0%})", ha="center", va="bottom", fontsize=9)
        ax.set_title(col)
        ax.set_ylabel("experiments")
        ax.set_ylim(0, counts.max() * 1.3)

    ax = axes.flat[3]
    finalized = meta[meta["machining_finalized"] == "yes"]
    table = pd.crosstab(finalized["tool_condition"], finalized["passed_visual_inspection"])
    sns.heatmap(table, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(
        f"tool_condition x passed_visual_inspection\n(finalized=yes only, n={len(finalized)})"
    )

    fig.suptitle(f"Label distribution at experiment level (N={len(meta)} -- very small)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lengths(all_rows: pd.DataFrame, meta: pd.DataFrame, path: Path) -> None:
    lengths = all_rows.groupby("No").size().rename("rows").reset_index()
    merged = lengths.merge(meta[["No", "tool_condition", "machining_finalized"]], on="No")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    colors = {"worn": "#d95f02", "unworn": "#1b9e77"}
    ax = axes[0]
    ax.bar(
        merged["No"].astype(str),
        merged["rows"],
        color=[colors[c] for c in merged["tool_condition"]],
    )
    for _, row in merged.iterrows():
        if row["machining_finalized"] == "no":
            ax.text(str(row["No"]), row["rows"], "*", ha="center", va="bottom", fontsize=14)
    ax.axhline(
        merged["rows"].median(), ls="--", c="grey", label=f"median = {merged['rows'].median():.0f}"
    )
    ax.set_xlabel("experiment No")
    ax.set_ylabel("rows (time steps)")
    ax.set_title("Rows per experiment (* = machining_finalized == no)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()]
    ax.legend(handles + [ax.lines[0]], list(colors) + [f"median={merged['rows'].median():.0f}"])

    ax = axes[1]
    sns.boxplot(
        data=merged,
        x="tool_condition",
        y="rows",
        hue="tool_condition",
        palette=colors,
        legend=False,
        ax=ax,
    )
    sns.stripplot(data=merged, x="tool_condition", y="rows", color="black", size=4, ax=ax)
    ax.set_title("Length vs tool_condition")

    fig.suptitle(
        f"Series length: min={merged['rows'].min()}, max={merged['rows'].max()}, "
        f"total={merged['rows'].sum()} rows"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_process(all_rows: pd.DataFrame, meta: pd.DataFrame, path: Path) -> None:
    df = all_rows[["No", "Machining_Process"]].copy()
    df["phase"] = normalize_process(df["Machining_Process"])
    share = pd.crosstab(df["No"], df["phase"], normalize="index")
    order = [
        c
        for c in [
            "Prep",
            "Starting",
            "Layer 1 Down",
            "Layer 1 Up",
            "Layer 2 Down",
            "Layer 2 Up",
            "Layer 3 Down",
            "Layer 3 Up",
            "Repositioning",
            "End",
        ]
        if c in share.columns
    ]
    share = share[order]

    fig, ax = plt.subplots(figsize=(13, 6))
    bottom = np.zeros(len(share))
    cmap = sns.color_palette("tab10", len(order))
    for color, col in zip(cmap, order, strict=False):
        ax.bar(share.index.astype(str), share[col].values, bottom=bottom, label=col, color=color)
        bottom += share[col].values
    ax.set_xlabel("experiment No")
    ax.set_ylabel("share of rows")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    end_share = share["End"] if "End" in share else pd.Series(dtype=float)
    worst = end_share.idxmax() if len(end_share) else None
    ax.set_title(
        "Machining_Process composition per experiment "
        f"(idle 'End' share up to {end_share.max():.0%} in exp {worst})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _ = meta


def plot_similarity(all_rows: pd.DataFrame, meta: pd.DataFrame, dupes: dict, path: Path) -> None:
    sig = experiment_signature(all_rows)
    mat = sig.to_numpy()
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    cos = (mat / np.where(norm == 0, 1, norm)) @ (mat / np.where(norm == 0, 1, norm)).T
    labels = meta.set_index("No")["tool_condition"]
    ticks = [f"{no} ({labels[no][0]})" for no in sig.index]

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(
        pd.DataFrame(cos, index=ticks, columns=ticks),
        cmap="rocket_r",
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "cosine similarity of experiment fingerprint"},
    )
    for group in dupes["identical_file_groups"]:
        for a, b in combinations(group, 2):
            i, j = list(sig.index).index(a), list(sig.index).index(b)
            for x, y in ((i, j), (j, i)):
                ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor="lime", lw=2.5))
    ax.set_title(
        "Experiment similarity (mean+std fingerprint); green = byte-identical file pairs\n"
        "tick label: No (w=worn / u=unworn)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    if not META_CSV.exists() or not EXP_DIR.exists():
        print(f"데이터 없음: {RAW_DIR} -- KAMP에서 원본을 내려받아야 합니다.")
        return 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    files = experiment_files()
    all_rows = load_experiments(files)

    label_summary = summarize_labels(meta)
    series_summary = summarize_series(all_rows, files)
    dupes = find_duplicate_experiments(files, all_rows)

    dup_label_conflicts = []
    for group in dupes["identical_file_groups"]:
        sub = meta[meta["No"].isin(group)][["No", *LABEL_COLS]]
        conflicting = [c for c in LABEL_COLS if sub[c].nunique(dropna=False) > 1]
        dup_label_conflicts.append(
            {"group": group, "conflicting_labels": conflicting, "rows": sub.to_dict("records")}
        )

    plot_labels(meta, FIG_DIR / "001_label_distribution.png")
    plot_lengths(all_rows, meta, FIG_DIR / "001_experiment_lengths.png")
    plot_process(all_rows, meta, FIG_DIR / "001_machining_process.png")
    plot_similarity(all_rows, meta, dupes, FIG_DIR / "001_experiment_similarity.png")

    report = {
        "labels": label_summary,
        "series": series_summary,
        "duplicates": dupes,
        "duplicate_label_conflicts": dup_label_conflicts,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
