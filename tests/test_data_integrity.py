"""Data-integrity regression tests.

These encode facts already verified by hand during initial data review
(see docs/decisions/001-data-source-strategy.md and
docs/failures/001-integrated-dataset-labels.md). If any of these start
failing, the raw data changed and the docs above need a re-check before
trusting anything built on top of it.
"""

import csv
from pathlib import Path

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPERIMENTS_DIR = DATA_RAW / "CNC 비식별화 원본데이터_1209" / "CNC Virtual Data set _v2"
TRAIN_META_CSV = DATA_RAW / "CNC 비식별화 원본데이터_1209" / "train.csv"
INTEGRATED_DIR = DATA_RAW / "CNC 학습통합데이터_1209"


def _data_row_count(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header


def test_25_experiment_files_present():
    experiment_files = sorted(EXPERIMENTS_DIR.glob("experiment_*.csv"))
    assert len(experiment_files) == 25


def test_train_metadata_covers_all_experiments():
    # train.csv pads values with spaces for alignment, so every field needs stripping.
    with TRAIN_META_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
    assert len(rows) == 25
    assert {r["No"] for r in rows} == {str(i) for i in range(1, 26)}
    expected_columns = {
        "No",
        "material",
        "feedrate",
        "clamp_pressure",
        "tool_condition",
        "machining_finalized",
        "passed_visual_inspection",
    }
    assert set(rows[0].keys()) == expected_columns


def test_integrated_dataset_is_the_experiments_concatenated():
    total_experiment_rows = sum(
        _data_row_count(f) for f in EXPERIMENTS_DIR.glob("experiment_*.csv")
    )
    x_train_rows = _data_row_count(INTEGRATED_DIR / "X_train.csv") + 1  # no header row
    x_test_rows = _data_row_count(INTEGRATED_DIR / "X_test.csv") + 1  # no header row
    assert total_experiment_rows == x_train_rows + x_test_rows


def test_integrated_y_test_is_single_class_known_issue():
    """Documented in docs/failures/001-integrated-dataset-labels.md.

    Y_test.csv currently has exactly one distinct label across all rows,
    which makes it useless for evaluating a classifier. If this test
    starts failing, the file has multiple classes now -- update the
    failure doc and reconsider using the integrated dataset directly.
    """
    with (INTEGRATED_DIR / "Y_test.csv").open(newline="", encoding="utf-8") as f:
        distinct_labels = {tuple(row) for row in csv.reader(f)}
    assert len(distinct_labels) == 1
