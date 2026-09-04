"""Cross-check that experiment logs reporting numbers are registered in
reports/metrics.json, the single source of truth for reported metrics
(see CLAUDE.md "필수: 검증 규율").

This only checks that a matching entry *exists* -- it can't verify the
number itself is correct without re-running the pipeline. It exists to
catch the common failure mode of a number appearing in a doc/notebook that
was never actually recorded (or copy-pasted from a stale run).

Run: python scripts/check_report.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_JSON = ROOT / "reports" / "metrics.json"
EXPERIMENTS_DIR = ROOT / "docs" / "experiments"

NUMBER_PATTERN = re.compile(r"\d")
ID_PATTERN = re.compile(r"^(\d+)-")
RESULT_SECTION_PATTERN = re.compile(r"## 결과\n(.*?)(?=\n## |\Z)", re.DOTALL)


def _experiment_id(path: Path) -> str | None:
    match = ID_PATTERN.match(path.name)
    return match.group(1) if match else None


def _result_section_has_number(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    match = RESULT_SECTION_PATTERN.search(text)
    if not match:
        return False
    return bool(NUMBER_PATTERN.search(match.group(1)))


def main() -> int:
    if not METRICS_JSON.exists():
        print(f"{METRICS_JSON.relative_to(ROOT)} 없음 -- 빈 객체 {{}}라도 만들어두세요.")
        return 1

    try:
        metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"{METRICS_JSON.relative_to(ROOT)}가 올바른 JSON이 아닙니다: {e}")
        return 1

    problems = []
    doc_ids = set()

    for doc in sorted(EXPERIMENTS_DIR.glob("*.md")):
        if doc.name == "TEMPLATE.md":
            continue
        exp_id = _experiment_id(doc)
        if exp_id is None:
            continue
        doc_ids.add(exp_id)
        if _result_section_has_number(doc) and exp_id not in metrics:
            problems.append(
                f"{doc.relative_to(ROOT)}: 결과에 수치가 있는데 "
                f"reports/metrics.json에 '{exp_id}' 항목이 없음"
            )

    for exp_id in metrics:
        if exp_id not in doc_ids:
            problems.append(
                f"reports/metrics.json의 '{exp_id}' 항목에 대응하는 "
                f"docs/experiments/{exp_id}-*.md 파일이 없음"
            )

    if problems:
        print("보고 수치 정합성 문제:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("보고 수치 정합성 문제 없음 (metrics.json in sync)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
