"""Flag stray temp/backup files that shouldn't be committed.

Run: python scripts/check_structure.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["src", "tests", "scripts", "notebooks"]
FORBIDDEN_PATTERNS = [
    "temp_*",
    "*_new.*",
    "*_old.*",
    "*_backup.*",
    "*_fix.*",
]


def find_violations() -> list[Path]:
    violations = []
    for scan_dir in SCAN_DIRS:
        base = ROOT / scan_dir
        if not base.exists():
            continue
        for pattern in FORBIDDEN_PATTERNS:
            violations.extend(base.rglob(pattern))
    return sorted(set(violations))


def main() -> int:
    violations = find_violations()
    if violations:
        print("구조 규칙 위반 파일 (structure rule violations):")
        for path in violations:
            print(f"  - {path.relative_to(ROOT)}")
        return 1

    print("구조 드리프트 없음 (no structure drift)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
