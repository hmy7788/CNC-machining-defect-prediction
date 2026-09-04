"""Check that every file path CLAUDE.md points to actually exists.

Run: python scripts/check_docs_drift.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAUDE_MD = ROOT / "CLAUDE.md"


def find_missing_paths() -> list[str]:
    """Only checks backticked strings that contain a '/', i.e. real paths --
    bare filenames mentioned in prose (e.g. `train.csv`, `experiment_NN.csv`)
    are just illustrative and aren't meant to resolve from the repo root.
    """
    text = CLAUDE_MD.read_text(encoding="utf-8")
    candidates = re.findall(r"`([\w./-]+/[\w./-]*)`", text)

    missing = []
    for candidate in candidates:
        path = candidate.rstrip("/")
        if not (ROOT / path).exists():
            missing.append(candidate)
    return sorted(set(missing))


def main() -> int:
    missing = find_missing_paths()
    if missing:
        print("CLAUDE.md references paths that don't exist:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print("문서 드리프트 없음 (no doc drift)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
