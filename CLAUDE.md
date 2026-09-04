# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 현재 상태

CNC 가공 결함/공구 마모 예측 프로젝트. 아직 분석/모델링 코드는 없고(`src/`, `notebooks/`, `models/`는
`.gitkeep`만 존재), 대신 에이전트가 안전하게 작업할 수 있도록 하네스(harness)부터 구축된 상태다. 첫
실제 코드를 추가할 때는 아래 "디렉토리 구조" 절의 관례를 따르면 된다.

**작업을 시작하기 전에 `docs/failures/INDEX.md`를 먼저 읽는다** — 이미 실패했던 접근을 반복하지
않기 위함이다.

## 하네스: CNC 분석 파이프라인

(이 섹션은 아래 "하네스 엔지니어링" 절과 다른 개념이다 — 저건 문서/린트/CI 등 프로젝트 운영 하네스,
이건 EDA→전처리→모델링→검증을 수행하는 에이전트 팀 구성이다. revfactory/harness 플러그인으로
생성됨.)

**목표:** CNC 가공 데이터에서 EDA·전처리·모델링·검증을 일관된 규율(라벨 소스, 분할 순서, 수치 등록)
아래 수행하는 에이전트 파이프라인 제공.

**트리거:** "CNC 파이프라인 실행해줘", "EDA부터 모델링까지 해줘", "전처리만 다시", "모델만 재학습",
"결과 업데이트해줘" 등 CNC 데이터 분석 작업 요청 시 `cnc-pipeline` 스킬을 사용하라. 단순 질문은
직접 응답 가능.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-09-04 | 초기 구성 (cnc-eda/cnc-preprocess/cnc-model/cnc-qa + cnc-pipeline 오케스트레이터) | `.claude/agents/`, `.claude/skills/` 전체 | 사용자가 kamp_2026 참고 후 revfactory/harness 플러그인으로 하네스 구성 요청 |

## 자주 쓰는 명령어

```bash
pip install -r requirements-dev.txt   # 분석용 + 개발용 패키지 설치
ruff check .                          # 린트
ruff format .                         # 포맷
pytest tests/ -v                      # 테스트
pre-commit install                    # 최초 1회, 커밋 전 자동 검사 활성화
python scripts/check_docs_drift.py    # CLAUDE.md가 가리키는 경로가 실제로 있는지 점검
python scripts/check_structure.py     # temp_*, *_old.* 같은 금지 파일 패턴 점검
python scripts/check_report.py        # docs/experiments/ 수치가 reports/metrics.json과 맞는지 점검
vulture src/                          # 미사용 코드 점검 (src/에 코드가 생긴 뒤부터 의미 있음)
```

## 디렉토리 구조 (관례)

```
data/raw/        # 원본 데이터. 절대 수정하지 않는다 (아래 "데이터 구조" 참고)
data/processed/  # 전처리 결과물 저장 위치
src/             # 코드 추가 시: data/(로딩·검증) · features/ · models/ · visualization/ 하위 구조 권장
notebooks/       # 탐색적 분석
models/          # 학습된 모델 아티팩트
reports/figures/ # 산출 그래프
reports/metrics.json  # 보고 수치의 단일 출처 (아래 "필수: 검증 규율" 참고)
docs/            # 지식 저장소 (아래 "지식 저장소" 참고)
tests/           # pytest
scripts/         # 하네스 점검용 스크립트
```

## 절대 금지
- `data/raw/` 아래 파일을 스크립트로 덮어쓰거나 수정하지 않는다. 전처리 결과는 `data/processed/`에 쓴다.
- `temp_`, `_new`, `_old`, `_backup`, `_fix` 이름의 파일을 만들지 않는다 (`scripts/check_structure.py`가
  감지한다).
- `data/raw/CNC 학습통합데이터_1209/Y_train.csv` / `Y_test.csv`를 라벨 소스로 그대로 쓰지 않는다 —
  이유는 `docs/decisions/001-data-source-strategy.md`와
  `docs/failures/001-integrated-dataset-labels.md`에 기록되어 있다.

## 필수: 전처리·실험 기록

**데이터 전처리를 하든, 모델을 학습/평가하든, 무언가를 실험하든 — 끝나면 반드시
`docs/experiments/`에 결과와 트러블슈팅을 기록한다.** 템플릿은
`docs/experiments/TEMPLATE.md`. 성공/실패 여부와 무관하게 기록한다 (실패도 기록해야 다음 세션이
같은 시도를 반복하지 않는다 — `docs/failures/`와 같은 취지). 기술적 선택을 내렸다면
`docs/decisions/`에도 별도로 남긴다. "나중에 기록하기"는 없다 — 세션이 끝나면 맥락이 사라진다.

## 필수: 검증 규율

(참고 사례: `docs/decisions/001-data-source-strategy.md`에서 이미 겪은 라벨 신뢰성 문제)

- **고성능 = 누수 의심 먼저.** 모델 성능이 기대보다 지나치게 좋으면(예: AUC/정확도가 비정상적으로
  높음) 곧바로 결과로 보고하지 않는다. 먼저 데이터 누수(분할 기준, 전처리 순서, 그룹/시간 유출)부터
  검증하고, 검증 과정을 `docs/experiments/`에 함께 남긴다. 시드 하나짜리 단일 실행 결과를 최종
  수치로 제시하지 않는다 — 최소한 여러 시드/폴드로 반복해서 편차를 같이 보고한다.
- **분할 먼저, 전처리는 그다음.** 전처리(정규화, 리샘플링 등)를 학습/검증 분할보다 먼저 하면 검증셋
  정보가 학습 과정에 새어 들어간다. 이 데이터는 실험(`No`) 단위 시계열이므로 행 단위 무작위 분할을
  쓰지 않는다 — 실험 단위 또는 시간 순서 기준으로 분할한다. 오버샘플링/언더샘플링은 학습 폴드
  내부에서만 수행한다 (검증/테스트 폴드에 적용 금지).
- **보고 수치는 `reports/metrics.json`에서만 가져온다.** 문서·노트북·리포트에 적는 모든 성능
  수치는 이 파일에 등록된 값이어야 한다. `docs/experiments/NNN-*.md`에 수치가 담긴 결과를 적었다면
  같은 번호(`NNN`)로 `reports/metrics.json`에도 등록한다. `python scripts/check_report.py`로
  등록 누락을 점검한다 (숫자 자체가 맞는지까지는 검증하지 못한다 — 그건 실제로 다시 실행해서
  확인해야 한다).

## 하네스 엔지니어링

이 프로젝트는 다음 5가지 구성 요소로 하네스를 구축했다 (참고: Notion `AI 하네스 엔지니어링`).

| 구성 요소 | 이 저장소에서의 실체 |
|---|---|
| 1. 지시 문서 | 이 `CLAUDE.md` 자체 |
| 2. 아키텍처 제약 | `pyproject.toml` (ruff 설정), `.pre-commit-config.yaml`, `.gitattributes` |
| 3. 피드백 루프 | `tests/`, `.github/workflows/ci.yml`, `reports/metrics.json` + `scripts/check_report.py` |
| 4. 지식 저장소 | `docs/decisions/`, `docs/failures/`(+`INDEX.md`), `docs/domain/`, `docs/experiments/` |
| 5. 가비지 컬렉션 | `scripts/check_docs_drift.py`, `scripts/check_structure.py`, `scripts/check_report.py`, `.github/workflows/housekeeping.yml` (매주 월요일 자동 실행) |

### 1. 지시 문서
이 파일이 그것이다. 프로젝트가 커져서 src/api/ 처럼 레이어가 생기면, 그 하위에 별도 `CLAUDE.md`를
두는 것을 고려한다 (지금은 아직 그 정도 구조가 아니다).

### 2. 아키텍처 제약
- `pyproject.toml`의 `[tool.ruff]`: `line-length = 100`, `target-version = "py310"` (설치된 Python이
  3.10이라 이에 맞춤). `E`(pycodestyle), `F`(pyflakes), `I`(isort), `N`(naming) 규칙만 켰다 — 데이터
  분석 코드라 mypy 같은 엄격한 타입 검사는 아직 붙이지 않았다(과한 마찰 방지 목적, 필요해지면 추가).
- 레이어 간 import 제약(`import-linter`)은 아직 `src/`에 api/services/models 같은 레이어가 없어서
  붙이지 않았다. 실제 모듈 구조가 잡히면 추가를 고려한다.
- `.gitattributes`로 텍스트 파일 줄바꿈을 LF로 고정했다 — Windows의 `core.autocrlf=true` 환경에서
  데이터 CSV 전체가 매번 "수정됨"으로 뜨는 문제를 막기 위함. 같은 파일에 `*.ipynb merge=binary`도
  설정해뒀다 — 노트북을 여러 세션/사람이 건드릴 때 자동 병합이 출력 셀을 조용히 깨뜨리는 대신 병합
  충돌로 드러나게 하기 위함.

### 3. 피드백 루프
- **가이드**: 아직 예제 코드가 없다 — 첫 모듈을 추가할 때 이 섹션에 패턴 예시를 채워 넣는다.
- **센서**: `tests/test_data_integrity.py`는 데이터 리뷰 중 직접 확인한 사실(실험 파일 25개, 원본
  실험 데이터 행 수 합계가 통합데이터 행 수와 일치, `Y_test.csv`가 단일 클래스뿐이라는 문제)을 그대로
  회귀 테스트로 박아뒀다. 이 테스트가 실패하면 원본 데이터가 바뀐 것이므로 `docs/decisions/`,
  `docs/failures/` 문서를 다시 확인해야 한다. `data/raw/`가 git에 없으므로(아래 "데이터 구조" 참고)
  로컬에 데이터가 없으면 이 테스트 파일 전체가 skip된다 — CI에서는 항상 skip 상태다.
- `.github/workflows/ci.yml`이 push/PR마다 `ruff check` + `ruff format --check` + `pytest`를 돌린다.
- `reports/metrics.json` + `scripts/check_report.py`: 보고 수치 정합성 센서. 위 "필수: 검증 규율"
  참고.

### 4. 지식 저장소
- `docs/decisions/001-data-source-strategy.md`: 통합데이터의 `Y_*`를 쓰지 않고 원본 `train.csv`에서
  라벨을 직접 만들기로 한 결정과 이유.
- `docs/failures/INDEX.md`: 실패 기록 색인 + 태그(`#data` `#eval` `#model` `#env` `#process` `#agent`).
  **작업 시작 전 필독.**
- `docs/failures/001-integrated-dataset-labels.md`: `Y_*` 라벨이 어떤 컬럼을 인코딩했는지 행 순서로
  역산하려다 실패한 기록 — 같은 시도를 반복하지 않기 위함.
- `docs/domain/glossary.md`: KAMP 데이터 출처, `train.csv`/`experiment_XX.csv` 컬럼 용어 정리.
- `docs/experiments/`: 전처리·모델 실험을 할 때마다 남기는 결과·트러블슈팅 로그
  (`docs/experiments/TEMPLATE.md` 형식, 위 "필수: 전처리·실험 기록" 참고).
- 앞으로 에이전트가 잘못된 방향을 제안했다가 정정되거나, 기술적 결정을 내릴 때마다 이 아래에 문서를
  추가한다.

### 5. 가비지 컬렉션
- `scripts/check_docs_drift.py`: 이 `CLAUDE.md`가 언급하는 파일 경로가 실제로 존재하는지 확인한다.
- `scripts/check_structure.py`: `src/`, `tests/`, `scripts/`, `notebooks/`에 금지 패턴(`temp_*` 등)
  파일이 있는지 확인한다.
- `scripts/check_report.py`: `docs/experiments/`에 수치가 있는데 `reports/metrics.json`에 등록이
  안 됐거나, 반대로 `reports/metrics.json`에 대응하는 실험 문서가 없는 경우를 잡는다.
- `.github/workflows/housekeeping.yml`이 매주 월요일 오전 9시(UTC) + 수동 실행으로 위 세 스크립트와
  `vulture`를 돌린다. `vulture`는 현재 `src/`가 비어 있어 `|| true`로 실패를 무시하도록 해뒀다 —
  실제 코드가 생기면 이 예외 처리를 제거한다.

## 데이터 구조

**`data/raw/`, `data/processed/`는 git에 커밋되지 않는다** (`.gitignore` 참고). 로컬 디스크에는 있지만
새로 clone하면 없다 — 원본은 [KAMP AI "CNC 머신 AI 데이터셋"](https://www.kamp-ai.kr)에서 직접
받아야 한다. `tests/test_data_integrity.py`도 이 데이터가 로컬에 없으면(즉 CI에서는 항상) 자동으로
skip되도록 만들어져 있다.

- `data/raw/CNC 비식별화 원본데이터_1209/` — 원본 실험별 데이터셋 (공개된 "CNC Mill Tool Wear"
  데이터셋과 동일한 구조이며, KAMP AI "CNC 머신 AI 데이터셋"의 원천 데이터다):
  - `CNC Virtual Data set _v2/experiment_01.csv` ~ `experiment_25.csv`: 가공 1회당 CSV 1개, 각각
    1000행 이상의 시계열 데이터. 컬럼 설명은 `docs/domain/glossary.md` 참고.
  - `train.csv`: 실험 25건에 대한 메타데이터 1행씩(총 25행), `No`(1~25, `experiment_NN.csv`와 매칭)를
    키로 `material`, `feedrate`, `clamp_pressure`, `tool_condition`, `machining_finalized`,
    `passed_visual_inspection`을 담고 있다. **라벨은 여기서 만든다.**
- `data/raw/CNC 학습통합데이터_1209/` — 위 25개 실험 CSV를 순서대로 이어붙여 만든 헤더 없는 flat
  CSV(`X_train.csv`/`X_test.csv`/`Y_train.csv`/`Y_test.csv`, 총 32,048행으로 원본과 행 수 일치 확인됨).
  **`Y_*`는 라벨 소스로 쓰지 않는다** — `docs/decisions/001-data-source-strategy.md` 참고.
- `data/processed/` — 비어 있음. 전처리/피처 엔지니어링 결과물을 저장할 위치.
- `docs/03. Guidebook_CNC.pdf` — CNC 데이터셋/도메인 배경 설명 문서.
