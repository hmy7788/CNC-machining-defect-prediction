# 001. CNC 원본 데이터 베이스라인 EDA

## 날짜
2026-09-04

## 한 일
`data/raw/CNC 비식별화 원본데이터_1209/`(25개 실험 CSV + `train.csv` 메타데이터)를 대상으로
`src/visualization/eda_baseline.py`를 작성해 실행했다. 스크립트는 데이터를 읽기 전용으로만 다루고,
아래 수치는 전부 그 스크립트의 stdout(JSON)에서 그대로 가져온 것이다 — 하드코딩 없음.

확인한 항목: 라벨 분포(실험 단위, N=25), 실험별 시계열 길이·`Machining_Process` 구성, 결측/상수/이상치
컬럼, 그리고 (체크리스트에는 없었지만 발견 즉시 추가한) **실험 파일 간 중복 여부**.

## 결과

### 1. 라벨 분포 (N=25, 매우 작은 표본)
- `tool_condition`: worn 14 / unworn 11
- `machining_finalized`: yes 19 / no 6 (no면 `passed_visual_inspection` 공란)
- `passed_visual_inspection` (finalized=yes인 19건 중): yes 13 / no 6
- `material`은 전 실험에서 상수(`aluminum`) — 피처로 쓸 수 없음.
- 그림: `reports/figures/001_label_distribution.png`

### 2. 시계열 특성
- 25개 파일, 총 32,048행, 48컬럼. 실험당 행 수는 462~2,332로 편차가 크다(중앙값 1296, 표준편차 733).
- `Z_CurrentFeedback`, `Z_DCBusVoltage`, `Z_OutputCurrent`, `Z_OutputVoltage` 4개 컬럼은 **전체
  데이터셋에서 값이 상수(분산 0)** — 모델링 시 제거 후보.
- `Machining_Process`는 대소문자 표기가 섞여 있다(`"End"` vs `"end"`) — 같은 단계인데 문자열이 달라
  그대로 두면 별개 카테고리로 잡힌다. 원본 라벨 목록: `End, Layer 1 Down, Layer 1 Up, Layer 2 Down,
  Layer 2 Up, Layer 3 Down, Layer 3 Up, Prep, Repositioning, Starting, end` (11개 raw 값 = 실제 9단계 + 대소문자 중복).
  `Starting` 라벨은 32,048행 중 **단 1행**뿐이다.
- 결측치는 0건(전체 컬럼 기준).
- 그림: `reports/figures/001_experiment_lengths.png`, `001_machining_process.png`

### 2.5. 중요 발견 — 실험 파일 중복 + 라벨 충돌 (체크리스트 밖에서 발견)
실험 파일 간 바이트 단위 해시(MD5)를 비교한 결과, **`experiment_14.csv`와 `experiment_24.csv`가
완전히 동일한 파일**이고, **`experiment_19.csv`와 `experiment_25.csv`도 완전히 동일한 파일**이다
(cell_match = 1.0, 즉 텔레메트리 데이터가 100% 일치). 그런데 `train.csv`에 적힌 라벨은 서로
**충돌**한다:

| 그룹 | No | tool_condition | passed_visual_inspection |
|---|---|---|---|
| A | 14 | worn | yes |
| A | 24 | **unworn** | yes |
| B | 19 | worn | **no** |
| B | 25 | worn | **yes** |

같은 텔레메트리 데이터에 서로 다른 정답이 붙어 있다는 뜻이다. 둘 중 하나는 라벨이 틀렸거나, 파일이
실수로 중복 제공된 것으로 보인다 — 어느 쪽이 맞는지는 이 데이터만으로 판단할 수 없다.
그림: `reports/figures/001_experiment_similarity.png` (초록 테두리가 바이트 동일 쌍).

부수적으로: 콘텐츠 전체가 아니라 길이/순서 유사도만 보면 (2,21), (5,23), (7,19/22/25), (8,20)
같은 쌍도 `Machining_Process` 진행 패턴은 100% 일치하지만 실제 셀 값은 46~63%만 일치한다 — 이건
"같은 가공 프로그램을 유사한 조건으로 반복 실행한 결과"로 보이며 (7,19,22,25 상호 간, 8,20 등)
중복은 아니다. **완전 바이트 일치는 (14,24), (19,25) 두 쌍뿐**이다.

### 3. 통합데이터 교차검증
25개 실험 파일의 총 행 수(32,048)가 `학습통합데이터`의 `X_train+X_test` 행 수와 일치한다는
기존 결정(`docs/decisions/001-data-source-strategy.md`)을 재확인했다. 이번 EDA에서 새로 발견한
중복 파일 문제는 통합데이터 쪽에도 그대로 들어있을 것이다(2배로 카운트됨) — 통합데이터를 참고할 때
이 점도 감안해야 한다.

## 트러블슈팅
- 초기 실행에서 EDA를 수행하던 서브 에이전트가 진행 중 멈춰(watchdog: 600초 무응답) 실패 처리됐다.
  스크립트 자체(`src/visualization/eda_baseline.py`)와 그림 4개는 이미 정상 생성돼 있었고, 이 문서
  작성 단계만 남아 있었다. 재실행 대신 오케스트레이터가 직접 스크립트를 재실행해 JSON 출력을 받아
  이 문서를 완성했다 — 서브 에이전트를 처음부터 다시 돌리는 낭비를 피함.
- `train.csv`는 값 앞뒤에 정렬용 공백이 있어 `skipinitialspace=True` + 문자열 컬럼 `.str.strip()`
  없이 읽으면 `"unworn"`과 `"unworn "`이 다른 카테고리로 잡힌다 (`docs/domain/glossary.md`에 이미
  기록된 함정, 실제로 처음 겪음 — strip 적용 후 해결).

## 다음 액션 / 에이전트 지침
- **`cnc-preprocess` 단계는 (14,24), (19,25) 중복 쌍을 반드시 처리 방침을 정하고 넘어가야 한다.**
  최소한 다음 중 하나를 명시적으로 선택하고 근거를 기록할 것: (a) 각 쌍에서 한쪽만 남기고 제외, (b)
  두 쌍 모두 학습에서 제외, (c) 라벨 충돌을 그대로 두고 노이즈로 취급 — 어떤 경우든 "몰랐다"는 안 됨.
  자세한 내용은 `docs/failures/002-duplicate-experiment-conflicting-labels.md` 참고.
- `Z_CurrentFeedback`/`Z_DCBusVoltage`/`Z_OutputCurrent`/`Z_OutputVoltage`는 전역 상수이므로 피처에서
  제외 후보로 검토.
- `Machining_Process`는 `.str.strip().str.title()` 등으로 대소문자를 통일한 뒤 사용할 것 (`End`/`end`
  중복 방지).
- `material`은 상수라 피처로 의미 없음.
- N=25(실험 단위)는 매우 작으므로, `cnc-model` 단계에서 교차검증 폴드 수를 신중히 정할 것 (많은 폴드는
  폴드당 표본이 1~2개로 줄어든다).
