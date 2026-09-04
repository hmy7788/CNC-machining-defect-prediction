# 002. 베이스라인 전처리 · 실험 단위 요약 피처(v1)

## 날짜
2026-09-04

## 한 일
`src/features/build_features_v1.py`를 작성해 실행했다 (`python src/features/build_features_v1.py`).
`data/raw/CNC 비식별화 원본데이터_1209/`를 읽기 전용으로만 읽고, 결과는 `data/processed/`에 쓴다.
아래 수치는 전부 그 스크립트의 stdout(JSON)에서 그대로 가져왔다 — 하드코딩 없음.

이번 단계의 결정 사항과 근거:

### (1) 중복 실험 쌍 4개(No=14, 19, 24, 25)를 데이터셋에서 전부 제외
`docs/failures/002-duplicate-experiment-conflicting-labels.md`에 기록된 대로
`experiment_14.csv == experiment_24.csv`, `experiment_19.csv == experiment_25.csv`가 바이트 단위로
동일한데 `train.csv`의 라벨은 서로 충돌한다(14=worn/24=unworn, 19=visual no/25=visual yes).

EDA 문서가 제시한 세 선택지 중 **(b) 두 쌍 모두 제외**를 택했다. 이유:
- 어느 쪽 라벨이 맞는지 판단할 근거가 데이터 안에 없다. 한쪽만 남기려면 "14가 맞다/24가 맞다"를
  임의로 골라야 하는데, 그 선택이 곧 라벨 노이즈로 남는다.
- 두 쌍을 그대로 두면(선택지 c) 동일한 X에 상반된 y를 학습시키는 셈이고, 쌍이 train/test로 갈릴 경우
  실험 단위 분할을 했는데도 사실상 같은 샘플이 양쪽에 존재하는 누수가 된다.
- N이 25 → 21로 줄어드는 손해는 있지만, 21개 중 4개(16%)를 잃는 대신 라벨 신뢰성과 누수 차단을
  얻는 쪽이 베이스라인으로는 보수적이고 안전하다.

**재검토 조건**: KAMP 원본 배포 페이지나 동일 원천 데이터셋(Kaggle "CNC Mill Tool Wear")의 공식
설명에서 이 중복/충돌에 대한 안내를 찾으면, 이 결정을 되돌려 (a) 한쪽만 남기는 방식으로 바꿀 수 있다.
그때는 `DUPLICATE_CONFLICT_NOS` 상수만 고치면 된다.

### (2) 1차 타겟은 `tool_condition`
21개 실험 전부에 값이 있어 표본을 잃지 않는다. `passed_visual_inspection`은
`machining_finalized == "yes"`인 실험에만 존재하므로 2차/보조 타겟으로 컬럼만 함께 저장하고
(결측은 채우지 않고 NaN 그대로), 이번 분할·평가 기준은 `tool_condition`으로 잡았다.

### (3) 분할: 실험(`No`) 단위 stratified hold-out, seed=42
`sklearn.model_selection.train_test_split`에 **실험 번호 배열**을 넣고 `tool_condition`으로 층화했다
(`test_size=5`, `random_state=42`). 이건 **행 단위 셔플이 아니다** — 한 실험의 모든 텔레메트리 행은
애초에 한 행으로 집계되고, 실험 자체가 통째로 train 또는 test 한쪽에만 들어간다. seed 42는 실험 배정
난수에만 쓰이며 스크립트 상수(`RANDOM_SEED`)로 고정돼 있다.

### (4) 피처: 실험 1건 = 1행짜리 요약 테이블
시계열 원시값(32,048행)을 그대로 쓰지 않고 실험 단위로 집계했다.
- 센서 채널별 `mean` / `std` / `min` / `max`.
- `Machining_Process` 단계별 소요 **비중**(행 수 비율). 실험별 길이 편차(462~2,332행)가 커서
  절대 개수 대신 비율을 썼다. 문자열은 `.str.strip().str.title()`로 정규화해 `"end"`/`"End"`를 합쳤다.
- 실행 길이 자체(`n_rows`)도 피처로 넣었다.
- 가공 조건 `feedrate`, `clamp_pressure`는 가공 전에 알 수 있는 값이라 피처로 포함했다.
- **제외**: `material`(전 실험 상수), `Z_CurrentFeedback`/`Z_DCBusVoltage`/`Z_OutputCurrent`/
  `Z_OutputVoltage`(EDA 001에서 확인된 전역 상수).

### (5) 스케일링은 파일에 적용하지 않고 train 통계만 별도 저장
저장된 CSV는 **원 스케일**이다(트리 계열 모델은 스케일링이 필요 없다). 스케일링이 필요한 모델을 쓸
때를 위해 **train 16개 실험만으로 계산한** 평균·표준편차를 `data/processed/features_v1_scaler.json`에
따로 저장해뒀다. 이 값을 train/test 양쪽에 적용하면 분할 이후 전처리 규율(CLAUDE.md "필수: 검증 규율")을
지킬 수 있다. test로 fit하지 말 것.

### (6) 오버샘플링 생략
train 라벨이 unworn 8 / worn 8로 완전 균형이고 test도 2/3이라 불균형 보정이 필요 없다. 표본이 16개뿐인
상황에서 합성 샘플(SMOTE 등)을 넣으면 오히려 원본 16개를 보간한 값들이 폴드 안에서 서로를 예측하는
형태가 되기 쉬워, 이득보다 위험이 크다고 판단했다.

## 결과

### 데이터셋 규모
- 실험 25건 → **21건**(No=14, 19, 24, 25 제외).
- 피처 **181개** (센서 통계 172개 + `Machining_Process` 단계 비중 10개 + `n_rows` +
  `feedrate` + `clamp_pressure` — 아래 zero-variance 제거 4개 반영 후 최종 181개).
- 단계 비중 컬럼 10개: `End`, `Prep`, `Starting`, `Repositioning`,
  `Layer_1_Down`, `Layer_1_Up`, `Layer_2_Down`, `Layer_2_Up`, `Layer_3_Down`, `Layer_3_Up`.
- 피처 결측 0건 (`passed_visual_inspection_yes`의 결측은 라벨이지 피처가 아니다).

### train/test 분할 (실험 단위, seed=42)

| split | 실험 수 | `No` 목록 | unworn(0) | worn(1) |
|---|---|---|---|---|
| train | 16 | 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 18, 20, 21, 22, 23 | 8 | 8 |
| test | 5 | 4, 8, 12, 15, 16 | 2 | 3 |

보조 타겟 `passed_visual_inspection_yes` 결측: train 4건(No=5, 7, 20, 23), test 2건(No=4, 16) —
모두 `machining_finalized == "no"`인 실험이다.

### train 기준 분산 0으로 판정돼 제거한 컬럼 (4개)
`M_CURRENT_FEEDRATE_max`, `M_sequence_number_min`, `S_OutputVoltage_min`, `S_SystemInertia_std`.
판정은 **train 16개 실험만으로** 했다 — test까지 보고 컬럼을 고르면 그 자체가 누수다.

### 산출물
- `data/processed/features_v1_train.csv` — 16행 × 185열
- `data/processed/features_v1_test.csv` — 5행 × 185열
- `data/processed/features_v1_scaler.json` — train만으로 fit한 평균/표준편차
- `src/features/build_features_v1.py` — 재실행 가능한 생성 스크립트

(185열 = `No` + 라벨 3개 + 피처 181개)

## 트러블슈팅
- `train.csv`의 값 앞뒤 공백 문제(`docs/domain/glossary.md`, EDA 001에서 이미 기록됨)를 그대로
  겪지 않도록 `skipinitialspace=True` + 전 문자열 컬럼 `.str.strip()`을 처음부터 적용했다. 추가로
  `machining_finalized == "no"`인 행의 `passed_visual_inspection`은 strip 후 빈 문자열 `""`이 되므로
  `replace("", np.nan)`으로 명시적 NaN 처리했다 — 안 하면 `""`가 하나의 카테고리처럼 남는다.
- `Z_OutputPower` 컬럼은 애초에 원본에 없다(Z축만 `OutputPower`가 빠져 있다). 제외 목록에 넣었다가
  `KeyError`가 날 뻔했으나, EDA가 지목한 전역 상수 4개만 제외 목록에 두어 문제없이 지나갔다.
  Z축 관련 컬럼 목록을 손으로 적을 때 주의할 것.
- `ruff format`이 `write_text(...)` 한 줄을 100자 한 줄로 합치도록 재포맷했다 — 기능 영향 없음.
- 스크립트 실행 시간은 수 초 수준(21개 CSV, 총 2.6만여 행)이라 별도 캐시는 두지 않았다.

## 다음 액션 / 에이전트 지침

### `cnc-model`이 쓸 스키마
파일: `data/processed/features_v1_train.csv`(16행), `data/processed/features_v1_test.csv`(5행).
두 파일은 컬럼 구성이 동일하다(185열).

| 컬럼 | dtype | 역할 |
|---|---|---|
| `No` | int64 | 실험 ID(1~25 중 21개). **피처 아님 — 반드시 제외하고 학습할 것.** |
| `tool_condition_worn` | int64 | **1차 타겟.** worn=1 / unworn=0 |
| `passed_visual_inspection_yes` | float64 | 2차 타겟. yes=1.0 / no=0.0 / NaN(미완료 가공) |
| `machining_finalized_yes` | int64 | 라벨 성격의 컬럼(가공 완료 여부). 피처로 쓰지 말 것 — 가공이 끝난 뒤에야 알 수 있는 값이라 누수 소지가 있다. |
| `feedrate` | int64 | 피처(가공 조건) |
| `clamp_pressure` | float64 | 피처(가공 조건) |
| `n_rows` | float64 | 피처(실행 길이) |
| `<채널>_{mean,std,min,max}` | float64 | 피처(센서 통계 172개) |
| `proc_frac_<단계>` | float64 | 피처(단계별 소요 비중 10개) |

즉 **피처 = 전체 컬럼에서 `No`, `tool_condition_worn`, `passed_visual_inspection_yes`,
`machining_finalized_yes` 4개를 뺀 181개**.

- train `No`: 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 18, 20, 21, 22, 23
- test `No`: 4, 8, 12, 15, 16

### 모델링 시 반드시 지킬 것
- **train 16개 / 피처 181개는 p ≫ n이다.** 규제가 강한 모델(로지스틱 + L1/L2, 얕은 트리)이나
  피처 선택/차원 축소를 먼저 고려하고, 어떤 경우든 피처 선택은 **train 안에서만** 수행할 것.
- **test 5개짜리 hold-out 하나로 최종 성능을 보고하지 말 것.** 실험 1개 차이가 20%p씩 흔든다.
  train 16개에 대한 실험 단위 CV(LOO 또는 4-fold stratified)를 여러 seed로 반복해 편차와 함께
  보고하고, 5개 hold-out은 마지막 확인용으로만 쓴다.
- 성능이 지나치게 좋으면 누수부터 의심할 것 (CLAUDE.md "필수: 검증 규율"). 특히 `No`나
  `machining_finalized_yes`가 실수로 피처에 섞였는지 먼저 확인.
- 스케일링이 필요하면 `data/processed/features_v1_scaler.json`(train으로만 fit)을 쓰거나,
  CV 안에서 폴드별로 다시 fit할 것. 전체 21행으로 fit하지 말 것.
- 오버샘플링은 이 데이터에서는 불필요하다(위 (6) 참고). 굳이 시도한다면 학습 폴드 내부에서만.

### 하지 말 것 / 다시 시도하지 말 것
- No=14, 19, 24, 25를 다시 넣지 말 것 — 근거 없이 되돌리면 위 (1)의 누수/노이즈 문제가 그대로 돌아온다.
  되돌리려면 KAMP/원천 데이터셋의 공식 설명을 먼저 확보할 것.
- 통합데이터(`CNC 학습통합데이터_1209`)의 `Y_*`를 라벨로 쓰지 말 것
  (`docs/decisions/001-data-source-strategy.md`).
- 행 단위(32,048행) 학습으로 되돌아갈 거라면 그건 v1이 아니라 별도 실험이며, 그 경우에도 분할은
  반드시 `No` 단위 group split이어야 한다.
