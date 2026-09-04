---
name: cnc-eda
description: "CNC 가공 결함/공구 마모 예측 프로젝트의 EDA(탐색적 데이터 분석) 절차. 원본 실험 CSV와 train.csv 메타데이터에서 라벨 분포, 클래스 불균형, 실험별 시계열 특성, 결측/이상치를 확인하는 방법을 담는다. 'CNC 데이터 EDA', '데이터 탐색해줘', '라벨 분포 확인', '클래스 불균형 확인' 요청 시 사용."
---

# CNC EDA 절차

이 프로젝트 데이터의 구조상 반드시 확인해야 하는 것들과, 확인하지 않으면 나중에 터지는 함정을 담은 체크리스트다.

## 왜 이 순서로 확인해야 하는가

이 데이터는 25개 실험(가공 1회당 1개 CSV)의 시계열이고, 진짜 라벨은 `train.csv`에만 있다(`docs/domain/glossary.md`). 통합 flat 파일의 `Y_*`를 먼저 열어보면 그럴듯한 라벨처럼 보이지만, 이미 `docs/failures/001-integrated-dataset-labels.md`에서 그 라벨의 의미를 신뢰할 수 없다는 게 확인됐다. EDA를 원본 `train.csv` 기준으로 시작해야, 애초에 잘못된 라벨로 분석을 쌓아 올리는 실수를 피한다.

## 체크리스트

1. **라벨 분포 (실험 단위, N=25)**
   - `tool_condition`(worn/unworn), `machining_finalized`(yes/no), `passed_visual_inspection`(yes/no, finalized=no면 공란) 각각의 분포를 센다.
   - N=25는 매우 작다 — "라벨이 균형 잡혀 있다"는 결론을 섣불리 내리지 않는다. 이후 모델 평가에서 이 작은 표본이 어떤 제약을 만드는지 기록한다.

2. **실험별 시계열 특성**
   - 각 `experiment_XX.csv`의 행 수(가공 시간)가 실험마다 다르다 — 고정 길이를 가정하지 않는다.
   - `Machining_Process` 컬럼(Prep/Starting/Layer N Down·Up/Repositioning/end)의 순서와 비중을 확인한다. 이 컬럼이 시계열 내 위치 정보를 담고 있으므로, 나중에 피처로 쓸지 결정할 근거가 된다.

3. **결측·이상치**
   - 컬럼별 결측 여부, 값이 상수인 컬럼(분산 0인 센서 채널 등)이 있는지 확인한다 — 있으면 모델링 단계에서 제거 후보로 기록한다.

4. **`train.csv` 파싱 함정**
   - 이 파일은 값 앞뒤에 정렬용 공백이 들어 있다 (`" 1, aluminum, ...        6"` 형태). `pandas.read_csv(..., skipinitialspace=True)` 또는 읽은 뒤 문자열 컬럼 `.str.strip()`을 반드시 적용한다. 안 하면 `"unworn"`과 `"unworn "`이 다른 카테고리로 잡힌다.

5. **통합데이터(`CNC 학습통합데이터_1209`) 참고 시 주의**
   - 참고용으로 열어볼 수는 있지만, 라벨(`Y_*`) 소스로 인용하지 않는다. 행 수가 원본 실험 CSV 합계와 일치하는지 정도만 교차검증용으로 쓴다.

## 산출물

- `reports/figures/`에 그림 저장 (라벨 분포, 시계열 예시, 결측 히트맵 등).
- `docs/experiments/NNN-eda-*.md`에 위 체크리스트 각 항목의 확인 결과를 기록 (`docs/experiments/TEMPLATE.md` 형식).
- 발견한 함정(예: 특정 실험의 이상치)은 향후 전처리 단계가 참고할 수 있도록 명시적으로 적는다.

## 후속 작업

"EDA 다시 해줘", "라벨 분포 다시 확인", "새 데이터로 EDA" 요청 시에도 이 스킬을 사용한다. 이전 `docs/experiments/NNN-eda-*.md`가 있으면 먼저 읽고 중복 분석을 피한다.
