# 003. 베이스라인 모델 학습·평가 (tool_condition, features_v1)

## 날짜
2026-09-04

## 한 일
`src/models/train_baseline.py`를 작성해 실행했다 (`python src/models/train_baseline.py`).
입력은 실험 002가 만든 실험 단위 요약 피처(`data/processed/features_v1_train.csv` 16행,
`features_v1_test.csv` 5행, 185열)이고, 타겟은 `tool_condition_worn`이다.
아래 수치는 전부 그 스크립트가 stdout에 출력한 JSON에서 그대로 가져왔다 — 하드코딩·추정 없음.
같은 수치를 `reports/metrics.json`의 `"003"` 키에 등록했다.

### (1) 피처 정의와 누수 차단
피처 = 전체 185열 − {`No`, `tool_condition_worn`, `passed_visual_inspection_yes`,
`machining_finalized_yes`} = **181개**. 스크립트의 `load_split()`이 train/test 양쪽에 대해
`assert banned not in features`로 4개 컬럼이 피처 목록에 없음을 매 실행마다 확인한다.
train/test의 피처 컬럼 순서가 동일한지도 `assert features == test_features`로 확인한다.

### (2) 하이퍼파라미터 탐색 생략 (고정값)
train 16개 / 피처 181개는 p ≫ n이다. 그리드서치를 돌리면 16개짜리 CV 점수로 설정을 고르게 되는데,
그 순간 그 CV 점수는 더 이상 해당 설정의 정직한 성능 추정치가 아니게 된다(선택 편향). 그래서 이번
v1은 **강한 정규화를 고정값으로** 쓰고 탐색을 하지 않았다.

- 주 모델: `LogisticRegression(penalty="l2", C=0.1, class_weight="balanced", max_iter=1000,
  solver="lbfgs")` — `StandardScaler`와 함께 `Pipeline`으로 묶었다.
- 비교 모델: `RandomForestClassifier(n_estimators=200, max_depth=3, class_weight="balanced")`.
  깊이 3으로 얕게 제한해 16개 표본에서의 과적합을 억제했다.

`class_weight="balanced"`는 train이 8/8로 이미 균형이라 사실상 무영향이지만, 폴드마다 15개로
학습하면서 8/7 또는 7/8이 되는 미세 불균형을 자동 보정하도록 그대로 뒀다.

### (3) 평가 절차: train 16개 Leave-One-Out CV
`sklearn.model_selection.LeaveOneOut`으로 train 16개 실험에 대해서만 CV를 돌렸다. 각 행이 곧
실험 1건이므로 이건 그대로 **실험 단위 CV**이며, test 5개는 이 과정에서 전혀 보지 않는다.

- **스케일링은 폴드 안에서.** `StandardScaler`를 Pipeline에 넣어 폴드마다 15개 실험으로만 fit하도록
  했다. `data/processed/features_v1_scaler.json`(train 16개 전체로 fit)을 CV에 쓰면 held-out 실험 1건이
  스케일 통계에 섞여 들어가므로 쓰지 않았다. 최종 모델(16개 전체 학습)의 스케일러는 결과적으로
  그 JSON과 같은 통계다.
- **LOO는 폴드별 지표를 낼 수 없다.** 폴드마다 held-out이 1개뿐이라 폴드 accuracy는 0 아니면 1이고
  F1/AUC는 정의되지 않는다. 그래서 16개 폴드의 예측을 **모아서(pooled) 한 번에** accuracy/F1/
  balanced accuracy/ROC AUC를 계산했다.
- **시드 반복**: 로지스틱 회귀(lbfgs + 고정 데이터)는 결정론적이라 시드를 바꿔도 같은 값이 나온다 —
  시드 반복이 의미가 없어 1회만 실행했고, 표준편차는 0으로 기록했다. RandomForest는 랜덤성이 있어
  **시드 5개(0~4)**로 LOO를 반복해 평균±표준편차를 보고한다.
- **지표 선택 근거**: accuracy 단독은 쓰지 않는다. train은 8/8 균형이지만 test는 2/3이고, 21개짜리
  데이터에서 accuracy는 클래스 비율에 쉽게 끌려간다. 주 지표는 **balanced accuracy**(두 클래스
  recall의 평균 — 불균형과 무관하게 "worn을 잡는 능력"과 "unworn을 안 틀리는 능력"을 같이 본다)로
  두고, worn(양성) 탐지 관점의 **F1**과 임계값과 무관한 순위 품질인 **ROC AUC**를 함께 보고한다.

### (4) 추가 안전장치: 라벨 셔플(permutation) 점검
컬럼 이름 assert만으로는 "우연히 라벨을 인코딩한 피처"를 못 잡는다. 그래서 라벨을 무작위로 섞은
상태에서 같은 LOO를 10회(시드 0~9) 돌려, 성능이 우연 수준으로 무너지는지 확인했다.

## 결과

**요약: 두 모델 모두 LOO에서 사실상 우연 수준(chance level)이다. 고성능이 나와서 누수를 의심해야
하는 상황이 아니라, 그 반대다.**

### 누수 점검 결과
- 피처 개수 181개, `banned_columns_in_features = []` — `No`, `machining_finalized_yes`,
  `passed_visual_inspection_yes`가 피처에 섞이지 않았음을 실행 시점에 확인(assert 통과).
- train/test 피처 컬럼 동일, 피처 결측 0건.
- 라벨 셔플 점검(로지스틱, 10회): LOO accuracy 평균 **0.406 ± 0.194** (최대 0.75).
  실제 라벨에서의 0.500이 이 귀무분포 안에 완전히 들어간다 — 모델이 붙잡은 신호가 없다는 뜻이며,
  동시에 라벨을 몰래 알려주는 피처도 없다는 뜻이다.
- CLAUDE.md "고성능 = 누수 의심 먼저" 규율의 발동 조건(0.95+)에 해당하는 수치는 하나도 없었다.

### LOO CV (train 16개 실험, 예측 pooled)

| 모델 | 시드 | accuracy | F1 | balanced accuracy | ROC AUC |
|---|---|---|---|---|---|
| LogisticRegression (L2, C=0.1) | 결정론적, 1회 | 0.500 ± 0.000 | 0.556 ± 0.000 | **0.500 ± 0.000** | 0.531 ± 0.000 |
| RandomForest (200 trees, depth 3) | 0~4 (5회) | 0.475 ± 0.031 | 0.522 ± 0.042 | **0.475 ± 0.031** | 0.469 ± 0.034 |

RandomForest 시드별 LOO accuracy: 0.500, 0.4375, 0.500, 0.4375, 0.500.

balanced accuracy 0.5 = 동전 던지기. F1이 0.5를 넘는 것은 모델이 worn 쪽으로 많이 찍어서 생기는
착시이지 성능이 아니다(8/8 균형에서 전부 worn이라 찍으면 F1 ≈ 0.667). ROC AUC도 0.469~0.531로
0.5 근방이다. **비선형 모델(RF)이 선형 모델보다 나을 것도 없었다.**

### Hold-out sanity check (test 5개 실험 — 최종 성능 아님)

train 16개 전체로 다시 학습한 모델을 test 5개(`No` = 4, 8, 12, 15, 16)에 **딱 1회** 적용했다.

| 모델 | accuracy | F1 | balanced accuracy | ROC AUC |
|---|---|---|---|---|
| LogisticRegression | 0.400 | 0.571 | 0.333 | 0.000 |
| RandomForest (seed 0) | 0.400 | 0.571 | 0.333 | 0.500 |

두 모델 다 `y_true = [0, 1, 0, 1, 1]`에 대해 `y_pred = [1, 0, 1, 1, 1]` — 2/5만 맞혔다.

**이 표는 "최종 성능"이 아니라 sanity check다.** n=5라 실험 1건이 accuracy를 20%p씩 흔들고,
정확도 0.4의 95% 신뢰구간은 사실상 0~0.8 전 구간에 걸쳐 있어 어떤 결론도 지지하지 못한다.
로지스틱의 hold-out ROC AUC = 0.000(양성 2개보다 음성 3개에 더 높은 확률을 준 완전 역순)도
n=5, 양성 3/음성 2에서는 우연히 충분히 나올 수 있는 값이다(무작위 순열에서 확률 1/10).
LOO 결과와 함께 읽으면 "우연 수준"이라는 같은 결론일 뿐, 별도의 나쁜 신호로 해석하지 않는다.

### 결론
`features_v1`(실험 단위 센서 요약 통계 181개) + train 16개 실험이라는 조건에서,
강하게 정규화된 선형 모델도 얕은 트리 앙상블도 `tool_condition`을 우연 이상으로 예측하지 못한다.
이건 모델 선택의 문제라기보다 **표본 수(16)와 피처 수(181)의 비율, 그리고 실험 단위 집계가
공구 마모 신호를 평균 속에 묻어버린 결과**일 가능성이 크다. 실패도 결과이므로 그대로 기록한다.

### 산출물
- `src/models/train_baseline.py` — 재실행 가능한 학습·평가 스크립트 (stdout에 JSON 리포트)
- `models/baseline_logreg_v1.pkl` — train 16개 전체로 학습한 로지스틱 파이프라인 (joblib)
- `models/baseline_rf_v1.pkl` — 같은 데이터, RandomForest seed=0 (joblib)
- `reports/metrics.json` `"003"` 키 — 위 수치 전부

## 트러블슈팅
- **LOO에서 폴드별 F1/AUC가 정의되지 않는 문제.** `cross_val_score(..., scoring="f1")`을
  LOO에 쓰면 폴드마다 표본이 1개라 F1이 0/1로만 나오고 AUC는 `ValueError: Only one class present
  in y_true`가 난다. 폴드별 점수를 평균 내는 대신 16개 held-out 예측을 모아 한 번에 지표를 계산하는
  방식(`loo_pooled()`)으로 우회했다. LOO에서는 이 pooled 방식이 표준이다.
- **`f1_score`의 zero-division 경고.** 폴드 예측이 전부 0인 경우가 있어 `zero_division=0`을 명시했다.
- **스케일러를 어디서 fit할지.** `features_v1_scaler.json`(train 16개 전체 통계)을 CV에 그대로 쓰면
  held-out 실험 1건의 정보가 스케일에 들어간다. Pipeline 안에 `StandardScaler`를 넣어 폴드마다
  15개로 다시 fit하도록 바꿨다. 이 JSON은 CV가 아닌 최종 모델/추론 경로용으로만 의미가 있다.
- **ruff `E731`/lambda 관련.** RF 시드별 팩토리를 넘길 때 `lambda s=seed: make_rf(s)`처럼 기본인자로
  시드를 바인딩해야 한다. 그냥 `lambda: make_rf(seed)`로 쓰면 late binding 때문에 5회 전부 마지막
  시드로 돌아간다 — 시드별 값이 미묘하게 다른 것으로 이 함정을 피했는지 확인했다.
- 전체 실행 시간은 1분 이내다(LOO 16회 × 모델 6종 + permutation 10회). 캐시는 두지 않았다.

## 다음 액션 / 에이전트 지침

### 다시 시도하지 말 것
- **같은 `features_v1` 위에서 모델만 바꿔 가며 성능을 짜내려 하지 말 것.** 선형(L2 로지스틱)과
  비선형(얕은 RF)이 모두 balanced accuracy 0.5 근방이고, 라벨 셔플 귀무분포(0.406 ± 0.194) 안에
  들어간다. XGBoost/SVM 등을 추가로 돌려서 0.6이 나온다면 그건 16개짜리 CV의 분산이지 개선이 아니다.
- **16개 데이터로 그리드서치/하이퍼파라미터 튜닝을 하지 말 것.** 위 (2) 참고. 탐색으로 얻은 CV
  점수는 보고 가능한 성능이 아니다.
- **hold-out 5개(No 4, 8, 12, 15, 16)를 반복해서 들여다보지 말 것.** 이번에 1회 썼다. 여러 번
  확인하면 그 5개가 사실상 검증셋이 되어 hold-out으로서의 가치를 잃는다.

### 다음에 해볼 것 (우선순위 순)
1. **표본 수를 늘리는 방향** — 실험 1건 = 1행(21행) 구조가 근본 제약이다. 실험을 시간 구간(window)
   단위로 쪼개 행 수를 늘리되, 분할·CV는 반드시 `No` 단위 group split
   (`GroupKFold`/`LeaveOneGroupOut`)으로 유지할 것. 행 단위 무작위 분할은 즉시 누수다.
2. **피처 수를 줄이는 방향** — 181개 중 상당수가 노이즈일 수 있다. 단, 피처 선택은 반드시 CV 폴드
   **안에서** 수행해야 한다(전체 train으로 고른 뒤 CV 돌리면 selection bias로 성능이 부풀려진다).
3. **타겟 재검토** — `tool_condition`(worn/unworn)이 실험 단위 요약 통계로 구분 가능한 양인지
   자체가 의심스럽다. `docs/domain/glossary.md`와 가이드북(`docs/03. Guidebook_CNC.pdf`)에서
   마모 판정 기준을 다시 확인하고, 필요하면 `passed_visual_inspection`을 보조 타겟으로 비교해볼 것.
4. 위 1~3 중 무엇을 하든 **이번 003의 우연 수준 결과가 비교 기준선**이다. 새 접근이
   balanced accuracy 0.5를 유의하게 넘지 못하면 개선이 아니다.
