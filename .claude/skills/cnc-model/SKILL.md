---
name: cnc-model
description: "CNC 프로젝트의 모델 학습·평가 절차. 불균형 데이터 지표 선택, 다중 시드 반복, reports/metrics.json 등록 순서를 강제한다. '모델 학습해줘', '모델 평가해줘', '성능 확인해줘', '하이퍼파라미터 튜닝해줘' 요청 시 사용."
---

# CNC 모델 학습·평가 절차

## 성능이 너무 잘 나올 때 (가장 먼저 읽을 것)

AUC/정확도가 비정상적으로 높으면(예: 0.98 이상) 좋은 모델을 만든 게 아니라 **데이터 누수를 만든 것**일 확률이 높다 (`docs/decisions/001-data-source-strategy.md`에서 이미 겪은 문제와 같은 패턴). 곧바로 결과로 보고하지 않는다:
1. `cnc-preprocess`가 분할을 실험 단위/시간 순서로 했는지, 전처리를 분할 이후에 했는지 직접 코드를 확인한다.
2. 의심이 남으면 `cnc-qa`에게 검증을 요청한다.
3. 검증을 거치지 않은 고성능 수치는 `docs/experiments/`나 `reports/metrics.json`에 최종 결과로 적지 않는다 — 검증 과정 자체를 기록에 남긴다.

## 평가 절차

1. **지표 선택** — 이 데이터는 클래스 불균형이 있다(`train.csv` 기준 25건 중 소수 클래스가 6~11건 수준). accuracy를 단독 주 지표로 쓰지 않는다. F1/precision-recall AUC/balanced accuracy 등에서 고른 근거를 기록한다.
2. **다중 시드** — 최소 3~5개 시드로 반복 학습·평가하고, 평균과 표준편차를 함께 보고한다. 시드 하나짜리 결과를 최종 수치로 제시하지 않는다.
3. **모델 저장** — 학습된 아티팩트는 `models/`에 저장한다(파일명에 실험 번호나 날짜를 포함해 어떤 실행인지 구분 가능하게).
4. **수치 등록** — 모든 보고 수치는 `reports/metrics.json`에 등록한다. 키는 대응하는 `docs/experiments/NNN-model-*.md`와 같은 번호(`NNN`)를 쓴다. 등록 후 `python scripts/check_report.py`로 스스로 확인한다.

## 산출물

- `models/*` (아티팩트)
- `reports/metrics.json`에 `{"NNN": {"metric_name": value, ...}}` 형태로 등록
- `docs/experiments/NNN-model-*.md`에 사용한 모델/하이퍼파라미터/시드/검증 결과 기록

## 후속 작업

"모델 다시 학습", "하이퍼파라미터만 바꿔줘", "이전 모델보다 개선해줘" 요청 시에도 이 스킬을 사용한다. 이전 `docs/experiments/NNN-model-*.md`와 `reports/metrics.json`을 먼저 읽어 기존 수치를 기준선으로 삼는다.
