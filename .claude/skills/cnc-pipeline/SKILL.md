---
name: cnc-pipeline
description: "CNC 가공 결함/공구 마모 예측 프로젝트의 EDA→전처리→모델링→검증 파이프라인을 조율하는 오케스트레이터. 'CNC 파이프라인 실행해줘', 'EDA부터 모델링까지 해줘', '전체 분석 진행해줘', '데이터 분석 파이프라인 돌려줘' 요청 시 사용. 후속 작업 요청('전처리만 다시', '모델만 재학습', '결과 업데이트해줘', '이전 결과 개선해줘', 'QA만 다시 돌려줘')에도 반드시 이 스킬을 사용한다."
---

# CNC Pipeline Orchestrator

CNC 가공 결함/공구 마모 예측 프로젝트의 EDA → 전처리 → 모델링 → 검증을 조율하는 통합 스킬.

## 실행 모드: 서브 에이전트 (파이프라인 + 생성-검증 게이트)

**왜 팀 모드가 아닌가:** 각 단계가 이전 단계의 파일 산출물에 순차 의존하고, 실시간 상호 토론이 결과 품질을 높이는 구조가 아니다(파이프라인은 팀 모드 이점이 제한적). 검증(QA)도 자동화 스크립트 실행 결과에 근거하므로, "산출물 제출 → 검증 → 필요시 재작업 요청"의 왕복이면 충분하고 실시간 채팅형 팀 통신이 필요 없다. 1인 프로젝트에서 팀 통신 오버헤드는 이득보다 크다.

**`_workspace/`를 쓰지 않는 이유:** 이 프로젝트는 이미 `docs/experiments/`(실험 로그), `data/processed/`(전처리 산출물), `models/`(모델 아티팩트), `reports/metrics.json`(수치 단일 출처)이라는 감사 추적 체계를 갖추고 있다. 별도의 임시 작업 폴더를 만들면 같은 목적의 저장소가 두 개 생겨 오히려 드리프트를 유발한다. 그래서 이 오케스트레이터는 각 단계의 실제 산출물 경로를 그대로 다음 단계의 입력으로 전달한다.

## 에이전트 구성

| 에이전트 | subagent_type | 역할 | 출력 |
|---------|--------------|------|------|
| cnc-eda | cnc-eda | 탐색적 데이터 분석 | `reports/figures/*`, `docs/experiments/NNN-eda-*.md` |
| cnc-preprocess | cnc-preprocess | 라벨 생성·분할·피처 엔지니어링 | `data/processed/*`, `docs/experiments/NNN-preprocess-*.md` |
| cnc-model | cnc-model | 모델 학습·평가 | `models/*`, `reports/metrics.json`, `docs/experiments/NNN-model-*.md` |
| cnc-qa | cnc-qa (general-purpose 기반) | 각 단계 산출물 검증 | 검증 보고, 필요 시 `docs/failures/NNN-*.md` |

모든 `Agent` 호출에 `model: "opus"`를 명시한다.

## 워크플로우

### Phase 0: 컨텍스트 확인 (후속 작업 지원)

1. `docs/failures/INDEX.md`를 먼저 읽는다 — 이미 실패한 접근을 반복하지 않기 위함.
2. `docs/experiments/`를 확인해 가장 번호가 큰 기존 기록을 찾는다.
3. 사용자 요청에 따라 분기한다:
   - 기존 기록 없음 → **초기 실행**: Phase 1(EDA)부터 순서대로 진행.
   - "전처리만 다시", "모델만 재학습" 등 특정 단계 재실행 요청 → 해당 에이전트만 재호출하고, 프롬프트에 이전 산출물 경로를 포함해 개선점을 반영하도록 지시. QA 게이트는 재실행된 단계에 대해서만 다시 수행.
   - "결과 개선"/"업데이트" 요청 + 새 입력 없음 → 마지막으로 완료된 단계 다음부터 재개.
   - 새 데이터/새 요구사항 제공 → Phase 1부터 새로 진행 (기존 `docs/experiments/` 기록은 지우지 않고 새 번호를 이어서 추가).

### Phase 1: EDA
`Agent(subagent_type: "cnc-eda", model: "opus", prompt: "<사용자 요청 요약 + 이전 EDA 기록 경로(있으면)>")` 호출. 완료 후 산출물 경로(`reports/figures/*`, `docs/experiments/NNN-eda-*.md`)를 확인한다.

### Phase 2: 전처리 (생성)
`Agent(subagent_type: "cnc-preprocess", model: "opus", prompt: "<Phase 1 EDA 기록 경로 포함>")` 호출. 완료 후 Phase 2.5로 진행.

### Phase 2.5: 전처리 검증 (QA 게이트)
`Agent(subagent_type: "cnc-qa", model: "opus", prompt: "<Phase 2 산출물 경로 + 분할/전처리 순서 검증 요청>")` 호출.
- **불합격**: 문제를 `cnc-preprocess`에게 전달해 재작업 요청 (같은 이유로 최대 2회). 2회 초과 시 사용자에게 보고하고 진행 여부 확인.
- **합격**: Phase 3으로 진행.

### Phase 3: 모델링 (생성)
`Agent(subagent_type: "cnc-model", model: "opus", prompt: "<검증된 Phase 2 산출물 경로 포함>")` 호출. 완료 후 Phase 3.5로 진행.

### Phase 3.5: 모델 결과 검증 (QA 게이트)
`Agent(subagent_type: "cnc-qa", model: "opus", prompt: "<Phase 3 산출물 + reports/metrics.json 정합성 + 고성능 의심 여부 검증 요청>")` 호출.
- **불합격**: `cnc-model`에게 재작업 요청 (최대 2회).
- **합격**: Phase 4로 진행.

### Phase 4: 종합 보고
`docs/experiments/`의 신규 기록, `reports/metrics.json`, `models/`의 산출물을 종합해 사용자에게 보고한다: 무엇을 했는지, 수치가 어디 등록됐는지, 남은 이슈가 있는지. 이어서 "결과에서 개선할 부분이 있나요?"라고 피드백 기회를 준다.

## 데이터 흐름

```
[cnc-eda] → docs/experiments/NNN-eda-*.md
                ↓ (경로 전달)
[cnc-preprocess] → data/processed/* + docs/experiments/NNN-preprocess-*.md
                ↓
        [cnc-qa 검증] --불합격--> cnc-preprocess 재작업 (최대 2회)
                ↓ 합격
[cnc-model] → models/* + reports/metrics.json + docs/experiments/NNN-model-*.md
                ↓
        [cnc-qa 검증] --불합격--> cnc-model 재작업 (최대 2회)
                ↓ 합격
        [오케스트레이터: 종합 보고]
```

## 에러 핸들링

| 상황 | 전략 |
|------|------|
| 데이터(`data/raw/`)가 로컬에 없음 | 재시도하지 않는다 — KAMP AI에서 데이터를 받아야 한다고 사용자에게 즉시 보고. |
| 에이전트 1개 실패 | 1회 재시도. 재실패 시 원인을 명시하고 해당 Phase에서 중단, 사용자에게 보고. |
| QA 게이트 반복 불합격 (2회 초과) | 사용자에게 보고하고 계속 진행할지, 접근을 바꿀지 확인한다. |
| 단계 간 산출물 불일치(예: 전처리 스키마와 모델 입력 불일치) | 삭제하지 않고 원인을 병기해 보고 — cnc-model이 임의로 데이터를 변형하지 않는다. |

## 테스트 시나리오

**정상 흐름**: "CNC 파이프라인 실행해줘" 요청 → Phase 0에서 초기 실행 판정 → EDA 완료 → 전처리 완료 → QA 통과 → 모델링 완료 → QA 통과 → 종합 보고, `reports/metrics.json`에 새 항목 등록됨.

**에러 흐름**: Phase 3.5에서 QA가 "성능이 비정상적으로 높음(AUC 0.999), Phase 2 분할이 실험 단위가 아니라 행 단위로 보임"을 발견 → `cnc-preprocess` 재작업 요청 → 재작업 후 재검증 → 통과 → Phase 3 재실행 → 정상 범위 수치로 종합 보고.
