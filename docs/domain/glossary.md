# 용어 사전

## 데이터 출처
- **KAMP**: 중소벤처기업부 산하 AI 제조 데이터 플랫폼(kamp-ai.kr). 이 프로젝트의 데이터는 KAMP의
  "CNC 머신 AI 데이터셋" (KAIST, UNIST, EPM Solutions 제공, 2020-12-14 게시).
- 인용 표기: `중소벤처기업부, KAMP, CNC 머신 AI 데이터셋, KAIST(UNIST, EPM Solutions), 2020.12.14,
  https://kamp-ai.kr`

## `train.csv` 컬럼 (공식 정의: `docs/03. Guidebook_CNC.pdf` 13쪽)
- **No**: 작업 번호 (`experiment_NN.csv`와 매칭).
- **material**: 작업 소재. 25개 실험 전부 `aluminum`으로 상수.
- **feedrate**: Tool 이동 속도, 단위 mm/s.
- **clamp_pressure**: 소재 Clamping 압력, 단위 bar.
- **tool_condition**: 공구 마모 상태. `worn`(닳음) / `unworn`(안 닳음). "일정시간 사용한 Tool,
  새로운 Tool".
- **machining_finalized** (가이드북 표기는 `machine_completed`): 가공이 끝까지 완료됐는지.
  `yes`/`no`. `no`인 실험은 `passed_visual_inspection`이 공란이다 (완료 안 된 부품은 육안검사
  자체를 안 함).
- **passed_visual_inspection**: 육안검사 합격 여부. `yes`/`no` (finalized=yes인 실험에서만 값 존재).

## 실험 텔레메트리 컬럼 (`experiment_XX.csv`, 총 48개 변수)
- 축별(`X_`, `Y_`, `Z_` = 이송축, `S_` = 스핀들) 서보 상태: `ActualPosition`(mm), `ActualVelocity`
  (mm/s), `ActualAcceleration`(mm/s/s), `SetPosition`(mm), `SetVelocity`(mm/s),
  `SetAcceleration`(mm/s/s), `CurrentFeedback`(A), `DCBusVoltage`(V), `OutputCurrent`(A),
  `OutputVoltage`(V), `OutputPower`(kw). `S_`에는 추가로 `S_SystemInertia`(토크 관성, kg·m²)가 있다.
- 기타 4개: `M_CURRENT_PROGRAM_NUMBER`(프로그램이 CNC에 나열된 번호), `M_sequence_number`(실행 중인
  G-code 라인), `M_CURRENT_FEEDRATE`(스핀들의 순간 공급 속도), `Machining_Process`.
- **Machining_Process**: 가공 공정 단계 라벨, 공식적으로 10가지: `Prep`, `Starting`,
  `Layer 1 Up`/`Down`, `Layer 2 Up`/`Down`, `Layer 3 Up`/`Down`, `Repositioning`, `End`(대소문자
  `end`도 섞여 있음 — 실제로는 같은 단계). 통합데이터(`학습통합데이터`)에서는 이 값이 정수(0~9)로
  인코딩되어 있다. `Starting`은 전체 32,048행 중 딱 1행뿐이다(EDA 001에서 확인).

## 학습통합데이터 라벨 제작 방식 (공식, `Guidebook_CNC.pdf` 27~40쪽)
`X_train/X_test/Y_train/Y_test`가 어떻게 만들어졌는지 원래 알 수 없었는데(`docs/failures/001`
참고), 공식 가이드북에 그대로 나와 있었다:

1. `train.csv`의 `machining_finalized` + `passed_visual_inspection`을 조합해 25개 실험을 3그룹으로
   나눈다: **공정완료+합격**(13개 실험, 행 약 22,654개, "양품") / **공정완료+불합격**(6개 실험,
   6,175행) / **공정미완료**(6개 실험, 3,228행). 뒤의 두 그룹을 합쳐 "불량품"(9,403행)으로 본다.
   **즉 `Y_*`의 진짜 의미는 `tool_condition`이 아니라 이 finalized+passed 조합이다.**
2. 양품·불량품 개수를 맞추려고 양품 데이터를 **앞에서부터 9,403행만 잘라** 불량품 9,403행과 합쳐
   `X_train`(18,806행)을 만든다. 실험 1·2번이 (우리가 검증한 대로) 순서 그대로 앞부분에 온 건
   이 때문이고, 3번째 실험부터 어긋나는 것도 이 절단 지점이 실험 중간에 걸려서다.
3. **`X_test`(13,242행)는 이 절단에서 남은 양품 데이터 전부**다 — 그래서 `Y_test`가 13,242행
   전부 "양품(0)"인 게 우리가 발견한 버그가 아니라 원래 그렇게 설계된 것이다. 정상적인
   train/test 분할이 아니라 "학습에 안 쓴 나머지 양품"일 뿐이다.
4. MinMaxScaler를 `X_train`과 `X_test`에 **각각 따로** `fit_transform`한다 — 즉 두 세트가 서로 다른
   스케일 기준을 쓴다는 뜻이라, 이것도 우리가 `Y_*`뿐 아니라 `X_*`도 라벨 소스/피처 소스로 그대로
   쓰지 않기로 한 결정(`docs/decisions/001-data-source-strategy.md`)을 뒷받침한다.
5. 가이드북 자체의 학습/검증 분할은 `fit()`의 `shuffle`/`validation_split` 옵션으로 **행 단위
   무작위**로 한다 — 같은 실험의 타임스텝이 학습/검증에 흩어져 들어갈 수 있어 CLAUDE.md의 "행 단위
   무작위 분할 금지" 규칙이 우려하는 종류의 누수다. 공식 결과(검증 99.6%, 평가 91.5%)가 높게 나온
   것도 이와 무관하지 않을 수 있다 — 우리 프로젝트가 실험 단위 분할을 고집하는 이유와 정확히 같은
   맥락.

## `train.csv` 파싱 시 주의
파일이 컬럼 정렬을 위해 값 앞뒤에 공백을 넣어뒀다 (예: `" 1, aluminum,        6,            4  , unworn"`).
`pandas.read_csv`로 읽을 때 `skipinitialspace=True`를 쓰거나, 읽은 뒤 문자열 컬럼에
`.str.strip()`을 적용해야 한다. 안 하면 `"unworn"`과 `"unworn "`이 다른 값으로 처리된다.

## 데이터셋 관계
- `CNC 학습통합데이터_1209/X_*.csv`는 `CNC 비식별화 원본데이터_1209`의 25개 `experiment_XX.csv`를
  이어붙인 것 (행 수 32,048로 확인). 자세한 내용과 주의사항은
  `docs/decisions/001-data-source-strategy.md`, `docs/failures/001-integrated-dataset-labels.md` 참고.
