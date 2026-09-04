# 용어 사전

## 데이터 출처
- **KAMP**: 중소벤처기업부 산하 AI 제조 데이터 플랫폼(kamp-ai.kr). 이 프로젝트의 데이터는 KAMP의
  "CNC 머신 AI 데이터셋" (KAIST, UNIST, EPM Solutions 제공, 2020-12-14 게시).
- 인용 표기: `중소벤처기업부, KAMP, CNC 머신 AI 데이터셋, KAIST(UNIST, EPM Solutions), 2020.12.14,
  https://kamp-ai.kr`

## `train.csv` 컬럼
- **tool_condition**: 공구 마모 상태. `worn`(닳음) / `unworn`(안 닳음).
- **machining_finalized**: 가공이 끝까지 완료됐는지. `yes`/`no`. `no`인 실험은
  `passed_visual_inspection`이 공란이다 (완료 안 된 부품은 육안검사 자체를 안 함).
- **passed_visual_inspection**: 육안검사 합격 여부. `yes`/`no` (finalized=yes인 실험에서만 값 존재).
- **material / feedrate / clamp_pressure**: 가공 조건 파라미터.

## 실험 텔레메트리 컬럼 (`experiment_XX.csv`)
- 축별(`X_`, `Y_`, `Z_` = 이송축, `S_` = 스핀들) 서보 상태: `ActualPosition`, `ActualVelocity`,
  `ActualAcceleration`, `SetPosition`, `SetVelocity`, `SetAcceleration`, `CurrentFeedback`,
  `DCBusVoltage`, `OutputCurrent`, `OutputVoltage`, `OutputPower`.
- **Machining_Process**: 가공 공정 단계 라벨. `Prep`, `Starting`, `Layer 1/2/3 Down/Up`,
  `Repositioning`, `end`. 통합데이터(`학습통합데이터`)에서는 이 값이 정수(0~8)로 인코딩되어 있다.

## `train.csv` 파싱 시 주의
파일이 컬럼 정렬을 위해 값 앞뒤에 공백을 넣어뒀다 (예: `" 1, aluminum,        6,            4  , unworn"`).
`pandas.read_csv`로 읽을 때 `skipinitialspace=True`를 쓰거나, 읽은 뒤 문자열 컬럼에
`.str.strip()`을 적용해야 한다. 안 하면 `"unworn"`과 `"unworn "`이 다른 값으로 처리된다.

## 데이터셋 관계
- `CNC 학습통합데이터_1209/X_*.csv`는 `CNC 비식별화 원본데이터_1209`의 25개 `experiment_XX.csv`를
  이어붙인 것 (행 수 32,048로 확인). 자세한 내용과 주의사항은
  `docs/decisions/001-data-source-strategy.md`, `docs/failures/001-integrated-dataset-labels.md` 참고.
