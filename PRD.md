# PRD.md — Phase 1 리팩토링

## 배경

현재 `train_PaiNN.py`와 `train_Equiformer.py`는 모델 생성 부분을 제외한 대부분(약 250줄:
argparse, 데이터셋/split 로딩, optimizer/scheduler 생성, 학습 루프, 체크포인트/예측 저장, 로깅)이
거의 동일하게 중복되어 있다. `train_Geoformer.py`는 PyTorch Lightning 기반의 별도 파이프라인
(`geoformer/module.py`의 `LNNP`)을 사용하며, Naive/GMM/FC 스펙트럼 타입 분기 로직이
`engine.py`(PaiNN/Equiformer 경로)와 `geoformer/module.py`(Geoformer 경로)에 각각 독립적으로
재구현되어 있다. `train.py`는 이 세 스크립트를 `os.system()`으로 서브프로세스 실행하는 얇은
디스패처일 뿐이다.

이 상태에서는:
- 스펙트럼 타입/타겟 목록 정의가 3곳(`train_PaiNN.py`, `train_Equiformer.py`, `geoformer/examples/*.yml`)에
  중복되어 있어 한 곳만 고치면 불일치가 발생한다.
- `engine.py`의 `train_one_step`/`evaluate`는 Equiformer 전용 forward 시그니처
  (`f_in`, `edge_d_index`, `edge_d_attr` 등)에 암묵적으로 결합되어 있고, PaiNN은 이 시그니처에
  맞추기 위한 어댑터 래퍼를 별도로 구현하고 있다.
- 새 아키텍처를 추가하려면 공통 인터페이스에 꽂는 것이 아니라 `train_XXX.py`를 통째로 새로 작성해야 한다.

본 리팩토링(Phase 1)은 이 구조적 문제를 **동작 변경 없이** 해소하는 것을 목표로 한다.

## 목표 (Goals)

### G1. Behavior Preservation
리팩토링 전후로 forward output, loss, gradient, optimizer step 결과, spectrum reconstruction 결과가
**수치적으로 동일**해야 한다. 이는 다른 모든 목표(G2~G6)보다 우선한다. G1과 다른 목표가 충돌하면
G1을 지키는 방향으로 구조 변경을 유보한다.

### G2. 실행 흐름 공통화 (Architecture-Independent Code Consolidation)
CLI 진입, dataset/split 로딩, dataloader 생성, checkpoint/prediction 저장, logging 등
**아키텍처에 독립적인 코드**를 공통 module로 이동한다.
- 대상 예시: `load_split_from_npz`, `save_pred`, `warmup_exponential_decay`
  (`train_PaiNN.py`/`train_Equiformer.py`에 중복), dataset 선택 분기(`IrDB` vs `PtDB`),
  `OneBatchLoader`.
- 물리 계산(FC/GMM spectrum reconstruction, `spectrum/loss.py`)은 이미 `spectrum/`에 공유되어
  있으므로 G2의 범위가 아니다.
- Geoformer의 Lightning 기반 경로는 architecture-independent 코드가 다른 형태(Lightning
  hook)로 존재하므로, G2는 "동일한 함수를 억지로 공유"하는 것이 아니라 "중복된 로직을 한 곳에서
  정의하고 각 경로가 그것을 참조"하는 수준까지를 의미한다. Geoformer 경로를 PaiNN/Equiformer
  경로와 강제로 통합하지 않는다 (비목표 참고).

### G3. Model Registry / Adapter 도입
신규 아키텍처를 추가할 때 **`engine.py`, `train.py` 등 핵심 공통 코드를 수정하지 않고**
새 adapter 모듈 하나만 추가하면 되는 구조를 만든다.
- Adapter는 최소한 다음을 만족해야 한다:
  - 통일된 시그니처로 모델을 생성한다 (예: `build(args) -> nn.Module`).
  - 통일된 시그니처로 forward를 호출한다 (예: `forward(model, batch) -> Tensor`).
  - 현재 세 모델의 실제 호출 관례(Equiformer의 `f_in/edge_d_index/edge_d_attr`, PaiNN의
    `SimpleNamespace` 기반 fairchem 백본, Geoformer의 `z/pos` 기반 Lightning forward)를
    **adapter 내부로 캡슐화**하고, 외부에는 동일한 인터페이스만 노출한다.
- registry는 문자열 키(`"Geoformer"`, `"PaiNN"`, `"Equiformer"`) → adapter 매핑을 제공한다.
- G3는 "지금 네 번째 모델을 추가"하는 것이 아니라, **추가할 수 있는 골격을 만드는 것**까지가
  범위다 (비목표 참고).

### G4. Training / Inference 분리
spectrum prediction(추론)을 학습 루프로부터 독립된 Python API로 뽑아낼 수 있는 골격을 만든다.
- 목표는 `predict(model, batch, norm_factor, spec_type) -> spectrum` 형태의, 학습 상태(optimizer,
  scheduler, 학습 루프)에 의존하지 않는 순수 함수를 `engine.py` 또는 그 후신 모듈에서
  분리해내는 것이다.
- 현재 `evaluate()`(`engine.py:99`)는 loss 계산과 예측 수집이 섞여 있다. 이를 "예측 생성"과
  "평가 지표 계산"으로 분리 가능한 형태로 정리한다.
- G4는 실제 추론 API를 최종 완성하는 것이 아니라 **분리 가능한 골격(skeleton)**을 만드는 것까지가
  범위다 (Step 9 참고).

### G5. 라이센스 경계 보존
`spectrum/` 폴더와 그 외 폴더 사이의 소스 이동/복사/병합을 **발생시키지 않는다.**
모든 Step은 이 제약을 최우선으로 검증한다 (`CLAUDE.md` 제약 1 참고).

### G6. 회귀 안전망 구축
Characterization/regression test를 **CPU에서 재현 가능한 형태**로 `tests/`에 마련하여,
Phase 1 이후의 모든 구조 변경이 자동으로 검증되게 한다.
- 테스트는 실제 데이터셋 전체가 아니라, 최소 원자 수 / batch size 1~2의 tiny fixture 또는
  mock 데이터를 사용한다 (CPU 환경 제약, `CLAUDE.md` 참고).
- 세 모델(Geoformer/PaiNN/Equiformer) 각각에 대해 model construction, forward, physics
  함수, gradient, optimizer step 수준의 오라클 테스트를 확보한다.
- 이 안전망을 만드는 작업 자체와, 이 안전망 위에서 진행하는 모든 이후 리팩토링은
  [`.claude/TDD/SKILL.md`](.claude/TDD/SKILL.md)의 RED → GREEN → REVIEW 사이클을 따른다.
  순수 구조 이동 사이클에서는 characterization test가 곧 오라클이므로 "새로 실패하는 테스트"
  대신 "이동 전후로 계속 GREEN"이 검증 기준이 되고, 신규 인터페이스(G3 adapter, G4 `predict()`)
  도입 사이클에서는 통상적인 RED(아직 없어서 실패하는 테스트)부터 시작한다.

## 비목표 (Non-Goals)

- **모델 성능 개선, hyperparameter 튜닝.** 이 리팩토링은 순수 구조 개선이다.
- **Geoformer를 PaiNN/Equiformer와 동일한 수동 학습 루프로 강제 이전하는 것.**
  Geoformer는 PyTorch Lightning 기반 실행 경로를 그대로 유지한다.
- **`spectrum/`과 외부 폴더(Geoformer/PaiNN/Equiformer/fairchem) 간 코드 통합.**
  (G5 참고 — 라이센스 경계상 절대 금지)
- **GPU 환경에서의 실제 재현 성능 벤치마크.** 이 Phase의 검증은 CPU 환경에서의
  behavior-preserving 여부에 한정된다.
- **EquiformerV2/V3 등 신규 아키텍처 실제 도입.** G3에서 만드는 것은 "추가하기 쉬운 골격"이지,
  실제 신규 모델 추가 작업 자체가 아니다.
- **새로운 GNN 백본 추가.** (이전 대화에서 논의된 주제이지만, 이번 Phase 1의 범위가 아니다.)
- **Agent Tool화.** (이전 대화에서 논의된 주제이지만, 이번 Phase 1의 범위가 아니다.)

## 성공 기준

- [ ] G1: 세 모델 모두 legacy 코드 대비 forward/loss/gradient/optimizer-step 수치가
      `tests/` 회귀 테스트에서 동일함을 확인.
- [ ] G2: `load_split_from_npz`, `save_pred`, `warmup_exponential_decay` 등 중복 함수가
      단일 정의로 통합되고, `train_PaiNN.py`/`train_Equiformer.py`는 이를 import해서 사용.
- [ ] G3: registry를 통해 문자열 키로 세 모델의 adapter를 조회할 수 있고, adapter는 공통
      인터페이스(`build`, `forward`)를 만족.
- [ ] G4: 학습 루프 없이 "모델 + 배치 → 예측"만 수행하는 순수 함수가 분리되어 있고 테스트로 커버됨.
- [ ] G5: 전체 리팩토링 기간 동안 `spectrum/`과 그 외 폴더 사이의 `git diff` 상 코드 이동/병합이
      0건.
- [ ] G6: `tests/` 하위에 세 모델 각각의 characterization test가 존재하고
      `venv_spectrum_cpu`에서 `pytest tests/`로 재현 가능.
