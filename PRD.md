<<<<<<< HEAD
# PRD — Phase 2: Spectrum Agent

## 배경
Phase 1 리팩토링으로 세 백본(Geoformer/PaiNN/Equiformer)의 학습·추론 코드가 `common/`(adapters, data,
training_utils, inference)과 `spectrum/physics`로 정리되었다. Phase 2에서는 이를 **tool**로 노출하고,
그 tool을 사용하는 **AGNO 기반 Agent**와 **간단한 UI**를 개발한다.

핵심 질문은 "성능"이 아니라 **"현재 코드를 tool 형식으로 만들 수 있고, 그것을 사용하는 Agent가 실제로
동작하는가"** 이다.

## 목표 (Goals)

- **G1. 학습 tool**: Agent/UI가 호출할 수 있는 spectrum 학습 tool.
  - 모델(PaiNN/Geoformer/Equiformer), spectrum type(FC/GMM/Naive), batch size, 데이터셋 등 파라미터를
    검증해서 받는다. 기본값을 조회해 사용자에게 보여줄 수 있어야 한다.
  - 학습은 오래 걸리므로 백그라운드로 실행하고 상태/로그를 조회할 수 있다.
- **G2. 예측 tool**: 학습된 체크포인트로 IrDB 예시 데이터의 분자에 대해 spectrum 곡선을 예측한다.
  - 체크포인트가 없으면 예측하지 않고 "학습이 먼저 필요함"을 구조화된 응답으로 돌려준다.
- **G3. AGNO Agent**: 반드시 AGNO로 구현하고, LLM은 AGNO의 `VLLM` 모델 클래스로 연결한다.
  - "그냥 학습해줘" → 기본 파라미터를 보여주고 사용자 확인 후 학습, 또는 모델 선택을 되묻는다.
  - 예측 요청인데 학습된 모델이 없으면 학습을 먼저 제안한다.
  - **특수 목적 Agent**: 학습/예측과 무관한 질문에는 답할 수 없다고 말하고, 관련 질문만 하도록 유도한다.
    예) "오늘 날씨가 뭐야?", "반도체는 뭐지", "OLED의 정의는"은 거절. "PaiNN으로 학습해줘",
    "이 분자의 spectrum 예측해줘" 같은 학습/예측 요청만 동작한다(현재 버전).
  - 범위 밖 판별은 **LLM 호출 전 코드 가드**(1차)와 Agent instructions(2차)로 이중 방어한다.
    1차 가드는 vLLM 서버 없이도 테스트 가능해야 한다.
- **G4. 간단한 UI**: 채팅(Agent)과, 학습/예측을 직접 실행하는 화면. UI에서 spectrum 학습과 예측이
  가능해야 한다.
  - 채팅 탭은 vLLM 서버에 연결되어야 동작한다. 미연결 시 안내 메시지를 보이고, 학습/예측 탭은 LLM 없이 동작한다.
  - 채팅 입력창 아래에 클릭 가능한 예시 질문(동작하는 질문 + 거절 시연용 질문)을 제공한다.
- **G5. 로컬 CPU 테스트**: 모든 테스트는 Windows 로컬 CPU에서 통과한다. vLLM 서버·GPU 없이도 테스트가
  통과해야 한다(LLM은 mock으로 대체).
- **G6. 기존 동작 보존**: 기존 학습/모델/physics 코드의 수치 동작을 바꾸지 않는다(Phase 1 oracle
  테스트가 계속 통과).

## 확정된 결정

- 입력 범위: **IrDB 예시 데이터의 분자**만 대상. 학습도, 예측도 이 데이터 기준.
- LLM 연결: AGNO `VLLM(id, base_url, api_key)`. vLLM 서버는 OpenAI 호환 엔드포인트로 **분리**되어 있고
  `VLLM_BASE_URL`, `VLLM_MODEL` 환경변수로 설정한다. 서버를 띄우는 방법은 이 프로젝트의 범위 밖이다
  (Windows 네이티브 vLLM 미지원, GPU 없음).
- **Docker는 사용하지 않는다.**
- 환경: `venv_spectrum_cpu` **하나**로 통일한다(개발용 환경). agno/openai/gradio 등을 여기에 추가한다.
- 신규 코드는 `agent/`(+ `common/inference` 확장)에 둔다. `spectrum/`은 import로만 참조한다.

## 비목표 (Non-goals)

- 모델 성능 개선, 하이퍼파라미터 기본값 변경, 알고리즘 변경.
- GPU 환경 지원 작업, 대규모 학습.
- 임의 SMILES/SDF 입력으로부터의 예측(3D 구조 생성 필요).
- Docker, 배포용 서비스화, 인증/멀티유저.
- 범용 챗봇 기능.
- 실제 vLLM 서버의 로컬 구축(연동 지점만 제공하고, 연동 smoke는 선택).

## 제약

- CLAUDE.md의 라이센스 경계(`spectrum/` ↔ 그 외), 한글 테스트명/커밋, 자동 커밋, TDD 규칙을 따른다.
- 기존 학습 스크립트의 기본값은 바꾸지 않는다. tool은 CLI 인자를 통해 값을 **전달**할 뿐이다
  (smoke 테스트용 작은 설정은 호출 시 override).
- 테스트 시간: 기존 46개 테스트 전체가 약 56초(Equiformer 모델 생성 ~8초/건). 신규 테스트는 LLM mock,
  subprocess mock, tiny 모델을 사용하고, 실제 학습을 도는 테스트는 `slow` 마커로 분리한다.

## 성공 기준

1. UI(또는 tool 직접 호출)로 tiny 설정 학습 → 체크포인트 생성 → IrDB 분자 예측 → 곡선 반환이 CPU에서
   끝까지 동작한다.
2. mock LLM 기반 Agent 테스트가 위 시나리오(기본값 제시/모델 질문/학습 선행 안내/범위 밖 거절)를 검증한다.
3. `VLLM_BASE_URL`이 주어지면 동일 Agent가 vLLM 서버에 연결된다(선택 smoke).
4. 기존 Phase 1 테스트가 모두 통과한다.
=======
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
>>>>>>> parent of 6ec37fe (doc: 기존의 PRD.md와 Plan.md 삭제)
