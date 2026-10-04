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
- **G4. 간단한 UI**: 채팅(Agent)과, 학습/예측을 직접 실행하는 화면. UI에서 spectrum 학습과 예측이
  가능해야 한다.
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
