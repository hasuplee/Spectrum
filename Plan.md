# Plan — Phase 2: Spectrum Agent

목표/비목표는 [PRD.md](PRD.md)(G1~G6)를 따른다. 모든 Step은 Windows CPU(`venv_spectrum_cpu`)에서 검증한다.
Step 0은 환경 세팅, Step 1~7은 `.claude/TDD/SKILL.md`의 RED → GREEN → REVIEW 사이클로 진행한다.
커밋은 CLAUDE.md 제약 5/6에 따라 Step마다 **RED 종료 / GREEN 종료 / REVIEW 종료의 세 번**, 제목은
`Step X RED: ...` / `Step X GREEN: ...` / `Step X REVIEW: ...` 형식으로 하며, **Claude는 커밋 메시지만 작성하고, 커밋은 사용자가 직접 한다.**
(Step 0은 TDD 대상이 아니므로 논리 변경 단위별로 커밋했고, 그때는 자동 커밋이었다.)

공통 테스트 원칙: LLM은 mock, 학습 subprocess는 mock, 실제 학습은 tiny 설정 + `slow` 마커.
각 Step 완료 시 기존 Phase 1 테스트(46개)가 계속 통과해야 한다(G6).

---

## Step 0. 환경 세팅 (비 TDD)
- 목표: `venv_spectrum_cpu` 하나에 Agent 개발 환경을 갖춘다 (G5).
- 작업 (논리 단위별 별도 커밋)
  1. 설치 전 `pip freeze > requirements-dev-before-agent.txt` 백업.
  2. `agno`, `openai`, `gradio` 설치 (agno가 pydantic 1.10 → 2.x 업그레이드를 동반함을 dry-run으로 확인).
  3. 설치 직후 `pytest tests/` 전체 통과 확인. 깨지면 버전 고정 등 대응책을 정한다.
  4. AGNO `VLLM` 클래스의 import 경로/생성자/tool-calling 요구사항을 설치된 소스에서 확인 → 이 문서에 기록.
  5. CPU 학습 실측: tiny 설정(`--embed-dim 8 --num-layers 1 --num-basis 8`, 작은 batch, `--workers 0`)에서
     PaiNN/Equiformer step당 시간 측정 → smoke 학습 step 수와 `slow` 테스트 예산 확정.
  6. `requirements-agent.txt`(추가 의존성 목록), `pytest.ini`에 `slow`, `vllm` 마커 등록.
- 완료 조건: 기존 테스트 전체 통과, 실측 결과 기록, `from agno.models.vllm import VLLM` 성공.
- **완료 결과 (Step 0 종료)**
  - 설치: agno 3.1.1, openai 3.24.0, gradio 6.29.1, pydantic 2.13.5 (`requirements-agent.txt`).
    백업은 `requirements-dev-before-agent.txt`.
  - gradio가 `huggingface-hub`를 2.x로 올려 transformers 4.56(`<1.0` 요구)과 충돌 → 0.36.2로 되돌려 고정.
    그 결과 `pip check`는 gradio의 `huggingface-hub>=1.16` 요구 위반을 경고하지만, gradio import/Blocks 구성/
    서버 응답(HTTP 200)은 확인함. UI 단계(Step 6)에서 문제가 생기면 이 지점을 먼저 의심한다.
  - 회귀: 설치 후 기존 테스트 46개 전체 통과 (55.9초).
  - `from agno.models.vllm import VLLM` 성공. 생성자 주요 인자: `id`, `base_url`, `api_key`, `temperature`(기본 0.7),
    `top_p`(0.8), `presence_penalty`(1.5), `max_tokens`, `timeout` 등 (OpenAI 호환 클라이언트 기반).
  - CPU 학습 실측 (`--workers 0`, split 0.0: train 818 / val 103 / test 103, 평가 시 val+test 206개 전체 순회):
    - PaiNN tiny (`--embed-dim 8 --num-layers 1 --num-basis 8 --batch-size 4`): 2 step 11초, 20 step 약 10초 (평가 포함).
      체크포인트 `checkpoint_best.ckpt`(약 44KB, `model`/`optimizer`/`args` 저장), `pred.csv`, `p_spec.csv` 생성 확인.
    - Equiformer (모델 크기 인자 없음, 기본 3.3M 파라미터): 2 step 56초 → 실학습 테스트에서 제외.
  - 결정: `slow` 실학습 테스트는 PaiNN tiny, `--train-steps 5 --eval-steps 5` 수준 1건(약 10초 예산)으로 한다.
    Equiformer/Geoformer는 ckpt fixture(tiny 모델 저장)로만 테스트한다.
  - 참고(Step 2): ckpt의 `args`는 pickle된 Namespace이며 task_mean/std는 저장되지 않는다.
  - `pytest.ini`에 `slow`, `vllm` 마커 등록.

## Step 1. 체크포인트 registry (신규 인터페이스)
- 목표: `results_*/` 아래에서 학습된 체크포인트를 찾아 "예측 가능 여부"를 판정 (G2 전제).
- 범위: `agent/tools/registry.py`
  - 포함
    - `SUPPORTED_BASE_MODELS = ("PaiNN", "Equiformer", "Geoformer")`
    - `CheckpointInfo`(dataclass: `base_model`, `path`, `modified_time`)
    - `find_checkpoints(results_root=".", base_model=None) -> list[CheckpointInfo]` — 수정 시각 최신순 정렬.
    - `get_latest_checkpoint(results_root=".", base_model=None) -> CheckpointInfo | None`
    - `has_trained_model(results_root=".", base_model=None) -> bool`
    - 지원하지 않는 `base_model`은 `ValueError`(메시지에 지원 목록 포함).
  - 탐색 규칙(기존 학습 스크립트의 저장 경로 그대로, 코드는 읽기만 하고 수정하지 않음)
    - PaiNN/Equiformer: `{results_root}/results_{모델}/{seed}/{fold}/checkpoint_best.ckpt`
    - Geoformer: `{results_root}/results_Geoformer/{seed}/{fold}/checkpoints/*.ckpt` (`last.ckpt` 포함)
  - 미포함: ckpt 내용 로드/검증(Step 2), 어떤 ckpt가 "최고 성능"인지 판정(수정 시각 기준 최신만 제공),
    `agent/` 패키지 외 기존 코드 수정, tool 래핑/AGNO 등록(Step 3~5).
- 테스트 계획 (`tests/agent_tools/test_registry.py`, `tmp_path`에 빈 파일로 디렉터리 구조만 만듦 — 모델/torch 불필요).
  테스트 디렉터리 이름을 `agent_tools`로 하는 이유: `tests/agent/`로 두면 pytest가 `agent`라는 모듈 이름을
  잡아 신규 `agent` 패키지를 가릴 수 있음.
  1. `test_results_폴더가_없으면_체크포인트_목록이_비어있다`
  2. `test_PaiNN_checkpoint_best를_찾는다`
  3. `test_Equiformer_checkpoint_best를_찾는다`
  4. `test_Geoformer_checkpoints_폴더의_ckpt를_모두_찾는다` (last.ckpt 포함)
  5. `test_모델을_지정하면_해당_모델의_체크포인트만_반환한다`
  6. `test_여러_체크포인트는_수정_시각_최신순으로_정렬된다` (`os.utime`으로 시각 지정)
  7. `test_최신_체크포인트를_반환한다`
  8. `test_체크포인트가_없으면_최신_체크포인트는_None이다`
  9. `test_학습된_모델이_있으면_True이고_없으면_False이다` (모델 지정/미지정 각각)
  10. `test_지원하지_않는_모델을_지정하면_ValueError가_발생한다`
  11. `test_다른_모델의_폴더에_있는_ckpt는_무시한다` (예: `results_PaiNN`에 Geoformer 형식 파일만 있으면 PaiNN은 없음으로 판정)
- RED 검증 기준: 모든 테스트가 `ModuleNotFoundError: agent`(기능 부재)로 실패해야 한다 (오타/설정 오류 아님).
- 완료 조건(REVIEW 종료 시): 위 테스트 전체 통과 + 기존 46개 회귀 통과, `agent/`는 `spectrum/`을 import하지 않음.
- **완료 결과 (Step 1 종료)**
  - RED(`a01afd0`): 11개 테스트가 `ModuleNotFoundError: agent`로 실패함을 확인.
  - GREEN(`1d028b7`): `agent/tools/registry.py` 구현, 신규 11개 통과 + 전체 57개(기존 46 + 신규 11) 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안(체크포인트 내용 로드/성능 판정/tool 래핑 없음). 스코프 크리프 없음.
    - 라이센스 경계/수치 불변: `agent/`에 `spectrum` 참조 없음(grep 확인), 기존 코드 수정 없음(신규 파일만).
    - 실제 산출물 검증: Step 0 smoke에서 만든 실제 PaiNN `checkpoint_best.ckpt`를 `results_PaiNN/0/0/`에 두면
      registry가 인식함. 저장소 루트(현재 results_* 없음)에서는 `has_trained_model(".")`가 False.
    - 리팩토링: 코드 변경 없음. 관찰만 기록 — (a) PaiNN/Equiformer의 glob 패턴이 같은 문자열이라 상수로 묶을
      수 있으나 모델별 규칙이 달라질 여지를 남기려고 그대로 둠. (b) `results_root` 기본값 `"."`은 현재 작업
      디렉터리에 의존하므로 Step 3/4의 tool은 저장소 루트를 명시적으로 넘겨야 한다.
  - Step 2로 넘기는 사항: registry는 경로만 제공하며 ckpt 로드/모델 재구성은 `common/inference`(Step 2)가 담당.

## Step 2. 곡선 복원 + `common/inference` 확장 (신규 인터페이스) — 2A~2D로 분할
- 목표: ckpt → 모델 재구성 → 정규화 복원 → spectrum 곡선 반환 (G2).
- 분할: 각 하위 Step(2A~2D)마다 RED/GREEN/REVIEW 세 번 커밋하며, 제목은 `Step 2A RED: ...` 형식을 따른다.
  2A 곡선 복원 → 2B PaiNN → 2C Equiformer → 2D Geoformer 순서. 2B 이후의 세부 계획은 각 하위 Step의 RED에서 확정한다.
- 조사로 확인된 사실 (2B~2D 설계 근거)
  - PaiNN/Equiformer ckpt: `{'model', 'optimizer', 'args'}` (`args`는 pickle된 Namespace: targets, radius, num_basis,
    embed_dim, num_layers, data_path, split_index_npz, spectrum_type, lineshape, beta, n_mode 등).
  - PaiNN ckpt에는 task_mean/std가 없다 → 학습 split(`common.data.load_dataset_splits`)으로 재계산해야 한다.
    Equiformer는 `args.task_mean/task_std`가 ckpt `args`에 들어 있다.
  - Geoformer ckpt는 Lightning 형식(`state_dict`의 키가 `model.` 접두사, `hyper_parameters`에 mean/std 포함)이고,
    모델 내부에 `mean`/`std` 버퍼가 있어 출력이 **이미 역정규화**되어 나온다. 따라서 Geoformer 예측에는 기존
    `common.inference.predict()`에 실제 mean/std를 넣으면 이중 역정규화가 된다 → norm_factor는 `[0, 1]`을 사용한다.
  - 곡선 복원 로직은 현재 `spectrum/write.py`의 `save_spectrum`에 CSV 저장과 함께 섞여 있다.
- 결정(사용자 확인 완료): 곡선 복원은 **`spectrum/` 안에** 순수 함수로 추가하고 `save_spectrum`이 그것을 호출하도록
  `spectrum/` 내부에서 재구성한다(제약 1: spectrum/ 내부 재구성은 자유). `common/`·`agent/`는 import만 한다.

### Step 2A. 곡선 복원 함수 (`spectrum/` 내부 재구성 + 신규 함수)
- 목표: 예측 파라미터 → 곡선(400~800nm, 0.5nm 간격, 800점)을 CSV 저장 없이 반환하는 함수를 `spectrum/` 안에 둔다.
  `save_spectrum`의 CSV 출력은 수치/형식이 이전과 완전히 동일해야 한다 (제약 2, 3).
- 범위
  - 포함
    - `spectrum/reconstruct.py` 신규: `wavelength_grid_nm()`(400~800nm, 0.5 간격 torch/numpy 격자 — save_spectrum과 동일 값),
      `reconstruct_spectrum(preds, spectrum_type='FC', kernel_kind='gaussian', beta=2.0) -> torch.Tensor (B, 800)`.
      `spectrum_type`별 처리(Naive min-max, GMM 8열, FC `n_S=(열수-2)//2`)와 미지원 타입 예외 메시지
      (`"Undefined spectrum type"`)는 `save_spectrum`의 기존 동작 그대로.
    - `spectrum/write.py`의 `save_spectrum`이 `reconstruct_spectrum`을 호출하도록 재구성 (CSV 컬럼/포맷 불변).
  - 미포함: `common/`·`agent/` 연동(2B 이후), `spectrum/physics/` 수정, 포맷/수치 변경.
- 테스트 계획 (`tests/refactor/test_spectrum_reconstruct.py`)
  - 특성화(현재 코드에서 이미 통과해야 하는 오라클, 골든 파일 사용 — `tests/support/golden.py`의 2회 실행 규칙):
    1. `test_save_spectrum_FC_CSV가_리팩토링_전과_동일하다`
    2. `test_save_spectrum_GMM_CSV가_리팩토링_전과_동일하다`
    3. `test_save_spectrum_Naive_CSV가_리팩토링_전과_동일하다`
    4. `test_save_spectrum_미지원_타입은_Undefined_spectrum_type_예외를_낸다`
  - 신규(실패해야 함 — `spectrum.reconstruct` 부재):
    5. `test_FC_곡선복원은_save_spectrum_CSV값과_일치한다`
    6. `test_GMM_곡선복원은_save_spectrum_CSV값과_일치한다`
    7. `test_Naive_곡선복원은_save_spectrum_CSV값과_일치한다`
    8. `test_FC_n_mode가_2일때도_곡선을_복원한다` (열 수 6)
    9. `test_파장_격자는_400에서_800nm까지_0점5nm_간격_800점이다`
    10. `test_미지원_spectrum_type은_예외를_낸다`
- RED 검증 기준: 특성화 4개는 (골든 생성 후 재실행 시) 통과, 신규 6개(5~10)는 `ModuleNotFoundError: spectrum.reconstruct`로 실패.
- 완료 조건: 위 테스트 전체 통과 + 기존 테스트 회귀 통과, `git diff`에서 `save_spectrum`의 출력 로직이 순수 이동으로 보임,
  `spectrum/` 변경에 외부 폴더 변경이 섞이지 않음.

- **완료 결과 (Step 2A 종료)**
  - RED(`6681374`): 특성화 4개 통과 + 신규 6개가 `ModuleNotFoundError: spectrum.reconstruct`로 실패.
  - GREEN(`c00ecca`): `spectrum/reconstruct.py` 신규, `save_spectrum`이 이를 호출. 신규 10개 통과, 전체 67개 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안(`spectrum/` 내부 재구성 + 신규 함수 2개). `common/`·`agent/` 연동, `spectrum/physics/` 수정 없음.
    - 라이센스 경계: 변경 파일이 `spectrum/reconstruct.py`, `spectrum/write.py` 두 개뿐이며 외부 폴더 변경이 섞이지 않음.
      외부로의 코드 복사 없음(`common/`·`agent/`는 아직 `spectrum`을 참조하지 않음).
    - 동작 보존(실제 산출물 오라클): Step 0 smoke의 `pred.csv`(103개 분자)로 리팩토링 전(`6681374`의 `write.py`)과 후의
      `save_spectrum`을 같은 입력으로 실행 → **CSV가 바이트 단위로 동일**. 구 코드를 두 번 실행해도 동일(결정적).
    - 참고: Step 0에서 학습 중 생성된 원본 `p_spec.csv`와는 Intensity 값이 최대 1e-6(CSV 6번째 소수 자리 한 칸) 차이가
      났다. 구 코드로 재생성해도 동일하게 차이가 나므로 리팩토링 원인이 아니라, 학습 중 메모리의 preds와 `pred.csv`
      텍스트 왕복(또는 실행 간 부동소수 차이) 때문으로 판단한다. 이 차이는 이후 테스트에서 허용오차(atol 1e-6)로 다룬다.
    - 미세 차이: 기존 `expand(len(ids), -1)`이 `expand(len(preds), -1)`로 바뀜. 모든 호출처(`train_PaiNN.py`,
      `train_Equiformer.py`, `geoformer/module.py`)에서 `len(ids) == len(preds)`이므로 정상 입력에서 동작 동일.
    - 리팩토링: 코드 변경 없음. 관찰만 기록 — `spectrum/write.py`의 `import torch`가 더는 쓰이지 않음(제거는 선택 사항,
      이번 사이클에서는 하지 않음).
  - Step 2B로 넘기는 사항: `reconstruct_spectrum`은 `spectrum.reconstruct`에서 import해 사용한다
    (`common/inference`는 import만, 로직 복사 금지).

### Step 2B. PaiNN: 체크포인트 로드 + 예측 + 곡선 (세부 계획은 2B RED에서 확정)
- 범위(예정): `common/inference.py`에 `load_checkpoint(path, base_model)`(PaiNN: ckpt `args`로 `painn_adapter.build`,
  state_dict 로드, `eval()`), task_mean/std 재계산, 배치 → 파라미터 → `reconstruct_spectrum` 곡선 반환.
- 테스트(예정): tiny PaiNN 저장→로드→`predict()` 수치 일치, 정규화 복원, 곡선 shape/유한성.

### Step 2C. Equiformer (세부 계획은 2C RED에서 확정)
- 범위(예정): ckpt `args.task_mean/std` 사용, `equiformer_adapter.build`로 재구성. Equiformer 테스트는 느리므로(약 8초/건) 최소 케이스.

### Step 2D. Geoformer (세부 계획은 2D RED에서 확정)
- 범위(예정): Lightning ckpt의 `state_dict`에서 `model.` 접두사 제거, `hyper_parameters`로 모델 재구성,
  norm_factor는 `[0, 1]`(모델 내부 역정규화).

## Step 3. 예측 tool (신규 인터페이스)
- 목표: IrDB 분자(ID 또는 인덱스) → 예측 spectrum (G2).
- 범위: `agent/tools/predict_tool.py` — `predict_spectrum(molecule_id, base_model=None, ...)`.
  ckpt가 없으면 `{"status": "needs_training", ...}` 구조화 응답. 존재하지 않는 ID는 오류 응답.
- 테스트: ckpt 없음 → needs_training, 잘못된 ID, tiny ckpt fixture로 정상 예측.

## Step 4. 학습 tool (신규 인터페이스)
- 목표: 파라미터 검증/기본값 조회/백그라운드 학습/상태 조회 (G1).
- 범위
  - `agent/tools/train_tool.py` — `get_training_defaults(base_model)`(argparse 기본값을 그대로 노출,
    **기본값 불변**), `start_training(...)`(검증 후 Popen), `get_training_status()`(프로세스/로그 tail).
  - train.py가 노출하지 않는 인자(train-steps 등)는 tool이 직접 커맨드를 조립해 전달(기존 스크립트
    기본값은 건드리지 않음). `train.py`의 `build_command()` 재사용 가능 부분은 재사용.
- 테스트: 잘못된 모델/spectrum type 거부, 커맨드 조립(subprocess mock), 상태 조회,
  `slow`: tiny PaiNN 실학습 1건으로 ckpt 생성 확인.

## Step 5. AGNO Agent (신규 인터페이스)
- 목표: tool을 사용하는 특수 목적 Agent (G3).
- 범위: `agent/agent.py` — `build_agent(model=None)`; 기본은 `VLLM(id=$VLLM_MODEL, base_url=$VLLM_BASE_URL)`,
  테스트에서는 mock 모델 주입. instructions: 학습/예측 이외 질문 거절, 학습 요청 시 기본값 제시 후
  확인/모델 선택 질문, 예측 시 ckpt 없으면 학습 선행 안내. tool 등록: Step 1/3/4의 함수.
  **범위 가드(1차 방어)**: `agent/guard.py` — `is_in_scope(text)`가 학습/예측 관련 키워드(학습, 예측, spectrum,
  PaiNN/Geoformer/Equiformer, 분자 ID, 기본값, 상태 등)로 판별. 범위 밖이면 LLM을 호출하지 않고 고정 거절
  문구를 반환. instructions(2차 방어)는 가드를 통과한 애매한 질문(예: "PaiNN이 뭐야?")도 학습/예측 요청이
  아니면 거절하도록 명시.
- 테스트(mock 모델): tool 등록 목록, 환경변수 설정 → VLLM 객체 생성, tool-call 시나리오별 dispatch,
  instructions에 도메인 제한 규칙 포함. 범위 밖 질문 거절은 mock 응답 흐름 + (선택) 실제 vLLM smoke.
  가드 테스트(vLLM 불필요): 거절 — "오늘 날씨가 뭐야?", "반도체는 뭐지", "OLED의 정의는", "파이썬 코드 짜줘";
  통과 — "PaiNN으로 학습해줘", "그냥 학습해줘", "이 분자의 spectrum 예측해줘", "학습 상태 알려줘".
  가드가 거절하면 mock 모델이 호출되지 않음을 확인.

## Step 6. UI (신규 인터페이스)
- 목표: 간단한 UI에서 학습/예측 수행 (G4).
- 범위: `agent/ui.py` (Gradio) — 채팅 탭(Agent) + 학습 탭 + 예측 탭(곡선 plot). 탭은 tool 함수를 직접 호출
  하므로 LLM 없이도 동작. 로직은 UI 파일과 분리해 테스트 가능한 함수로 둔다.
  채팅 탭: `VLLM_BASE_URL` 미설정/연결 불가 시 안내 메시지. 입력창 아래 예시 질문(클릭 시 입력) —
  동작: "PaiNN으로 학습해줘", "그냥 학습해줘", "학습 기본값 보여줘", "학습 상태 알려줘",
  "이 분자의 spectrum 예측해줘"(예시에는 실제 IrDB 분자 ID 포함), "사용 가능한 모델 알려줘";
  거절 시연: "오늘 날씨가 뭐야?", "반도체는 뭐지?", "OLED의 정의는?", "파이썬 코드 짜줘".
- 테스트: UI 핸들러 함수 단위 테스트(실제 브라우저 불필요), `build_ui()`가 예외 없이 구성되는지,
  예시 질문 목록이 가드 기준과 일치하는지(동작 예시는 in-scope, 거절 예시는 out-of-scope).
  브라우저 수동 확인은 REVIEW에서 1회.

## Step 7. E2E smoke 및 정리
- 목표: 성공 기준 1~4 확인 (G1~G6).
- 작업: `slow` E2E(tiny 학습 → registry → 예측 → 곡선), 선택적 `vllm` 마커 smoke(`VLLM_BASE_URL` 없으면 skip),
  README에 Agent 실행 방법과 환경변수 문서화, 최종 전체 테스트 실행.
- 완료 조건: 전체 테스트 통과(slow 포함), 문서 간 Step/G 번호 일치.

---

## 의존 관계
Step 0 → 1 → 2 → 3 → 4(독립 가능, 3 이후 권장) → 5(1,3,4 필요) → 6(5 필요) → 7.
