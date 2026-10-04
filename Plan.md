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

### Step 2B. PaiNN: 체크포인트 로드 + 예측 + 곡선 (`common/inference.py` 확장)
- 목표: PaiNN 체크포인트에서 모델을 복원하고, 배치를 넣으면 역정규화된 파라미터를 거쳐 스펙트럼 곡선(B, 800)을 반환한다 (G2).
- 조사 근거: 실제 ckpt(`torch.load(..., weights_only=False)`; `weights_only=True`는 pickle된 `Namespace` 때문에 `UnpicklingError`)는
  `{'model', 'optimizer', 'args'}`이고, `args`에 `out_channels, radius, num_basis, embed_dim, num_layers, targets, standardize,
  spectrum_type, lineshape, beta, n_mode, data_path, split_index_npz, seed`가 모두 있다. state_dict 키는 `backbone.*`(DDP 접두사 없음).
- 범위
  - 포함 (`common/inference.py`에 추가, 기존 `predict()`는 변경하지 않음)
    - `LoadedCheckpoint`(dataclass: `model`, `base_model`, `args`, `norm_factor`)
    - `load_checkpoint(path, base_model, norm_factor=None) -> LoadedCheckpoint`
      - CPU(`map_location="cpu"`)로 로드, `painn_adapter.build(args)`로 재구성 후 `load_state_dict`, `eval()`.
      - `norm_factor`를 직접 주면 그대로 사용(재계산 안 함, 데이터셋 접근 안 함).
      - 안 주면: `args.standardize`가 False면 `[zeros, ones]`, True면 `common.data.load_dataset_splits(args)`로 학습 split의
        mean/std를 재계산해 `[mean, std]` 텐서로 만든다.
      - 지원하지 않는 모델 → `ValueError`(2B 시점에서는 `PaiNN`만 지원), 파일 없음 → `FileNotFoundError`.
    - `predict_curves(loaded, batch) -> torch.Tensor (B, 800)`: 기존 `predict()`로 역정규화된 파라미터를 얻고
      `spectrum.reconstruct.reconstruct_spectrum(params, args.spectrum_type, kernel_kind=args.lineshape, beta=args.beta)` 호출.
  - 미포함: Equiformer/Geoformer 로드(2C/2D), 분자 ID → 배치 변환·IrDB 조회(Step 3), 체크포인트 탐색(Step 1 registry),
    `spectrum/` 수정(`spectrum.reconstruct`는 import만), 학습 코드 수정.
  - 신뢰 전제: `weights_only=False`는 임의 pickle 실행이 가능하므로 이 저장소가 직접 학습해 만든 ckpt에만 사용한다
    (docstring에 명시).
- 테스트 계획 (`tests/refactor/test_inference_checkpoint.py`; tiny PaiNN `embed_dim=8, num_layers=1, num_basis=8`을 `tmp_path`에 저장해 사용,
  새 함수 import는 각 테스트 안에서 하여 개별 실패로 확인)
  1. `test_PaiNN_체크포인트를_로드하면_원본_모델과_같은_출력을_낸다` (state_dict 복원 + `eval()` 모드, `predict()` 결과 수치 일치)
  2. `test_standardize가_False이면_norm_factor는_0과_1이다`
  3. `test_standardize가_True이면_학습_split에서_평균_표준편차를_다시_계산한다` (실제 IrDB + `splits.0.0.npz`, 기대값은 기존
     `load_dataset_splits`)
  4. `test_norm_factor를_직접_주면_데이터셋을_읽지_않는다` (존재하지 않는 split 경로여도 오류 없음)
  5. `test_predict_curves는_역정규화한_파라미터로_곡선을_복원한다` (shape `(2, 800)`, 유한값, 수동으로 `raw*std+mean`→
     `reconstruct_spectrum`한 결과와 일치; 물리적으로 말이 되는 평균/표준편차를 주입해 NaN 방지)
  6. `test_지원하지_않는_모델이면_ValueError가_발생한다`
  7. `test_체크포인트_파일이_없으면_FileNotFoundError가_발생한다`
- RED 검증 기준: 7개 모두 `ImportError`(`load_checkpoint`/`predict_curves` 부재, 오타 아님)로 실패해야 한다.
- 완료 조건(REVIEW 종료 시): 위 7개 + 기존 67개 통과, Step 0 smoke의 실제 PaiNN ckpt로 로드→곡선 반환 확인(독립 검증),
  `common/`이 `spectrum.reconstruct`를 import만 하고 로직을 복사하지 않음.

- **완료 결과 (Step 2B 종료)**
  - RED(`823b6db`): 7개가 `ImportError: load_checkpoint`로 실패. GREEN(`390fa9b`): `common/inference.py`에
    `LoadedCheckpoint`/`load_checkpoint`/`predict_curves` 추가, 신규 7개 통과 + 전체 74개 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안(PaiNN만, 분자 조회/다른 모델 없음). 기존 `predict()` 본문은 변경 없음.
    - 라이센스 경계: `common/`은 `spectrum.reconstruct`를 import만 함(로직 복사 없음). `spectrum/` 변경 없음.
    - 실제 산출물 검증: Step 0에서 학습한 실제 PaiNN ckpt를 `load_checkpoint`(mean/std 재계산 포함)로 로드해 test split 103개를
      예측 → 학습 스크립트가 저장한 `pred.csv`와 파라미터 최대 절대차 1.9e-6(값 범위 약 38, float32 오차 수준),
      `p_spec.csv`와 곡선 최대 절대차 3.7e-6(CSV 6자리 반올림 + 파라미터 오차 전파). 분자 순서 동일, 곡선 (103, 800) 유한.
    - 성능: 재계산 포함 `load_checkpoint` 0.20초, `norm_factor` 직접 지정 시 0.01초.
    - 리팩토링: 코드 변경 없음. 단, 이번 변경으로 틀려진 모듈 docstring("곡선 복원은 호출자에게 남김")만 현재 동작에 맞게 고쳤다
      (주석 변경만, 동작 영향 없음).
  - Step 3~4로 넘기는 사항 (주의)
    - `load_checkpoint`의 mean/std 재계산(`load_dataset_splits`)은 ckpt `args`의 상대 경로(`data_path='IrDB'`,
      `split_index_npz`)를 쓰므로 **작업 디렉터리가 저장소 루트여야 한다**(다른 곳에서는 `FileNotFoundError`). Step 3/4의 tool은 루트를
      명시적으로 지정(chdir 또는 경로 인자)해야 한다.
    - `load_dataset_splits`는 내부에서 전역 `torch.manual_seed`/`np.random.seed`를 `args.seed`로 재설정한다(기존 동작).
      재계산 경로를 호출하면 호출자의 전역 난수 상태가 바뀐다 — 필요하면 tool 쪽에서 `norm_factor`를 캐시해 직접 넘긴다.
    - 분자 ID → 배치 변환(IrDB 조회, `DataLoader`)은 Step 3에서 구현한다(이번 검증에서 쓴 방식:
      `load_dataset_splits(args)`의 test_dataset + `torch_geometric.loader.DataLoader`).

### Step 2C. Equiformer: 체크포인트 로드 (`common/inference.py` 확장)
- 목표: Equiformer 체크포인트에서 모델을 복원하고 `load_checkpoint`/`predict_curves`가 PaiNN과 똑같이 동작하게 한다 (G2).
- 조사 근거
  - Equiformer ckpt(`Step 0` 실측 ckpt 확인): `{'model', 'optimizer', 'args'}`, `args`에 `model_name, input_irreps(None), radius, num_basis,
    out_channels, drop_path, targets, standardize, spectrum_type, lineshape, beta, data_path, split_index_npz, seed` +
    **`task_mean`/`task_std`(Tensor)**가 들어 있다(`train_Equiformer.py`가 `args.task_mean/std`를 설정한 뒤 저장). `atomref`는 없음.
  - Equiformer 모델은 `task_mean/std`를 속성으로 저장만 하고 forward에서 쓰지 않는다 → 정규화는 PaiNN처럼 모델 밖
    (`predict()`)에서 적용된다(이중 역정규화 없음).
  - `equiformer_adapter.build(args)`가 `getattr(args, "task_mean", None)` 등을 읽으므로 ckpt `args`로 그대로 재구성 가능.
  - 모델 1개 생성에 약 8초가 걸린다 → 테스트는 로드 횟수를 최소화(로드한 모델을 테스트 간에 재사용).
- 범위
  - 포함 (`common/inference.py`)
    - `_CHECKPOINT_BUILDERS`에 `"Equiformer": equiformer_adapter.build` 등록.
    - `_restore_norm_factor` 우선순위 변경: ckpt `args`에 `task_mean`/`task_std`가 **있으면 그 값을 사용**(`float32` CPU 텐서로 변환,
      데이터셋 접근 없음) → 없으면 기존 방식(`standardize` False면 `[0, 1]`, True면 `load_dataset_splits`로 재계산).
      PaiNN ckpt `args`에는 이 속성이 없으므로 PaiNN 동작은 그대로(2B 테스트 7개가 회귀 오라클).
    - 오류 메시지의 지원 목록과 `load_checkpoint` docstring을 "PaiNN, Equiformer"로 갱신.
  - 미포함: Geoformer(2D), 분자 조회(Step 3), `Equiformer/` 모델 코드와 `train_Equiformer.py` 수정, `predict()` 수정.
- 테스트 계획 (`tests/refactor/test_inference_checkpoint_equiformer.py`; 기본 크기 Equiformer에 `num_basis=8`로 저장한 ckpt 1개를
  모듈 범위에서 만들고, 로드는 캐시해 한 번만 수행. 새 동작 호출은 각 테스트 안에서 하여 개별 실패로 확인)
  1. `test_Equiformer_체크포인트를_로드하면_원본_모델과_같은_출력을_낸다` (state_dict 복원 + `eval()`, `predict()` 수치 일치)
  2. `test_Equiformer는_체크포인트_args의_task_mean_std를_norm_factor로_쓴다` (ckpt에 `standardize=True`와 존재하지 않는 split 경로를
     넣어, 데이터셋을 읽지 않고 저장된 값을 쓴다는 것을 함께 검증)
  3. `test_Equiformer_predict_curves는_역정규화한_파라미터로_곡선을_복원한다` (shape `(2, 800)`, 유한값, 수동 계산과 일치;
     ckpt `args.task_mean/std`에 물리적으로 말이 되는 값을 저장)
- RED 검증 기준: 3개 모두 `ValueError: Unsupported base_model: 'Equiformer'`(기능 부재)로 실패. 기존 74개는 영향 없이 통과.
- 완료 조건(REVIEW 종료 시): 위 3개 + 기존 74개 통과, Step 0 smoke의 실제 Equiformer ckpt(`train_Equiformer`가 만든 것, 기본 모델 크기)로
  test split 103개를 예측해 학습 스크립트의 `pred.csv`/`p_spec.csv`와 비교, `Equiformer/`·`train_Equiformer.py` 변경 없음.

- **완료 결과 (Step 2C 종료)**
  - RED(`25056f7`): 3개가 `ValueError: Unsupported base_model: 'Equiformer'`로 실패. GREEN(`ff6290a`): `_CHECKPOINT_BUILDERS`에
    Equiformer 등록 + `_restore_norm_factor`가 ckpt `args`의 `task_mean/task_std`를 우선 사용. 신규 3개 + PaiNN 2B 7개 통과, 전체 77개 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안(Equiformer 등록 + norm_factor 우선순위). `Equiformer/`, `train_Equiformer.py`, `predict()`, `spectrum/` 변경 없음
      (GREEN 커밋의 변경 파일은 `common/inference.py` 하나).
    - 실제 산출물 검증: Step 0에서 `train_Equiformer`로 학습한 실제 ckpt(기본 모델 크기, 약 3.3M 파라미터)를 로드해 test split 103개 예측.
      - ckpt `args`의 `task_mean/std`가 학습 split 재계산 값과 정확히 일치(데이터셋을 읽지 않고도 같은 값). dtype float32, CPU.
      - 파라미터 vs `pred.csv`: 최대 절대차 6.6e-7(값 범위 약 1.6). 분자 순서 동일.
      - 곡선 vs `p_spec.csv`: 최대 절대차 1.4e-5. 같은 파라미터(`pred.csv`)로 `reconstruct_spectrum`하면 `p_spec.csv`와 5.4e-7(CSV 반올림 수준)
        까지 일치하고, 파라미터에 ±6.6e-7 섭동만 줘도 곡선이 최대 8.3e-5 변하므로(선폭 C 최솟값 0.036) 관측된 차이는 float32 파라미터 오차가
        곡선에서 증폭된 정상 범위다. 곡선 (103, 800), 유한.
    - 성능(CPU): `load_checkpoint` 6.6초(모델 생성이 대부분), 103개 forward 21.8초(분자당 약 0.2초).
    - 리팩토링: 제안 없음(코드 변경 없음).
  - Step 3~4로 넘기는 사항: 예측 tool은 `LoadedCheckpoint`를 **캐시**해 요청마다 모델을 다시 만들지 않아야 한다(Equiformer는 로드 약 7초).
    단일 분자 예측은 forward 약 0.2초 수준이다. 정규화 값 출처: PaiNN은 학습 split 재계산(0.2초, 전역 난수 시드 재설정 부작용 — 2B 기록 참고),
    Equiformer는 ckpt `args`에 저장된 값.

### Step 2D. Geoformer: Lightning 체크포인트 로드 (`common/inference.py` 확장)
- 목표: Geoformer(Lightning) 체크포인트에서 모델을 복원하고 `load_checkpoint`/`predict_curves`가 세 백본 모두에서 같은 방식으로 동작하게 한다 (G2).
- 조사 근거 (`train_Geoformer`를 CPU에서 tiny 설정으로 실행해 만든 **실제** ckpt 확인: `--embedding-dim 8 --ffn-embedding-dim 16 --num-layers 1
  --num-heads 2 --num-rbf 8 --num-steps 4`, 12초)
  - ckpt 최상위 키: `epoch, global_step, pytorch-lightning_version, state_dict, loops, callbacks, optimizer_states, lr_schedulers, hparams_name,
    hyper_parameters, datamodule_hyper_parameters`. `epoch=N-val_loss=X.ckpt`와 `last.ckpt`가 같은 구조.
  - `hyper_parameters`는 `Namespace`가 아니라 **일반 dict**(66개 키): `max_z, embedding_dim, ffn_embedding_dim, num_layers, num_heads, cutoff, num_rbf,
    trainable_rbf, norm_type, decoder_type, aggr, dataset_root, dataset_arg, prior_model, num_classes, pad_token_id, mean, std, lineshape, beta, n_mode,
    standardize, splits` 등 — `geoformer_adapter.build`(=`create_model`)가 읽는 필드가 모두 있다.
  - **`spectrum_type`이 없고 `spec_loss_type`('FC' 등)이 그 역할**을 한다. `predict_curves`는 `args.spectrum_type`을 읽으므로 매핑이 필요하다.
  - `state_dict` 키는 전부 `model.` 접두사(LNNP의 `self.model`)이며 `model.mean`, `model.std` 버퍼가 포함된다 → 접두사를 제거해 `GeoformerForEnergyRegression`에 로드.
  - 모델이 `logits * std + mean`을 내부에서 적용해 **출력이 이미 역정규화**되어 있다 → `norm_factor`는 항상 `[0, 1]`. 현재 `_restore_norm_factor`는
    `standardize=True`면 학습 split으로 재계산하므로 Geoformer에 그대로 쓰면 이중 역정규화가 된다.
- 범위
  - 포함 (`common/inference.py`)
    - 모델별 로드 방식을 분리: PaiNN/Equiformer는 기존(`checkpoint['args']`, `checkpoint['model']`), Geoformer는
      `hyper_parameters`(dict)를 속성 접근이 되는 args로 바꾸고 `spectrum_type = spec_loss_type`을 채우며,
      `state_dict`에서 `model.` 접두사를 제거해 `geoformer_adapter.build(args)`로 만든 모델에 로드(`eval()`).
    - Geoformer의 기본 `norm_factor`는 `[zeros(len(dataset_arg)), ones(len(dataset_arg))]`(데이터셋 접근 없음). 호출자가 `norm_factor`를 직접
      넘기면 그대로 사용(기존 규칙).
    - 지원 목록·docstring 갱신.
  - 미포함: 분자 조회(Step 3), 어떤 ckpt(`last` vs best)를 쓸지 선택하는 정책(Step 3/registry 쪽 결정), `geoformer/`·`train_Geoformer.py`·`predict()` 수정,
    `Equiformer`/`PaiNN` 동작 변경(2B/2C 테스트 10개가 회귀 오라클).
- 테스트 계획 (`tests/refactor/test_inference_checkpoint_geoformer.py`; tiny Geoformer를 실제 Lightning ckpt 구조(`state_dict`의 `model.` 접두사 +
  `hyper_parameters` dict)로 `tmp_path`에 저장. 모델 `mean/std` 버퍼에 물리적으로 말이 되는 값을 주어 NaN 방지. 새 동작 호출은 각 테스트 안에서)
  1. `test_Geoformer_Lightning_체크포인트를_로드하면_원본_모델과_같은_출력을_낸다` (접두사 제거 + `eval()`, `predict()`를 `[0, 1]`로 호출한 결과가 일치)
  2. `test_Geoformer는_모델이_이미_역정규화하므로_norm_factor가_0과_1이다` (ckpt에 `standardize=True`와 존재하지 않는 split 경로를 넣어 데이터셋을 읽지 않음도 검증)
  3. `test_Geoformer_hyper_parameters는_속성_접근이_가능한_args로_복원되고_spectrum_type은_spec_loss_type에서_온다`
  4. `test_Geoformer_predict_curves는_모델이_출력한_파라미터로_곡선을_복원한다` (shape `(2, 800)`, 유한값, 원본 모델 출력 → `reconstruct_spectrum` 결과와 일치)
  5. `test_Geoformer_norm_factor를_직접_주면_그대로_사용한다` (명시적 지정 우선 규칙 확인)
- RED 검증 기준: 5개 모두 `ValueError: Unsupported base_model: 'Geoformer'`로 실패. 기존 77개는 통과.
- 완료 조건(REVIEW 종료 시): 신규 5개 + 기존 77개 통과, 실제 Geoformer ckpt(위 tiny 학습 산출물, `epoch=*.ckpt`와 `last.ckpt` 모두)를 로드해 test split 103개를
  예측하고 학습 스크립트가 만든 `p.csv`/`p_spec.csv`와 비교, `geoformer/`·`train_Geoformer.py` 변경 없음.

- **완료 결과 (Step 2D 종료)**
  - RED(`5481b3a`): 5개가 `ValueError: Unsupported base_model: 'Geoformer'`로 실패. GREEN(`ea91ef8`): 모델별 로더(`_CHECKPOINT_LOADERS`)로 분리하고
    Geoformer 추가(`hyper_parameters` dict → `Namespace`, `spec_loss_type` → `spectrum_type`, `model.` 접두사 제거, 기본 `norm_factor` `[0, 1]`).
    신규 5개 + 2B/2C 10개 통과, 전체 82개 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안. `geoformer/`, `train_Geoformer.py`, `predict()`, `spectrum/` 변경 없음(GREEN 커밋의 변경 파일은 `common/inference.py` 하나).
      PaiNN/Equiformer는 로드 경로를 함수로 감쌌을 뿐 로직 동일(2B/2C 테스트 10개가 오라클).
    - 실제 산출물 검증: `train_Geoformer`를 CPU tiny 설정으로 실행해 만든 **실제** Lightning ckpt(`last.ckpt`, `epoch=0-val_loss=12.1353.ckpt` 둘 다)를
      로드해 test split 103개를 `GeoformerDataCollator`로 배치화해 예측.
      - 분자 순서 동일, `strict=True` 로드 성공, `spectrum_type='FC'`, `norm_factor=[0, 1]`(모델 내부 역정규화).
      - 파라미터 vs `p.csv`: 최대 절대차 3.0e-8(값 범위 2.2, 사실상 비트 동일). 곡선 vs `p_spec.csv`: 최대 절대차 6.0e-7(CSV 6자리 반올림 수준).
        곡선 (103, 800), 유한. 두 ckpt 결과 동일(4 step 학습이라 last와 best가 같은 가중치).
    - 리팩토링: 제안 없음(코드 변경 없음).
  - Step 3으로 넘기는 사항
    - registry는 Geoformer에 대해 `last.ckpt`와 `epoch=*.ckpt`를 모두 반환한다. 예측에 어느 것을 쓸지(`last` vs val_loss 최소)는 아직 정해지지 않았다 —
      Step 3의 예측 tool에서 정책을 정한다(기본 제안: 수정 시각 최신 = `get_latest_checkpoint`).
    - 이번 실측은 tiny 모델(로드 0.01초)이다. 기본 크기 Geoformer(9레이어, 256차원)의 로드/추론 시간은 아직 측정하지 않았다.
    - 세 백본 모두 `load_checkpoint`/`predict_curves`로 같은 방식으로 호출할 수 있고, 입력 배치만 다르다(PaiNN/Equiformer: PyG `Batch`, Geoformer:
      `GeoformerDataCollator` dict) — 분자 ID → 배치 변환은 모델별로 Step 3에서 구현한다.
- **Step 2 전체 완료**: 2A(곡선 복원) → 2B(PaiNN) → 2C(Equiformer) → 2D(Geoformer). 전체 82개 테스트 통과.

## Step 3. 예측 tool (신규 인터페이스) — 3A~3B로 분할
- 목표: IrDB 분자 ID → 학습된 체크포인트로 예측한 스펙트럼 곡선을 **구조화된 dict**로 반환 (G2). 체크포인트가 없으면 "학습 필요" 응답.
- 분할(사용자 확정): 각 하위 Step마다 RED/GREEN/REVIEW 세 번 커밋(제목 `Step 3A RED: ...`). **IrDB에 대해 확실하게 동작하는 것을 목표**로 하며,
  다른 데이터셋(`IrDB_uff`/`IrDB_murcko`/`PtDB`)은 로컬에 `processed`가 없어 이번 범위에서 검증하지 않는다.
  3A 모델별 입력 배치 구성 → 3B 예측 tool(+분자 목록 tool).
- 조사 근거
  - IrDB는 분자 1024개, 분자 ID는 `Data.name`(예: `cn1_cn1_nn1`)이며 중복 없음, 데이터셋 로드 0.03초 → 조회 비용 부담 없음.
  - 모델별 입력 배치: PaiNN/Equiformer는 PyG `Batch`. Geoformer는 `GeoformerDataCollator`가 만드는 dict이며, 실제 학습 파이프라인(`geoformer/data.py`)은
    PyG `Data` 객체 목록을 이 콜레이터에 **그대로** 넘긴다(콜레이터가 `z, pos, y, spec_x, spec_y, name`을 읽음 — IrDB `Data`에 모두 있음).
  - `load_checkpoint`(2B~2D)의 mean/std 재계산과 데이터셋 접근은 **작업 디렉터리가 저장소 루트**여야 한다(2B 기록). Equiformer 로드 약 7초 → 캐시 필요(2C 기록).
  - 곡선 800점은 LLM 컨텍스트에 넣기에 크다 → Step 3의 함수는 전체 곡선을 반환하고(UI·테스트용), Step 5에서 Agent용으로 요약만 돌려주는 얇은 래퍼를 둔다.

### Step 3A. 모델별 입력 배치 구성 (`common/inference.py` 확장)
- 목표: 데이터셋의 `Data` 목록을 각 백본이 기대하는 배치로 만든다(예측 tool이 모델 종류를 몰라도 되게).
- 범위
  - 포함: `build_batch(base_model, data_list)` — PaiNN/Equiformer는 `torch_geometric.data.Batch.from_data_list(data_list)`, Geoformer는
    `GeoformerDataCollator(max_nodes=None)(data_list)`(학습 파이프라인과 동일한 호출). 미지원 모델은 `ValueError`(지원 목록 포함).
  - 미포함: 데이터셋 조회·분자 ID 처리(3B), 체크포인트 로드/예측(2B~2D에서 완료), `predict()`·`geoformer/` 수정.
- 테스트 계획 (`tests/refactor/test_inference_build_batch.py`; 새 함수 import는 각 테스트 안에서 하여 개별 실패로 확인)
  1. `test_PaiNN과_Equiformer_배치는_PyG_Batch로_묶인다` (`make_tiny_pyg_batch().to_data_list()` → 원본 배치와 `pos`/`z`/`batch` 일치; parametrize로 두 모델 모두)
  2. `test_Geoformer_배치는_z와_pos를_가진_dict이다` (원본 `make_tiny_geoformer_batch`와 `z`/`pos` 일치)
  3. `test_실제_IrDB_분자로_세_모델의_배치를_만들_수_있다` (실제 `IrDB` 분자 2개: PyG 배치의 그래프 수 2·노드 수 합, Geoformer 배치의 `z` 행 수 2와 `pos` 마지막 축 3)
  4. `test_지원하지_않는_모델이면_ValueError가_발생한다`
  (pytest 케이스 수: 1번이 parametrize로 2개 → 총 5개)
- RED 검증 기준: 5개 모두 `ImportError: cannot import name 'build_batch'`로 실패. 기존 82개는 영향 없이 통과.
- 완료 조건(REVIEW 종료 시): 신규 5개 + 기존 82개 통과, Step 2에서 쓴 실제 체크포인트 3종과 `build_batch`로 만든 IrDB 배치로 예측해 학습 스크립트 결과와
  일치(PaiNN/Equiformer는 PyG `DataLoader` 배치와, Geoformer는 콜레이터 배치와 같은 결과).

- **완료 결과 (Step 3A 종료)**
  - RED(`b0a6741`): 5개가 `ImportError: cannot import name 'build_batch'`로 실패. GREEN(`235128c`): `common/inference.py`에 `build_batch` 추가(+15줄),
    신규 5개 통과 + 전체 87개 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안(`build_batch`만). `predict()`/`load_checkpoint`/`geoformer/`/`spectrum/` 변경 없음(GREEN 커밋의 변경 파일은 `common/inference.py` 하나).
    - 실제 산출물 검증: 실제 체크포인트 3종(PaiNN `painn_tiny`, Equiformer `eq_tiny`, Geoformer `geo_tiny/last.ckpt`)을 로드하고, IrDB test split 103개를
      `build_batch`로 배치화해 예측 → 학습 스크립트의 `pred.csv`/`p.csv`·`p_spec.csv`와 비교. 세 모델 모두 분자 순서 동일, 곡선 (103, 800) 유한.
      파라미터 최대 절대차 PaiNN 1.9e-6 / Equiformer 6.6e-7 / Geoformer 3.0e-8, 곡선 최대 절대차 3.7e-6 / 1.4e-5 / 6.0e-7 — Step 2B~2D에서 PyG `DataLoader`/콜레이터로
      만든 배치로 측정한 값과 동일(`build_batch`가 기존 경로와 같은 배치를 만든다는 증거).
    - 배치 독립성(3B 단일 분자 예측의 전제): 원자 수가 다른 분자(37/69/49/51) 4개를 한 배치로 예측한 결과와 각 분자를 단독 배치로 예측한 결과가 일치 —
      최대 절대차 PaiNN 1.9e-6 / Equiformer 1.1e-6(float32 오차 수준) / Geoformer 0(패딩 영향 없음).
    - 성능(CPU): `build_batch` 0.01초 이하. 103개 예측 PaiNN 0.1초 / Geoformer(tiny) 0.1초 / Equiformer 19.8초.
    - 리팩토링: 제안 없음(코드 변경 없음).
  - Step 3B로 넘기는 사항: 단일 분자 예측은 `build_batch(model, [data])`로 충분하다. 빈 `data_list`는 `build_batch`에서 처리하지 않으므로(PyG/콜레이터 오류) 3B가 분자 조회 결과를
    확인한 뒤에만 호출해야 한다.

### Step 3B. 예측 tool + 분자 목록 tool (`agent/tools/predict_tool.py`)
- 목표: IrDB 분자 ID → 예측 곡선 dict, 분자 ID 조회 (G2).
- 범위
  - 포함 (`agent/tools/predict_tool.py`)
    - `predict_spectrum(molecule_id, base_model=None, *, results_root=저장소루트, project_root=저장소루트) -> dict`
      - 체크포인트 선택: `base_model` 지정 시 해당 모델 중 최신, 미지정 시 전 모델 중 최신(`registry.get_latest_checkpoint`; Geoformer의 `last` vs best 정책은
        "수정 시각 최신"으로 확정).
      - 응답(JSON 직렬화 가능한 dict, 예외 대신 상태값으로 반환 — LLM tool 결과로 쓰기 위함)
        - `{"status": "ok", "molecule_id", "base_model", "checkpoint", "spectrum_type", "wavelength_nm": [800], "intensity": [800], "peak_wavelength_nm"}`
        - `{"status": "needs_training", "requested_base_model": 모델 또는 None, "message"}` (해당 조건의 체크포인트가 없음)
        - `{"status": "error", "error": "unsupported_base_model", "supported": [...], "message"}`
        - `{"status": "error", "error": "unknown_molecule", "molecule_id", "message"}`
      - 로드한 `LoadedCheckpoint`는 `(경로, 수정 시각)` 키로 캐시해 같은 체크포인트로 재예측 시 다시 로드하지 않는다.
      - 호출 동안 작업 디렉터리를 `project_root`로 바꾸고(`contextlib.chdir`) 끝나면 복원한다 — 호출자의 cwd와 무관하게 동작.
      - 데이터셋은 ckpt `args.data_path`를 따른다(이번 범위에서 검증하는 것은 `IrDB`). 분자 배치는 3A의 `build_batch`로 만들고, 곡선은 `common.inference.predict_curves`로 계산.
    - `list_molecules(query="", limit=20, *, project_root=저장소루트) -> dict` — 사용자가 유효한 분자 ID를 고를 수 있게 하는 조회 tool:
      `{"status": "ok", "total_matches", "molecule_ids": [...]}` (대소문자 무시 부분일치), `limit`이 1~100을 벗어나면 `{"status": "error", ...}`.
  - 미포함: AGNO 등록과 Agent용 요약 래퍼(Step 5), 학습 tool(Step 4), UI 곡선 plot과 실험 스펙트럼 비교(Step 6; 데이터에 `spec_x/spec_y`가 있어 가능),
    SMILES/SDF 입력(PRD 비목표), `common/data.py` 수정(데이터셋 클래스 선택은 6줄이라 tool 안에서 같은 규칙으로 처리 — 중복은 REVIEW에서 정리 후보로만 기록),
    Equiformer 실제 예측 테스트(로드는 2C에서 검증, 느림), 스레드 안전성(`chdir`는 전역 상태이므로 동시 호출은 지원하지 않음 — 문서화).
- 테스트 계획 (`tests/agent_tools/test_predict_tool.py`; tiny PaiNN/Geoformer 체크포인트를 `tmp_path/results_*`에 저장해 사용. PaiNN은 ckpt `args`를
  `standardize=True`와 실제 split으로 저장해 평균/표준편차를 재계산하게 하고, Geoformer는 모델 `mean/std` 버퍼에 물리적인 값을 주어 곡선이 유한하도록 한다)
  1. `test_학습된_모델이_없으면_needs_training을_반환한다`
  2. `test_모델을_지정했는데_해당_모델만_없으면_needs_training을_반환한다`
  3. `test_지원하지_않는_모델을_지정하면_error를_반환한다`
  4. `test_PaiNN_체크포인트로_IrDB_분자의_스펙트럼을_예측한다` (status ok, 곡선 800점, 파장 400~799.5, 유한값, 최댓값 1, 피크 파장이 격자 위의 값, JSON 직렬화 가능)
  5. `test_예측_곡선은_직접_계산한_값과_일치한다` (`common.inference`와 데이터셋에서 직접 만든 배치로 계산한 곡선과 일치 — 분자가 맞게 선택되는지 검증)
  6. `test_Geoformer_체크포인트로도_예측한다`
  7. `test_모델을_지정하지_않으면_가장_최근_체크포인트의_모델을_쓴다` (`os.utime`으로 수정 시각을 달리한 두 경우)
  8. `test_존재하지_않는_분자_ID는_error를_반환한다`
  9. `test_같은_체크포인트로_다시_예측하면_모델을_다시_로드하지_않는다` (`load_checkpoint` 호출 횟수 계측)
  10. `test_작업_디렉터리가_달라도_예측하고_원래_디렉터리로_복원한다`
  11. `test_분자_ID_목록을_limit만큼_반환한다`
  12. `test_query로_분자_ID를_대소문자_무시하고_부분일치_검색한다`
  13. `test_limit이_범위를_벗어나면_error를_반환한다`
- RED 검증 기준: 13개 모두 `agent.tools.predict_tool` 부재로 실패(12개는 `ModuleNotFoundError`, 캐시 테스트 1개는 `from agent.tools import predict_tool`의 `ImportError`).
- 완료 조건(REVIEW 종료 시): 신규 13개 + 기존 테스트 통과, Step 0/2에서 만든 **실제 체크포인트 3종**(PaiNN/Equiformer/Geoformer)으로 `predict_spectrum`을 호출해
  학습 스크립트가 만든 같은 분자의 곡선(`p_spec.csv`)과 비교, `agent/`가 곡선 복원 로직을 직접 갖지 않고 `common.inference`를 통해 사용
  (파장 격자만 `spectrum.reconstruct.wavelength_grid_nm` import 허용), 기존 학습 코드 변경 없음.

- **완료 결과 (Step 3B 종료)**
  - RED(`Step 3B RED` 커밋): 13개가 `agent.tools.predict_tool` 부재로 실패. GREEN(`e342f89`): `agent/tools/predict_tool.py` 신규(113줄) — `predict_spectrum`,
    `list_molecules`. 신규 13개 통과 + 전체 100개 통과(REVIEW에서 테스트 1개 추가 → 14개, 전체 101개).
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안. 기존 파일 변경 없음(GREEN 커밋은 신규 파일 1개). `agent/`가 `spectrum`에서 가져오는 것은 `wavelength_grid_nm`뿐이고 곡선 복원은
      `common.inference`를 통해서만 수행(라이센스 경계 유지).
    - 실제 산출물 검증: 실제 체크포인트 3종을 `results_*` 구조로 놓고(저장소 밖 임시 디렉터리), **작업 디렉터리를 저장소 밖으로 둔 채** `predict_spectrum`을
      분자 3개씩(test split의 처음/중간/마지막) 호출해 학습 스크립트의 `p_spec.csv` 곡선과 비교.
      곡선 최대 절대차: PaiNN 5.0e-7~4.1e-6, Equiformer 5.0e-7~1.5e-6, Geoformer 5.0e-7(CSV 6자리 반올림 + float32 오차 수준). 호출 후 작업 디렉터리 복원 확인.
    - 성능(CPU, 캐시 적용): 첫 호출(모델 로드 포함) PaiNN 0.4초 / Equiformer 7.7초 / Geoformer(tiny) 0.2초, 이후 호출 PaiNN 0.25초 / Equiformer 0.78초 / Geoformer 0.21초.
      `list_molecules` 0.18초. 호출당 약 0.1~0.2초는 분자 이름 목록(1024개 순회) 생성 비용.
    - 모델 미지정 시 가장 최근 체크포인트의 모델(Geoformer, 수정 시각 최신)을 선택. 저장소 기본 `results_root`(현재 학습 산출물 없음)에서는 `needs_training`.
    - 응답 크기: 한 번의 `ok` 결과 JSON이 약 19~22KB(곡선 800점 ×2) → LLM 컨텍스트에는 너무 크므로 Step 5의 Agent용 요약 래퍼가 필수(계획대로).
  - 리뷰에서 나온 개선 사항
    1. **(REVIEW에서 처리함)** 로드 캐시(`_loaded_checkpoint_cache`)에 크기 제한이 없어, 재학습으로 체크포인트 수정 시각이 바뀔 때마다 이전 모델이 메모리에 남았다
       (장시간 실행되는 UI 세션에서 누적). 사용자 승인 하에 REVIEW 안에서 TDD 미니 사이클로 처리: 테스트
       `test_재학습으로_체크포인트가_바뀌면_새로_로드하고_이전_모델은_메모리에서_해제한다`를 먼저 추가해 실패(이전 모델이 해제되지 않음)를 확인한 뒤,
       캐시를 `base_model -> (키, LoadedCheckpoint)`로 바꿔 모델 종류마다 최신 1개만 유지하도록 수정(`_load_cached` 약 6줄). 이후 신규 포함 14개 + 전체 101개 통과.
  - 남은 개선 후보 (코드 변경 없이 기록만, 진행 여부는 사용자가 결정)
    2. 데이터셋 클래스 선택(`IrDB`/`PtDB` 분기, 6줄)이 `common/data.py`의 `load_dataset_splits`와 중복된다. 제안: `common/data.py`로 순수 추출 후 양쪽에서 사용
       (`tests/refactor/test_common_data.py`가 오라클). 구조 변경이므로 별도 커밋/사이클이 필요.
    3. 손상된 체크포인트 등 `load_checkpoint`의 예외는 현재 tool 밖으로 그대로 전파된다(`status` dict가 아님). Step 5에서 Agent 래퍼가 처리하거나, tool에
       `{"status": "error", "error": "checkpoint_load_failed"}`를 추가할 수 있다.
  - Step 4~5로 넘기는 사항: 학습 tool이 만드는 체크포인트 경로가 `registry`의 탐색 규칙(`results_{모델}/{seed}/{fold}/...`)과 일치해야 한다. Agent용 래퍼는 곡선을 요약
    (피크 파장, 필요 시 일부 샘플)해서 돌려준다.
- **Step 3 전체 완료**: 3A(`build_batch`) → 3B(`predict_spectrum`/`list_molecules`). 전체 101개 테스트 통과.

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
