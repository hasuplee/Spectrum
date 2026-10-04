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

## Step 4. 학습 tool (신규 인터페이스) — 4A~4B로 분할
- 목표: Agent/UI가 호출할 학습 tool — 기본값 조회, 파라미터 검증, 사용자 확인 후 백그라운드 학습 실행, 상태 조회 (G1).
- 분할: 각 하위 Step마다 RED/GREEN/REVIEW 세 번 커밋(제목 `Step 4A RED: ...`).
  4A 기본값 조회 + 요청 검증 + 명령 조립(프로세스 없음, 순수 함수) → 4B 백그라운드 실행 + 상태 조회(+실제 학습 smoke).
- 조사 근거
  - `train.py`가 실제로 노출하는 인자는 `--base-model/--spectrum-type/--batch-size/--data-path` 4개뿐이며(`batch-size` 기본 16), seed=0, fold=0으로 고정되어 하위 스크립트를
    `os.system("python -m train_X ...")`로 **블로킹 실행**한다. `build_command(args, i_seed, i_fold, split_npz)`는 순수 함수이고(CPU면 Geoformer에 `--accelerator cpu --ndevices 1` 추가),
    출력 경로는 `results_{모델}/{seed}/{fold}`로 Step 1 registry의 탐색 규칙과 일치한다. 단, 명령이 `python`으로 시작하므로 가상환경 인터프리터는 `sys.executable`로 바꿔야 한다.
  - 기본값의 출처가 모델마다 다르다. PaiNN/Equiformer는 `get_args_parser()`(add_help=False)가 부작용 없이 `parse_args([])`로 기본값을 돌려준다
    (PaiNN: lr 1e-3, train-steps 10000, eval-steps 100, embed-dim 512, num-layers 6, num-basis 128, workers 4. Equiformer: lr 5e-4, num-basis 128 …).
    Geoformer는 `get_args()`가 `sys.argv`를 파싱하고 `log_dir/input.yaml`을 쓰는 부작용이 있어 부적합 → 실제로 train.py가 쓰는 `geoformer/examples/{spectrum_type}.yml`을 읽는다
    (FC.yml: num_steps 10000, lr 2e-4, eval_every 100, embedding_dim 256, num_layers 9, num_workers 6 …). train.py는 `--batch-size`를 항상 넘기므로 batch_size 기본은 16.
  - 기본값을 코드에 복사하면 스크립트와 어긋날 수 있으므로 **스크립트/yml에서 읽는다**(테스트도 같은 출처와 비교).
  - Geoformer yml(`FC.yml`/`GMM.yml`/`Naive.yml`)에서 tool이 노출하는 값(`num_steps, eval_every, lr, num_workers`, 모델 크기)은 세 스펙트럼 종류에서 **모두 동일**하다
    (다른 것은 `spec_loss_type`, `num_classes`, `dataset_arg`뿐) → `get_training_defaults`는 `spectrum_type`과 무관하게 한 번만 읽는다. `spectrum_type`을 바꾸면 명령의
    `--conf` 경로(`{type}.yml`)만 달라진다.
  - CPU에서 기본 설정(10000 step, 512차원 6레이어)은 비현실적으로 오래 걸린다(Step 0 실측: Equiformer 기본 모델 2 step 56초). 그래서 사용자가 step 수와 모델 크기를 줄일 수 있어야 한다.
  - 제약 2(수치 불변): 스크립트의 기본값은 바꾸지 않는다. tool은 기본값을 **그대로 보여주고**, 사용자가 명시적으로 지정한 값만 CLI 인자로 덮어쓴다. 학습률 등 최적화 하이퍼파라미터는 읽기 전용(덮어쓰기 불가).

### Step 4A. 기본값 조회 + 요청 검증 + 명령 조립 (`agent/tools/train_tool.py`)
- 목표: 프로세스를 띄우지 않고, "어떤 설정으로 학습하는가"를 조회·검증·명령으로 조립한다.
- 범위
  - 포함
    - `get_training_defaults(base_model) -> dict`: `{"status": "ok", "base_model", "spectrum_type": "FC", "data_path": "IrDB", "batch_size": 16, "train_steps", "eval_steps",
      "workers", "learning_rate", "model_size": {...}, "seed": 0, "fold": 0, "device": "cpu"|"gpu"}`. 값은 스크립트 파서/yml에서 읽는다. 미지원 모델은
      `{"status": "error", "error": "unsupported_base_model", "supported": [...]}`.
    - `validate_training_request(base_model, overrides=None) -> dict`: 기본값 위에 overrides를 적용한 **최종 설정**을 반환
      (`{"status": "ok", "request": {...전체 설정...}}`) 또는 오류(`{"status": "error", "error": "unsupported_base_model"|"unknown_parameter"|"invalid_value", "parameter", "message"}`).
      덮어쓸 수 있는 파라미터(허용 목록): 공통 `spectrum_type`(Naive/GMM/FC), `data_path`(train.py의 선택지), `batch_size`, `train_steps`, `eval_steps`, `workers`(양의 정수, workers는 0 이상);
      모델 크기 — PaiNN `embed_dim, num_layers, num_basis` / Equiformer `num_basis` / Geoformer `embedding_dim, ffn_embedding_dim, num_layers, num_heads, num_rbf`. 그 외(lr, seed 등)는 `unknown_parameter`.
    - `build_training_command(request) -> list[str]`: `train.build_command`를 재사용(문자열을 `shlex.split`, 첫 토큰 `python`을 `sys.executable`로 교체)해 핵심 인자를 만들고, 기본값과 다른 override만
      모델별 CLI 플래그로 덧붙인다(PaiNN/Equiformer `--train-steps --eval-steps --workers --embed-dim --num-layers --num-basis`, Geoformer `--num-steps --eval-every --num-workers --embedding-dim
      --ffn-embedding-dim --num-layers --num-heads --num-rbf`).
    - `training_output_dir(base_model) -> str`: `results_{모델}/0/0` (Step 1 registry와 같은 규칙).
  - 미포함: 프로세스 실행/상태/로그/중복 실행 방지(4B), 사용자 확인 절차(4B), AGNO 등록(Step 5), `train.py`·`train_*.py` 수정, 학습률 등 하이퍼파라미터 덮어쓰기.
- 테스트 계획 (`tests/agent_tools/test_train_tool.py`; 새 함수 import는 각 테스트 안에서 하여 개별 실패로 확인)
  1. `test_PaiNN_기본값은_학습_스크립트의_기본값과_같다` (`train_PaiNN.get_args_parser().parse_args([])`와 `train.py` 기본값(batch 16, FC, IrDB)과 비교)
  2. `test_Equiformer_기본값은_학습_스크립트의_기본값과_같다`
  3. `test_Geoformer_기본값은_yml_설정과_같다` (`geoformer/examples/FC.yml`을 직접 읽어 비교, batch_size는 train.py 기본 16)
  4. `test_지원하지_않는_모델의_기본값_조회는_error를_반환한다`
  5. `test_override가_없으면_검증_결과는_기본값과_같다`
  6. `test_override는_기본값_위에_적용된다` (train_steps/eval_steps/모델 크기)
  7. `test_허용되지_않는_파라미터는_unknown_parameter_error를_반환한다` (예: `lr`, `seed`, Equiformer의 `embed_dim`)
  8. `test_잘못된_값은_invalid_value_error를_반환한다` (parametrize: `batch_size=0`, `train_steps="abc"`, `spectrum_type="XYZ"`, `data_path="Foo"`, `workers=-1`)
  9. `test_PaiNN_기본_명령은_학습_스크립트_모듈과_train_py_인자로_조립된다` (`[sys.executable, -m, train_PaiNN, --spectrum-type FC, --batch-size 16, --data-path IrDB, --output-dir results_PaiNN/0/0, --seed 0 ...]`)
  10. `test_Equiformer_기본_명령은_학습_스크립트_모듈과_train_py_인자로_조립된다`
  11. `test_Geoformer_기본_명령은_yml과_CPU_플래그를_포함한다` (`--conf geoformer/examples/FC.yml`, CPU 환경이면 `--accelerator cpu --ndevices 1`)
  12. `test_기본_명령의_핵심_인자는_train_py의_build_command와_같다` (드리프트 방지)
  13. `test_override는_모델별_CLI_플래그로_변환된다` (PaiNN/Equiformer와 Geoformer의 서로 다른 플래그 이름)
  14. `test_학습_출력_경로는_registry가_찾는_경로와_같다` (`training_output_dir`에 체크포인트 파일을 만들면 `find_checkpoints`가 찾음)
- RED 검증 기준: 테스트 함수 14개(parametrize 포함 26개 케이스) 모두 `ModuleNotFoundError: agent.tools.train_tool`로 실패. 기존 101개는 영향 없이 통과.
- 완료 조건(REVIEW 종료 시): 신규 테스트 + 기존 101개 통과, 실제 `train_*.py`로 만든 명령이 `--help`/인자 파싱 단계에서 거부되지 않음을 확인(세 모델의 argparse가 조립된 인자를 모두 수용),
  `train.py`·`train_*.py` 변경 없음, 기본값이 스크립트 값과 일치(복사본 없음).

- **완료 결과 (Step 4A 종료)**
  - RED(`49c25ba`): 26개 케이스가 `ModuleNotFoundError: agent.tools.train_tool`로 실패. GREEN(`f5b39d4`): `agent/tools/train_tool.py` 신규(184줄), 신규 26개 통과 + 전체 127개 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안(프로세스 실행/상태/확인 절차 없음). `train.py`·`train_*.py` 변경 없음(GREEN 커밋은 신규 파일 1개). 기본값은 스크립트/yml에서 읽고 복사본 없음.
    - 실제 학습 검증: 도구가 만든 명령을 **저장소 루트에서 그대로 실행**(워커 수는 기본값 그대로, step 수·모델 크기만 tiny로 override). 세 모델 모두 종료 코드 0이고
      체크포인트가 `training_output_dir`/registry의 탐색 경로에 생성됨 — PaiNN `results_PaiNN/0/0/checkpoint_best.ckpt`, Equiformer `results_Equiformer/0/0/checkpoint_best.ckpt`,
      Geoformer `results_Geoformer/0/0/checkpoints/{last,epoch=...}.ckpt`(registry의 `find_checkpoints`가 모두 찾음). 즉 argparse가 조립된 인자를 모두 수용하고 경로 규칙이 일치한다.
      (실행으로 만든 `results_*`는 확인 후 삭제함. 모두 `.gitignore` 대상.)
    - **워커 수 기본값 관찰 (4B 설계의 결정 사항)**: 같은 tiny 설정에서 기본 워커 수와 `workers=0`의 소요 시간이 크게 다르다.
      | 모델 | 기본 workers | 기본값 실행 | workers=0 실행(Step 0/2D 실측) |
      |---|---|---|---|
      | PaiNN (5 step) | 4 | 35.5초 | 약 10초(20 step) |
      | Geoformer (4 step) | 6 | **217.8초** | 12초 |
      | Equiformer (2 step) | 4 | 79.2초 | 56초 |
      Windows에서는 DataLoader 워커 프로세스 생성(spawn) 비용이 평가 때마다 반복되어 CPU 학습이 크게 느려진다(특히 Geoformer yml의 6 워커). 결과는 정상 종료하므로 동작 문제는 아니고
      성능 문제다.
    - 허용 파라미터 검증의 한계(의도된 것): 필드 간 조합(예: Geoformer에서 `embedding_dim`이 `num_heads`로 나누어떨어져야 함)은 사전 검증하지 않는다. 잘못된 조합은 학습 프로세스가
      실패하며, 4B의 상태 조회가 `failed`와 로그 tail로 알려준다.
    - 리팩토링: 제안 없음(코드 변경 없음). 참고: `_train_py_defaults`가 `sys.argv`를 잠시 바꾼다(호출 동안만, 스레드 안전하지 않음 — 문서화됨).
  - **4B 전에 결정할 사항 (사용자 확인 필요)**: CPU 환경에서 `workers` 기본값을 어떻게 할 것인가.
    - 안 A: 스크립트 기본값 유지(PaiNN 4, Geoformer 6) — 제약 2에 가장 엄격. 대신 CPU 사용자는 `workers=0`을 직접 지정해야 하고 Agent/UI가 이를 안내해야 한다.
    - 안 B: CPU일 때만 tool 기본 `workers`를 0으로 한다(`train.build_command`가 CPU일 때 Geoformer에 `--accelerator cpu`를 붙이는 것과 같은 성격의 "CPU 환경 적응").
      GPU에서는 스크립트 기본값 그대로. 워커 수는 학습 데이터 순서/수치에 영향을 주지 않는 런타임 설정이지만(제약 2에 명시된 값 목록에 없음), 기본값을 바꾸는 것이므로 승인이 필요하다.

### Step 4B. CPU 기본 워커 수 + 백그라운드 실행 + 상태 조회 (`agent/tools/train_tool.py` 확장)
- 목표: 사용자 확인을 거친 학습을 백그라운드로 실행하고 진행 상태를 조회한다 (G1). 그 전에 4A 리뷰에서 결정된 CPU 기본 워커 수 정책을 반영한다.
- 결정(사용자 승인, 안 B): 실제 구동은 GPU일 수 있으나 이번 계획의 개발·테스트는 처음부터 끝까지 CPU에서 진행한다. 4A 리뷰에서 기본 워커 수(PaiNN 4, Geoformer 6)로는 Windows CPU의
  tiny 학습이 `workers=0`보다 크게 느렸으므로(Geoformer 217.8초 vs 12초), **CPU일 때만** tool의 기본 `workers`를 0으로 한다. GPU에서는 스크립트 기본값을 그대로 쓴다.
  워커 수는 학습 수치(데이터 순서·값)에 영향이 없는 런타임 설정이며, `train.build_command`가 CPU일 때 `--accelerator cpu`를 붙이는 것과 같은 "CPU 환경 적응"이다(제약 2에 명시된 값 목록에 없음).
- 4A 코드 영향
  - `get_training_defaults`: `device == "cpu"`이면 `workers = 0`, `"gpu"`이면 스크립트 기본값(PaiNN/Equiformer `--workers`, Geoformer yml `num_workers`).
  - `build_training_command`: 덮어쓰기 플래그 여부를 **tool 기본값이 아니라 스크립트 기본값과 비교**해 결정한다(CPU에서는 `--workers 0`/`--num-workers 0`이 붙고, GPU에서는 붙지 않는다).
    그렇지 않으면 tool 기본값 0이 스크립트 기본값 4/6으로 조용히 되돌아간다.
  - 4A 테스트 중 `workers` 기본값과 "기본 명령에는 덮어쓰기 플래그가 없다"를 검증하던 부분을 새 명세에 맞게 수정하고, GPU 경로(`torch.cuda.is_available`을 True로 대체)를 검증하는 테스트를 추가한다.
- 범위
  - 포함
    - `start_training(base_model, overrides=None, *, confirmed=False, overwrite=False, project_root=저장소루트) -> dict`
      1. `validate_training_request` 결과가 오류이면 그대로 반환(프로세스 없음).
      2. 이미 실행 중인 작업이 있으면 `{"status": "busy", "job_id", "message"}` (한 번에 한 작업만).
      3. `confirmed=False`이면 실행하지 않고 `{"status": "needs_confirmation", "settings": 최종 설정, "output_dir", "will_overwrite": 기존 체크포인트 존재 여부, "message"}` —
         "기본값을 보여 주고 확인 후 학습" 절차를 tool 수준에서 강제한다.
      4. `confirmed=True`인데 해당 모델의 체크포인트가 이미 있고 `overwrite=False`이면 `{"status": "already_trained", "checkpoint", "message"}`.
      5. 그 외에는 시작: `overwrite=True`이면 기존 출력 디렉터리(`results_{모델}/0/0`)를 **삭제**한 뒤 시작한다(Geoformer의 `last.ckpt` 자동 이어 학습과 이전 체크포인트 혼입을 막고
         세 모델의 의미를 같게 하기 위함; `confirmed`와 `overwrite`를 둘 다 명시해야만 도달). `subprocess.Popen(명령, cwd=project_root, stdout/stderr → 로그 파일)`로 실행하고
         `{"status": "started", "job_id", "base_model", "command", "log_path", "output_dir", "settings"}`를 반환. 로그 파일은 `{project_root}/results_agent_logs/{job_id}.log`
         (`.gitignore`의 `results_*/`에 포함, registry 탐색 대상 아님). 자식 프로세스는 환경변수 `PYTHONUTF8=1`(Windows 기본 cp949로 기록되면 UTF-8로 읽을 수 없음 —
         실측 확인; 임시 경로의 한글도 깨짐)과 `PYTHONUNBUFFERED=1`(stdout이 파일이면 출력이 버퍼링되어 실행 중 로그가 보이지 않음 — 실측 확인)로 실행한다.
    - `get_training_status(job_id=None, *, tail_lines=20, project_root=저장소루트) -> dict`: `{"status": "ok", "job_id", "base_model", "state": "running"|"finished"|"failed", "return_code",
      "elapsed_seconds", "log_tail": [마지막 N줄], "log_path", "output_dir", "has_checkpoint", "settings"}` (`has_checkpoint`는 registry로 확인). `job_id` 생략 시 가장 최근 작업.
      작업이 없으면 `{"status": "no_job"}`, 모르는 `job_id`는 `{"status": "error", "error": "unknown_job"}`.
  - 미포함: 작업 중지/취소 tool, 여러 작업 큐, 프로세스 재시작 후 작업 복구(작업 목록은 프로세스 메모리에만 있음), AGNO 등록(Step 5), UI(Step 6), 학습 진행률 파싱(로그 tail만 제공),
    GPU 실제 실행 검증(개발·테스트는 CPU만).
  - 제약: 학습 프로세스는 `cwd=project_root` 기준 상대 경로(`IrDB/...`, `results_*`)를 쓰므로 실제 학습의 `project_root`는 저장소 루트여야 한다(테스트는 가짜 명령으로 임시 디렉터리를 쓴다).
- 테스트 계획
  - 4A 수정/추가 (`tests/agent_tools/test_train_tool.py`)
    - 수정: 세 모델의 기본값 테스트(CPU에서 `workers == 0`, GPU면 스크립트 기본값), 세 모델의 기본 명령 테스트(CPU에서 `--workers 0`/`--num-workers 0` 포함, GPU면 없음),
      `train_py` 드리프트 방지 테스트(핵심 인자는 동일하고 CPU면 workers 플래그만 추가).
    - 추가: `test_GPU_환경에서는_workers_기본값이_스크립트_기본값이고_workers_플래그가_없다` (`torch.cuda.is_available`을 True로 대체)
  - 4B (`tests/agent_tools/test_train_job.py`; 학습 명령은 `build_training_command`를 `[sys.executable, "-c", ...]`로 대체해 **실제 서브프로세스**를 쓴다. 작업 목록 `_jobs`는 테스트마다 초기화)
    1. `test_확인하지_않으면_실행하지_않고_needs_confirmation을_반환한다` (프로세스가 시작되지 않았음을 파일 흔적 부재로 확인, `settings`/`output_dir`/`will_overwrite`)
    2. `test_잘못된_요청은_검증_오류를_그대로_반환하고_실행하지_않는다`
    3. `test_확인하면_백그라운드로_시작하고_작업_디렉터리와_로그를_남긴다` (`started`, 로그 파일 생성, 프로세스의 cwd가 project_root, 로그에 출력 기록)
    4. `test_작업이_끝나면_finished_상태와_로그_tail을_알려준다`
    5. `test_작업이_실패하면_failed_상태와_return_code와_로그를_알려준다`
    6. `test_실행_중에_다시_시작하면_busy를_반환한다`
    7. `test_이미_학습된_체크포인트가_있으면_overwrite_없이는_already_trained를_반환한다` (`needs_confirmation`의 `will_overwrite`도 True)
    8. `test_overwrite를_주면_기존_출력_디렉터리를_지우고_시작한다`
    9. `test_학습이_체크포인트를_만들면_has_checkpoint가_True이다` (가짜 학습이 `training_output_dir`에 체크포인트 파일 생성 → registry 연동)
    10. `test_log_tail은_tail_lines만큼만_반환한다`
    10-1. `test_실행_중에도_로그_tail에서_진행_상황을_볼_수_있다` (flush 없는 한글 출력이 실행 중 `log_tail`에 보임 — 버퍼링 없음 + UTF-8)
    11. `test_작업이_없으면_no_job을_모르는_job_id는_unknown_job_error를_반환한다`
    12. `test_실제_PaiNN_tiny_학습이_체크포인트를_만들고_예측_tool로_이어진다` (`@pytest.mark.slow`; 실제 명령, CPU 기본 워커 0; 저장소 루트의 `results_PaiNN`이 이미 있으면 skip하고, 이 테스트가 만든 산출물만 정리)
- RED 검증 기준: 4A 수정/추가 중 새 명세에 의존하는 테스트와 4B 테스트가 모두 실패(4B는 `start_training`/`get_training_status` 부재). 새 명세와 무관한 4A 테스트와 기존 테스트는 통과.
- 완료 조건(REVIEW 종료 시): 신규/수정 테스트와 기존 테스트 통과(slow 포함), 도구로 CPU 기본 워커 수의 실제 학습(tiny)이 4A 리뷰의 기본값 실행보다 빠르게 끝남을 실측, 학습 후 `predict_spectrum`으로 예측까지 연결,
  `train.py`·`train_*.py` 변경 없음, 실행으로 만든 산출물 정리.

- **완료 결과 (Step 4B 종료)**
  - RED(`a74911d`): 4A 수정 테스트 9개 + 4B 신규 13개 실패. GREEN(`3790e47`): `agent/tools/train_tool.py` 확장(CPU 기본 workers 0, `start_training`, `get_training_status`),
    4A/4B 테스트 42개(slow 포함) 통과 + 전체 143개 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안. 작업 취소/큐/재시작 후 복구/진행률 파싱 없음. `train.py`·`train_*.py` 변경 없음(GREEN 커밋은 `train_tool.py` 하나).
    - 실제 학습 검증: 도구(`start_training`)로 세 모델을 **CPU 기본 워커(0)**로 학습 → 상태 조회 → 예측 tool까지 연결(저장소 루트에서 실행, 확인 후 산출물 삭제).
      | 모델 (tiny 설정) | 4A 리뷰: 스크립트 기본 워커 | 4B: CPU 기본 워커 0 |
      |---|---|---|
      | PaiNN (5 step) | 35.5초 | **8.0초** |
      | Geoformer (4 step) | 217.8초 | **8.5초** |
      | Equiformer (2 step) | 79.2초 | **52.2초** |
      세 모델 모두 `needs_confirmation`(`will_overwrite=False`) → `started` → 실행 중 로그 관찰됨 → `finished`(return code 0, `has_checkpoint=True`) → 같은 모델 재시작 시
      `already_trained` → `predict_spectrum` `ok`(곡선 800점)로 끝까지 동작. 시작 명령에 CPU 기본 `--workers 0`/`--num-workers 0`이 실제로 들어감.
    - 관찰(코드 변경 없음): Geoformer의 로그 tail은 Lightning tqdm 진행 막대 프레임이라 정보량이 낮다. Step 6 UI에서 진행 막대 줄을 걸러서 보여줄 수 있다.
  - 리팩토링 (사용자 지시로 REVIEW에서 한꺼번에 처리; 동작 불변, 테스트가 오라클)
    1. `train_tool._value_problem`의 쓰이지 않는 인자 `size_parameters` 제거.
    2. `build_training_command`의 "공통 파라미터 대 모델 크기 파라미터" 조회 분기를 `_setting(settings, parameter)` 헬퍼로 정리.
    3. `training_output_dir`가 경로 계산에 필요한 `train.py` 기본값만 읽도록 단순화(스크립트/yml 전체 파싱 제거).
    4. `start_training`(약 65줄)에서 확인 응답(`_needs_confirmation`)과 프로세스 실행(`_launch_job`)을 분리해 판단 로직만 남김(약 39줄).
    5. `predict_tool`과 `train_tool`에 중복되던 `PROJECT_ROOT`와 `unsupported_base_model` 응답을 `agent/tools/_shared.py`로 공통화.
    6. `predict_tool`의 데이터셋 클래스 선택(`IrDB`/`PtDB` 분기)이 `common/data.py`의 `load_dataset_splits`와 중복되던 것을 `common.data.build_dataset(data_path, targets)`로 추출해
       양쪽에서 사용(3B 리뷰의 개선 후보 2번). 기존 예외 타입/메시지는 그대로(`Exception("data_path must be start 'IrDB' or 'PtDB'")`). 새 공개 함수라 테스트를 먼저
       추가해 실패(`ImportError`)를 확인한 뒤 구현(`test_build_dataset은_data_path_접두사로_IrDB를_선택한다`).
    - 리팩토링 중 발견한 회귀: 6번에서 `list_molecules`를 `build_dataset(절대경로)`로 바꾸자 `build_dataset`이 `IrDB`로 *시작하는* 경로를 요구해 거부함(테스트가 즉시 포착).
      `predict_spectrum`과 같이 `project_root`로 `chdir`한 뒤 상대 이름 `"IrDB"`를 넘기도록 수정. 이후 미사용 import 없음(AST 점검), 전체 144개 통과.
  - 리뷰에서 검토했으나 변경하지 않은 것
    - `get_training_status(project_root=...)`는 작업이 시작될 때의 `project_root`를 기록해 두면 불필요한 인자이지만, 4B RED에서 승인된 인터페이스(테스트가 인자를 넘김)를 바꾸지 않았다.
    - 로그를 매 상태 조회마다 전체 읽기: 현재 로그 크기(최대 수백 KB 추정)에서는 문제없어 최적화하지 않았다(필요해지면 끝부분만 읽도록 개선).
    - `_script_defaults`가 Geoformer 기본값을 `FC.yml`에서만 읽음: 노출 파라미터는 FC/GMM/Naive가 동일함을 4A에서 확인(주석/Plan에 기록).
  - Step 5로 넘기는 사항
    - Agent 래퍼는 `predict_spectrum` 결과의 곡선(약 20KB)을 요약해야 하고(Step 3B 기록), `start_training`의 `needs_confirmation`을 사용자에게 보여 주고 확인을 받은 뒤에만
      `confirmed=True`로 호출해야 한다. `overwrite=True`는 기존 학습 결과를 **삭제**하므로 사용자에게 분명히 확인해야 한다.
    - 학습 tool 결과 중 `message`는 한국어이며 LLM이 그대로 사용자에게 전달할 수 있다. 오류는 모두 `status` dict(`busy`/`already_trained`/`error` 등)이므로 예외 처리가 필요 없다.
    - Geoformer 체크포인트 파일 이름(`epoch=0-val_loss=12.1353.ckpt`)에는 `=`가 있어 경로 인자로 쓸 때 주의(현재 코드는 경로를 인자 문자열 없이 `Path`로만 다룸).
- **Step 4 전체 완료**: 4A(기본값 조회/검증/명령 조립) → 4B(CPU 기본 워커, 확인 절차, 백그라운드 실행, 상태 조회). 전체 144개 테스트 통과.

## Step 5. AGNO Agent (신규 인터페이스) — 5A~5C로 분할
- 목표: Step 1~4의 tool을 사용하는 **학습/예측 전용** Agent. AGNO로 구현하고 LLM은 `agno.models.vllm.VLLM`으로 연결한다 (G3, G5).
  "그냥 학습해줘" → 기본값을 보여 주고 모델을 물은 뒤 사용자 확인 후 학습, 예측인데 학습된 모델이 없으면 학습을 먼저 제안, 학습/예측과 무관한 질문은 거절.
- 분할: 각 하위 Step마다 RED/GREEN/REVIEW 세 번 커밋(제목 `Step 5A RED: ...`). 5A 범위 가드 → 5B Agent용 tool 래퍼 → 5C Agent 조립 + 대화 시나리오 테스트.
- **LLM 방침(사용자 확정)**: 이 프로젝트의 개발·테스트 중에는 진짜 LLM/API 키 없이 **mock**(가짜 OpenAI 호환 서버)만 쓴다. 진짜 LLM 연결은 프로젝트 종료 후 내부 LLM API(URL, 토큰)를
  환경변수(`VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_MODEL`)만 바꿔 연결한다. 사용자가 이전에 AGNO `VLLM`을 `base_url`/API 키로 연결해 본 경험이 있어 연결 방식 자체는 검증된 것으로 본다.
  mock으로 확인되는 것은 연결·tool 실행·확인 절차·가드·다중 턴 기억이며, "LLM이 instructions를 잘 따르는가/tool을 올바르게 고르는가"는 확인할 수 없다(실제 LLM 연결 후 `vllm` 마커 선택
  테스트로 확인). 내부 LLM에서 미리 확인할 것은 **tool 호출(function calling) 지원 여부**(vLLM이면 `--enable-auto-tool-choice --tool-call-parser <모델별>`)이며, Step 7의 README에 연결 가이드로 포함한다.
- 조사 근거 (agno 3.1.1 소스 확인 + 로컬 실험, 실험 스크립트는 저장소 밖)
  - `VLLM(OpenAILike)`은 `api_key`가 없으면 환경변수 `VLLM_API_KEY`를 요구하고 없으면 `ModelAuthenticationError`를 낸다(인증이 없는 vLLM 서버에도 더미 키가 필요). `base_url`은 인자 또는
    `VLLM_BASE_URL`(기본 `http://localhost:8000/v1/`). → 우리 `build_model()`은 `VLLM_API_KEY`가 없으면 `"EMPTY"`를 쓴다.
  - **가드레일**: `Agent(pre_hooks=[BaseGuardrail 하위 클래스])`에서 `InputCheckError`를 던지면 `agent.run()`이 예외 없이 `RunOutput(status=error, content=<오류 메시지>)`를 반환하고
    **LLM 서버는 호출되지 않는다**(요청 수 0 확인). → 범위 가드를 AGNO 가드레일로 구현하면 "AGNO Agent의 일부"이면서 vLLM 서버 없이 테스트할 수 있다.
  - **테스트용 LLM**: 로컬 `http.server`로 만든 가짜 OpenAI 호환 서버에 **실제 `VLLM` 클래스**를 연결하면 tool 호출 흐름 전체가 돈다(요청 1: `tools`와 `instructions`(system 메시지) 포함 →
    tool 실행 → 요청 2: `tool` 결과 포함 → 최종 텍스트). 즉 mock 모델 클래스를 따로 만들지 않고도 `VLLM` 연결 경로까지 검증된다.
  - **대화 기억**: `add_history_to_context=True`만으로는 이전 턴이 다음 요청에 들어가지 않는다(DB 없음). `db=agno.db.in_memory.InMemoryDb()`를 붙이면 이전 `user`/`assistant` 턴이
    들어가고 `session_id`가 다르면 기억이 분리된다 → "그냥 학습해줘 → 모델 질문 → PaiNN으로" 같은 다중 턴에 필요. (프로세스 메모리 안에서만 유지)
  - 가드가 현재 메시지만 보면 후속 답변("응", "PaiNN", "취소")이 도메인 키워드가 없어 막힌다 → 가드는 짧은 후속 답변 허용 목록을 가져야 한다.
  - 한 번의 `predict_spectrum` 결과(곡선 800점×2, 약 20KB)는 LLM 컨텍스트에 부적합 → Agent용 tool은 요약만 돌려준다(3B/4B 기록).

### Step 5A. 범위 가드 + 가짜 LLM 서버 테스트 기반 (`agent/guard.py`, `tests/support/fake_openai.py`)
- 목표: 학습/예측과 무관한 질문을 **LLM 호출 전에** 거절한다 (G3).
- 범위
  - 포함
    - `agent/guard.py`
      - `is_in_scope(text) -> bool`: 휴리스틱(키워드 + 짧은 후속 답변 + 정의형 질문 제외). 판정 규칙:
        1. 빈 입력/공백만 → False.
        2. 정의·설명을 묻는 표현(`뭐야/뭐지/뭔가요/무엇/정의/설명/이란/what is/define`)이 있고 **행동 표현**(`해줘/해 줘/해주세요/시작/실행/돌려/보여/알려/조회/확인/취소/중단`)이 없으면 False
           (예: "PaiNN이 뭐야?", "스펙트럼이 뭐야?").
        3. 짧은 후속 답변 허용 목록과 **정확히 일치**하면 True (`응/네/예/아니/아니요/좋아/좋아요/그래/맞아/취소/중단/진행/확인/ok/yes/no/y/n`, 구두점·공백 제외).
        4. 도메인 키워드가 하나라도 있으면 True: `학습/훈련/train/예측/predict/스펙트럼/spectrum/painn/geoformer/equiformer/모델/체크포인트/checkpoint/분자/molecule/irdb/step/스텝/배치/batch/
           기본값/파라미터/parameter/상태/로그/워커/worker` 및 분자 ID 형태(예: `cn1_cn1_nn1`).
        5. 그 외 False.
      - `REFUSAL_MESSAGE`: 고정 거절 문구(할 수 있는 일과 예시 질문 포함).
      - `ScopeGuardrail(BaseGuardrail)`: `check`/`async_check`에서 `is_in_scope`가 False이면 `InputCheckError(REFUSAL_MESSAGE)`.
      - `ALLOWED_EXAMPLES`, `REFUSED_EXAMPLES`: UI(Step 6)가 쓰는 예시 질문의 단일 출처. 테스트가 둘 다 가드 판정과 일치함을 검증.
    - `tests/support/fake_openai.py` + `tests/conftest.py`의 `fake_llm` fixture: 스크립트된 응답을 돌려주는 가짜 OpenAI 호환 서버(`FakeOpenAIServer`: `base_url`, 받은 요청 기록 `requests`,
      응답 큐 `queue_text`/`queue_tool_calls`, 큐가 비었는데 요청이 오면 500). 5A~5C와 Step 6/7이 공유. 테스트 코드이므로 RED 커밋에 포함하고, 구현 없이 동작을 별도로 점검했다
      (텍스트 응답, tool 호출 2회 요청 흐름, 가드 차단 시 요청 0건).
  - 미포함: tool 래퍼(5B), Agent 조립/instructions/VLLM 환경변수(5C), 실제 vLLM 서버, 의미 기반(임베딩/LLM) 범위 판별.
  - 한계(의도된 것): 키워드 휴리스틱이므로 정의형 질문이 행동 표현과 함께 오면("PaiNN 정의 알려줘") 가드를 통과한다 → 5C의 instructions가 두 번째 방어선이다.
- 테스트 계획 (`tests/agent_tools/test_guard.py`, `tests/support/test_fake_openai.py` 성격의 검증은 가드 테스트 안에서 함께; 새 함수 import는 각 테스트 안에서)
  1. `test_학습과_예측_요청은_범위_안이다` (parametrize: "PaiNN으로 학습해줘", "그냥 학습해줘", "Geoformer로 학습 시작해줘", "학습 기본값 보여줘", "학습 상태 알려줘", "이 분자의 spectrum 예측해줘",
     "cn1_cn1_nn1 스펙트럼 예측해줘", "사용 가능한 모델 알려줘", "분자 목록 보여줘", "20 step으로 해줘", "배치 크기는 8로 해줘")
  2. `test_짧은_후속_답변은_범위_안이다` (parametrize: "응", "네", "네, 진행해줘", "아니요", "취소", "PaiNN", "yes", "OK!")
  3. `test_학습과_무관한_질문은_범위_밖이다` (parametrize: "오늘 날씨가 뭐야?", "반도체는 뭐지", "OLED의 정의는", "파이썬 코드 짜줘", "너는 누구야", "맛집 추천해줘", "", "   ")
  4. `test_키워드가_있어도_정의를_묻는_질문은_범위_밖이다` (parametrize: "PaiNN이 뭐야?", "스펙트럼이 뭐야?", "학습이란 무엇인가요")
  5. `test_예시_질문은_가드_판정과_일치한다` (`ALLOWED_EXAMPLES`는 모두 통과, `REFUSED_EXAMPLES`는 모두 거절, 각각 최소 4개)
  6. `test_ScopeGuardrail은_범위_밖_입력에_InputCheckError를_범위_안_입력은_통과시킨다`
  7. `test_범위_밖_질문은_LLM_서버를_호출하지_않고_거절_문구를_반환한다` (실제 `Agent` + `VLLM` + 가짜 서버: 서버 요청 수 0, `RunOutput.content == REFUSAL_MESSAGE`)
  8. `test_범위_안_질문은_LLM_서버로_전달된다` (서버 요청 수 1, 응답 텍스트 반환)
- RED 검증 기준: 테스트 함수 8개(parametrize 포함 34 케이스) 모두 `agent.guard` 부재(`ModuleNotFoundError`)로 실패. 기존 144개는 영향 없이 통과(새 `tests/conftest.py` 포함).
- 완료 조건(REVIEW 종료 시): 신규 + 기존 테스트 통과, **실험으로 쓴 질문 전체**(PRD의 거절 예시와 동작 예시)가 기대대로 판정됨, `agno` 로그 출력 소음 확인(거절 시 ERROR 로그 줄이 남는지 — 필요하면 조용히 처리할지 결정).

- **완료 결과 (Step 5A 종료)**
  - RED(`c50c10e`): 가드 테스트 8개(34 케이스)가 `agent.guard` 부재로 실패 + 가짜 OpenAI 호환 서버 테스트 기반. GREEN(`2bbeb82`): `agent/guard.py` 신규(`is_in_scope`, `REFUSAL_MESSAGE`,
    `ScopeGuardrail`, `ALLOWED_EXAMPLES`/`REFUSED_EXAMPLES`), 신규 34 케이스 통과 + 전체 178개 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안(가드 + 테스트 기반만). tool 래퍼/Agent 조립 없음. 기존 파일 변경 없음(GREEN 커밋은 신규 파일 1개). 라이센스 경계: `spectrum/` 미참조.
    - 폭넓은 문장 시험(구현 외부의 일회성 스크립트, 64문장): 기대와 다른 판정 3건.
      - 거짓 거절 1건: "진행 상황 알려줘"(정상 상태 문의인데 키워드 목록에 `진행 상황`이 없음).
      - 알려진 한계 2건(계획에 명시): "Geoformer 논문 요약해줘", "FC 모델 설명해줘"는 도메인 키워드 + 행동 표현("해줘") 때문에 가드를 통과 → 5C의 instructions가 두 번째 방어선.
      - 사용자가 직접 말한 거절 예시("오늘 날씨가 뭐야?", "반도체는 뭐지", "OLED의 정의는")와 동작 예시("PaiNN으로 학습해줘", "이 분자의 spectrum 예측해줘")는 모두 기대대로. UI 예시 질문 목록과 가드 판정도 일치.
    - **발견된 약점(변경하지 않고 기록): 가드는 현재 메시지만 보므로 대화 중 도메인 단어가 없는 후속 질문이 거절된다.** 관찰된 거절: "기본으로 해줘", "기본 설정으로 진행해줘",
      "default로 해줘", "얼마나 걸려?", "끝났어?", "다시 해줘", "아까 그걸로 해줘", "그만해줘", "멈춰줘", "결과 보여줘", "그래프 보여줘", "왜 실패했어?", "뭘 할 수 있어?", "사용법 알려줘".
      (거절되어도 `REFUSAL_MESSAGE`가 할 수 있는 일과 예시를 안내하므로 사용자가 다시 물을 수 있다.) 반면 "에러 로그 보여줘", "Geoformer 말고 PaiNN으로"는 통과.
      - 개선안(동작 변경이라 사용자 승인 필요): (a) 키워드/표현 보강 — `진행 상황`, `설정`, `default`, `결과`, `그래프`, `실패`, `에러`, `얼마나`, `끝났`, `사용법`, `뭘 할 수 있` 등,
        (b) 대화 이력이 있을 때만 짧은 문장을 완화하는 이력 인지 가드(범위 밖 짧은 질문이 통과하므로 instructions 의존도가 커짐).
      - 결정: **5A에서는 변경하지 않는다.** 가드와 instructions, 대화 기억이 함께 있어야 적절한 균형을 판단할 수 있으므로 5C(데모 서버로 직접 채팅해 보며)에서 한 번에 조정한다.
    - AGNO 로그: 가드가 거절할 때마다 `agno` 로거가 `ERROR   Validation failed: <거절 문구> | Check trigger: INPUT_NOT_ALLOWED` 한 줄을 남긴다(범위 안 질문에서는 없음). 반환값(`RunOutput.content`)은
      정상이며 stdout/stderr 버퍼가 아니라 로깅 핸들러로 출력된다. 동작 문제는 아니고 서버 콘솔의 소음이다. 5C에서 `build_agent`가 이 메시지를 걸러낼지(로깅 필터) 결정한다.
    - 리팩토링 검토: **필요 없음, 변경하지 않았다.** `guard.py`(약 90줄)는 상수(키워드/표지/답변 어휘) → 판정 함수 → 보조 함수 1개 → 가드레일 클래스로 구성되어 책임이 분명하고 중복·죽은 코드가 없다
      (`ScopeGuardrail.async_check`는 `check`에 위임, 미사용 import는 GREEN에서 이미 제거). `tests/support/fake_openai.py`와 `tests/conftest.py`도 작고 5B/5C에서 그대로 재사용된다.
  - Step 5B/5C로 넘기는 사항
    - UI의 예시 질문은 `agent.guard.ALLOWED_EXAMPLES`/`REFUSED_EXAMPLES`를 단일 출처로 쓴다. 가드 응답은 `agent.run()`에서 `RunOutput(status=error, content=REFUSAL_MESSAGE)`이므로 5C의 `chat()`이
      문자열로 정규화한다.
    - 5C에서 가드의 후속 질문 보강(위 개선안 a/b)과 `ERROR` 로그 처리를 결정한다.

### Step 5B. Agent용 tool 래퍼 (`agent/assistant_tools.py`)
- 목표: Step 1~4의 tool을 LLM이 쓰기 좋은 형태(간결한 인자, 요약된 결과, 확인 절차 강제)로 감싼다 (G1~G3).
- 조사 근거 (agno 3.1.1, 로컬 실험)
  - AGNO는 함수의 시그니처·docstring(Args 항목이 파라미터 설명)으로 tool 스키마를 만들고, 가짜 OpenAI 서버를 거친 tool 호출에서 `dict`, `bool`, `Optional`, `int` 인자가 그대로 전달된다
    (생략된 선택 인자는 기본값).
  - `Dict[str, Any]`는 값 스키마가 `{"type": "object"}`로 변환되어 값이 정수여야 하는 `settings`에서 실제 LLM을 혼동시킬 수 있다. `Dict[str, Union[int, str]]`는
    `anyOf [integer, string]`으로 올바르게 변환된다 → `settings`는 `Optional[Dict[str, Union[int, str]]]`로 선언한다.
  - 한 번의 `predict_spectrum` 결과(곡선 800점 ×2, 약 20KB)는 LLM 컨텍스트에 부적합하고, 학습 결과의 `command`(인터프리터 절대 경로 포함)·`log_path`도 LLM에 불필요하다 → 래퍼가 요약한다.
  - 훈련과 예측의 경로가 한 곳이어야 한다: 학습은 `project_root`(cwd와 `results_*`)에, 예측은 `results_root`에서 체크포인트를 찾고 `project_root`에서 데이터셋(`IrDB`)을 연다.
    실제 사용에서는 둘 다 저장소 루트이고, 테스트에서만 다르게 줄 수 있다(가짜 학습은 임시 `project_root`, 예측은 임시 `results_root` + 실제 데이터셋).
- 범위
  - 포함: `build_assistant_tools(project_root=저장소루트, results_root=None) -> list[callable]` (`results_root` 기본값은 `project_root`). 호출마다 **독립적인 상태**(마지막 학습 미리보기)를 가진
    클로저 7개를 돌려준다. 결과는 모두 `status`가 있는 JSON 직렬화 가능한 dict.
    1. `list_trained_models()` → `{"status": "ok", "models": {모델: {"trained": bool, "checkpoint": 파일 이름 또는 None}}, "any_trained": bool}` (registry 기반).
    2. `show_training_defaults(base_model)` → `get_training_defaults` 결과 + `changeable_parameters`(사용자가 바꿀 수 있는 파라미터 이름 목록: 공통 6개 + 그 모델의 모델 크기 파라미터). 미지원 모델은 error.
    3. `preview_training(base_model, settings=None)` → `start_training(confirmed=False)` 결과(`needs_confirmation` + 최종 설정 + `will_overwrite`) 또는 검증 오류/`busy`. `needs_confirmation`이면 이 미리보기를 기억한다.
    4. `start_training_confirmed(base_model, settings=None, overwrite=False)` → **기억한 미리보기와 같은 모델·같은 최종 설정**일 때만 `start_training(confirmed=True)`를 실행하고, 아니면
       `{"status": "error", "error": "not_previewed", "message"}`. 시작에 성공하면(`started`) 미리보기를 소모한다(재시작에는 새 미리보기 필요). `already_trained`/`busy`면 미리보기를 유지한다
       (사용자가 덮어쓰기를 확인한 뒤 `overwrite=True`로 다시 호출할 수 있게). 결과는 `{"status": "started", "job_id", "base_model", "output_dir", "settings", "message"}`로 요약(`command`·`log_path` 제외).
       ※ 이 장치는 "LLM이 설정을 사용자에게 보여 주는 단계를 건너뛰는 것"을 막을 뿐이며, 사용자가 "응"이라고 답했는지는 검증하지 않는다(instructions의 몫).
    5. `check_training_status(job_id=None)` → `{"status": "ok", "job_id", "base_model", "state", "return_code", "elapsed_seconds"(정수), "log_tail"(마지막 5줄, 줄당 200자), "has_checkpoint"}`.
       `no_job`/`unknown_job`은 그대로 전달.
    6. `list_molecule_ids(query="", limit=10)` → `list_molecules` 결과(그대로).
    7. `predict_molecule_spectrum(molecule_id, base_model=None)` → 성공 시 **요약만**:
       `{"status": "ok", "molecule_id", "base_model", "checkpoint"(파일 이름), "spectrum_type", "peak_wavelength_nm", "samples": [{"wavelength_nm", "intensity"} ×8]}`
       (400~750nm를 50nm 간격으로 샘플링한 상대 강도, 소수 셋째 자리). `needs_training`/`error`는 그대로 전달.
  - 미포함: Agent 조립·instructions·`VLLM` 환경변수(5C), 가드와의 연결(5C), UI용 전체 곡선(UI는 `predict_spectrum`을 직접 호출), 작업 취소 tool.
- 테스트 계획 (`tests/agent_tools/test_assistant_tools.py`; 학습 명령은 4B 방식으로 `[sys.executable, "-c", ...]`로 대체해 실제 서브프로세스를 쓰고, 예측은 tiny PaiNN 체크포인트와 실제 IrDB를 쓴다.
  `train_tool._jobs`는 테스트마다 초기화. 새 모듈 import는 각 테스트의 도우미 안에서 하여 개별 실패로 확인)
  1. `test_도구_목록은_계획한_7개이고_AGNO_스키마로_변환된다` (이름 집합, 모두 docstring 보유, `Function.from_callable`의 파라미터가 시그니처와 일치, `settings`가 `anyOf [integer, string]`)
  2. `test_학습된_모델_목록은_모델별_체크포인트_유무를_알려준다` (없을 때/PaiNN만 있을 때)
  3. `test_학습_기본값에는_바꿀_수_있는_파라미터_목록이_포함된다` (PaiNN `embed_dim` 포함, Equiformer는 `embed_dim` 없음·`num_basis` 있음, `lr` 없음, 미지원 모델 error)
  4. `test_미리보기는_실행하지_않고_최종_설정을_보여_준다` (프로세스 미시작 확인)
  5. `test_미리보기는_잘못된_설정을_검증_오류로_돌려준다`
  6. `test_미리보기_없이_확인_실행하면_not_previewed를_반환하고_실행하지_않는다`
  7. `test_미리보기와_다른_모델이나_설정으로_확인_실행하면_not_previewed를_반환한다`
  8. `test_미리보기와_같은_최종_설정이면_표현이_달라도_실행된다` (예: 미리보기 `{"train_steps": 5}`, 실행 `{"train_steps": 5, "eval_steps": <기본값>}`)
  9. `test_미리보기_후_확인_실행하면_요약된_started를_반환하고_미리보기를_소모한다` (`command`/`log_path` 없음, 완료 후 `check_training_status`로 finished, 다시 실행하면 `not_previewed`)
  10. `test_이미_학습된_모델은_overwrite_확인_후에만_다시_학습한다` (`will_overwrite` → `already_trained` → 같은 미리보기로 `overwrite=True` → started, 이전 산출물 삭제)
  11. `test_실행_중에_다시_시작하면_busy를_반환하고_미리보기를_유지한다`
  12. `test_학습_상태는_로그를_5줄_200자로_요약한다` (50줄×500자 출력, 정수 경과 시간, 실패 작업의 `failed`/`return_code`, `no_job`/`unknown_job` 전달)
  13. `test_분자_ID_목록은_그대로_전달한다`
  14. `test_예측_결과는_요약만_돌려주고_전체_곡선과_일치한다` (8개 샘플의 파장 400~750, 강도가 `predict_spectrum` 전체 곡선의 같은 파장 값과 일치, JSON 2KB 미만, 전체 곡선 키 없음)
  15. `test_학습된_모델이_없으면_예측은_needs_training을_전달한다`
  16. `test_도구_상태는_build_assistant_tools_호출마다_독립적이다` (A에서 미리보기, B에서 확인 실행 → `not_previewed`)
  17. `test_AGNO_Agent가_가짜_LLM_서버를_통해_미리보기와_확인_실행을_호출할_수_있다` (실제 `Agent` + `VLLM` + 가짜 서버: LLM이 `preview_training`(dict 인자) → `start_training_confirmed` 순서로 tool을
      호출하고, 요청에 7개 tool이 실리며, 학습 프로세스가 실제로 실행됨)
- RED 검증 기준: 모두 `agent.assistant_tools` 부재(`ModuleNotFoundError`)로 실패. 기존 178개는 영향 없이 통과.
- 완료 조건(REVIEW 종료 시): 신규 + 기존 테스트 통과, 실제 tiny 학습 → 상태 조회 → 예측 요약까지 래퍼만으로 연결(저장소 루트에서 실제 PaiNN tiny 학습), 응답 크기 측정(예측 요약 약 1KB 이하),
  `agent/`가 곡선 계산 로직을 직접 갖지 않음, 리팩토링 필요성 검토.

- **완료 결과 (Step 5B 종료)**
  - RED(`1e7ba6f`): tool 래퍼 테스트 17개가 `agent.assistant_tools` 부재로 실패. GREEN(`02f7631`): `agent/assistant_tools.py` 신규(170줄) + `train_tool.changeable_parameters` 공개 함수(+6/-1줄),
    신규 17개 통과 + 전체 195개 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안. Agent 조립/instructions/가드 연결/작업 취소 tool 없음. `agent/`에 곡선 계산 로직이나 `spectrum` 참조 없음(예측은 `predict_spectrum` 결과를 요약만 함).
    - 실제 동작 검증: 저장소 루트에서 **래퍼 7개만으로** PaiNN tiny 학습부터 예측 요약까지 연결(확인 후 산출물 삭제).
      학습 전 `list_trained_models`는 `any_trained=False`, 예측은 `needs_training` → 미리보기 없이 실행은 `not_previewed` → `preview_training`(`will_overwrite=False`) → `start_training_confirmed`
      (`started`) → 상태 폴링(실행 중 로그 관찰됨, 8.5초 만에 `finished`, 체크포인트 생성) → `predict_molecule_spectrum`이 `ok`(PaiNN, `checkpoint_best.ckpt`, 피크 파장, 샘플 8개)
      → 재학습 시도: `preview`가 `will_overwrite=True`와 삭제 경고를 안내, `overwrite` 없이는 `already_trained`, `overwrite=True`로 다시 `started` → 재학습 후 예측 `ok`.
    - 응답 크기(LLM 컨텍스트 부담): 모든 래퍼 응답이 1KB 미만 — 학습 전 목록 206B, 기본값 439B, 미리보기 514B, 시작 496B, 상태 424B, **예측 요약 521B(전체 곡선 결과는 22,597B로 약 43배 작음)**.
      LLM에 실리는 tool 스키마 전체(7개)는 약 3.7KB(tool당 306~789B).
    - 리팩토링 검토: **필요 없음, 변경하지 않았다.** `assistant_tools.py`는 tool별로 작은 클로저 7개와 단일 상태 항목(`state["preview"]`)으로 구성되어 책임이 분명하다. `start_training_confirmed`가 검증을
      한 번 더 하는 중복은 비용이 없고 `start_training`의 독립적 검증과 의도가 다르다. 두 테스트 파일에 중복된 도우미(`_use_fake_training` 5줄)는 시그니처가 다른 `_wait_until_done`과 함께 묶을 만큼
      크지 않다. 미사용 import 없음(AST 점검), `train_tool.changeable_parameters`로 허용 목록의 중복은 이미 제거됨.
  - 관찰한 개선 후보 (동작/문구 변경이라 리팩토링이 아니므로 변경하지 않고 기록)
    1. 문구: 학습된 모델이 하나도 없을 때 예측의 `needs_training` 메시지가 "학습된 어떤 모델의 체크포인트가 없습니다"로 어색하다(`predict_tool.py`의 `target` 기본 문구). "학습된 모델이 없습니다"가 자연스럽다
       — 테스트는 메시지가 비어 있지 않은지만 확인하므로 안전하게 고칠 수 있다(Step 7 정리 때 함께 처리 가능).
    2. LLM이 선택 인자를 빈 문자열로 보내는 경우: `check_training_status(job_id="")`는 `unknown_job`이 된다. 실제 LLM 연결 후 관찰해 필요하면 빈 문자열을 `None`으로 취급한다.
    3. Geoformer 학습 로그의 마지막 줄은 Lightning 진행 막대 프레임이라 정보량이 낮다(4B 기록). UI(Step 6)에서 걸러서 보여줄 수 있다.
  - Step 5C로 넘기는 사항
    - 미리보기 상태는 `build_assistant_tools()` 인스턴스(= 대화 세션)별이고, 작업 목록(`train_tool._jobs`)은 프로세스 전역이다 → 5C의 `build_agent`는 **세션마다 도구 인스턴스를 만들어** 한 사용자의 미리보기가 다른 세션의
      확인 실행에 쓰이지 않게 하고, 동시 학습 제한(`busy`)은 세션 간에 공유된다.
    - instructions는 tool 호출 순서(`show_training_defaults` → `preview_training` → 사용자 확인 → `start_training_confirmed`; 덮어쓰기는 `overwrite=True` + 삭제 경고)와 `not_previewed`/`already_trained`/`busy`/
      `needs_training` 응답을 사용자에게 어떻게 전달할지를 담는다. 사용자의 "응"이 있었는지는 tool이 검증하지 않으므로 instructions의 책임이다.
    - 가드의 후속 질문 보강과 `ERROR` 로그 처리(5A 기록)도 5C에서 결정한다.

### Step 5C. Agent 조립 + 대화 시나리오 — 5C-1~5C-2로 분할
- 목표: 가드(5A)·tool(5B)·instructions·대화 기억·`VLLM`을 합친 Agent와 `chat()`, 그리고 키 없이 직접 채팅해 볼 수 있는 데모 서버 (G3, G5).
- 분할(제안): 5C-1 Agent 조립 + 가짜 서버 시나리오 테스트 → 5C-2 규칙 기반 데모 서버(`agent/demo/`) + `python -m agent.demo` + 선택적 실제 LLM 연결 테스트.
  각 하위 Step마다 RED/GREEN/REVIEW 세 번 커밋(제목 `Step 5C-1 RED: ...`). 5C-2의 세부 계획은 5C-2 RED에서 확정한다.
- 조사 근거 (agno 3.1.1, 로컬 실험)
  - **LLM 연결 실패**: 서버에 연결할 수 없을 때 `agent.run()`은 예외를 던지지 않고 `RunOutput(status=error, content="Connection error.")`를 반환하며 SDK 기본 재시도 때문에 **약 7.5초** 걸린다.
    서버가 500을 돌려줘도 같은 형태(재시도 포함 약 1.5초)다 → `chat()`이 오류 응답을 사용자용 문구로 바꿔야 하고, `build_model`은 재시도를 줄인다(`max_retries=1`).
    가드 거절도 같은 `status=error`이므로 내용이 `REFUSAL_MESSAGE`이면 그대로 돌려준다.
  - **다중 턴 기억**: `InMemoryDb` + `add_history_to_context`이면 다음 턴 요청에 이전 턴의 `assistant(tool_calls)`, `tool` 결과, `assistant` 텍스트가 모두 들어간다
    (역할 순서 `user, assistant, tool, assistant, user`). "미리보기 → 응 → 시작" 흐름에 필요한 정보가 전달된다.
  - **세션 분리**: Agent 인스턴스마다 `InMemoryDb`를 두면 같은 `session_id`여도 기억이 공유되지 않는다. 5B의 미리보기 상태도 도구 인스턴스별이므로 **세션마다 Agent를 하나씩 만든다**
    (UI는 브라우저 세션당 Agent 하나를 유지).
  - `VLLM`은 `max_retries`, `timeout`, `default_headers`를 받는다(내부 LLM의 토큰 헤더 등 대응 가능).

### Step 5C-1. Agent 조립 (`agent/agent.py`) + 가드 보강
- 목표: `build_model`/`INSTRUCTIONS`/`build_agent`/`chat`을 구현하고, 가짜 OpenAI 서버로 다중 턴 대화 시나리오를 검증한다.
- 범위
  - 포함
    - `build_model(environ=None) -> VLLM`: `VLLM_BASE_URL`, `VLLM_MODEL`은 **필수**(없으면 누락된 변수 이름을 담은 `ModelConfigError(ValueError)`; 기본 localhost로 조용히 붙지 않게 함),
      `VLLM_API_KEY`는 없으면 `"EMPTY"`(AGNO `VLLM`이 키를 요구하므로 인증 없는 서버용 더미). `max_retries=1`. 나머지는 AGNO 기본값.
    - `INSTRUCTIONS`(한국어): ① 학습/예측 전용, 그 외는 정중히 거절하고 할 수 있는 일을 안내 ② 학습 요청: 모델이 정해지지 않았으면 PaiNN/Geoformer/Equiformer 중 무엇인지 되묻기 →
      `show_training_defaults`로 기본값을 보여 주기(CPU면 step이 많으면 오래 걸림을 알림) → `preview_training` → 최종 설정을 보여 주고 확인 → **사용자가 명시적으로 동의한 뒤에만**
      `start_training_confirmed`(같은 인자) ③ `will_overwrite`면 기존 결과가 삭제됨을 알리고 별도 확인 후 `overwrite=True` ④ 학습 상태는 `check_training_status`로 확인해 알려 주고 완료되면 예측 안내,
      실패하면 로그 마지막 줄 전달 ⑤ 예측은 분자 ID가 필요하며 모르면 `list_molecule_ids`로 도움, `needs_training`이면 학습을 먼저 제안, 결과는 피크 파장과 샘플 강도로 설명하고 수치를 지어내지 않기
      ⑥ `busy`/`already_trained`/`not_previewed` 응답은 이유를 사용자에게 전달하기.
    - `build_agent(model=None, project_root=저장소루트, results_root=None) -> Agent`: `Agent(model, tools=build_assistant_tools(...)(새 인스턴스), instructions=INSTRUCTIONS, pre_hooks=[ScopeGuardrail()],
      db=InMemoryDb(), add_history_to_context=True, num_history_runs=10, tool_call_limit=8, telemetry=False)`. `model`이 없으면 `build_model()`.
    - `chat(agent, message, session_id="default") -> str`: `agent.run`의 결과를 문자열로 정규화 — 정상이면 `content`, 가드 거절이면 `REFUSAL_MESSAGE`, 그 외 오류(`status=error`)면 LLM 서버 호출 실패를
      알리는 문구(원인 `content` 포함, `VLLM_BASE_URL` 확인 안내).
    - **가드 보강(5A 리뷰의 개선안 a, 사용자 승인 필요)**: 실제 대화에서 쓰일 만한 후속 표현이 거절되는 것을 막기 위해 허용 표현을 보수적으로 추가 —
      `진행 상황`, `설정`, `default`, `결과`, `그래프`, `실패`, `얼마나`, `끝났`, `뭘 할 수`, `사용법`, `기본으로`. 정의형 질문 거절 규칙(예: "결과가 뭐야?")은 그대로 유지.
      이력 인지 가드(개선안 b)는 이번에 하지 않는다. `ERROR` 로그 한 줄(5A 기록)은 서버 콘솔 소음일 뿐이라 변경하지 않는다.
  - 미포함: 데모 서버와 `python -m agent.demo`(5C-2), UI(Step 6), 실제 LLM 연결 테스트(5C-2), LLM 응답 품질 평가.
- 테스트 계획 (`tests/agent_tools/test_agent.py`, `tests/agent_tools/test_guard.py` 추가; 가짜 OpenAI 서버 `fake_llm` 사용, 학습은 4B 방식의 가짜 명령)
  1. `test_build_model은_필수_환경변수가_없으면_누락된_이름을_알려준다` / `test_build_model은_환경변수로_VLLM을_구성한다` (API 키 기본 `"EMPTY"`, `max_retries == 1`, 값 전달)
  2. `test_instructions에는_학습_확인_절차와_덮어쓰기_경고와_범위_제한이_담겨_있다` (tool 이름, 확인/동의, `overwrite`/삭제, 거절 문구 등 핵심 규칙 포함)
  3. `test_build_agent는_tool_7개_가드_대화기억을_갖춘다` (tool 이름, `pre_hooks`에 `ScopeGuardrail`, `InMemoryDb`, `telemetry` 꺼짐)
  4. `test_Agent_요청에는_instructions와_tool_7개가_실린다` (가짜 서버가 받은 요청의 system 메시지와 `tools`)
  5. `test_범위_밖_질문은_chat이_거절_문구를_돌려주고_LLM을_호출하지_않는다`
  6. `test_학습_시나리오_미리보기_후_사용자_확인으로_학습이_시작된다` (1턴: LLM이 `preview_training` 호출 후 확인 질문, 2턴 "응": 이전 턴의 tool 호출/결과가 요청에 포함되고 LLM이
     `start_training_confirmed` 호출 → 학습 프로세스 실행)
  7. `test_미리보기_없이_학습_시작을_시도하면_tool이_거부한다` (LLM이 바로 `start_training_confirmed` 호출 → 요청에 `not_previewed` 결과가 실리고 프로세스는 시작되지 않음)
  8. `test_학습된_모델이_없을_때_예측_시나리오는_needs_training_결과를_LLM에_전달한다`
  9. `test_세션마다_Agent가_독립적이다` (A에서 미리보기, B에서 확인 실행 → `not_previewed`, 기억도 분리)
  10. `test_chat은_LLM_서버_오류를_사용자용_문구로_바꾼다` (닫힌 포트 + `max_retries=0`로 빠르게)
  11. `test_chat은_정상_응답_텍스트를_그대로_돌려준다`
  12. (test_guard 추가) `test_대화에서_쓰이는_후속_표현은_범위_안이다` (parametrize: "진행 상황 알려줘", "기본 설정으로 진행해줘", "default로 해줘", "얼마나 걸려?", "끝났어?", "결과 보여줘", "그래프 보여줘",
      "왜 실패했어?", "뭘 할 수 있어?", "사용법 알려줘", "기본으로 해줘") + 기존 거절 예시가 여전히 거절되는지(`test_학습과_무관한_질문은_범위_밖이다`, `test_키워드가_있어도_정의를_묻는_질문은_범위_밖이다`)
- RED 검증 기준: `agent.agent` 부재(`ModuleNotFoundError`)로 실패하는 테스트와, 가드 보강 테스트(새 허용 표현이 현재 거절됨)가 실패. 기존 195개는 영향 없이 통과.
- 완료 조건(REVIEW 종료 시): 신규 + 기존 테스트 통과, 가드 폭넓은 문장 시험(5A REVIEW의 64문장 + 후속 표현)에서 오판정 재점검, instructions를 한 번 읽고 tool 이름·응답 상태 이름이 실제 코드와 일치하는지 대조.

- **완료 결과 (Step 5C-1 종료)**
  - RED(`339801e`): `test_agent` 12개가 `agent.agent` 부재로, 가드 후속 표현 11개가 현재 규칙으로 실패. GREEN(`c601d5b`): `agent/agent.py` 신규(`build_model`, `ModelConfigError`, `INSTRUCTIONS`,
    `build_agent`, `chat`) + `agent/guard.py` 허용 표현 11개 보강. 신규 24 케이스 통과 + 전체 219개 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안(데모 서버·UI·실제 LLM 연결 테스트 없음). 기존 코드 변경은 가드의 키워드 목록뿐. `agent/`에 곡선 계산 로직이나 `spectrum` 참조 없음.
    - **실제 tool 스택 + 가짜 LLM의 다중 턴 대화**(저장소 루트, 실제 PaiNN tiny 학습, `chat()` 사용, 확인 후 산출물 삭제): 1턴 "PaiNN으로 학습해줘"(LLM이 `show_training_defaults` → `preview_training` 호출 후 확인 질문) →
      2턴 "응"(요청에 이전 턴의 `assistant(tool_calls)`/`tool` 결과/`assistant`가 모두 포함, `start_training_confirmed`로 학습 시작) → 3턴 "끝났어?"(5C-1에서 허용한 후속 표현, `check_training_status`) →
      4턴 "cn1_cn1_nn1 스펙트럼 예측해줘"(`predict_molecule_spectrum`, 결과 521B). 모든 응답이 `chat()`을 거쳐 문자열로 반환됨.
    - instructions ↔ 코드 대조: instructions에 나온 snake_case 식별자 11개(tool 6개 + `will_overwrite`, `train_steps`, `needs_training`, `not_previewed`, `already_trained`)가 **모두 실제 tool 이름이거나 응답 키**이고,
      언급한 상태 값(`running/finished/failed`, `busy`, `samples`, `device`, `overwrite`)도 코드에 실재. 규칙 9개, 1,290자(2,478B).
    - 크기/지연: 4턴 누적 후 마지막 LLM 요청이 10.6KB(system instructions 약 2.5KB + tool 스키마 약 3.7KB + 메시지 18개). 기본 `build_model`(`max_retries=1`)로 연결 불가 서버에서 `chat()`이 4.5초 만에
      "LLM 서버 호출에 실패했습니다 … VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY 설정과 서버 상태를 확인해 주세요"를 반환(SDK 기본 설정은 7.5초).
    - 가드 재시험(5A REVIEW의 64문장): 기대와 다른 판정 3건 → 2건(남은 것은 알려진 한계 "Geoformer 논문 요약해줘", "FC 모델 설명해줘"). 새로 추가한 후속 표현("기본으로 해줘", "얼마나 걸려?", "끝났어?",
      "결과 보여줘", "그래프 보여줘", "왜 실패했어?", "뭘 할 수 있어?", "사용법 알려줘" 등)은 통과하고 "결과가 뭐야?" 같은 정의형 질문은 계속 거절.
    - 리팩토링(테스트 코드, 동작 불변): 세 테스트 파일(`test_train_job`, `test_assistant_tools`, `test_agent`)에 똑같이 복사되어 있던 `_fresh_jobs` fixture와 `_use_fake_training` 도우미를 공용으로 추출 —
      `tests/agent_tools/conftest.py`(자동 fixture), `tests/support/fake_training.py`(`use_fake_training`). 세 번 반복되었고 Step 6 UI 테스트에서도 필요해질 것이라 판단. 이후 미사용 import 정리,
      테스트 수 219개 그대로 통과, 줄 수 71줄 삭제/31줄 추가. 프로덕션 코드(`agent/`)는 리팩토링할 것이 없다고 판단해 변경하지 않음(`agent.py`는 함수 4개와 상수 1개로 책임이 분명).
  - 관찰한 개선 후보 (동작/문구 변경이라 리팩토링이 아니므로 변경하지 않고 기록)
    1. instructions에 `list_trained_models`가 언급되지 않았다(LLM은 tool 스키마 설명으로 알 수 있음). 필요하면 "예측 전에 학습된 모델이 있는지 확인할 때 사용" 한 줄을 추가.
    2. 여전히 거절되는 대화 표현: "다시 해줘", "그만해줘", "멈춰줘", "처음부터 다시", "더 자세히 알려줘", "도와줘". 실제 대화에서 자주 나오면 같은 방식(테스트 선행)으로 보강.
    3. 대화 기억(`num_history_runs=10`)이 누적되면 요청이 커진다(4턴 후 10.6KB). 컨텍스트가 작은 LLM에서는 기억 턴 수를 줄이거나 tool 결과를 기억에서 제외하는 설정을 검토.
    4. 가드 거절 때 `agno` 로거의 `ERROR` 한 줄(5A 기록)은 변경하지 않음(서버 콘솔 소음일 뿐).
  - Step 5C-2로 넘기는 사항
    - AGNO는 이전 턴의 `assistant(tool_calls)`와 `tool` 결과를 다음 요청에 모두 실어 보낸다(검증됨) → 규칙 기반 데모 서버는 **상태 없이** 메시지 이력만으로 동작할 수 있다
      (예: 사용자가 "응"이면 이력에서 마지막 `preview_training` 호출의 인자를 찾아 `start_training_confirmed`로 그대로 전달).
    - 데모 서버는 진짜 LLM이 아님을 문서와 출력에 표시하고, `agent/demo/`로 분리한다. 같은 서버로 다중 턴 시나리오 테스트를 구동한다. 선택적 `vllm` 마커 테스트는 `VLLM_BASE_URL`이 있을 때만 실행.

### Step 5C-2. 규칙 기반 데모 서버 + 실제 LLM 연결 선택 테스트
- 목표: LLM 키·서버 없이 Agent 전체(가드, tool, 학습 확인 절차, 백그라운드 학습, 예측 요약)를 **직접 채팅으로** 확인할 수 있게 하고, 같은 서버로 다중 턴 시나리오를 구동해 검증한다. 진짜 LLM 연결용 선택 테스트(`vllm` 마커)를 둔다 (G3, G5).
- 조사 근거 (agno 3.1.1, 로컬 실험)
  - AGNO는 이전 턴의 `assistant(tool_calls)`와 `tool` 결과를 다음 요청에 모두 싣는다 → 데모 서버는 **상태 없이** 메시지 이력만으로 대화 맥락(대기 중인 미리보기, 직전 질문)을 알 수 있다.
  - 요청의 `tool` 메시지 `content`는 JSON이 아니라 **Python 표현 문자열**(`{'status': 'needs_confirmation', 'will_overwrite': True, 'return_code': None, ...}`)이다.
    `json.loads`는 실패하고 `ast.literal_eval`은 한글·따옴표·줄바꿈·`True`/`None`을 정상 파싱한다. 반면 `assistant`의 `tool_calls[].function.arguments`는 JSON 문자열이다.
  - AGNO는 기본적으로 비스트리밍 요청을 보내므로 서버는 `/v1/chat/completions`의 일반 JSON 응답만 지원하면 된다.
- 범위
  - 포함
    - `agent/demo/mock_llm.py`
      - `respond(messages) -> dict`: **순수 함수**(HTTP 없음). OpenAI chat completion 형식의 응답(텍스트 `finish_reason="stop"` 또는 `tool_calls` `finish_reason="tool_calls"`)을 돌려준다.
        현재 턴(마지막 `user` 메시지 이후)의 tool 호출/결과 진행 상황과 이력을 보고 다음 행동을 정한다. 모든 텍스트 응답은 `"[데모] "`로 시작해 진짜 LLM으로 오해하지 않게 한다.
      - `MockLLMServer`: `respond`를 `/v1/chat/completions`로 서비스하는 로컬 HTTP 서버(컨텍스트 매니저, `base_url`, 받은 `requests` 기록, 포트 0이면 자동 할당).
    - 대화 규칙 (키워드 기반, 의도는 현재 메시지 + 이력으로 판정, 위에서부터 우선):
      1. **동의**("응/네/예/좋아/그래/진행해줘/시작해줘/ok/yes")이고 이력에 **대기 중인 미리보기**(마지막 `preview_training` 결과가 `needs_confirmation`이고 그 뒤에 `start_training_confirmed` 호출이 없음)가 있으면
         → 같은 인자로 `start_training_confirmed`(미리보기의 `will_overwrite`가 true였으면 `overwrite=True`). 결과(`started`/`already_trained`/`busy`/`not_previewed`)를 `message`와 함께 안내.
      2. **거부**("아니/취소/그만/멈춰")이고 대기 중인 미리보기가 있으면 → 학습을 시작하지 않고 취소를 안내(tool 호출 없음).
      3. **학습 + 모델 이름**(PaiNN/Geoformer/Equiformer) → `show_training_defaults` → `preview_training`(문장의 "N step"→`train_steps`, "배치 N"→`batch_size`를 `settings`로) → 최종 설정을 보여 주고 "진행할까요?"
         (`will_overwrite`면 기존 결과 삭제 경고 포함).
      4. **학습, 모델 이름 없음** → "PaiNN, Geoformer, Equiformer 중 무엇으로 학습할까요?"; 직전 assistant 메시지가 이 질문이고 사용자가 모델 이름만 답하면 3으로 이어간다.
      5. **상태/끝났/진행 상황/로그** → `check_training_status` → `running`/`finished`(체크포인트 유무)/`failed`(로그 마지막 줄) 요약.
      6. **예측/스펙트럼 + 분자 ID**(`cn1_cn1_nn1` 형태) → `predict_molecule_spectrum` → 성공이면 피크 파장과 샘플 강도 요약, `needs_training`이면 "먼저 학습이 필요하다"고 학습 제안, 오류면 `message` 전달.
         ID가 없으면 `list_molecule_ids(limit=5)`로 예시 ID를 안내.
      7. **분자 목록/어떤 분자** → `list_molecule_ids`; **학습된 모델** → `list_trained_models`; **기본값/설정 + 모델** → `show_training_defaults`(모델 없으면 모델을 되묻기).
         ※ `기본값`이나 `설정`에 `보여/알려`가 붙은 **조회 질문**("PaiNN 학습 기본값 보여줘")은 "학습 + 모델 이름"(3)보다 우선한다 — 기본값만 알려 주고 미리보기로 이어가지 않는다(테스트 작성 중 규칙 충돌을 발견해 확정).
      8. **뭘 할 수 있어/도움/사용법** → 가능한 일 안내.
      9. 그 외 → "이해하지 못했어요. 데모 서버는 규칙 기반이라 정해진 표현만 알아듣습니다"와 예시.
    - `agent/demo/__main__.py`: `python -m agent.demo [--port N] [--serve-only]`. 기본은 서버를 띄우고 그 서버에 연결된 Agent(`build_agent`)로 콘솔 채팅(종료: `종료`/`exit`/`quit`),
      시작 시 **"데모 모드: 진짜 LLM이 아닌 규칙 기반 가짜 서버입니다"** 배너를 출력. `--serve-only`는 서버만 실행(Step 6 UI나 다른 클라이언트가 `VLLM_BASE_URL=http://127.0.0.1:<port>/v1`로 연결).
    - 실제 LLM 연결 선택 테스트: `@pytest.mark.vllm`, `VLLM_BASE_URL`(과 `VLLM_MODEL`)이 없으면 skip.
  - 미포함: UI(Step 6), 진짜 LLM 품질 평가, 스트리밍 응답, 규칙에 없는 표현의 유연한 이해, 데모 서버를 이용한 성능 측정. 데모 서버는 프로젝트의 정식 기능이 아니라 개발·시연 도구이며 `agent/demo/`에 분리한다.
- 테스트 계획
  - `tests/agent_tools/test_demo_mock_llm.py` — `respond`를 메시지 이력만으로 직접 호출(빠른 순수 함수 테스트; tool 결과 메시지는 AGNO와 같은 Python 표현 문자열로 구성)
    1. `test_응답은_OpenAI_chat_completion_형식이다` (텍스트/tool_calls 두 경우의 `choices[0].message`, `finish_reason`, tool_call `id`/`arguments`가 JSON 문자열)
    2. `test_모델이_정해지지_않은_학습_요청은_모델을_되묻는다`
    3. `test_모델이_정해진_학습_요청은_기본값_조회_후_미리보기를_호출하고_확인을_묻는다` (3단계: `show_training_defaults` → `preview_training` → 확인 텍스트, `[데모]` 접두사)
    4. `test_step_수와_배치_크기를_말하면_미리보기_설정에_반영한다`
    5. `test_모델만_답하면_직전_모델_질문의_학습_요청으로_이어간다`
    6. `test_사용자가_동의하면_직전_미리보기와_같은_인자로_학습을_시작한다`
    7. `test_덮어쓰기_미리보기에는_삭제를_경고하고_동의하면_overwrite로_시작한다`
    8. `test_거부하면_학습을_시작하지_않고_취소를_알린다`
    9. `test_대기_중인_미리보기가_없으면_동의_표현에도_학습을_시작하지_않는다`
    10. `test_학습_상태_질문은_상태를_조회하고_running_finished_failed를_요약한다`
    11. `test_예측_요청은_분자_ID로_예측하고_결과_또는_학습_안내를_전달한다` (ok의 피크 파장 수치가 결과와 일치, `needs_training`이면 학습 제안)
    12. `test_분자_ID가_없는_예측_요청은_분자_목록을_보여준다`
    13. `test_목록_기본값_학습된_모델_질문은_해당_tool을_호출한다` (parametrize)
    13-1. `test_기본값_조회_질문은_미리보기로_이어가지_않고_기본값만_알려준다`
    14. `test_도움_요청은_가능한_일을_안내한다` / `test_규칙에_없는_말은_이해하지_못했다고_답한다`
    15. `test_tool_오류_응답의_message를_사용자에게_전달한다` (`busy`, `already_trained`, `not_previewed`)
  - `tests/agent_tools/test_demo_scenarios.py` — 실제 `build_agent` + `VLLM` + `MockLLMServer`(키 불필요), 학습은 가짜 명령
    16. `test_대화_시나리오_모델_되묻기부터_미리보기_동의_학습_시작_상태_확인까지_이어진다` (그냥 학습해줘 → PaiNN → 미리보기 → 응 → 학습 프로세스 실제 실행 → 끝났어?)
    17. `test_학습된_모델이_있으면_예측_시나리오가_피크_파장을_알려준다` (tiny PaiNN 체크포인트 + 실제 IrDB, 피크 파장이 `predict_spectrum`과 일치)
    18. `test_학습된_모델이_없으면_예측_시나리오는_학습을_먼저_제안한다`
    19. `test_이미_학습된_모델은_삭제_경고_후_동의해야_다시_학습한다`
    20. `test_범위_밖_질문은_데모_서버를_호출하지_않고_거절된다` (서버 요청 수 0)
  - `tests/agent_tools/test_demo_cli.py`
    21. `test_python_m_agent_demo는_데모_배너를_출력하고_질문에_답한다` (실제 서브프로세스, stdin으로 "PaiNN 학습 기본값 보여줘"와 "종료")
  - `tests/agent_tools/test_agent_real_llm.py` (`vllm` 마커, 환경변수 없으면 skip)
    22. `test_실제_LLM_서버에_연결해_범위_안_질문에_응답한다` / `test_실제_LLM_서버에서도_범위_밖_질문은_가드가_거절한다`
- RED 검증 기준: 1~21은 `agent.demo` 부재(`ModuleNotFoundError`)로 실패(순수 함수 테스트는 parametrize 포함 26 케이스, 시나리오 5개, CLI 1개), 22는 환경변수가 없어 skip(실제 LLM 연결 후에는 사용자가 직접 실행).
  기존 219개는 영향 없이 통과. 테스트 이력은 AGNO가 실제로 보내는 메시지와 구조·형식이 같음을 구현 없이 별도로 대조했다(역할 순서, 키, `arguments` 문자열, tool 결과 `literal_eval` 파싱).
- 완료 조건(REVIEW 종료 시): 신규 + 기존 테스트 통과, `python -m agent.demo`를 실제로 실행해 직접 대화(그냥 학습해줘 → PaiNN → 응 → 끝났어? → 예측)하며 화면 확인, 대화 규칙 표와 실제 동작 대조,
  데모 응답이 "진짜 LLM이 아님"을 항상 표시하는지 확인, 리팩토링 필요성 검토.

- **완료 결과 (Step 5C-2 종료)**
  - RED(`87a66a0`): 데모 순수 함수 26 케이스 + 시나리오 5개 + CLI 1개가 `agent.demo` 부재로 실패, 실제 LLM 연결 테스트 2개는 skip. GREEN(`ea40fda`): `agent/demo/`(`mock_llm.py`의 `respond`/`MockLLMServer`,
    `__main__.py`, `__init__.py`) 신규 350줄, 신규 32개 통과(2개 skip) + 전체 251개 통과.
  - REVIEW 확인 사항
    - 스코프: Plan 범위 안(UI 없음). 프로덕션 코드(`agent/`의 데모 외, `common/`)가 `agent.demo`를 참조하지 않음(grep 확인) — 폴더만 지우면 데모를 제거할 수 있다. 기존 코드 변경 없음.
    - **실제 대화 검증**(데모 서버 + 실제 tool 스택, 저장소 루트, 실제 PaiNN 학습(기본 모델 크기, 2 step, 21초), 확인 후 산출물 삭제): "그냥 학습해줘"(모델 되묻기) → "PaiNN"(기본값 조회 후 미리보기, 진행 여부 질문) → "아니요"(취소) →
      "PaiNN으로 2 step 배치 2로 학습해줘"(설정 반영) → "응"(학습 시작) → "끝났어?"(끝났다고 안내 + 예측 제안) → "cn1_cn1_nn1 스펙트럼 예측해줘"(피크 파장과 파장별 강도) → "PaiNN으로 학습해줘"(이미 학습됨: **삭제 경고** 포함) →
      "아니요"(취소) → "오늘 날씨가 뭐야?"(가드 거절, LLM 호출 없음) → "분자 목록 보여줘" → "학습된 모델 있어?" → "블라블라"(가드 거절). 모든 데모 응답이 `[데모]`로 시작하고 LLM 요청은 22건.
    - **`--serve-only` 서버 + 환경변수 연결**(내부 LLM을 붙일 때와 같은 `build_model(environ)` 경로): `python -m agent.demo --serve-only --port 8799`를 별도 프로세스로 띄우고 `VLLM_BASE_URL`/`VLLM_MODEL`만으로 `build_agent`를 만들어
      "PaiNN 학습 기본값 보여줘"에 정상 응답. 서버 출력에 "진짜 LLM이 아니라 규칙 기반 가짜 서버" 배너가 표시됨.
    - 대화 규칙 ↔ 실제 동작 대조(`respond` 직접 호출): 규칙표대로 라우팅됨(예: "학습 상태/진행 상황"→상태, "PaiNN으로 학습해줘 기본값으로"→학습 흐름, "Geoformer 학습 설정 보여줘"→기본값 조회, "예측 결과 보여줘"→분자 목록,
      "모델 목록/사용 가능한 모델"→학습된 모델, "학습된 모델로 cn1_cn1_nn1 예측해줘"→예측). 가드는 통과하지만 데모가 알아듣지 못하는 표현은 "이해하지 못했어요"(예: "그래프 보여줘", "결과 보여줘", "도와줘") — 규칙 기반의 의도된 한계.
    - **관찰: 2 step만 학습한 모델의 예측 요약에서 상대 강도가 959처럼 1을 훌쩍 넘었다.** 원인을 같은 조건으로 재현해 확인: 예측 파라미터가 `S1=-10.3` 등 비물리적이어서 정규화 전 곡선이 **모든 점에서 음수**이고, `spectrum_fc`가 곡선을 음수인 최댓값으로 나눠
      모든 값이 1 이상(최솟값 정확히 1.0, 최댓값 969)이 된다. `spectrum/`의 정규화를 그대로 따른 결과이며 도구·데모의 버그가 아니다(충분히 학습된 모델에서는 해소). 사용자에게 혼동을 줄 수 있어 아래 개선 후보로 기록.
    - 리팩토링 검토: `mock_llm.py`(284줄)의 `_start`(약 45줄)와 `_continue`(약 55줄)는 짧은 분기가 평평하게 늘어선 규칙표이고 26개 테스트가 커버하므로, tool 이름별 디스패치 딕셔너리로 바꾸면 간접 참조만 늘고 코드가 줄지 않아 **변경하지 않았다**.
      단 한 곳을 정리: "기본값 보여줘"(모델 없음)에서 `_MODEL_QUESTION.replace("학습할까요", "기본값을 보여 드릴까요")`로 만들던 임시 처리를 이름 있는 상수 `_DEFAULTS_MODEL_QUESTION`으로 교체
      (기존의 어색한 문장 "무엇으로 기본값을 보여 드릴까요?"가 "어떤 모델의 기본값을 보여 드릴까요?"로 자연스러워짐, 테스트에 영향 없음). `guard.py`와 데모의 토큰화 중복(3줄)은 서로 다른 어휘 집합을 쓰는 독립 모듈이라 그대로 둠.
  - 관찰한 개선 후보 (동작/문구 변경이라 리팩토링이 아니므로 변경하지 않고 기록)
    1. 비정상 곡선 경고: 예측 요약에서 최댓값이 1을 크게 넘거나 음수가 섞이면 `warning: "예측 곡선이 비정상입니다. 학습이 충분하지 않을 수 있습니다"`를 덧붙이기(5B 래퍼와 UI에서 유용, 테스트 선행 필요).
    2. 데모의 미리보기 문구에 CPU 안내 추가: 기본 10000 step은 CPU에서 매우 오래 걸린다는 점과 "20 step"처럼 줄이는 방법(`device`가 미리보기 설정에 포함되어 있어 구현 가능).
    3. `respond`는 `user` 메시지가 없는 요청에 `ValueError`를 내고 서버 핸들러가 연결을 끊는다(AGNO는 항상 user 메시지를 보내므로 실사용 영향 없음). 필요하면 400 응답으로 처리.
  - 사용자 확인: 사용자가 `python -m agent.demo`를 직접 실행해 동작을 확인했다.
- **Step 5 전체 완료**: 5A(범위 가드, 가짜 OpenAI 서버 기반) → 5B(Agent용 tool 래퍼 7개) → 5C-1(Agent 조립, 가드 보강) → 5C-2(규칙 기반 데모 서버). 전체 251개 테스트 통과(2개 `vllm` 선택 테스트 skip).
  mock/데모로 확인된 것: 연결·tool 실행·확인 절차 강제·가드·대화 기억·세션 분리·학습/예측 흐름. 확인되지 않은 것: 진짜 LLM의 지시 이행과 tool 선택(내부 LLM 연결 후 `pytest -m vllm`과 대화 로그로 확인).
  - Step 6(UI)로 넘기는 사항
    - UI 채팅 탭은 브라우저 세션당 `build_agent()` 하나를 유지하고(미리보기와 대화 기억이 세션별), 예시 질문은 `agent.guard.ALLOWED_EXAMPLES`/`REFUSED_EXAMPLES`를 쓴다. 가드 거절과 LLM 서버 오류는 `chat()`이 문자열로 정규화한다.
    - LLM 연결: `VLLM_BASE_URL`/`VLLM_MODEL`이 없으면 채팅 탭은 안내 메시지를 보이고(학습/예측 탭은 LLM 없이 tool 함수를 직접 호출), 데모 모드는 `MockLLMServer`를 프로세스 안에서 띄우거나 `python -m agent.demo --serve-only` 서버에 연결.
    - 학습 탭은 `get_training_defaults`/`validate_training_request`/`start_training`/`get_training_status`, 예측 탭은 `list_molecules`/`predict_spectrum`(전체 곡선)을 직접 사용. Geoformer 로그의 진행 막대 프레임은 걸러서 표시.
    - 기본 설정(10000 step)은 CPU에서 매우 오래 걸리므로 UI 학습 탭의 기본 입력은 작게 시작하도록 안내(제약 2: 스크립트 기본값은 변경하지 않고 tool의 override로 지정).

## Step 6. UI (신규 인터페이스) — 6A~6B로 분할
- 목표: Gradio UI에서 **채팅(Agent)**, **학습**, **예측**을 쓸 수 있게 한다 (G4). 학습/예측 탭은 LLM 없이 tool 함수를 직접 호출하므로 LLM이 없어도 동작하고, 채팅 탭만 LLM(실제 또는 데모)이 필요하다.
- 분할: 6A UI 로직(핸들러, Gradio 비의존) → 6B Gradio 화면 구성 + 실행 진입점(`python -m agent.ui`). 각 하위 Step마다 RED/GREEN/REVIEW 세 번 커밋(제목 `Step 6A RED: ...`).
- 조사 근거 (gradio 6.29.1, 로컬 시제품)
  - `Blocks`/`Tabs`/`Chatbot`/`State`/`Examples`/`LinePlot`/`Timer`가 모두 동작한다. **Gradio 6의 `Chatbot`에는 `type` 인자가 없고** 메시지(`{"role", "content"}`) 형식이 기본이다.
    `api_name`을 준 이벤트는 `gradio_client.Client`로 호출할 수 있다(`huggingface-hub` 버전 충돌(Step 0 기록)이 있어도 호출 성공) → **브라우저 없이** 화면 구성과 이벤트 연결을 테스트할 수 있다.
  - 실험 스펙트럼: IrDB의 `spec_x`는 eV 오름차순 800점(1.551~3.100), `spec_y`는 0~1이다. `nm = 1240/eV`로 바꾸면 예측 곡선(400~799.5nm)과 같은 축에 겹쳐 그릴 수 있다(예측은 nm에서 균등, 실험은 eV에서 균등이라
    x 값이 달라 두 계열을 별도 x로 가진 긴 형식 표로 그린다).
  - 비정상 곡선(5C-2 리뷰): 학습이 거의 안 된 모델은 상대 강도가 1을 훌쩍 넘는 곡선을 낼 수 있다(음수 최댓값으로 정규화) → 예측 탭은 이를 경고로 표시한다.
  - 학습 로그의 Lightning 진행 막대 프레임은 정보량이 낮으므로(4B 기록) 학습 탭에서 걸러서 보여 준다.
  - Agent는 대화 세션마다 하나여야 한다(미리보기 상태·대화 기억이 Agent별, 5C 기록) → 채팅 탭은 브라우저 세션별 `gr.State`에 Agent를 지연 생성해 보관한다.

### Step 6A. UI 로직 (`agent/ui_logic.py`, Gradio 비의존)
- 목표: 화면과 무관하게 테스트할 수 있는 핸들러 함수들. 화면(6B)은 이 함수를 이벤트에 연결할 뿐이다.
- 범위
  - 포함
    - **LLM 상태**: `resolve_llm(environ=None, demo=False) -> LlmStatus(mode, message, base_url, model_id)` — `demo=True`면 프로세스 안에서 `MockLLMServer`를 한 번만 띄워(`mode="demo"`, 진짜 LLM이 아님을 알리는 메시지),
      아니면 `VLLM_BASE_URL`/`VLLM_MODEL`이 모두 있을 때 `mode="real"`, 하나라도 없으면 `mode="disconnected"`(누락된 변수 이름과 `python -m agent.ui --demo` 안내). 시작 시 서버 접속 가능 여부는 확인하지 않는다(응답 지연 방지; 호출 실패는 `chat()`이 안내).
    - **채팅**: `new_agent(llm_status, project_root, results_root) -> Agent | None`(`disconnected`면 None), `chat_turn(message, history, agent, llm_status, session_id) -> (history, agent)` —
      빈 입력은 무시, Agent가 없으면(지연 생성) 만들고, `disconnected`면 assistant 안내 메시지(LLM이 필요하다는 설명과 학습/예측 탭은 쓸 수 있다는 안내), 그 외에는 `chat()` 결과를 assistant 메시지로 추가.
    - **학습**: `quick_settings(base_model)`(CPU에서 빠르게 확인할 수 있는 작은 설정: 모델별 step/배치/모델 크기), `parse_custom_settings(text) -> (dict, 오류)`(JSON 객체만, 값은 정수/문자열), `training_defaults_markdown(base_model)`(기본값과 바꿀 수 있는 파라미터 표),
      `training_preview(base_model, use_quick, custom_json, project_root)`(빠른 설정 위에 사용자 JSON을 덮어 최종 설정과 덮어쓰기 경고를 마크다운으로, 실행하지 않음), `training_start(base_model, use_quick, custom_json, confirmed, overwrite, project_root)`
      (**화면 수준 확인**: `confirmed`가 False면 시작하지 않고 확인 안내, 기존 결과가 있으면 `overwrite` 체크 없이는 시작하지 않음 — `start_training`의 `confirmed`/`overwrite` 의미 그대로), `training_status_view(project_root) -> (상태 마크다운, 로그 텍스트, 실행 중 여부)`
      (진행 막대 프레임을 거른 최근 로그 20줄).
    - **예측**: `molecule_choices(query, limit=50)`(분자 ID 목록), `prediction_view(molecule_id, base_model, show_experimental, results_root, project_root) -> (곡선 표 DataFrame, 요약 마크다운)` — 곡선 표는 열 `wavelength_nm`, `intensity`, `종류`(`예측`/`실험`),
      요약은 분자·모델·체크포인트·피크 파장, **비정상 곡선(최댓값 > 1.001 또는 최솟값 < -0.05) 경고**, 학습된 모델이 없으면 학습 탭 안내, 알 수 없는 분자 ID는 오류 메시지. 실험 곡선은 `nm = 1240/eV`로 변환해 파장 오름차순.
  - 미포함: Gradio 컴포넌트와 이벤트 연결(6B), 실행 진입점(6B), 브라우저 확인(6B REVIEW), 비동기/스트리밍 응답, 여러 사용자 인증.
- 테스트 계획 (`tests/agent_tools/test_ui_logic.py`; 새 모듈 import는 각 테스트의 도우미 안에서. LLM은 `fake_llm`/`MockLLMServer`, 학습은 가짜 명령, 예측은 tiny PaiNN 체크포인트 + 실제 IrDB)
  1. `test_demo_모드는_데모_서버를_한_번만_띄우고_진짜_LLM이_아님을_알린다`
  2. `test_실제_모드는_필수_환경변수가_모두_있어야_하고_없으면_누락된_이름과_데모_실행법을_안내한다`
  3. `test_연결되지_않은_상태의_채팅은_LLM이_필요하다는_안내를_돌려주고_Agent를_만들지_않는다`
  4. `test_채팅은_첫_메시지에서_Agent를_만들고_세션_동안_재사용하며_대화를_누적한다` (가짜 LLM으로 두 턴, 같은 Agent 객체, history에 user/assistant 메시지가 쌓임)
  5. `test_빈_메시지는_무시한다` / `test_범위_밖_질문은_채팅에서도_거절_문구로_답한다`
  6. `test_빠른_설정은_세_모델_모두_검증을_통과하고_기본값보다_작다`
  7. `test_사용자_JSON은_객체만_허용하고_잘못된_입력은_오류_문구를_돌려준다` (JSON이 아님, 배열, 중첩 값, 빈 문자열은 빈 설정)
  8. `test_학습_기본값_표시에는_기본값과_바꿀_수_있는_파라미터가_담긴다`
  9. `test_학습_미리보기는_실행하지_않고_최종_설정과_덮어쓰기_경고를_보여_준다` (빠른 설정 + 사용자 JSON 병합, 사용자 JSON이 우선)
  10. `test_확인_체크_없이는_학습을_시작하지_않는다` / `test_확인하면_학습이_시작되고_기존_결과가_있으면_덮어쓰기_체크가_필요하다`
  11. `test_학습_상태_보기는_상태와_진행_막대를_걸러낸_로그를_돌려준다` (진행 막대 프레임이 섞인 가짜 학습 로그, 20줄 제한, 실행 중 여부, 작업 없음)
  12. `test_분자_선택_목록은_검색어로_걸러진다`
  13. `test_예측_보기는_예측_곡선_표와_요약을_돌려준다` (tiny 체크포인트, `종류=예측` 800행, 피크 파장이 `predict_spectrum`과 일치)
  14. `test_실험_스펙트럼을_함께_보면_파장으로_변환한_실험_곡선이_추가된다` (`종류=실험`, 파장 400~800 범위, 강도 0~1)
  15. `test_비정상_곡선은_경고를_표시한다` / `test_학습된_모델이_없으면_학습_탭을_안내한다` / `test_알_수_없는_분자는_오류_메시지를_돌려준다`
- RED 검증 기준: 모두 `agent.ui_logic` 부재(`ModuleNotFoundError`)로 실패. 기존 251개는 영향 없이 통과.
- 완료 조건(REVIEW 종료 시): 신규 + 기존 테스트 통과, 실제 tiny 학습 → 상태 → 예측 표시를 로직 함수만으로 연결, 로그 필터를 실제 Geoformer 학습 로그로 확인, 리팩토링 필요성 검토.

### Step 6B. Gradio 화면 + 실행 진입점 (`agent/ui.py`)
- 범위(예정, 세부는 6B RED에서 확정)
  - `build_ui(llm_status, project_root, results_root) -> gr.Blocks`: 상단에 LLM 상태 배너(데모 모드는 "진짜 LLM이 아님" 강조), 탭 3개 —
    ① 채팅: `Chatbot` + 입력창 + `gr.Examples`(`agent.guard.ALLOWED_EXAMPLES`와 거절 시연용 `REFUSED_EXAMPLES`), 세션별 `gr.State`에 Agent, ② 학습: 모델 선택·기본값 표시·"작은 설정으로 빠르게 시험" 체크(CPU 기본)·사용자 JSON·미리보기 버튼·
    확인/덮어쓰기 체크·시작 버튼·`gr.Timer`로 갱신하는 상태/로그, ③ 예측: 분자 검색·선택, 모델 선택(자동/세 모델), 실험 스펙트럼 함께 보기 체크, 예측 버튼, `LinePlot`(예측/실험 색 구분)과 요약.
  - 이벤트에 `api_name`을 부여해 `gradio_client`로 테스트: `/chat`, `/training_defaults`, `/training_preview`, `/training_start`, `/training_status`, `/molecules`, `/predict`.
  - `python -m agent.ui [--demo] [--port 7860]`: `--demo`이면 `resolve_llm(demo=True)`, 아니면 환경변수. 서버는 `127.0.0.1`에만 바인딩.
  - 테스트(예정): `build_ui` 구성, 엔드포인트 목록, 실제 `launch`(`prevent_thread_lock`) + `gradio_client`로 `/predict`, `/training_preview`→`/training_start`(가짜 학습), `/chat`(데모 서버) 호출, `python -m agent.ui --demo` 서브프로세스 기동 스모크(HTTP 200), REVIEW에서 브라우저로 화면 확인.

## Step 7. E2E smoke 및 정리
- 목표: 성공 기준 1~4 확인 (G1~G6).
- 작업: `slow` E2E(tiny 학습 → registry → 예측 → 곡선), 선택적 `vllm` 마커 smoke(`VLLM_BASE_URL` 없으면 skip),
  README에 Agent 실행 방법과 환경변수 문서화, 최종 전체 테스트 실행.
- 완료 조건: 전체 테스트 통과(slow 포함), 문서 간 Step/G 번호 일치.

---

## 의존 관계
Step 0 → 1 → 2 → 3 → 4(독립 가능, 3 이후 권장) → 5(1,3,4 필요) → 6(5 필요) → 7.
