# Plan.md — Phase 1 리팩토링 실행 계획

목표(G1~G6)와 비목표는 [PRD.md](PRD.md), 절대 제약은 [CLAUDE.md](CLAUDE.md)를 따른다.
각 Step은 **완료 조건(Definition of Done)을 만족하고 회귀 테스트가 통과해야만** 다음 Step으로
넘어간다. 각 Step은 원칙적으로 별도 커밋(또는 커밋 그룹)으로 분리한다.

## 이 문서와 TDD 사이클의 관계

모든 작업은 [`.claude/TDD/SKILL.md`](.claude/TDD/SKILL.md)의 RED → GREEN → REVIEW 사이클을 따른다.
각 Step을 진행할 때, RED 단계에서 작성/갱신하는 "이번 사이클의 목표/범위/테스트 계획"은 **새 파일이
아니라 해당 Step 섹션 아래에 하위 항목으로 기록**한다. 커밋은 RED 종료 시점과 REVIEW 종료 시점에
자동으로 실행된다 (CLAUDE.md 제약 6).

각 Step은 아래 두 성격 중 하나로 표시되어 있다:
- **[구조 이동]**: 파일/함수 위치만 바꾸고 동작은 바꾸지 않는다. Step 1에서 만든
  characterization test가 오라클이므로, 새로 실패하는 테스트를 억지로 만들지 않는다 — 이동
  전후로 기존 테스트가 계속 GREEN을 유지하는지가 검증 기준이다.
- **[신규 인터페이스]**: 지금까지 없던 공개 인터페이스(adapter, `predict()` 등)를 추가한다.
  통상적인 TDD RED(아직 없어서 실패하는 테스트)부터 시작한다.

## 0. 사전 확정 사항 (환경)

- 작업 가상환경: `venv_spectrum_cpu` (Python 3.12.10, Windows, CPU-only).
  모든 테스트/스모크 실행은 이 환경에서 수행한다.
- 데이터셋(`IrDB/`, `IrDB_uff/`, `IrDB_murcko/`, `PtDB/`, `dataset/IrDB.py`)은 리팩토링 대상이
  아니다. 손대지 않는다.
- CPU 환경 이슈 (`CLAUDE.md` 참고):
  - `engine.py:84`의 `torch.cuda.synchronize()` 가드 필요.
  - `train_Geoformer.py`의 `--accelerator` 기본값이 `gpu`이므로 CPU 스모크 실행 시
    `--accelerator cpu --ndevices 1` 명시 필요.

---

## Step 0 — Legacy 보존 + 스냅샷 태깅 [환경 설정 — TDD 사이클 대상 아님]

**목적:** 리팩토링 시작 시점의 동작을 오라클로 고정한다.

**작업:**
1. 현재 `main` 브랜치 HEAD에 태그(예: `pre-refactor-snapshot`)를 남긴다.
2. `venv_spectrum_cpu` 가상환경을 생성하고 `README.md`의 GPU용 설치 절차를 CPU용으로
   변형하여 의존성을 설치한다 (torch CPU wheel 등). 이 절차는 `README.md`를 건드리지 않고
   별도로 기록한다 (예: 이 문서 하단 부록 또는 별도 `docs/dev_setup_cpu.md` — 필요 시 후속 Step에서 결정).
3. `engine.py:84`의 `torch.cuda.synchronize()`에 `if torch.cuda.is_available():` 가드를 추가한다.
   (`CLAUDE.md` 제약 2의 명시적 예외. 단일 커밋으로 분리.)

**완료 조건:**
- [ ] `pre-refactor-snapshot` 태그 존재.
- [ ] `venv_spectrum_cpu`에서 `python -c "import torch; print(torch.__version__)"` 성공.
- [ ] `engine.py`의 `torch.cuda.synchronize()` 가드 추가 커밋 1건, 다른 변경 없음.

---

## Step 1 — Characterization Test 작성 [오라클 구축 — 이후 모든 구조 이동 Step의 전제조건]

**목적:** 이후 모든 구조 변경의 오라클이 될 회귀 테스트를 **구조 변경 이전에** 확보한다.
이 Step에서는 프로덕션 코드를 옮기지 않는다. 오직 `tests/`만 추가한다.

**작업:** 각 모델(Geoformer, PaiNN, Equiformer)에 대해 CPU에서 수 초 내로 끝나는 tiny fixture
(원자 수 최소, batch size 1~2, `num-layers`/`embedding-dim` 등은 legacy 기본값을 그대로 쓰되
fixture 자체는 별도의 합성 mock 분자 좌표/원자번호를 사용— 실제 데이터셋 파일에 의존하지 않는다)로
다음을 캡처한다:

1. **Model construction**: parameter 개수, parameter 이름 목록, 각 parameter의 shape.
2. **고정 seed + 고정 tiny batch에서의 forward output** (수치 그대로 `.npy`/`.pt`로 저장하거나,
   테스트 내부에서 legacy import 경로로 직접 재계산하여 비교).
3. **FC/GMM physics 함수의 입력 → 출력**: `spectrum/loss.py`의 `spectrum_fc`, `spectrum_gmm`,
   `fc_loss`, `gmm_loss`를 고정 입력에 대해 캡처. (이 부분은 `spectrum/` 내부이므로 G5 라이센스
   경계상 테스트 코드에서 import만 하고 로직을 복제하지 않는다.)
4. **1회 backward 후 gradient**: 고정 batch로 loss를 1회 backward한 뒤 주요 parameter의
   `.grad` 값.
5. **1회 optimizer step 후 parameter**: legacy `optim_factory.create_optimizer` (또는
   Geoformer의 `AdamW` 구성)로 1 step 진행한 뒤 parameter 값.

**주의:**
- Equiformer/PaiNN 경로는 `engine.py`의 `train_one_step`/`evaluate`를 오라클로 그대로 호출해서
  캡처한다 (아직 옮기지 않았으므로 legacy 그 자체).
- Geoformer 경로는 `geoformer/module.py`의 `LNNP.training_step`/`spectrum_step`을 오라클로
  그대로 호출한다.
- 테스트 함수명은 `def test_한글설명` 컨벤션을 따른다 (예: `def test_PaiNN_forward_output_고정됨`).
- 파일 배치 예시: `tests/characterization/test_geoformer_oracle.py`,
  `tests/characterization/test_painn_oracle.py`, `tests/characterization/test_equiformer_oracle.py`,
  `tests/characterization/test_spectrum_physics_oracle.py`.

**완료 조건:**
- [x] 세 모델 모두 construction/forward/gradient/optimizer-step 오라클 테스트 존재.
- [x] FC/GMM physics 함수 오라클 테스트 존재.
- [x] `venv_spectrum_cpu`에서 `pytest tests/` 전체 통과 (13 passed, ~30초).
- [x] 이 Step의 커밋에는 `tests/`와 `pytest.ini` 추가만 포함되고 프로덕션 코드 변경은 없다.

**실제 구현 노트 (계획과 달라진 부분):**
- 합성 fixture는 `tests/support/tiny_batches.py`에 모아뒀다. 분자는 탄소/산소 2종 원자로 이루어진
  2~3원자짜리 초소형 분자 2개(batch size 2)이며, 모든 모델의 cutoff(5.0Å)보다 훨씬 가깝게
  배치해 `radius_graph`가 빈 그래프를 만들지 않도록 했다.
- Equiformer/PaiNN 순방향은 f_in(원자 feature)과 edge_d_index/edge_d_attr을 실제로 사용하지
  않는다는 것을 코드로 확인했다 (Equiformer는 `atom_embed(node_atom)`만 사용, PaiNN 래퍼는
  자체 `radius_graph`로 그래프를 재계산). 따라서 fixture의 `x`/`edge_d_*`는 인터페이스를 맞추기
  위한 더미 값이다.
- gradient/optimizer-step 오라클은 "주요 parameter의 `.grad` 값"을 개별적으로 저장하는 대신,
  전체 parameter에 대한 **gradient L2 norm**과 **step 이후 parameter L2 norm** 스칼라로
  압축해서 저장했다 (Equiformer는 파라미터가 3백만 개 이상이라 개별 저장이 비현실적).
  이 스칼라 하나가 깨지면 회귀가 발생했다는 신호로는 충분하며, 어떤 값이 깨졌는지 상세 진단은
  이후 Step에서 필요하면 추가한다.
- optimizer는 `optim_factory.create_optimizer`(CLI arg 파싱에 의존)가 아니라 `torch.optim.AdamW`를
  직접 사용했다. `optim_factory.py`는 Plan.md의 Step으로 명시적으로 다루지 않으므로 범위 밖으로
  판단했다.
- Geoformer는 `geoformer/module.py`의 `LNNP`(Lightning 래퍼) 대신, `LNNP.forward`/
  `spectrum_step`이 실제로 호출하는 `geoformer.model.modeling_geoformer.create_model`과
  `spectrum.loss.fc_loss`를 직접 호출했다. Lightning `Trainer` 전체를 오라클 테스트마다 띄우는
  것은 무겁고, 두 함수 호출이 LNNP의 핵심 수치 로직 전부이기 때문이다. 전체 CLI 경로는 Step 0에서
  이미 스모크 테스트로 별도 검증했다.
- 오라클 값은 하드코딩 대신 `tests/support/golden.py`의 golden-fixture 패턴으로 저장한다: 처음
  실행 시 골든 파일이 없으면 현재(legacy) 출력을 `tests/characterization/golden/`에 저장하고
  일부러 실패(RED)하며, 재실행하면 그 파일과 비교해 통과(GREEN)한다. 실제로 모든 테스트를 이
  2단계로 실행해 RED→GREEN 전환을 확인했다.
- Equiformer는 named_parameters가 매우 많아(수백 개) construction 구조 골든 테스트는 생략했다
  (PaiNN/Geoformer만 파라미터 이름/shape 구조를 골든으로 남겼다). forward/gradient/
  optimizer-step/evaluate 골든은 세 모델 모두 존재한다.

---

## Step 2 — Configuration 분리 [구조 이동]

**목적:** 세 `train_XXX.py`에 흩어진 argparse 정의 중 **아키텍처 독립적인 인자**
(batch-size, seed, output-dir, spectrum-type, n-mode, lineshape, beta, split-index-npz,
loss, workers, pin-mem, 분산학습 인자 등)를 공통 모듈로 추출한다.

- 아키텍처 고유 인자(예: Equiformer의 `--model-name`/`--input-irreps`, PaiNN/Equiformer의
  `--radius`/`--num-basis`, Geoformer의 Lightning 관련 인자)는 그대로 각 스크립트/adapter에 남긴다.
- 인자 이름 불일치(`--dataset-root` vs `--data-path`, `--spec-loss-type` vs `--spectrum-type`)는
  이 Step에서 **강제로 통일하지 않는다** — 이름 통일은 CLI 동작(도움말, 에러 메시지)을 바꾸는
  것이므로 별도로 사용자와 논의 후 결정한다. 우선은 공통 로직(파싱된 값의 후처리, 예:
  `n_mode`에 따른 target list 생성)만 공유 함수로 뽑는다.
- 대상 함수 예시: `load_split_from_npz`, target list 생성 로직
  (`train_PaiNN.py:358-373` / `train_Equiformer.py:297-314`에서 거의 동일하게 반복되는 부분).

**완료 조건:**
- [x] 공통 config/타겟 생성 로직이 단일 모듈(`common/training_utils.py`)에 존재.
- [x] `train_PaiNN.py`, `train_Equiformer.py`가 이를 import해서 사용, 중복 코드 삭제.
- [x] Step 1의 characterization test 재실행 통과 (23 passed) + PaiNN CLI 스모크 테스트로
      Step 0과 완전히 동일한 수치(Training set mean/std, val/test MAE·loss) 확인.

**이번 TDD 사이클 (완료):**
- 목표: `load_split_from_npz`, `save_pred`, `warmup_exponential_decay`, 그리고
  `train_PaiNN.py`/`train_Equiformer.py`의 `__main__`에 중복된 spectrum-type→target list
  생성 로직(`build_spectrum_targets`로 명명)을 `common/training_utils.py`로 이동한다.
- 범위: 이 4개 함수의 이동만. `--dataset-root`/`--data-path` 등 인자 이름 통일은 포함하지 않는다.
- RED: `tests/refactor/test_common_training_utils.py` — 아직 없는 `common.training_utils`
  모듈을 import하여 `ModuleNotFoundError`로 실패하는 것을 확인.
- GREEN: `common/training_utils.py`에 4개 함수를 그대로(값 변경 없이) 옮겨 구현, 10개 테스트 통과.
- REVIEW: `train_PaiNN.py`/`train_Equiformer.py`에서 중복 정의 삭제 후 `common.training_utils`
  import로 교체, `__main__`의 spectrum-type 분기 18줄을 `build_spectrum_targets` 호출 1줄로 교체.
  더 이상 쓰이지 않게 된 `import pandas as pd`도 함께 제거(같은 이동의 직접적 귀결이라 범위 내로 판단).
  Step 1 오라클 23개 + Step 2 신규 10개 테스트 모두 통과, PaiNN CLI 스모크 테스트로 Step 0과
  동일한 수치 재확인.

---

## Step 3 — Model Construction 분리 (G3: Registry/Adapter) [신규 인터페이스]

**목적:** `build_painn`(`train_PaiNN.py:99-106`), `model_entrypoint`/`Equiformer` 레지스트리
호출(`train_Equiformer.py:176-183`), `create_model`(`geoformer/model/modeling_geoformer.py`)을
공통 registry 뒤로 캡슐화한다.

**작업:**
1. adapter 인터페이스를 정의한다 (예: `build(args) -> nn.Module`, 그리고 PaiNN/Equiformer처럼
   `engine.py` 스타일 forward를 쓰는 경로에 대해 `forward(model, batch) -> Tensor`).
2. 기존 `PaiNN` 래퍼 클래스(`train_PaiNN.py:46-97`, `SimpleNamespace` 기반 fairchem 어댑팅
   포함)와 Equiformer의 `model_entrypoint` 호출을 각각 adapter 모듈로 이동한다.
3. registry는 `{"Geoformer": ..., "PaiNN": ..., "Equiformer": ...}` 형태의 매핑을 제공한다.
   Geoformer는 Lightning 경로이므로 registry에는 "adapter 존재"만 등록하고, 실제 학습
   진입점은 여전히 `train_Geoformer.py`/`geoformer/module.py`를 그대로 사용한다 (비목표: 강제
   통합 금지).

**완료 조건:**
- [x] `train_PaiNN.py`, `train_Equiformer.py`에서 모델 생성 코드가 각각의 adapter 모듈 호출
      한 줄로 축소.
- [x] registry를 통해 문자열 키로 adapter를 조회하는 테스트(`tests/refactor/test_model_adapters.py`) 존재.
- [x] Step 1 오라클 대비 forward output 동일함을 회귀 테스트로 확인 (5개 테스트 모두 첫 실행에서
      바로 골든과 일치, 즉 adapter가 legacy와 100% 동일한 계산을 수행).

**이번 TDD 사이클 (완료):**
- 목표/범위: RED 노트와 동일 (adapter 모듈 신설 + registry, train_XXX.py의 모델 생성 줄만 교체).
- RED: `common.adapters`가 없어 `ModuleNotFoundError`로 실패 확인.
- GREEN: `common/adapters/{painn,equiformer,geoformer}_adapter.py` + `__init__.py`의
  `ADAPTERS`/`get_adapter()`를 추가. PaiNN 래퍼 클래스는 `train_PaiNN.py`에서 그대로(값 변경 없이)
  옮겼고, Equiformer/Geoformer adapter는 각각 `model_entrypoint`/`create_model`을 얇게 감쌌다.
  5개 테스트 모두 통과 (Step 1 golden과 정확히 일치).
- REVIEW:
  - `train_PaiNN.py`: `PaiNN` 클래스/`_rbf`/`build_painn`와 이제 쓰이지 않는
    `SimpleNamespace`/`torch_cluster.radius_graph`/`fairchem...PaiNN`/`torch.nn` import를 삭제하고
    `common.adapters.painn_adapter.build(args)` 호출로 교체 (`args.out_channels = len(args.targets)`
    를 먼저 설정).
  - `train_Equiformer.py`: `model_entrypoint` 직접 호출을 `common.adapters.equiformer_adapter.build(args)`
    호출로 교체, `args.out_channels`/`args.task_mean`/`args.task_std`를 먼저 설정. 더 이상 쓰이지
    않는 `import Equiformer`/`from Equiformer import model_entrypoint` 제거.
  - `tests/characterization/test_painn_oracle.py`의 `from train_PaiNN import PaiNN`을
    `from common.adapters.painn_adapter import PaiNN`로 갱신 (클래스가 옮겨졌으므로 이동의
    직접적 귀결).
  - 검증: `pytest tests/` 28개 전체 통과. PaiNN·Equiformer CLI 스모크 테스트 재실행 결과, PaiNN은
    Step 0/2와 완전히 동일한 수치. Equiformer는 MAE는 완전히 동일했으나 FC loss 값이 소수
    8번째 자리에서 흔들렸다 — 같은(리팩터링 후) 커맨드를 다시 한 번 더 실행해보니 그때도 또
    다른 값이 나와, **이 흔들림은 이번 변경과 무관하게 이미 존재하던 현상**임을 확인했다
    (torch_cluster의 CPU 병렬 reduction이 완전히 결정적이지 않고, `fc_loss`의 hinge 계수가
    1e6이라 그 미세한 float32 오차가 손실 값에서 상대오차 ~1e-8 수준으로 눈에 보이게 증폭됨).
    이후 Step에서 실제 데이터셋으로 스모크 비교를 할 때는 MAE(또는 hinge 계수가 없는 지표)를
    기준으로 비교하는 것이 더 안정적이다.

---

## Step 4 — Physics Module 정리 (spectrum 내부 전용) [구조 이동]

**목적:** `spectrum/loss.py`를 `spectrum/` **내부에서만** 재구성한다 (G5 라이센스 경계 — 외부
폴더와의 이동/병합 없음, `spectrum/` 내부 재구성은 자유).

**작업:**
- `spectrum/loss.py` → `spectrum/physics/fc.py`(`spectrum_fc`, `fc_loss`),
  `spectrum/physics/gmm.py`(`spectrum_gmm`, `gmm_loss`), 공통 유틸(`fn_spec_loss`, 상수:
  `coeff_hinge`, `eps`, `min_cut`, `max_cut`, `min_ev`, `max_ev`)은
  `spectrum/physics/common.py` 등으로 분리하는 방식을 우선 고려한다 (정확한 분할은 작업 시점에
  가독성 기준으로 결정 가능 — 단, 로직/값 자체는 변경 금지).
- `spectrum/loss.py`를 import하는 외부 파일(`engine.py`, `geoformer/module.py`,
  `train_PaiNN.py`, `train_Equiformer.py`)의 import 경로를 갱신한다. **이 파일들의 로직은
  변경하지 않고 import 문만 갱신한다.**

**완료 조건:**
- [x] `spectrum/loss.py`의 함수가 새 위치로 이동, import처 갱신.
- [x] Step 1의 physics 오라클 테스트가 새 import 경로로도 동일 수치 반환.
- [x] `git diff`에 `spectrum/` 외부 폴더의 로직 변경이 없고 import 문 변경만 존재함을 확인.

**이번 TDD 사이클 (완료):**
- RED: `tests/refactor/test_spectrum_physics_relocated.py`로 `spectrum.physics`가 없어
  `ModuleNotFoundError`로 실패 확인.
- GREEN: `spectrum/physics/{common,fc,gmm}.py` + `__init__.py`(재노출)를 추가, 로직/상수는
  `spectrum/loss.py`에서 그대로 옮김. 3개 테스트 모두 Step 1 golden과 즉시 일치.
- REVIEW: `engine.py`, `geoformer/module.py`, `spectrum/write.py`, 두 characterization 테스트의
  `from spectrum.loss import ...`를 `from spectrum.physics import ...`로 교체(로직은 손대지 않음).
  `train_PaiNN.py`/`train_Equiformer.py`는 `spectrum.loss`를 직접 import하지 않음을 grep으로
  재확인(변경 없음). `spectrum/loss.py` 삭제, 그리고 Step 1 오라클과 완전히 중복이 된
  `test_spectrum_physics_relocated.py`도 함께 삭제(같은 것을 같은 golden으로 두 번 검증할
  이유가 없어짐 — 영구 오라클은 `test_spectrum_physics_oracle.py` 하나로 유지).
  검증: `pytest tests/` 28개 전체 통과, PaiNN CLI 스모크 테스트가 Step 0/2/3과 완전히 동일한
  수치. `git diff` 확인 결과 `spectrum/` 외부 파일은 import 문 한 줄씩만 바뀌었다.

---

## Step 5 — Data Layer 분리 [구조 이동]

**목적:** dataset 선택(`IrDB` vs `PtDB`), split 로딩, `DataLoader` 생성 로직을 공통화한다.

**작업:**
- `train_PaiNN.py:212-222`/`train_Equiformer.py:144-154`의 dataset 선택 분기와
  `train_loader`/`val_loader`/`test_loader` 생성 로직(distributed sampler 분기 포함)을
  공통 함수로 추출한다.
- Geoformer의 `geoformer/data.py`(`DataModule`)는 Lightning `DataModule` 인터페이스이므로
  강제 통합하지 않는다. 다만 "동일한 split npz를 읽어 동일한 idx_train/val/test를 만든다"는
  불변식이 있는지 characterization test로 교차 검증한다 (이미 Step 1에서 다뤘다면 생략 가능).

**완료 조건:**
- [x] PaiNN/Equiformer 경로의 dataset/dataloader 생성 코드 중복 제거.
- [x] Step 1 오라클 대비 동일 batch 구성 확인 (PaiNN CLI 스모크 테스트로 val/test MAE·loss가
      Step 0/2/3/4와 완전히 동일함을 재확인).

**이번 TDD 사이클 (완료):**
- RED: `tests/refactor/test_common_data.py`로 `common.data`가 없어 `ModuleNotFoundError`로
  실패 확인.
- GREEN: `common/data.py`에 `load_dataset_splits(args)`/`build_dataloaders(args, ...)`를
  로직 그대로 구현. "since dataset needs random" 주석의 seed 재설정을
  `load_dataset_splits`가 반환하기 직전에 유지(원본과 동일한 시점). 4개 테스트 모두 통과.
- REVIEW: `train_PaiNN.py`/`train_Equiformer.py`의 `''' Dataset '''`/`''' Data Loader '''`
  블록을 각각 두 함수 호출로 교체. 더 이상 쓰이지 않게 된 `from dataset.IrDB import IrDB, PtDB`,
  `from torch_geometric.loader import DataLoader`, `common.training_utils.load_split_from_npz`
  import를 제거(같은 이동의 직접적 귀결). `utils.get_world_size/get_rank` 직접 호출도
  `common/data.py` 내부로만 남음.
  검증: `pytest tests/` 32개 전체 통과, PaiNN CLI 스모크 테스트가 Step 0/2/3/4와 완전히 동일한
  수치 (behavior-preserving).

---

## Step 6 — Evaluation 공통화 [구조 이동 — 외부 시그니처 불변]

**목적:** `engine.py`의 `evaluate()`(현재 loss 계산 + 예측 수집이 혼재)를 정리하여 G4(Training/
Inference 분리)의 기반을 만든다.

**작업:**
- `evaluate()`의 "모델 forward → 예측 생성" 부분과 "예측 vs 정답으로 loss/지표 계산" 부분을
  내부적으로 분리 가능한 형태로 리팩토링한다 (완전히 새 함수로 나누되, 외부에서 호출하는
  `evaluate(...)` 시그니처/반환값(`mae_metric.avg, loss_metric.avg, preds, ids`)은 그대로 유지
  — G1 위반 방지).
- Adapter의 `forward(model, batch)` (Step 3)를 `evaluate()` 내부에서 사용하도록 교체한다.

**완료 조건:**
- [x] `evaluate()`가 내부적으로 "예측 생성" 함수를 호출하는 구조로 변경, 외부 시그니처/반환값 불변.
- [x] Step 1 오라클 대비 evaluate 결과(MAE, loss, preds) 동일 (34개 테스트 전체 통과,
      기존 `test_PaiNN_evaluate가_고정된다`/`test_Equiformer_evaluate가_고정된다`가 그대로 통과).

**이번 TDD 사이클 (완료):**
- RED: `tests/refactor/test_engine_style_forward.py`로 `common.adapters._shared`가 없어
  `ModuleNotFoundError`로 실패 확인.
- GREEN: `common/adapters/_shared.py`에 `engine_style_forward(model, batch)`를 추가하고,
  `painn_adapter.py`/`equiformer_adapter.py`의 중복된 `forward` 함수 정의를
  `from common.adapters._shared import engine_style_forward as forward`로 교체(진짜 동일
  함수 객체가 되었음을 `is` 비교로 확인). 2개 테스트 통과.
- REVIEW: `engine.py`에 `predict_batch(model, data, task_mean, task_std)`(forward+unnormalize)와
  `compute_spec_loss(data, pred, criterion, loss_type, spec_type, line_shape, beta)`(지표 계산)
  헬퍼를 추가하고, `evaluate()` 내부의 `model(f_in=...,...)` 직접 호출과 spec_type 분기를 각각
  이 헬퍼 호출로 교체했다. `evaluate()`의 외부 시그니처/반환값(`mae_metric.avg, loss_metric.avg,
  preds, ids`)은 그대로다. `train_one_step()`은 이번 Step에서 건드리지 않음(Step 7에서 같은
  헬퍼를 재사용할 예정).
  검증: `pytest tests/` 34개 전체 통과, PaiNN CLI 스모크 테스트가 이전 Step들과 완전히 동일한
  수치 (behavior-preserving).

---

## Step 7 — Training Step 공통화 [구조 이동 — Step 3 adapter를 사용하도록 배선만 교체]

**목적:** `engine.py`의 `train_one_step()`도 Step 6과 동일한 방식으로 adapter의
`forward(model, batch)`를 사용하도록 정리하고, PaiNN/Equiformer 경로에서 옵티마이저 스텝
전후의 로직(grad clip, loss 분기)을 공통화한다.

**작업:**
- `train_one_step()` 내부의 `model(f_in=..., pos=..., ...)` 직접 호출을
  adapter의 `forward(model, batch)` 호출로 교체한다 (Step 3 registry 사용).
- Naive/GMM/FC loss 분기(`engine.py:66-71`, `:132-137`)는 이미 `spec_type` 문자열 분기로
  공통화되어 있으므로 구조는 유지하되, 위치만 필요 시 정리한다.

**완료 조건:**
- [x] `train_one_step()`이 특정 모델의 forward 시그니처에 더 이상 직접 결합되지 않음
      (adapter를 통해서만 호출).
- [x] Step 1 오라클 대비 1 step 학습 후 loss/gradient/parameter 값 동일.

**이번 사이클 (단일 커밋 — 새 모듈 없음):**
Step 6에서 이미 만든 `predict_batch`/`compute_spec_loss` 헬퍼(내부적으로
`common.adapters._shared.engine_style_forward` 사용)를 `train_one_step()`도 그대로
재사용하도록 배선만 바꿨다. 새로운 인터페이스가 생기지 않는 순수 배선 교체이므로,
별도의 실패 테스트를 새로 만들지 않고 Step 1의 기존 golden 테스트
(`test_PaiNN_1스텝_학습후_...`/`test_Equiformer_1스텝_학습후_...`)가 계속 GREEN을
유지하는지로 검증했다 (34개 테스트 전체 통과 + PaiNN CLI 스모크 테스트 수치 불변).
RED/REVIEW로 나눌 대상이 없어 한 커밋으로 처리한다.

---

## Step 8 — train.py 통합 [구조 이동 — 최종 학습 결과 불변]

**목적:** `train.py`가 `os.system()`으로 서브프로세스 문자열을 조립하는 대신, 공통화된 구성
요소(Step 2~7)를 사용해 PaiNN/Equiformer 경로를 일관되게 기동하도록 정리한다.

**작업:**
- Geoformer 경로는 Lightning 기반이므로 기존처럼 `train_Geoformer.py`를 별도 진입점으로 유지한다
  (비목표: 강제 통합 금지). `train.py`의 디스패치 방식(서브프로세스 vs 직접 함수 호출)은 구조
  변경 허용 범위이나, **최종적으로 동일한 CLI 인자에 대해 동일한 학습 결과가 나와야 한다** (G1).
- CPU 환경에서 `train.py --base-model Geoformer`가 실패하는 문제(`CLAUDE.md` 참고,
  `--accelerator` 미전달)는 이 Step에서 수정 대상으로 삼되, **GPU 기본 동작은 바꾸지 않고**
  CPU에서 override 가능하게 하는 방식으로 처리한다 (예: 환경 감지 후 CPU일 때만
  `--accelerator cpu --ndevices 1` 자동 추가, 또는 명시적 플래그 추가 — 방식은 구현 시점에 결정).

**완료 조건:**
- [x] `train.py --base-model {Geoformer|PaiNN|Equiformer}`가 GPU 환경에서 기존과 동일한
      최종 커맨드/설정으로 귀결됨을 확인 (회귀 테스트로 리터럴 비교).
- [x] CPU 환경에서 `train.py --base-model Geoformer`가 최소 smoke 수준으로 동작 (accelerator
      플래그 자동 추가 확인; PaiNN 경로는 실제 서브프로세스 dispatch까지 end-to-end로 확인).
- [x] Step 1 오라클 대비 전체 동작 동일.

**이번 TDD 사이클 (완료):**
- RED: `tests/refactor/test_train_dispatch.py`로 `build_command`/`resolve_split_npz`가 없어
  `ImportError`로 실패 확인.
- GREEN: `train.py`에 두 순수 함수를 추가. `build_command`는 GPU에서 legacy와 100% 동일한
  문자열을, CPU에서만 Geoformer 커맨드에 `--accelerator cpu --ndevices 1`을 추가(GPU 동작
  불변, CLAUDE.md 제약 2 예외). 8개 테스트 통과.
- REVIEW: `main()`이 두 함수를 호출하도록 재배선. 더 이상 쓰이지 않는(원래도 미사용이던)
  `train_file` 지역변수를 이 블록을 다시 쓰는 김에 함께 제거(작은 부수 정리로 커밋 메시지에
  명시).
  검증: `pytest tests/` 42개 전체 통과. 실제 `train.py --base-model PaiNN --batch-size 2
  --data-path IrDB`를 venv 파이썬이 PATH에 잡힌 상태로 실행해 `python -m train_PaiNN`
  서브프로세스가 정상 기동하고, 로그에 찍힌 Training set mean/std가 이전 모든 Step의
  스모크 테스트와 완전히 동일함을 확인 (behavior-preserving). Geoformer 경로는
  `torch.cuda.is_available()`을 CPU로 monkeypatch한 단위 테스트로 accelerator 플래그
  추가를 확인했다 (Lightning 전체 구동까지 매 Step마다 반복하는 대신, Step 0에서 이미
  Geoformer CLI 스모크 테스트를 별도로 완료했으므로 충분하다고 판단).

---

## Step 9 — Inference Skeleton 추가 [신규 인터페이스]

**목적:** G4의 "training/inference 분리" 골격을 완성한다.

**작업:**
- Step 6에서 분리한 "예측 생성" 함수를 기반으로, 학습 루프/optimizer/scheduler에 의존하지 않는
  `predict(model, batch, norm_factor, spec_type) -> Tensor` 형태의 얇은 Python API를 노출한다.
- 이 Step은 실제 프로덕션 추론 서비스를 만드는 것이 아니라, **다음 Phase(Agent Tool화 등)가
  이 API 위에 바로 얹힐 수 있는 골격**을 만드는 것이 목표다 (PRD 비목표 참고).

**완료 조건:**
- [ ] `predict()` 함수가 세 모델 모두에 대해(Geoformer는 Lightning `forward` 경로를 감싸는
      형태로) 존재하고 테스트로 커버됨.
- [ ] Step 1 오라클 대비 `predict()` 출력이 legacy forward 출력과 동일.

---

## 각 Step 공통 체크리스트 (CLAUDE.md 재확인용)

- [ ] 이 Step에서 값(하이퍼파라미터/수치 상수)이 하나도 바뀌지 않았는가?
- [ ] `spectrum/`과 외부 폴더 사이에 코드 이동/병합이 발생하지 않았는가?
- [ ] Step 1에서 만든 characterization test가 이 Step 이후에도 통과하는가?
- [ ] 이 Step의 커밋이 dead code 제거/이동/rename/engine 통합/config 이동 중 **하나의 성격만**
      담고 있는가?
- [ ] 테스트 함수명이 `def test_한글설명` 컨벤션을 따르는가?
