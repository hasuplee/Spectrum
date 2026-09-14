# Plan.md — Phase 1 리팩토링 실행 계획

목표(G1~G6)와 비목표는 [PRD.md](PRD.md), 절대 제약은 [CLAUDE.md](CLAUDE.md)를 따른다.
각 Step은 **완료 조건(Definition of Done)을 만족하고 회귀 테스트가 통과해야만** 다음 Step으로
넘어간다. 각 Step은 원칙적으로 별도 커밋(또는 커밋 그룹)으로 분리한다.

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

## Step 0 — Legacy 보존 + 스냅샷 태깅

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

## Step 1 — Characterization Test 작성

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
- [ ] 세 모델 모두 construction/forward/gradient/optimizer-step 오라클 테스트 존재.
- [ ] FC/GMM physics 함수 오라클 테스트 존재.
- [ ] `venv_spectrum_cpu`에서 `pytest tests/` 전체 통과, 각 테스트 수 초 이내로 종료.
- [ ] 이 Step의 커밋에는 `tests/`와 (필요 시) `pytest.ini`/`pyproject.toml` 추가만 포함되고
      프로덕션 코드 변경은 없다.

---

## Step 2 — Configuration 분리

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
- [ ] 공통 config/타겟 생성 로직이 단일 모듈(예: `common/spectrum_targets.py` — 정확한 위치는
      G5 라이센스 경계를 지키는 선에서 결정. `spectrum/`이 아닌 위치에 둔다)에 존재.
- [ ] `train_PaiNN.py`, `train_Equiformer.py`가 이를 import해서 사용, 중복 코드 삭제.
- [ ] Step 1의 characterization test 재실행 통과 (동일 CLI 인자로 동일 target list/동일 동작).

---

## Step 3 — Model Construction 분리 (G3: Registry/Adapter)

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
- [ ] `train_PaiNN.py`, `train_Equiformer.py`에서 모델 생성 코드가 각각의 adapter 모듈 호출
      한 줄로 축소.
- [ ] registry를 통해 문자열 키로 adapter를 조회하는 테스트(`tests/`) 존재.
- [ ] Step 1 오라클 대비 model construction(parameter 이름/shape) 및 forward output 동일함을
      회귀 테스트로 확인.

---

## Step 4 — Physics Module 정리 (spectrum 내부 전용)

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
- [ ] `spectrum/loss.py`의 함수가 새 위치로 이동, import처 갱신.
- [ ] Step 1의 physics 오라클 테스트가 새 import 경로로도 동일 수치 반환.
- [ ] `git diff`에 `spectrum/` 외부 폴더의 로직 변경이 없고 import 문 변경만 존재함을 확인.

---

## Step 5 — Data Layer 분리

**목적:** dataset 선택(`IrDB` vs `PtDB`), split 로딩, `DataLoader` 생성 로직을 공통화한다.

**작업:**
- `train_PaiNN.py:212-222`/`train_Equiformer.py:144-154`의 dataset 선택 분기와
  `train_loader`/`val_loader`/`test_loader` 생성 로직(distributed sampler 분기 포함)을
  공통 함수로 추출한다.
- Geoformer의 `geoformer/data.py`(`DataModule`)는 Lightning `DataModule` 인터페이스이므로
  강제 통합하지 않는다. 다만 "동일한 split npz를 읽어 동일한 idx_train/val/test를 만든다"는
  불변식이 있는지 characterization test로 교차 검증한다 (이미 Step 1에서 다뤘다면 생략 가능).

**완료 조건:**
- [ ] PaiNN/Equiformer 경로의 dataset/dataloader 생성 코드 중복 제거.
- [ ] Step 1 오라클 대비 동일 idx_train/val/test, 동일 batch 구성 확인.

---

## Step 6 — Evaluation 공통화

**목적:** `engine.py`의 `evaluate()`(현재 loss 계산 + 예측 수집이 혼재)를 정리하여 G4(Training/
Inference 분리)의 기반을 만든다.

**작업:**
- `evaluate()`의 "모델 forward → 예측 생성" 부분과 "예측 vs 정답으로 loss/지표 계산" 부분을
  내부적으로 분리 가능한 형태로 리팩토링한다 (완전히 새 함수로 나누되, 외부에서 호출하는
  `evaluate(...)` 시그니처/반환값(`mae_metric.avg, loss_metric.avg, preds, ids`)은 그대로 유지
  — G1 위반 방지).
- Adapter의 `forward(model, batch)` (Step 3)를 `evaluate()` 내부에서 사용하도록 교체한다.

**완료 조건:**
- [ ] `evaluate()`가 내부적으로 "예측 생성" 함수를 호출하는 구조로 변경, 외부 시그니처/반환값 불변.
- [ ] Step 1 오라클 대비 evaluate 결과(MAE, loss, preds) 동일.

---

## Step 7 — Training Step 공통화

**목적:** `engine.py`의 `train_one_step()`도 Step 6과 동일한 방식으로 adapter의
`forward(model, batch)`를 사용하도록 정리하고, PaiNN/Equiformer 경로에서 옵티마이저 스텝
전후의 로직(grad clip, loss 분기)을 공통화한다.

**작업:**
- `train_one_step()` 내부의 `model(f_in=..., pos=..., ...)` 직접 호출을
  adapter의 `forward(model, batch)` 호출로 교체한다 (Step 3 registry 사용).
- Naive/GMM/FC loss 분기(`engine.py:66-71`, `:132-137`)는 이미 `spec_type` 문자열 분기로
  공통화되어 있으므로 구조는 유지하되, 위치만 필요 시 정리한다.

**완료 조건:**
- [ ] `train_one_step()`이 특정 모델의 forward 시그니처에 더 이상 직접 결합되지 않음
      (adapter를 통해서만 호출).
- [ ] Step 1 오라클 대비 1 step 학습 후 loss/gradient/parameter 값 동일.

---

## Step 8 — train.py 통합

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
- [ ] `train.py --base-model {Geoformer|PaiNN|Equiformer}`가 GPU 환경에서 기존과 동일한
      최종 커맨드/설정으로 귀결됨을 확인 (회귀 테스트 또는 dry-run 비교).
- [ ] CPU 환경에서 `train.py --base-model Geoformer`가 최소 smoke 수준으로 동작.
- [ ] Step 1 오라클 대비 전체 동작 동일.

---

## Step 9 — Inference Skeleton 추가

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
