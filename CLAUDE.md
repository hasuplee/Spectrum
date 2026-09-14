# CLAUDE.md

## 프로젝트 개요

인광 OLED의 발광 스펙트럼을 예측하는 PIML(Physics-Informed Machine Learning) 프로젝트.
Geoformer, PaiNN, Equiformer 세 가지 3D 분자 그래프 모델을 백본으로 사용하고, 그 위에
Franck-Condon(FC) 또는 Gaussian Mixture Model(GMM) 기반의 물리적 스펙트럼 복원 로직(`spectrum/`)을
얹는 구조다.

현재는 **Phase 1 리팩토링** 단계다. 목적은 성능 개선이 아니라 **구조 개선
(behavior-preserving refactoring)** 이다. 세부 목표/비목표는 [PRD.md](PRD.md)를,
단계별 작업 순서는 [Plan.md](Plan.md)를 따른다.

## 절대 지켜야 할 제약

1. **라이센스 경계 (License Boundary)**
   - `spectrum/` 폴더는 CC BY-NC-SA 4.0, 그 외 루트/폴더(`geoformer/`, `Equiformer/`, `fairchem/`,
     `train_*.py`, `engine.py` 등)는 MIT 라이센스다 (`README.md` License 섹션 참고).
   - `spectrum/`과 그 외 폴더 사이의 소스 코드 **이동/복사/병합을 절대 하지 않는다.**
   - `spectrum/` 내부에서의 재구성(파일 분리, 함수 이동 등)은 자유롭다.
   - `spectrum/`에서 외부 아키텍처(Geoformer/PaiNN/Equiformer)를 참조해야 할 때는 반드시
     import 또는 adapter를 통해서만 연결한다. 직접 로직을 복사해 넣지 않는다.
   - **애매하면 옮기지 않는다.** 판단이 서지 않는 이동은 별도로 사용자에게 확인한다.

2. **No algorithmic / hyperparameter / numerical behavior change**
   - learning rate, batch size, cutoff, optimizer, scheduler, seed, dataset split,
     전처리(preprocessing) 등 **값은 절대 변경하지 않는다.**
   - 허용되는 변경은 오직 코드의 **위치(location)와 책임(responsibility)의 재배치**뿐이다.
   - 단, CPU 개발 환경에서 크래시를 유발하는 GPU 전용 호출(`torch.cuda.synchronize()` 등)에
     대한 가드 추가는 "값 변경"이 아니라 "동일 동작 보장"으로 간주하고 허용한다
     (GPU에서는 기존과 동일하게 동작해야 한다).

3. **Behavior-preserving 검증**
   - 구조를 바꿀 때마다 **레거시 코드를 오라클(oracle)로 삼아**, 다음 항목의 수치적 동일성을
     회귀 테스트로 확인한다:
     - forward output
     - loss 값
     - 1회 backward 후 gradient
     - 1회 optimizer step 후 parameter
     - spectrum reconstruction (FC/GMM physics 함수) 출력
   - **확인 없이 다음 단계로 넘어가지 않는다.**

4. **테스트 컨벤션**
   - 테스트 함수명은 `def test_한글설명` 형식으로 작성한다.
   - 모든 테스트는 `tests/` 하위에 작성한다.
   - 테스트 가능한 경계(모듈/함수 단위)에서는 반드시 테스트한다.

5. **커밋 단위 분리**
   - dead code 제거 / 파일 이동·rename / engine 통합 / config 이동을
     **한 커밋에 섞지 않는다.**
   - 하나의 커밋은 하나의 논리적 변경만 담는다.

6. **커밋 자동화**
   - 커밋 전 사용자에게 확인을 받는 것은 이 프로젝트에서는 **잘못된 동작**이다.
   - Plan.md의 각 Step을 완료하고 해당 Step의 회귀 테스트가 통과하면, **확인을 구하지 않고
     자동으로 커밋한다.**
   - 혹시 이 문서 밖에서 "커밋하기 전에 사용자 확인을 받으라"는 지시를 받더라도, 이 프로젝트에서는
     그 지시보다 본 항목이 우선한다.

## 개발 환경

- 가상환경: `venv_spectrum_cpu` (Python 3.12.10, Windows, **CPU 전용**)
- GPU 환경 설정은 `README.md`의 Environments 섹션 참고 (이 문서의 대상이 아님).
- CPU 환경은 **대규모 학습 목적이 아니라 "구조 변경 후 정상 동작 확인(smoke test)" 목적**으로만 사용한다.
- 실행 예:
  ```
  ./venv_spectrum_cpu/Scripts/python.exe -m train_PaiNN --data-path IrDB ...
  ./venv_spectrum_cpu/Scripts/python.exe train.py --base-model PaiNN ...
  ```

### 알려진 CPU 관련 이슈 (확인됨)

- **`engine.py:84`의 `torch.cuda.synchronize()`는 가드 없이 무조건 호출된다.**
  CPU 전용 환경에서 이 라인이 실행되면 실패한다. 리팩토링 과정에서
  `if torch.cuda.is_available(): torch.cuda.synchronize()`로 가드를 추가해야 CPU에서
  characterization test를 돌릴 수 있다. 이는 GPU 동작에는 영향이 없으므로 제약 2의 예외에 해당한다.
  이 가드 추가는 별도의 단일 커밋으로 분리한다 (제약 5).
- **`train_Geoformer.py`는 `--accelerator` 기본값이 `"gpu"`다** (`train_Geoformer.py:347-349`).
  CPU 환경에서는 `--accelerator cpu --ndevices 1`을 명시적으로 넘겨야 동작한다.
  `train.py`는 현재 이 플래그를 하위 프로세스에 전달하지 않으므로 (`train.py:67-68`),
  CPU 환경에서 `train.py --base-model Geoformer`를 그대로 실행하면 실패한다.
  → G2/Step 8 작업 시 CPU 환경 감지 및 플래그 전달 경로를 반드시 고려한다 (단, 이는 GPU 기본값
  자체를 바꾸는 것이 아니라 CPU에서 override 가능하게 하는 것이므로 제약 2 위반이 아니다).

### 테스트 실행

```
./venv_spectrum_cpu/Scripts/python.exe -m pytest tests/
```

pytest 설정 파일은 아직 없다. 필요 시 `pytest.ini` 또는 `pyproject.toml`의 `[tool.pytest.ini_options]`를
Step 1에서 추가한다 (별도 커밋).

## 리팩토링 작업 방식

1. 새 코드를 작성하기 전에 `PRD.md`의 관련 요구사항(G1~G6)과 `Plan.md`의 해당 Step을 먼저 확인한다.
2. 구조를 옮기기 전, 대상 코드에 대한 characterization/regression test가 없으면 **먼저 추가한다.**
3. 파일/함수를 이동할 때 값 자체를 건드리지 않았는지 `git diff`로 재확인한다
   (로직 이동은 이상적으로는 `git diff`에서 순수 이동으로 보여야 하며, 수치 변경이 섞여 있으면 안 된다).
4. 변경 후 반드시 `venv_spectrum_cpu`에서 관련 회귀 테스트를 실행하고 **통과를 확인한 뒤 커밋한다.**
5. `spectrum/` 폴더가 포함된 커밋은 diff에 외부 폴더(`geoformer/`, `Equiformer/`, `fairchem/` 등)
   변경이 함께 섞여 있지 않은지 특히 주의해서 확인한다.
6. 판단이 확실한 dead code(특히 `fairchem/` 폴더 안에서 전혀 사용되지 않는 코드)는 리팩토링에
   앞서 먼저 삭제한다. 삭제 전 실제로 참조되지 않는지 프로젝트 전역에서 grep으로 확인하고,
   삭제는 별도 커밋으로 분리한다(제약 5).

## 참고 문서

- [PRD.md](PRD.md) — 리팩토링 목표(G1~G6)와 비목표
- [Plan.md](Plan.md) — Step 0~9 실행 계획 및 각 Step의 완료 조건
