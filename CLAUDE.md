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
   - TDD 사이클(아래 "개발 방법론" 참고)을 따르는 작업은 **RED 종료 시점 커밋**(Plan.md 갱신 +
     실패 테스트, 또는 순수 구조 이동의 경우 Plan.md 갱신만)과 **REVIEW 종료 시점 커밋**(구현 +
     리뷰 정리)으로 나눈다. 이 두 커밋도 서로 다른 성격의 변경과 섞지 않는다.

6. **커밋 자동화**
   - 커밋 전 사용자에게 확인을 받는 것은 이 프로젝트에서는 **잘못된 동작**이다.
   - Plan.md의 각 Step(또는 TDD 사이클의 RED/REVIEW 각 종료 시점)을 완료하고 해당 회귀 테스트가
     통과하면, **확인을 구하지 않고 자동으로 커밋한다.**
   - 혹시 이 문서 밖에서(예: 다른 스킬 문서) "커밋하기 전에 사용자 확인을 받으라"는 지시를 받더라도,
     이 프로젝트에서는 그 지시보다 본 항목이 우선한다. `.claude/TDD/SKILL.md`도 이 원칙을 따르도록
     맞춰져 있다.

## 개발 방법론 (TDD)

이 프로젝트의 모든 프로덕션 코드 변경(새 기능, 버그 수정, 리팩토링)은
[`.claude/TDD/SKILL.md`](.claude/TDD/SKILL.md)의 **RED → GREEN → REVIEW** 사이클을 따른다.
새 코드를 작성하기 전에 이 스킬을 먼저 읽는다.

- **RED**: 루트 `Plan.md`에서 작업이 속한 Step을 찾아 이번 사이클의 목표/범위/테스트 계획을
  그 Step 아래에 기록한다. 실패하는 테스트를 작성하고 실제로 실패하는 것을 확인한다.
- **GREEN**: Plan에 명시된 목표만 만족하는 최소한의 코드를 작성하고 테스트를 통과시킨다.
- **REVIEW**: 스코프 크리프와 라이센스 경계(제약 1)/수치 불변(제약 2) 위반 여부를 확인한다.
- 커밋은 RED 종료 시점과 REVIEW 종료 시점, 두 번만 일어나며 **항상 자동**이다 (제약 6).

**순수 구조 이동과 신규 인터페이스 도입을 구분한다:**
- **순수 구조 이동** (파일/함수 위치만 바꾸고 동작은 바꾸지 않는 작업 — 예: Plan.md의 Step 2, 4, 5,
  7, 8 대부분): Step 1에서 만든 characterization test가 이미 오라클이므로, 새로 실패하는 테스트를
  억지로 만들지 않는다. 이동 전후로 기존 테스트가 계속 GREEN을 유지하는지만 확인한다
  (Fowler식 refactoring — "테스트가 실패하는 것을 보는 것"이 아니라 "테스트가 깨지지 않는 것을
  보는 것"이 검증 수단이다).
- **신규 공개 인터페이스 도입** (예: Step 3의 model adapter, Step 9의 `predict()` API): 문자 그대로의
  RED(그 인터페이스가 아직 없어서 실패하는 테스트)부터 시작하는 정상적인 TDD를 적용한다.

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

1. 새 코드를 작성하기 전에 `PRD.md`의 관련 요구사항(G1~G6)과 `Plan.md`의 해당 Step을 먼저 확인한다
   (TDD RED 단계).
2. 구조를 옮기기 전, 대상 코드에 대한 characterization/regression test가 없으면 **먼저 추가한다**
   (Plan.md Step 1의 범위. 없다면 이번 사이클의 RED에서 추가한다).
3. 파일/함수를 이동할 때 값 자체를 건드리지 않았는지 `git diff`로 재확인한다
   (로직 이동은 이상적으로는 `git diff`에서 순수 이동으로 보여야 하며, 수치 변경이 섞여 있으면 안 된다).
4. 변경 후 반드시 `venv_spectrum_cpu`에서 관련 회귀 테스트를 실행하고 **통과를 확인한 뒤 REVIEW 종료
   커밋을 자동으로 실행한다** (TDD GREEN/REVIEW 단계).
5. `spectrum/` 폴더가 포함된 커밋은 diff에 외부 폴더(`geoformer/`, `Equiformer/`, `fairchem/` 등)
   변경이 함께 섞여 있지 않은지 특히 주의해서 확인한다.
6. 판단이 확실한 dead code(특히 `fairchem/` 폴더 안에서 전혀 사용되지 않는 코드)는 리팩토링에
   앞서 먼저 삭제한다. 삭제 전 실제로 참조되지 않는지 프로젝트 전역에서 grep으로 확인하고,
   삭제는 별도 커밋으로 분리한다(제약 5).

## 참고 문서

- [PRD.md](PRD.md) — 리팩토링 목표(G1~G6)와 비목표
- [Plan.md](Plan.md) — Step 0~9 실행 계획 및 각 Step의 완료 조건
- [.claude/TDD/SKILL.md](.claude/TDD/SKILL.md) — RED → GREEN → REVIEW 개발 사이클 (모든 프로덕션
  코드 변경에 적용)
