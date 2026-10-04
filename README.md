# Physics-Informed Machine Learning for Spectrum Prediction in Phosphorescent OLEDs
## Overview
Authors: Hasup Lee, Hyuntae Cho, Hwidong Na, Kuhwan Jeong, Sang Ha Park, Kisoo Kwon, MiYoung Jang, Eun Hyun Cho, Sanghyun Yoo, Hyun Koo, Changjin Oh, and Sun-Jae Lee

<img src="overview.PNG" width=100%> 

## Environments

- Install the dependencies

```shell
# Recommended environment: Python 3.12 and CUDA 12.4
python3 -m venv venv_spectrum
source venv_spectrum/bin/activate

# Install dependencies
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

pip install e3nn==0.5.1 torch_geometric==2.6.1 transformers==4.56.0 \
    ogb==1.3.6 ase==3.25.0 pytorch-lightning==2.5.1.post0 \
    rdkit==2025.3.5 einops==0.8.1 tensorboard==2.15.1 timm==1.0.19 lmdb==1.7.3 \
    "pydantic<2"
```

## Getting started

To train for spectrum prediction, just run:

```shell
python train.py --base-model {Geoformer|PaiNN|Equiformer} [--spectrum-type {Naive|GMM|FC} (default: FC)] [--batch-size <int> (default: 16)] [--data-path {IrDB|PtDB|IrDB_uff|IrDB_murcko} (default: IrDB)]
```
#### model 
 - Geoformer
 - PaiNN
 - Equiformer

#### spectrum_type 
 - Naive: Not consider Spectrum Loss.
 - GMM: Spectrum prediction based on Gaussian Mixture Model
 - FC: Spectrum prediction based on Franck-Condon progression

## Output

Spectrum prediction results are saved to "p_spec.csv"
 - p_spec.csv: id,Wavelength,Intensity (Normalized)

## Spectrum Agent (학습/예측 Agent와 UI)

학습과 예측을 tool로 노출하고, 이를 [AGNO](https://docs.agno.com) Agent(LLM은 AGNO `VLLM` 모델 클래스)와 Gradio 화면에서 사용할 수 있다.
입력 데이터는 저장소의 IrDB 예시 데이터로 한정한다. 개발과 테스트는 Windows CPU 환경(`venv_spectrum_cpu`, Python 3.12)에서 했다.

### 설치

위 Environments의 기본 의존성(torch 등)을 설치한 같은 가상환경에 Agent용 의존성을 추가한다.

```shell
pip install -r requirements-agent.txt
```

> **주의**: AGNO가 `pydantic` 2.x를 요구하므로 위 Environments의 `"pydantic<2"`가 2.x로 올라간다.
> 이 조합은 CPU 환경에서 기존 테스트 전체가 통과함을 확인했다 (GPU 환경은 확인하지 않음).
> `requirements-agent.txt`는 `transformers`와의 충돌을 피하려고 `huggingface-hub==0.36.2`도 고정한다 (`pip check`가 gradio 쪽 경고를 낼 수 있으나 동작에는 문제가 없었다).
> 설치 전 상태는 `requirements-dev-before-agent.txt`에 있다.

### 실행

프로젝트 루트에서, 가상환경의 파이썬으로 실행한다.

```shell
# LLM 없이 둘러보기 (규칙 기반 데모 서버; 진짜 LLM이 아님)
python -m agent.ui --demo

# 실제 LLM에 연결 (아래 "실제 LLM 연결" 참고)
python -m agent.ui

# 포트 변경 (기본 7860)
python -m agent.ui --demo --port 7871
```

터미널에 `Running on local URL: http://127.0.0.1:7860`이 나오면 브라우저에서 그 주소를 연다 (브라우저는 자동으로 열리지 않는다).
서버가 터미널을 계속 사용하므로 종료는 `Ctrl+C`이다. 서버는 보안상 `127.0.0.1`(실행한 컴퓨터)에만 열려 있어, 다른 컴퓨터에서 접속하려면 포트 포워딩이 필요하다
(예: `ssh -L 7860:127.0.0.1:7860 사용자@서버`). 같은 컴퓨터에서 실행하는 한 누가 받아 실행해도 주소는 `http://127.0.0.1:7860`으로 같다
(해당 포트가 이미 쓰이고 있으면 `--port`로 바꾼다).

화면은 세 탭이다.

| 탭 | 내용 | LLM 필요 |
|---|---|---|
| 채팅 | Agent와 대화. 하단의 예시 질문을 누르면 입력창에 채워진다. 학습/예측 이외의 질문은 거절한다. | 필요 (데모 모드는 규칙 기반 서버) |
| 학습 | 모델 선택(PaiNN/Geoformer/Equiformer), 기본값 확인, 미리보기, 확인 후 학습 시작, 진행 상태와 로그 | 불필요 |
| 예측 | IrDB 분자 선택, 예측 스펙트럼과 실험 스펙트럼을 겹쳐 보기 | 불필요 |

- 학습 탭의 "작은 설정으로 빠르게 시험"(기본 체크)은 CPU에서 약 10초 안에 끝나는 작은 모델/step 수를 쓴다. 체크를 풀면 스크립트의 기본 설정(예: PaiNN 10000 step)으로 학습하며 CPU에서는 매우 오래 걸린다.
- 학습 결과(체크포인트)는 기존 스크립트와 같은 `results_<모델>/` 폴더에 저장된다. 이미 결과가 있으면 삭제하고 덮어쓴다는 확인을 받는다.
- 학습된 모델이 없을 때 예측하면 먼저 학습하라는 안내가 나온다.
- 학습 step 수가 작으면 예측 곡선의 모양은 의미가 없다. 이 화면의 목적은 성능이 아니라 학습/예측/Agent 흐름이 동작하는지 확인하는 것이다.

### 실제 LLM 연결

Agent는 **OpenAI 호환 서버**(vLLM 등)에 연결한다. 코드를 고치지 않고 환경변수만 설정한다 (`agent/agent.py`의 `build_model`).

| 환경변수 | 필수 | 설명 |
|---|---|---|
| `VLLM_BASE_URL` | 필수 | 서버 주소 (예: `http://서버:8000/v1`) |
| `VLLM_MODEL` | 필수 | 서버가 제공하는 모델 이름 |
| `VLLM_API_KEY` | 선택 | 인증 토큰. 없으면 더미 값을 쓴다 (AGNO `VLLM`이 키를 요구하기 때문) |

```shell
# Git Bash / Linux / macOS
export VLLM_BASE_URL=http://서버:8000/v1
export VLLM_MODEL=모델이름
export VLLM_API_KEY=토큰
python -m agent.ui
```

```powershell
# PowerShell
$env:VLLM_BASE_URL = "http://서버:8000/v1"
$env:VLLM_MODEL = "모델이름"
$env:VLLM_API_KEY = "토큰"
python -m agent.ui
```

환경변수가 없으면 화면 상단 배너에 "LLM 연결 안 됨"이 표시되고 채팅 탭은 안내만 보여 준다 (학습/예측 탭은 계속 쓸 수 있다). 연결되면 배너가 "🟢 실제 LLM"과 서버 주소를 보여 준다.

**서버 요건**
- OpenAI 호환 `/v1/chat/completions`를 제공하고, **tool calling(function calling)을 지원**해야 한다. vLLM이면 서버를 `--enable-auto-tool-choice --tool-call-parser <모델에 맞는 값>` 옵션으로 띄운다.
  tool calling이 켜져 있지 않거나 약한 모델은 학습/예측 tool을 호출하지 못하거나 잘못 호출할 수 있다.
- 토큰은 `Authorization: Bearer` 방식(`VLLM_API_KEY`)으로 전달된다.

**연결 확인**
```shell
pytest -m vllm     # 환경변수가 설정된 때만 실행되고, 없으면 skip. 연결/응답과 범위 밖 질문 거절만 확인한다.
```
LLM이 지시를 잘 따르는지(모델/기본값 질문, 확인 후 학습 등)는 모델마다 달라 자동 테스트로 판정하지 않으므로 채팅 탭에서 직접 확인한다.

**환경변수만으로 안 될 때** — `agent/agent.py`의 `build_model` 안 `VLLM(...)` 생성부를 수정한다.
- 인증이 `Authorization: Bearer`가 아니라 별도 헤더인 경우(사내 게이트웨이 등): `VLLM(...)`에 `default_headers={"헤더이름": "값"}`을 추가한다.
- 사내 인증서/프록시가 필요한 경우: `http_client`에 설정한 `httpx.Client`를 넘긴다.
- 생성 인자(`temperature`, `max_tokens`, `timeout` 등)는 AGNO `VLLM`의 인자를 그대로 쓴다.

### Agent가 하는 일

- 도구: 학습 기본값 조회, 학습 미리보기와 시작(확인 후), 학습 상태/로그 조회, 학습된 모델 목록, 분자 ID 검색, 스펙트럼 예측.
- "그냥 학습해줘"처럼 모델이나 설정이 정해지지 않으면 PaiNN/Geoformer/Equiformer 중 무엇으로 할지 되묻고, 기본 설정을 보여 준 뒤 사용자가 확인해야 학습을 시작한다.
- 학습된 모델이 없을 때의 예측 요청에는 먼저 학습을 제안한다.
- 학습/예측과 무관한 질문(예: 날씨, 일반 지식, 코딩)은 LLM을 호출하기 전에 가드가 거절한다. 가드는 키워드 기반이라 완벽하지 않으며, 통과한 질문은 Agent의 instructions가 두 번째 방어선이다.

### 테스트

```shell
pytest tests/                   # 전체 (실제 학습 E2E 포함, 약 3분)
pytest tests/ -m "not slow"     # 실제 학습을 도는 테스트 제외
pytest -m vllm                  # 실제 LLM 서버 연결 테스트 (환경변수가 있을 때만 실행)
```

모든 테스트는 LLM 서버나 키 없이 통과한다 (LLM은 mock/로컬 가짜 서버로 대체). 데이터는 IrDB 예시만 쓴다.

### 라이센스 경계

`agent/`는 `spectrum/`의 physics 코드를 import로만 사용한다. `spectrum/`(CC BY-NC-SA 4.0)과 그 외 폴더(MIT) 사이의 소스 코드 이동/복사는 하지 않는다.
설계와 단계별 작업 기록은 [PRD.md](PRD.md), [Plan.md](Plan.md)를 참고한다.

## Scripts

Python utilities for spectrum metrics are located in scripts/:
 - calc_SID.py: Spectral Information Divergence (SID), Spectral Information Similarity (SIS), Jensen-Shannon Divergence (JSD), Earth Mover's Distance (EMD)
 - calc_peak_fwhm.py: Peak position (Peak), Full Width at Half Maximum (FWHM), PL center, Full Width at Quarter Maximum (FWQM)
 - calc_ITPL_center.py: Intensity-Threshold PL center (IPTL center)
 - calc_FWXM.py: Generalized width at X% of maximum (FWXM) for Intensity-Threshold
 - clustering_murcko.py: Grouping based on Murcko scaffold

## Contact

Please contact Hasup Lee for technical support. (hasup.lee@samsung.com)

## License

This project contains code under multiple licenses:

1. **excluding spectrum/**:
   - Based on [Geoformer](https://github.com/microsoft/AI2BMD/tree/Geoformer), [PaiNN](https://github.com/facebookresearch/fairchem/blob/977a80328f2be44649b414a9907a1d6ef2f81e95/src/fairchem/core/models/painn/painn.py), and [Equiformer](https://github.com/atomicarchitects/equiformer).
   - Licensed under the **MIT License**. See [LICENSE-MIT.txt](https://github.com/samsungDS-PoCs/Spectrum/blob/main/LICENSE-MIT.txt) for the full terms.

2. **spectrum/**:
   - License under **CC BY-NC-SA 4.0**.
   - See [LICENSE-CCBYNCSA.txt](https://github.com/samsungDS-PoCs/Spectrum/blob/main/LICENSE-CCBYNCSA.txt) for details and contact information.

## Citation

If you use this code or data in your research, please cite our paper:

Lee, H. *et al.* Physics-informed machine learning for spectrum prediction in phosphorescent OLEDs.  
*npj Comput Mater* (2026). [https://doi.org/10.1038/s41524-026-02078-x](https://doi.org/10.1038/s41524-026-02078-x)

## Acknowledgements
This work builds upon the following models:

  - **Geometric Transformer with Interatomic Positional Encoding**
    - Authors: Yusong Wang, Shaoning Li, Tong Wang, Bin Shao, Nanning Zheng, Tie-Yan Liu
    - Article: [https://neurips.cc/virtual/2023/poster/72577](https://neurips.cc/virtual/2023/poster/72577)
    - Github: [https://github.com/microsoft/AI2BMD/tree/Geoformer](https://github.com/microsoft/AI2BMD/tree/Geoformer)

  - **PaiNN: Polarizable Atom Interaction Neural Network**
    - Authors: Kristof T. Schütt, Oliver Unke, Michael Gastegger
    - Article: [https://arxiv.org/abs/2102.03150](https://arxiv.org/abs/2102.03150)
    - Original Github: [https://github.com/atomistic-machine-learning/schnetpack](https://github.com/atomistic-machine-learning/schnetpack)
    - Implementation adapted from: [facebookresearch/fairchem (commit 977a803, `src/fairchem/core/models/painn/painn.py`)](https://github.com/facebookresearch/fairchem/blob/977a80328f2be44649b414a9907a1d6ef2f81e95/src/fairchem/core/models/painn/painn.py)

  - **Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs**
    - Authors: Yi-Lun Liao, Tess Smidt
    - Article: [https://arxiv.org/abs/2206.11990](https://arxiv.org/abs/2206.11990)
    - Github: [https://github.com/atomicarchitects/equiformer](https://github.com/atomicarchitects/equiformer)

