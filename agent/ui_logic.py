"""UI 로직: 화면(Gradio)과 무관하게 테스트할 수 있는 핸들러 모음 (Plan.md Step 6A, G4).

학습/예측 탭은 tool 함수(agent.tools.*)를 직접 호출하므로 LLM 없이 동작하고, 채팅 탭만 LLM(실제 또는 데모)이 필요하다.
화면(agent.ui, Step 6B)은 이 함수들을 이벤트에 연결할 뿐이다. 결과는 화면에 바로 보여 줄 수 있는 마크다운 문자열/표로 돌려준다.
"""

import contextlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from agent.agent import build_agent, build_model, chat
from agent.demo.mock_llm import DEMO_MODEL_ID, MockLLMServer
from agent.tools._shared import PROJECT_ROOT
from agent.tools.predict_tool import list_molecules, predict_spectrum
from agent.tools.train_tool import changeable_parameters, get_training_defaults, get_training_status, start_training
from common.data import build_dataset

CURVE_COLUMNS = ["wavelength_nm", "intensity", "종류"]
_EV_NM = 1240.0  # nm = 1240 / eV (spectrum.reconstruct와 같은 환산 관례)
_LOG_LINES = 20
_PROGRESS_FRAME = re.compile(r"\d+%\|.*\||it/s\]|s/it\]")  # Lightning/tqdm 진행 막대 프레임

_QUICK_SETTINGS = {
    # CPU에서 빠르게 동작을 확인하기 위한 작은 설정 (스크립트 기본값은 바꾸지 않고 tool의 override로만 지정)
    "PaiNN": {"train_steps": 20, "eval_steps": 20, "batch_size": 4, "embed_dim": 8, "num_layers": 1, "num_basis": 8},
    "Equiformer": {"train_steps": 5, "eval_steps": 5, "batch_size": 4, "num_basis": 8},  # 모델 크기 인자가 num_basis뿐이라 step을 더 줄인다
    "Geoformer": {"train_steps": 20, "eval_steps": 20, "batch_size": 4, "embedding_dim": 8, "ffn_embedding_dim": 16,
                  "num_layers": 1, "num_heads": 2, "num_rbf": 8},
}


# --- LLM 상태 ------------------------------------------------------------------------------


@dataclass
class LlmStatus:
    mode: str  # "real" | "demo" | "disconnected"
    message: str
    base_url: Optional[str] = None
    model_id: Optional[str] = None


_demo_server = None


def _ensure_demo_server() -> MockLLMServer:
    global _demo_server
    if _demo_server is None:
        _demo_server = MockLLMServer().__enter__()
    return _demo_server


def shutdown_demo_server() -> None:
    global _demo_server
    if _demo_server is not None:
        _demo_server.__exit__(None, None, None)
        _demo_server = None


def resolve_llm(environ=None, demo=False) -> LlmStatus:
    """채팅 탭이 쓸 LLM 상태. demo면 프로세스 안에서 규칙 기반 데모 서버를 한 번만 띄운다.

    서버 접속 가능 여부는 시작 시 확인하지 않는다(응답 지연 방지) — 호출이 실패하면 chat()이 안내한다.
    """
    if demo:
        return LlmStatus(
            mode="demo", base_url=_ensure_demo_server().base_url, model_id=DEMO_MODEL_ID,
            message="[데모 모드] 진짜 LLM이 아니라 규칙 기반 가짜 서버입니다. 정해진 표현만 알아듣습니다.")
    environ = os.environ if environ is None else environ
    missing = [name for name in ("VLLM_BASE_URL", "VLLM_MODEL") if not environ.get(name)]
    if missing:
        return LlmStatus(
            mode="disconnected",
            message=(f"LLM 서버가 설정되지 않았습니다 (누락: {', '.join(missing)}). 채팅에는 LLM 서버가 필요합니다. "
                     "환경변수 VLLM_BASE_URL, VLLM_MODEL(인증이 필요하면 VLLM_API_KEY)을 설정하거나 `python -m agent.ui --demo`로 데모 모드를 쓰세요. "
                     "학습/예측 탭은 LLM 없이 사용할 수 있습니다."))
    return LlmStatus(mode="real", base_url=environ["VLLM_BASE_URL"], model_id=environ["VLLM_MODEL"],
                     message=f"LLM 서버: {environ['VLLM_BASE_URL']} (모델 {environ['VLLM_MODEL']})")


# --- 채팅 ----------------------------------------------------------------------------------


def new_agent(llm_status, project_root=PROJECT_ROOT, results_root=None):
    """대화 세션 하나를 위한 Agent. LLM이 연결되지 않았으면 None."""
    if llm_status.mode == "disconnected":
        return None
    model = build_model({"VLLM_BASE_URL": llm_status.base_url, "VLLM_MODEL": llm_status.model_id,
                         "VLLM_API_KEY": os.environ.get("VLLM_API_KEY")})
    return build_agent(model=model, project_root=project_root, results_root=results_root)


def chat_turn(message, history, agent, llm_status, session_id="default", project_root=PROJECT_ROOT, results_root=None):
    """한 턴의 채팅. (새 history, Agent)를 돌려준다 — 입력 history는 변경하지 않고, Agent는 첫 메시지에서 만들어 세션 동안 재사용한다."""
    history = list(history or [])
    text = (message or "").strip()
    if not text:
        return history, agent
    history = history + [{"role": "user", "content": text}]
    if llm_status.mode == "disconnected":
        return history + [{"role": "assistant", "content": llm_status.message}], None
    if agent is None:
        agent = new_agent(llm_status, project_root, results_root)
    reply = chat(agent, text, session_id=session_id)
    return history + [{"role": "assistant", "content": reply}], agent


# --- 학습: 설정 ----------------------------------------------------------------------------


def quick_settings(base_model) -> dict:
    """CPU에서 빠르게 시험할 수 있는 작은 설정 (모델별 step/배치/모델 크기)."""
    return dict(_QUICK_SETTINGS[base_model])


def parse_custom_settings(text):
    """사용자가 입력한 JSON 설정을 (설정 dict, 오류 문구)로 돌려준다. 빈 입력은 빈 설정."""
    if not text or not text.strip():
        return {}, None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}, '설정은 JSON 형식이어야 합니다 (예: {"train_steps": 20}).'
    if not isinstance(parsed, dict):
        return {}, '설정은 JSON 객체여야 합니다 (예: {"train_steps": 20}).'
    for key, value in parsed.items():
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            return {}, f"설정 값은 정수 또는 문자열이어야 합니다: {key}={value!r}"
    return parsed, None


def training_defaults_markdown(base_model) -> str:
    defaults = get_training_defaults(base_model)
    if defaults["status"] != "ok":
        return f"⚠ {defaults['message']}"
    rows = [("spectrum_type", defaults["spectrum_type"]), ("data_path", defaults["data_path"]), ("batch_size", defaults["batch_size"]),
            ("train_steps", defaults["train_steps"]), ("eval_steps", defaults["eval_steps"]), ("workers", defaults["workers"]),
            ("learning_rate (읽기 전용)", defaults["learning_rate"]), *defaults["model_size"].items(), ("device", defaults["device"])]
    table = "\n".join(f"| {name} | {value} |" for name, value in rows)
    return (f"### {base_model} 기본 설정\n\n| 항목 | 기본값 |\n|---|---|\n{table}\n\n"
            f"바꿀 수 있는 파라미터: {', '.join(changeable_parameters(base_model))}")


def _merged_settings(base_model, use_quick, custom_json):
    custom, error = parse_custom_settings(custom_json)
    if error:
        return None, error
    return {**(quick_settings(base_model) if use_quick else {}), **custom}, None  # 사용자 JSON이 빠른 설정을 덮는다


def _settings_table(settings) -> str:
    rows = [("base_model", settings["base_model"]), ("spectrum_type", settings["spectrum_type"]), ("data_path", settings["data_path"]),
            ("batch_size", settings["batch_size"]), ("train_steps", settings["train_steps"]), ("eval_steps", settings["eval_steps"]),
            ("workers", settings["workers"]), *settings["model_size"].items()]
    return "| 항목 | 값 |\n|---|---|\n" + "\n".join(f"| {name} | {value} |" for name, value in rows)


# --- 학습: 미리보기 / 시작 -------------------------------------------------------------------


def training_preview(base_model, use_quick, custom_json, project_root=PROJECT_ROOT) -> str:
    """학습을 시작하지 않고 최종 설정과 덮어쓰기 경고를 보여 준다."""
    settings, error = _merged_settings(base_model, use_quick, custom_json)
    if error:
        return f"⚠ {error}"
    result = start_training(base_model, settings, confirmed=False, project_root=project_root)
    if result["status"] != "needs_confirmation":
        return f"⚠ {result['message']}"
    text = f"### 학습 미리보기\n\n{_settings_table(result['settings'])}\n\n저장 위치: `{result['output_dir']}`"
    if result["will_overwrite"]:
        text += "\n\n⚠ 이미 학습된 결과가 있어 다시 학습하면 기존 결과가 삭제됩니다."
    return text + "\n\n설정을 확인했으면 '확인'을 체크하고 학습 시작을 누르세요."


def training_start(base_model, use_quick, custom_json, confirmed, overwrite, project_root=PROJECT_ROOT) -> str:
    """사용자가 화면에서 확인 체크를 한 경우에만 학습을 시작한다. 기존 결과가 있으면 덮어쓰기 체크도 필요하다."""
    settings, error = _merged_settings(base_model, use_quick, custom_json)
    if error:
        return f"⚠ {error}"
    if not confirmed:
        return "⚠ 학습을 시작하기 전에 설정을 확인하고 '확인' 체크박스를 선택해 주세요. (미리보기로 최종 설정을 먼저 볼 수 있습니다.)"
    result = start_training(base_model, settings, confirmed=True, overwrite=overwrite, project_root=project_root)
    if result["status"] == "started":
        return f"✅ 학습을 시작했습니다 (작업 {result['job_id']}). 아래 학습 상태에서 진행 상황을 볼 수 있습니다."
    if result["status"] == "already_trained":
        return "⚠ 이미 학습된 결과가 있습니다. 다시 학습하려면 '덮어쓰기'를 체크해야 하며, 기존 결과는 삭제됩니다."
    return f"⚠ {result['message']}"


# --- 학습: 상태 ----------------------------------------------------------------------------


def training_status_view(project_root=PROJECT_ROOT):
    """(상태 마크다운, 로그 텍스트, 실행 중 여부). 로그는 진행 막대 프레임을 걸러낸 최근 20줄."""
    status = get_training_status(None, tail_lines=300, project_root=project_root)  # 진행 막대 프레임을 걸러도 20줄이 남도록 넉넉히 가져온다
    if status["status"] != "ok":
        return "**작업 없음** — 아직 시작한 학습이 없습니다.", "", False
    label = {"running": "진행 중", "finished": "완료", "failed": "실패"}[status["state"]]
    text = f"**{label}** — {status['base_model']}, 경과 {int(status['elapsed_seconds'])}초 (작업 {status['job_id']})"
    if status["state"] == "failed":
        text += f", 종료 코드 {status['return_code']}"
    if status["state"] == "finished":
        text += ", 체크포인트 생성됨" if status["has_checkpoint"] else ", 체크포인트 없음"
    lines = [line for line in status["log_tail"] if not _PROGRESS_FRAME.search(line)]
    return text, "\n".join(lines[-_LOG_LINES:]), status["state"] == "running"


# --- 예측 ----------------------------------------------------------------------------------


def molecule_choices(query="", limit=50, project_root=PROJECT_ROOT) -> list:
    result = list_molecules(query, limit, project_root=project_root)
    return result["molecule_ids"] if result["status"] == "ok" else []


def _empty_curves() -> pd.DataFrame:
    return pd.DataFrame(columns=CURVE_COLUMNS)


def _experimental_curve(molecule_id, project_root) -> pd.DataFrame:
    """IrDB의 실험 스펙트럼(eV 균등)을 파장(nm) 오름차순으로 바꾼 표."""
    with contextlib.chdir(project_root):
        dataset = build_dataset("IrDB", ["S1"])
        data = next(item for item in dataset if item.name == molecule_id)
    frame = pd.DataFrame({"wavelength_nm": (_EV_NM / data.spec_x[0]).tolist(), "intensity": data.spec_y[0].tolist(), "종류": "실험"})
    return frame.sort_values("wavelength_nm").reset_index(drop=True)


def prediction_view(molecule_id, base_model, show_experimental, results_root=None, project_root=PROJECT_ROOT):
    """(곡선 표, 요약 마크다운). 곡선 표는 예측(과 선택 시 실험) 곡선을 `종류` 열로 구분한 긴 형식."""
    model = None if base_model in (None, "", "자동") else base_model
    result = predict_spectrum(molecule_id, model, results_root=results_root if results_root is not None else project_root,
                              project_root=project_root)
    if result["status"] == "needs_training":
        return _empty_curves(), "⚠ 학습된 모델이 없습니다. 먼저 학습 탭에서 학습해 주세요."
    if result["status"] != "ok":
        return _empty_curves(), f"⚠ {result['message']}"

    intensity = result["intensity"]
    frame = pd.DataFrame({"wavelength_nm": result["wavelength_nm"], "intensity": intensity, "종류": "예측"})
    summary = (f"**{result['molecule_id']}** — {result['base_model']} ({result['spectrum_type']}), 체크포인트 {Path(result['checkpoint']).name}\n\n"
               f"피크 파장: **{result['peak_wavelength_nm']:.1f} nm**")
    if show_experimental:
        frame = pd.concat([frame, _experimental_curve(molecule_id, project_root)], ignore_index=True)
        summary += "\n\n실험 스펙트럼을 함께 표시했습니다."
    if max(intensity) > 1.001 or min(intensity) < -0.05:
        summary += (f"\n\n⚠ 예측 곡선이 비정상입니다 (최댓값 {max(intensity):.3g}, 최솟값 {min(intensity):.3g}). "
                    "학습이 충분하지 않은 모델의 출력일 수 있습니다. 학습 탭에서 step 수를 늘려 다시 학습해 보세요.")
    return frame, summary
