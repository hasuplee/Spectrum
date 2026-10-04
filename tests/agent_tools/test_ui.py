"""Plan.md Step 6B [신규 인터페이스], G4: Gradio 화면 `agent.ui.build_ui`와 `python -m agent.ui`.

agent.ui가 아직 없으므로 실패해야 한다 (RED). 화면 구성(탭, 배너, 예시 질문)은 launch 없이 Blocks 설정(config)으로,
이벤트 동작은 실제 서버를 띄우고 `gradio_client`로 호출해 **브라우저 없이** 검증한다. LLM은 데모 서버(키 불필요)이고
학습은 4B 방식의 가짜 명령, 예측은 tiny PaiNN 체크포인트와 실제 IrDB를 쓴다.
`/chat`은 Gradio 6의 Chatbot 형식(메시지 content가 `{"type": "text", "text": ...}` 조각 목록)으로 돌려주므로 `_text`로 문자열을 꺼낸다.
새 모듈 import는 `launch_ui`/`_build` 안에서 하여 테스트마다 개별 실패로 확인한다.
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from tests.agent_tools.test_predict_tool import _save_painn_ckpt
from tests.support.fake_training import use_fake_training

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ENDPOINTS = {"/chat", "/training_defaults", "/training_preview", "/training_start", "/training_status", "/molecules", "/predict"}


@pytest.fixture
def demo_llm():
    from agent import ui_logic

    yield ui_logic.resolve_llm({}, demo=True)
    ui_logic.shutdown_demo_server()


@pytest.fixture
def launch_ui():
    """build_ui로 만든 화면을 실제로 띄우고 gradio_client를 돌려주는 팩토리 (테스트가 끝나면 서버를 닫는다)."""
    launched = []

    def _launch(llm_status, **kwargs):
        from gradio_client import Client

        from agent.ui import build_ui

        demo = build_ui(llm_status, **kwargs)
        _, url, _ = demo.launch(prevent_thread_lock=True, server_name="127.0.0.1", quiet=True)
        launched.append(demo)
        return Client(url, verbose=False)

    yield _launch
    for demo in launched:
        demo.close()


def _build(llm_status, **kwargs):
    from agent.ui import build_ui

    return build_ui(llm_status, **kwargs)


def _disconnected():
    from agent.ui_logic import resolve_llm

    return resolve_llm({})


def _text(message):
    content = message["content"]
    return content if isinstance(content, str) else "".join(part["text"] for part in content if part["type"] == "text")


def _wait_for_training(project_root, timeout=60):
    from agent.tools.train_tool import get_training_status

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = get_training_status(project_root=project_root)
        if status["status"] == "ok" and status["state"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError("학습 작업이 끝나지 않았다")


# --- 화면 구성 -----------------------------------------------------------------------------


def test_화면은_이름_붙은_엔드포인트를_모두_갖는다(launch_ui, tmp_path):
    client = launch_ui(_disconnected(), project_root=tmp_path)

    endpoints = set(client.view_api(return_format="dict", print_info=False)["named_endpoints"])

    assert REQUIRED_ENDPOINTS <= endpoints


def test_탭은_채팅_학습_예측_순서이고_LLM_상태_배너와_예시_질문이_있다(demo_llm, tmp_path):
    from agent.guard import ALLOWED_EXAMPLES, REFUSED_EXAMPLES
    from agent.ui_logic import LlmStatus

    config = _build(demo_llm, project_root=tmp_path).get_config_file()["components"]

    tab_labels = [component["props"].get("label") for component in config if component["type"] == "tabitem"]
    banner = next(component["props"]["value"] for component in config if component["props"].get("elem_id") == "llm-banner")
    example_groups = {component["props"].get("label"): [sample[0] for sample in component["props"]["samples"]]
                      for component in config if component["type"] == "dataset"}
    assert tab_labels == ["채팅", "학습", "예측"]
    assert "진짜 LLM" in banner  # 데모 모드는 진짜 LLM이 아님을 상단에 분명히 알린다
    assert list(example_groups.values()) == [list(ALLOWED_EXAMPLES), list(REFUSED_EXAMPLES)]  # 동작하는 예시와 거절 시연 예시 (guard가 단일 출처)
    datasets = [component["props"] for component in config if component["type"] == "dataset"]
    assert all(props["samples_per_page"] >= len(props["samples"]) for props in datasets)  # 예시가 여러 쪽으로 나뉘어 숨지 않는다

    for status in (_disconnected(), LlmStatus(mode="real", message="LLM 서버: http://internal:8000/v1 (모델 m)", base_url="http://internal:8000/v1", model_id="m")):
        other = _build(status, project_root=tmp_path).get_config_file()["components"]
        assert status.message in next(c["props"]["value"] for c in other if c["props"].get("elem_id") == "llm-banner")


# --- 채팅 탭 -------------------------------------------------------------------------------


def test_채팅은_데모_모드에서_다중_턴_대화를_이어간다(launch_ui, demo_llm, tmp_path):
    client = launch_ui(demo_llm, project_root=tmp_path)

    first_history, cleared_input = client.predict("그냥 학습해줘", [], api_name="/chat")
    second_history, _ = client.predict("PaiNN", first_history, api_name="/chat")

    assert cleared_input == ""  # 전송 후 입력창이 비워진다
    assert [m["role"] for m in first_history] == ["user", "assistant"]
    assert "무엇으로 학습" in _text(first_history[-1])
    assert [m["role"] for m in second_history] == ["user", "assistant", "user", "assistant"]
    assert "진행할까요" in _text(second_history[-1])  # 세션의 Agent가 유지되어 이전 턴의 맥락이 이어진다


def test_연결되지_않은_상태의_채팅은_안내를_보여_준다(launch_ui, tmp_path):
    status = _disconnected()
    client = launch_ui(status, project_root=tmp_path)

    history, _ = client.predict("학습 상태 알려줘", [], api_name="/chat")

    reply = _text(history[-1])
    assert "LLM" in reply and "학습" in reply and "예측" in reply  # 학습/예측 탭은 LLM 없이 쓸 수 있다는 안내


def test_범위_밖_질문은_화면에서도_거절된다(launch_ui, demo_llm, tmp_path):
    from agent.guard import REFUSAL_MESSAGE

    client = launch_ui(demo_llm, project_root=tmp_path)

    history, _ = client.predict("오늘 날씨가 뭐야?", [], api_name="/chat")

    assert _text(history[-1]) == REFUSAL_MESSAGE


# --- 학습 탭 -------------------------------------------------------------------------------


def test_학습_기본값은_모델_선택으로_조회된다(launch_ui, tmp_path):
    client = launch_ui(_disconnected(), project_root=tmp_path)

    painn = client.predict("PaiNN", api_name="/training_defaults")
    geoformer = client.predict("Geoformer", api_name="/training_defaults")

    assert "10000" in painn and "embed_dim" in painn
    assert "embedding_dim" in geoformer


def test_학습_미리보기와_시작과_상태가_화면_이벤트로_동작한다(launch_ui, tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "print('학습 로그 줄')")
    client = launch_ui(_disconnected(), project_root=tmp_path)

    preview = client.predict("PaiNN", True, "", api_name="/training_preview")
    refused = client.predict("PaiNN", True, "", False, False, api_name="/training_start")  # 확인 체크 없음
    started = client.predict("PaiNN", True, "", True, False, api_name="/training_start")
    _wait_for_training(tmp_path)
    status_text, log_text = client.predict(api_name="/training_status")

    assert "train_steps" in preview and "20" in preview  # 빠른 설정의 step 수
    assert "확인" in refused and "시작했습니다" not in refused
    assert "시작했습니다" in started
    assert "완료" in status_text
    assert "학습 로그 줄" in log_text


# --- 예측 탭 -------------------------------------------------------------------------------


def test_분자_검색은_선택_목록을_갱신한다(launch_ui):
    client = launch_ui(_disconnected(), project_root=PROJECT_ROOT)  # 분자 목록은 저장소의 IrDB 데이터셋에서 읽는다

    update = client.predict("NN1", api_name="/molecules")

    values = [value for _, value in update["choices"]]
    assert 1 <= len(values) <= 50 and all("nn1" in value for value in values)


def test_예측_이벤트는_곡선과_요약을_돌려준다(launch_ui, tmp_path):
    _save_painn_ckpt(tmp_path)  # 체크포인트는 임시 results_root에, 데이터셋은 실제 저장소 루트에서 연다
    client = launch_ui(_disconnected(), project_root=PROJECT_ROOT, results_root=tmp_path)

    plot, summary = client.predict("cn1_cn1_nn1", "자동", True, api_name="/predict")

    assert plot["mark"] == "line" and plot["columns"] == ["wavelength_nm", "intensity", "종류"]
    assert len(plot["data"]) == 1600  # 예측 800점 + 실험 800점
    assert "cn1_cn1_nn1" in summary and "실험" in summary


def test_학습된_모델이_없을_때_예측은_학습_탭을_안내하고_빈_곡선을_돌려준다(launch_ui, tmp_path):
    client = launch_ui(_disconnected(), project_root=PROJECT_ROOT, results_root=tmp_path)

    plot, summary = client.predict("cn1_cn1_nn1", "자동", False, api_name="/predict")

    assert plot["data"] == []
    assert "학습 탭" in summary


# --- 실행 진입점 ---------------------------------------------------------------------------


def test_python_m_agent_ui_demo는_서버를_띄우고_화면을_응답한다():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # 비어 있는 포트를 골라 충돌을 피한다
    process = subprocess.Popen(
        [sys.executable, "-m", "agent.ui", "--demo", "--port", str(port)], cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    try:
        status = None
        deadline = time.time() + 120
        while time.time() < deadline and status is None:
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    status = response.status
            except OSError:
                time.sleep(1)
    finally:
        process.terminate()
        output = process.communicate(timeout=30)[0]

    assert status == 200, output[-800:]
    assert "데모" in output  # 시작할 때 데모 모드임을 알린다
