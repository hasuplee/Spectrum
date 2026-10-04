"""Plan.md Step 7A: 화면 수준 E2E smoke (slow).

데모 LLM(키 불필요) + 실제 tiny 학습 + 실제 IrDB로, 학습 탭 → 채팅 → 예측 탭이 **같은 실제 체크포인트**로 이어지는지
`gradio_client`로 화면 이벤트를 호출해 확인한다. 개별 기능은 다른 테스트가 가짜 학습/tiny 체크포인트로 이미 검증하므로,
여기서는 실제 산출물이 화면 경로 전체를 통과하는 것만 본다. 기존 slow 테스트와 같이 저장소 루트에 결과 폴더가 이미 있으면 덮어쓰지 않고 건너뛴다.
"""

import shutil
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = [PROJECT_ROOT / "results_PaiNN", PROJECT_ROOT / "results_agent_logs"]


def _text(message):
    content = message["content"]
    return content if isinstance(content, str) else "".join(part["text"] for part in content if part["type"] == "text")


def _wait_until_finished(client, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status_text, log_text = client.predict(api_name="/training_status")
        if "진행 중" not in status_text:
            return status_text, log_text
        time.sleep(2)
    raise AssertionError("학습이 끝나지 않았다")


@pytest.mark.slow
def test_E2E_학습_탭에서_학습하고_채팅과_예측_탭이_같은_체크포인트로_예측한다():
    if any(path.exists() for path in OUTPUTS):
        pytest.skip("저장소 루트에 results_PaiNN 또는 results_agent_logs가 이미 있어 덮어쓰지 않도록 건너뜀")
    from gradio_client import Client

    from agent import ui_logic
    from agent.ui import build_ui

    demo = build_ui(ui_logic.resolve_llm({}, demo=True))
    try:
        _, url, _ = demo.launch(prevent_thread_lock=True, server_name="127.0.0.1", quiet=True)
        client = Client(url, verbose=False)

        started = client.predict("PaiNN", True, "", True, False, api_name="/training_start")  # 빠른 설정, 확인 체크
        assert "시작했습니다" in started, started
        status_text, log_text = _wait_until_finished(client)
        assert "완료" in status_text and "체크포인트" in status_text, (status_text, log_text[-500:])
        assert log_text.strip()

        history, _ = client.predict("cn1_cn1_nn1 스펙트럼 예측해줘", [], api_name="/chat")
        reply = _text(history[-1])
        assert "cn1_cn1_nn1" in reply and "피크 파장" in reply, reply  # 데모 LLM이 예측 tool을 호출해 방금 학습한 체크포인트로 답한다

        plot, summary = client.predict("cn1_cn1_nn1", "자동", True, api_name="/predict")
        assert len(plot["data"]) == 1600  # 예측 800 + 실험 800
        assert "PaiNN" in summary and "checkpoint" in summary
    finally:
        demo.close()
        ui_logic.shutdown_demo_server()
        for path in OUTPUTS:
            shutil.rmtree(path, ignore_errors=True)  # 이 테스트가 만든 산출물만 정리 (시작 전에 없음을 확인함)
