"""Plan.md Step 5C-2 [신규 인터페이스], G3, G5: 규칙 기반 데모 서버로 구동하는 Agent 대화 시나리오.

agent.demo가 아직 없으므로 실패해야 한다 (RED). 실제 `build_agent` + AGNO `VLLM` + `MockLLMServer`(LLM 키 불필요)를 쓰며,
학습은 4B 방식의 가짜 명령(실제 서브프로세스), 예측은 tiny PaiNN 체크포인트와 실제 IrDB를 쓴다.
데모 서버는 진짜 LLM이 아니므로 여기서 검증하는 것은 연결·tool 실행·확인 절차·가드·대화 이력 활용이며, LLM의 이해력은 검증하지 않는다.
"""

import time

import pytest

from tests.agent_tools.test_predict_tool import _save_painn_ckpt
from tests.support.fake_training import use_fake_training


@pytest.fixture
def demo_server():
    from agent.demo.mock_llm import MockLLMServer

    with MockLLMServer() as server:
        yield server


def _demo_agent(demo_server, **kwargs):
    from agno.models.vllm import VLLM

    from agent.agent import build_agent

    return build_agent(model=VLLM(id="demo-rule-based", base_url=demo_server.base_url, api_key="EMPTY", max_retries=0), **kwargs)


def _wait_for_training(project_root, timeout=60):
    from agent.tools.train_tool import get_training_status

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = get_training_status(project_root=project_root)
        if status["status"] == "ok" and status["state"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError("학습 작업이 끝나지 않았다")


def test_대화_시나리오_모델_되묻기부터_미리보기_동의_학습_시작_상태_확인까지_이어진다(demo_server, tmp_path, monkeypatch):
    from agent.agent import chat

    marker = tmp_path / "ran.txt"
    use_fake_training(monkeypatch, f"open({str(marker)!r}, 'w').write('x'); print('학습 실행')")
    agent = _demo_agent(demo_server, project_root=tmp_path)

    asked_model = chat(agent, "그냥 학습해줘", session_id="s")
    previewed = chat(agent, "PaiNN", session_id="s")
    started_too_early = marker.exists()
    started = chat(agent, "응", session_id="s")
    _wait_for_training(tmp_path)
    finished = chat(agent, "끝났어?", session_id="s")

    replies = [asked_model, previewed, started, finished]
    assert all(reply.startswith("[데모]") for reply in replies)  # 진짜 LLM이 아님이 항상 표시된다
    assert all(model in asked_model for model in ("PaiNN", "Geoformer", "Equiformer"))
    assert "진행할까요" in previewed and "10000" in previewed  # 기본값(CPU 환경의 step 수 등)을 보여 주고 확인을 묻는다
    assert not started_too_early  # 사용자 동의 전에는 학습이 시작되지 않는다
    assert "시작" in started and marker.exists()  # 동의 후에야 학습 프로세스가 실제로 실행된다
    assert "끝났" in finished


def test_학습된_모델이_있으면_예측_시나리오가_피크_파장을_알려준다(demo_server, tmp_path):
    from agent.agent import chat
    from agent.tools.predict_tool import predict_spectrum

    _save_painn_ckpt(tmp_path)
    agent = _demo_agent(demo_server, results_root=tmp_path)  # 체크포인트는 임시 results_root에, 데이터셋은 실제 저장소 루트에서 연다

    reply = chat(agent, "cn1_cn1_nn1 스펙트럼 예측해줘")

    expected_peak = predict_spectrum("cn1_cn1_nn1", results_root=tmp_path)["peak_wavelength_nm"]
    assert reply.startswith("[데모]") and "PaiNN" in reply and "cn1_cn1_nn1" in reply
    assert f"{expected_peak:.1f}" in reply  # tool 결과의 피크 파장이 그대로 전달된다


def test_학습된_모델이_없으면_예측_시나리오는_학습을_먼저_제안한다(demo_server, tmp_path):
    from agent.agent import chat

    agent = _demo_agent(demo_server, project_root=tmp_path, results_root=tmp_path)

    reply = chat(agent, "cn1_cn1_nn1 스펙트럼 예측해줘")

    assert reply.startswith("[데모]") and "먼저" in reply and "학습" in reply


def test_이미_학습된_모델은_삭제_경고_후_동의해야_다시_학습한다(demo_server, tmp_path, monkeypatch):
    from agent.agent import chat

    use_fake_training(monkeypatch, "import os; print('old_exists=' + str(os.path.exists('results_PaiNN/0/0/old.txt')))")
    old_output = tmp_path / "results_PaiNN" / "0" / "0"
    old_output.mkdir(parents=True)
    (old_output / "checkpoint_best.ckpt").write_bytes(b"")
    (old_output / "old.txt").write_text("이전 학습 산출물", encoding="utf-8")
    agent = _demo_agent(demo_server, project_root=tmp_path)

    warned = chat(agent, "PaiNN으로 학습해줘", session_id="s")
    old_kept_before_consent = (old_output / "old.txt").exists()
    started = chat(agent, "응", session_id="s")
    status = _wait_for_training(tmp_path)

    assert "삭제" in warned and "진행할까요" in warned
    assert old_kept_before_consent  # 동의 전에는 기존 결과가 그대로 있다
    assert "시작" in started
    assert "old_exists=False" in status["log_tail"]  # 동의(overwrite) 후에는 이전 산출물이 지워진 채 시작되었다


def test_범위_밖_질문은_데모_서버를_호출하지_않고_거절된다(demo_server, tmp_path):
    from agent.agent import chat
    from agent.guard import REFUSAL_MESSAGE

    agent = _demo_agent(demo_server, project_root=tmp_path)

    reply = chat(agent, "오늘 날씨가 뭐야?")

    assert reply == REFUSAL_MESSAGE
    assert len(demo_server.requests) == 0
