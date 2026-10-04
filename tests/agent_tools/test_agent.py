"""Plan.md Step 5C-1 [신규 인터페이스], G3: Agent 조립 (`agent.agent`: build_model / INSTRUCTIONS / build_agent / chat).

agent.agent가 아직 없으므로 실패해야 한다 (RED). LLM은 진짜가 아니라 가짜 OpenAI 호환 서버(`fake_llm`, API 키 불필요)이며,
학습 명령은 4B 방식으로 `[sys.executable, "-c", ...]`로 대체해 실제 서브프로세스를 쓴다. 따라서 여기서 검증하는 것은 연결·tool 실행·확인 절차·가드·대화 기억·세션 분리이고,
"진짜 LLM이 instructions를 잘 따르는가"는 검증하지 않는다(LLM의 응답은 테스트가 정한다).
새 모듈 import는 각 테스트 안에서 하여 개별 실패로 확인한다.
"""

import sys
import time

import pytest

EXPECTED_TOOLS = {
    "list_trained_models", "show_training_defaults", "preview_training", "start_training_confirmed",
    "check_training_status", "list_molecule_ids", "predict_molecule_spectrum",
}


@pytest.fixture(autouse=True)
def _fresh_jobs(monkeypatch):
    from agent.tools import train_tool

    monkeypatch.setattr(train_tool, "_jobs", {}, raising=False)


def _use_fake_training(monkeypatch, code):
    from agent.tools import train_tool

    monkeypatch.setattr(train_tool, "build_training_command", lambda request: [sys.executable, "-c", code])


def _fake_model(fake_llm):
    from agno.models.vllm import VLLM

    return VLLM(id="fake", base_url=fake_llm.base_url, api_key="EMPTY", max_retries=0)


def _wait_for_training(project_root, timeout=60):
    from agent.tools.train_tool import get_training_status

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = get_training_status(project_root=project_root)
        if status["status"] == "ok" and status["state"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError("학습 작업이 끝나지 않았다")


def _tool_results(request):
    return [message["content"] for message in request["messages"] if message["role"] == "tool"]


# --- 모델 설정 ---------------------------------------------------------------------------


def test_build_model은_필수_환경변수가_없으면_누락된_이름을_알려준다():
    from agent.agent import ModelConfigError, build_model

    with pytest.raises(ModelConfigError) as nothing_set:
        build_model({})
    with pytest.raises(ModelConfigError) as only_model:
        build_model({"VLLM_MODEL": "some-model"})

    assert issubclass(ModelConfigError, ValueError)
    assert "VLLM_BASE_URL" in str(nothing_set.value) and "VLLM_MODEL" in str(nothing_set.value)
    assert "VLLM_BASE_URL" in str(only_model.value) and "VLLM_MODEL" not in str(only_model.value)


def test_build_model은_환경변수로_VLLM을_구성한다(monkeypatch):
    from agno.models.vllm import VLLM

    from agent.agent import build_model

    configured = build_model({"VLLM_BASE_URL": "http://internal:8000/v1", "VLLM_MODEL": "my-model", "VLLM_API_KEY": "secret-token"})
    without_key = build_model({"VLLM_BASE_URL": "http://internal:8000/v1", "VLLM_MODEL": "my-model"})
    monkeypatch.setenv("VLLM_BASE_URL", "http://from-env:8000/v1")
    monkeypatch.setenv("VLLM_MODEL", "env-model")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    from_environ = build_model()  # 인자를 생략하면 os.environ을 읽는다

    assert isinstance(configured, VLLM)
    assert (configured.id, configured.base_url, configured.api_key) == ("my-model", "http://internal:8000/v1", "secret-token")
    assert without_key.api_key == "EMPTY"  # 인증 없는 vLLM 서버용 더미 키 (AGNO VLLM은 키를 요구한다)
    assert configured.max_retries == 1  # 연결 실패를 빠르게 드러낸다 (SDK 기본 재시도로는 약 7.5초)
    assert (from_environ.id, from_environ.base_url) == ("env-model", "http://from-env:8000/v1")


# --- instructions / 구성 -----------------------------------------------------------------


def test_instructions에는_학습_확인_절차와_덮어쓰기_경고와_범위_제한이_담겨_있다():
    from agent.agent import INSTRUCTIONS

    text = "\n".join(INSTRUCTIONS)

    for tool_name in ("show_training_defaults", "preview_training", "start_training_confirmed",
                      "check_training_status", "list_molecule_ids", "predict_molecule_spectrum"):
        assert tool_name in text, f"instructions에 tool 이름이 있어야 한다: {tool_name}"
    for phrase in ("PaiNN", "Geoformer", "Equiformer", "동의", "overwrite", "삭제", "거절", "needs_training"):
        assert phrase in text, f"instructions에 핵심 규칙이 있어야 한다: {phrase}"


def test_build_agent는_tool_7개_가드_대화기억을_갖춘다(fake_llm, tmp_path):
    from agno.db.in_memory import InMemoryDb

    from agent.agent import build_agent
    from agent.guard import ScopeGuardrail

    agent = build_agent(model=_fake_model(fake_llm), project_root=tmp_path)

    assert {tool.__name__ for tool in agent.tools} == EXPECTED_TOOLS
    assert any(isinstance(hook, ScopeGuardrail) for hook in agent.pre_hooks)
    assert isinstance(agent.db, InMemoryDb)
    assert agent.add_history_to_context is True and agent.num_history_runs >= 5
    assert agent.tool_call_limit is not None and agent.tool_call_limit >= 4  # tool 호출 무한 반복 방지
    assert agent.telemetry is False


def test_Agent_요청에는_instructions와_tool_7개가_실린다(fake_llm, tmp_path):
    from agent.agent import build_agent, chat

    fake_llm.queue_text("안녕하세요")
    agent = build_agent(model=_fake_model(fake_llm), project_root=tmp_path)

    chat(agent, "학습 상태 알려줘")

    request = fake_llm.requests[0]
    system_text = " ".join(m["content"] for m in request["messages"] if m["role"] == "system")
    assert "start_training_confirmed" in system_text and "동의" in system_text
    assert {tool["function"]["name"] for tool in request["tools"]} == EXPECTED_TOOLS


# --- chat() ------------------------------------------------------------------------------


def test_범위_밖_질문은_chat이_거절_문구를_돌려주고_LLM을_호출하지_않는다(fake_llm, tmp_path):
    from agent.agent import build_agent, chat
    from agent.guard import REFUSAL_MESSAGE

    fake_llm.queue_text("이 응답은 사용되면 안 된다")
    agent = build_agent(model=_fake_model(fake_llm), project_root=tmp_path)

    reply = chat(agent, "오늘 날씨가 뭐야?")

    assert reply == REFUSAL_MESSAGE
    assert len(fake_llm.requests) == 0


def test_chat은_정상_응답_텍스트를_그대로_돌려준다(fake_llm, tmp_path):
    from agent.agent import build_agent, chat

    fake_llm.queue_text("PaiNN, Geoformer, Equiformer 중 무엇으로 학습할까요?")
    agent = build_agent(model=_fake_model(fake_llm), project_root=tmp_path)

    reply = chat(agent, "그냥 학습해줘")

    assert reply == "PaiNN, Geoformer, Equiformer 중 무엇으로 학습할까요?"


def test_chat은_LLM_서버_오류를_사용자용_문구로_바꾼다(tmp_path):
    from agno.models.vllm import VLLM

    from agent.agent import build_agent, chat
    from agent.guard import REFUSAL_MESSAGE

    unreachable = VLLM(id="fake", base_url="http://127.0.0.1:9/v1", api_key="EMPTY", max_retries=0, timeout=3)
    agent = build_agent(model=unreachable, project_root=tmp_path)

    reply = chat(agent, "학습 상태 알려줘")

    assert isinstance(reply, str) and reply != REFUSAL_MESSAGE
    assert "LLM" in reply and "VLLM_BASE_URL" in reply  # 무엇이 잘못되었고 무엇을 확인할지 알려 준다


# --- 대화 시나리오 (가짜 LLM이 tool을 호출하도록 스크립트) -------------------------------------


def test_학습_시나리오_미리보기_후_사용자_확인으로_학습이_시작된다(fake_llm, tmp_path, monkeypatch):
    from agent.agent import build_agent, chat

    marker = tmp_path / "ran.txt"
    _use_fake_training(monkeypatch, f"open({str(marker)!r}, 'w').write('x'); print('학습 실행')")
    settings = {"train_steps": 5}
    agent = build_agent(model=_fake_model(fake_llm), project_root=tmp_path)

    # 1턴: LLM이 미리보기를 호출하고 사용자에게 확인을 묻는다
    fake_llm.queue_tool_calls(("preview_training", {"base_model": "PaiNN", "settings": settings}))
    fake_llm.queue_text("PaiNN, 5 step으로 학습합니다. 진행할까요?")
    first_reply = chat(agent, "PaiNN으로 5 step 학습해줘", session_id="s")
    started_too_early = marker.exists()

    # 2턴: 사용자가 동의하면 LLM이 같은 인자로 확인 실행을 호출한다
    fake_llm.queue_tool_calls(("start_training_confirmed", {"base_model": "PaiNN", "settings": settings}))
    fake_llm.queue_text("학습을 시작했어요")
    second_reply = chat(agent, "응", session_id="s")
    status = _wait_for_training(tmp_path)

    assert first_reply == "PaiNN, 5 step으로 학습합니다. 진행할까요?" and second_reply == "학습을 시작했어요"
    assert not started_too_early  # 사용자 확인 전에는 학습이 시작되지 않는다
    second_turn_first_request = fake_llm.requests[2]["messages"]  # 요청 0·1은 1턴, 2·3은 2턴
    assert [m["role"] for m in second_turn_first_request if m["role"] != "system"] == ["user", "assistant", "tool", "assistant", "user"]
    assert any("needs_confirmation" in content for content in _tool_results(fake_llm.requests[2]))  # 이전 턴의 tool 결과가 기억되어 전달됨
    assert second_turn_first_request[-1]["content"] == "응"
    assert any("started" in content for content in _tool_results(fake_llm.requests[3]))
    assert status["state"] == "finished" and marker.exists()


def test_미리보기_없이_학습_시작을_시도하면_tool이_거부한다(fake_llm, tmp_path, monkeypatch):
    from agent.agent import build_agent, chat

    marker = tmp_path / "ran.txt"
    _use_fake_training(monkeypatch, f"open({str(marker)!r}, 'w').write('x')")
    fake_llm.queue_tool_calls(("start_training_confirmed", {"base_model": "PaiNN"}))  # 설정을 보여 주는 단계를 건너뛰는 LLM
    fake_llm.queue_text("먼저 설정을 확인해야 해요")
    agent = build_agent(model=_fake_model(fake_llm), project_root=tmp_path)

    chat(agent, "PaiNN으로 학습해줘")
    time.sleep(1.0)

    assert any("not_previewed" in content for content in _tool_results(fake_llm.requests[1]))
    assert not marker.exists()


def test_학습된_모델이_없을_때_예측_시나리오는_needs_training_결과를_LLM에_전달한다(fake_llm, tmp_path):
    from agent.agent import build_agent, chat

    fake_llm.queue_tool_calls(("predict_molecule_spectrum", {"molecule_id": "cn1_cn1_nn1"}))
    fake_llm.queue_text("먼저 학습이 필요해요. PaiNN으로 학습할까요?")
    agent = build_agent(model=_fake_model(fake_llm), project_root=tmp_path, results_root=tmp_path)

    reply = chat(agent, "cn1_cn1_nn1 스펙트럼 예측해줘")

    assert any("needs_training" in content for content in _tool_results(fake_llm.requests[1]))
    assert reply == "먼저 학습이 필요해요. PaiNN으로 학습할까요?"


def test_세션마다_Agent가_독립적이다(fake_llm, tmp_path, monkeypatch):
    from agent.agent import build_agent, chat

    marker = tmp_path / "ran.txt"
    _use_fake_training(monkeypatch, f"open({str(marker)!r}, 'w').write('x')")
    agent_a = build_agent(model=_fake_model(fake_llm), project_root=tmp_path)
    agent_b = build_agent(model=_fake_model(fake_llm), project_root=tmp_path)

    # 사용자 A가 미리보기까지 했다
    fake_llm.queue_tool_calls(("preview_training", {"base_model": "PaiNN"}))
    fake_llm.queue_text("A: 진행할까요?")
    chat(agent_a, "A 사용자의 PaiNN 학습 요청", session_id="same")
    # 사용자 B(같은 session_id)가 확인 실행만 호출한다
    fake_llm.queue_tool_calls(("start_training_confirmed", {"base_model": "PaiNN"}))
    fake_llm.queue_text("B: 거부되었어요")
    chat(agent_b, "B 사용자의 학습 요청", session_id="same")
    time.sleep(1.0)

    assert any("not_previewed" in content for content in _tool_results(fake_llm.requests[3]))  # B에게는 A의 미리보기가 없다
    assert "A 사용자의 PaiNN 학습 요청" not in str(fake_llm.requests[2]["messages"])  # 대화 기억도 분리됨
    assert not marker.exists()
