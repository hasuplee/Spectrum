"""Plan.md Step 5A [신규 인터페이스], G3: 학습/예측과 무관한 질문을 LLM 호출 전에 거절하는 범위 가드.

agent.guard가 아직 없으므로 실패해야 한다 (RED). 판정은 키워드 휴리스틱이며(Plan.md 5A의 규칙 1~5),
의미 기반 판별이 아니므로 정의형 질문에 행동 표현이 붙은 경우 등은 5C의 Agent instructions가 두 번째 방어선이다.
LLM 호출 여부는 가짜 OpenAI 호환 서버(tests/support/fake_openai.py)의 요청 기록으로 검증한다 — API 키 불필요.
새 모듈 import는 각 테스트 안에서 하여 개별 실패로 확인한다.
"""

import asyncio

import pytest


def _agent_with_guard(fake_llm):
    from agno.agent import Agent
    from agno.models.vllm import VLLM

    from agent.guard import ScopeGuardrail

    return Agent(model=VLLM(id="fake", base_url=fake_llm.base_url, api_key="EMPTY"),
                 pre_hooks=[ScopeGuardrail()], telemetry=False)


# --- 판정 규칙 -----------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "PaiNN으로 학습해줘",
    "그냥 학습해줘",
    "Geoformer로 학습 시작해줘",
    "학습 기본값 보여줘",
    "학습 상태 알려줘",
    "이 분자의 spectrum 예측해줘",
    "cn1_cn1_nn1 스펙트럼 예측해줘",
    "사용 가능한 모델 알려줘",
    "분자 목록 보여줘",
    "20 step으로 해줘",
    "배치 크기는 8로 해줘",
])
def test_학습과_예측_요청은_범위_안이다(text):
    from agent.guard import is_in_scope

    assert is_in_scope(text) is True


@pytest.mark.parametrize("text", ["응", "네", "네, 진행해줘", "아니요", "취소", "PaiNN", "yes", "OK!"])
def test_짧은_후속_답변은_범위_안이다(text):
    from agent.guard import is_in_scope

    assert is_in_scope(text) is True


@pytest.mark.parametrize("text", [
    "오늘 날씨가 뭐야?",
    "반도체는 뭐지",
    "OLED의 정의는",
    "파이썬 코드 짜줘",
    "너는 누구야",
    "맛집 추천해줘",
    "",
    "   ",
])
def test_학습과_무관한_질문은_범위_밖이다(text):
    from agent.guard import is_in_scope

    assert is_in_scope(text) is False


@pytest.mark.parametrize("text", ["PaiNN이 뭐야?", "스펙트럼이 뭐야?", "학습이란 무엇인가요"])
def test_키워드가_있어도_정의를_묻는_질문은_범위_밖이다(text):
    from agent.guard import is_in_scope

    assert is_in_scope(text) is False


def test_예시_질문은_가드_판정과_일치한다():
    # UI(Step 6)가 보여 주는 예시 질문의 단일 출처. 동작하는 예시는 통과, 거절 시연용 예시는 거절되어야 한다.
    from agent.guard import ALLOWED_EXAMPLES, REFUSED_EXAMPLES, is_in_scope

    assert len(ALLOWED_EXAMPLES) >= 4 and len(REFUSED_EXAMPLES) >= 4
    assert all(is_in_scope(text) for text in ALLOWED_EXAMPLES)
    assert not any(is_in_scope(text) for text in REFUSED_EXAMPLES)


# --- AGNO 가드레일 ---------------------------------------------------------------------


def test_ScopeGuardrail은_범위_밖_입력에_InputCheckError를_범위_안_입력은_통과시킨다():
    from agno.exceptions import InputCheckError
    from agno.run.agent import RunInput

    from agent.guard import REFUSAL_MESSAGE, ScopeGuardrail

    guardrail = ScopeGuardrail()

    guardrail.check(RunInput(input_content="PaiNN으로 학습해줘"))  # 예외 없이 통과
    asyncio.run(guardrail.async_check(RunInput(input_content="PaiNN으로 학습해줘")))
    with pytest.raises(InputCheckError) as sync_error:
        guardrail.check(RunInput(input_content="오늘 날씨가 뭐야?"))
    with pytest.raises(InputCheckError):
        asyncio.run(guardrail.async_check(RunInput(input_content="오늘 날씨가 뭐야?")))

    assert sync_error.value.message == REFUSAL_MESSAGE


# --- Agent + VLLM + 가짜 서버: LLM 호출 여부 ----------------------------------------------


def test_범위_밖_질문은_LLM_서버를_호출하지_않고_거절_문구를_반환한다(fake_llm):
    from agent.guard import REFUSAL_MESSAGE

    fake_llm.queue_text("이 응답은 사용되면 안 된다")
    agent = _agent_with_guard(fake_llm)

    result = agent.run("오늘 날씨가 뭐야?")

    assert len(fake_llm.requests) == 0
    assert result.content == REFUSAL_MESSAGE


def test_범위_안_질문은_LLM_서버로_전달된다(fake_llm):
    fake_llm.queue_text("학습을 도와드릴게요")
    agent = _agent_with_guard(fake_llm)

    result = agent.run("PaiNN으로 학습해줘")

    assert len(fake_llm.requests) == 1
    assert "PaiNN으로 학습해줘" in str(fake_llm.requests[0]["messages"])
    assert result.content == "학습을 도와드릴게요"
