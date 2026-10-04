"""테스트 공용 fixture."""

import pytest

from tests.support.fake_openai import FakeOpenAIServer


@pytest.fixture
def fake_llm():
    """LLM API 키 없이 쓰는 가짜 OpenAI 호환 서버 (Plan.md Step 5A). `queue_text`/`queue_tool_calls`로 응답을 준비한다."""
    with FakeOpenAIServer() as server:
        yield server
