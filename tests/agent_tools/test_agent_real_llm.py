"""Plan.md Step 5C-2: 실제 LLM 서버(vLLM 등)에 연결하는 선택 테스트.

`VLLM_BASE_URL`과 `VLLM_MODEL`(필요하면 `VLLM_API_KEY`)이 설정된 환경에서만 실행되고, 없으면 건너뛴다 — 개발·CI에서는 LLM 키 없이 mock만 쓴다.
내부 LLM API(URL, 토큰)를 연결한 뒤 사용자가 직접 실행한다:
    VLLM_BASE_URL=... VLLM_MODEL=... VLLM_API_KEY=... pytest -m vllm
LLM의 응답 내용은 모델마다 달라 검증하지 않고, "연결이 되어 응답을 받는다"와 "범위 밖 질문은 가드가 LLM 호출 없이 거절한다"만 확인한다.
(LLM이 instructions를 잘 따르는가, tool 호출을 올바르게 고르는가는 대화 로그를 보고 사람이 판단한다. 서버는 tool 호출 지원이 필요하다:
vLLM이면 `--enable-auto-tool-choice --tool-call-parser <모델에 맞는 값>`.)
"""

import os

import pytest

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.skipif(
        not (os.environ.get("VLLM_BASE_URL") and os.environ.get("VLLM_MODEL")),
        reason="VLLM_BASE_URL/VLLM_MODEL이 설정되지 않아 건너뜀 (실제 LLM 서버를 연결했을 때 실행)",
    ),
]


def test_실제_LLM_서버에_연결해_범위_안_질문에_응답한다(tmp_path):
    from agent.agent import build_agent, chat

    agent = build_agent(project_root=tmp_path)  # 모델은 환경변수(VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY)에서 만든다

    reply = chat(agent, "학습 상태 알려줘")

    assert reply.strip()
    assert "LLM 서버 호출에 실패했습니다" not in reply  # 연결과 tool 호출 왕복이 성공했다


def test_실제_LLM_서버에서도_범위_밖_질문은_가드가_거절한다(tmp_path):
    from agent.agent import build_agent, chat
    from agent.guard import REFUSAL_MESSAGE

    agent = build_agent(project_root=tmp_path)

    assert chat(agent, "오늘 날씨가 뭐야?") == REFUSAL_MESSAGE
