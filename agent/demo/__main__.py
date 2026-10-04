"""`python -m agent.demo [--port N] [--serve-only]` — 규칙 기반 데모 서버와 콘솔 채팅 (Plan.md Step 5C-2).

진짜 LLM이 아니다. 기본은 데모 서버를 띄우고 그 서버에 연결한 Agent와 콘솔로 대화한다(종료: 종료/exit/quit).
--serve-only는 서버만 실행한다: 다른 클라이언트(UI 등)에서 VLLM_BASE_URL=http://127.0.0.1:<port>/v1, VLLM_MODEL=demo-rule-based로 연결한다.
"""

import argparse
import sys
import time

from agent.demo.mock_llm import DEMO_MODEL_ID, MockLLMServer

BANNER = (
    "=" * 64 + "\n"
    "[데모 모드] 진짜 LLM이 아니라 규칙 기반 가짜 서버입니다. 정해진 표현만 알아듣습니다.\n"
    "예: 'PaiNN으로 학습해줘', '학습 기본값 보여줘', '학습 상태 알려줘', '분자 목록 보여줘', 'cn1_cn1_nn1 스펙트럼 예측해줘'\n"
    "종료: 종료 / exit / quit\n" + "=" * 64
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.demo", description="규칙 기반 데모 LLM 서버와 콘솔 채팅 (진짜 LLM이 아님)")
    parser.add_argument("--port", type=int, default=8765, help="데모 서버 포트 (0이면 빈 포트를 자동 할당, 기본 8765)")
    parser.add_argument("--serve-only", action="store_true", help="서버만 실행한다 (콘솔 채팅 없음)")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    with MockLLMServer(port=args.port) as server:
        print(BANNER, flush=True)
        if args.serve_only:
            print(f"데모 서버 주소: {server.base_url}\n다른 클라이언트에서: VLLM_BASE_URL={server.base_url} VLLM_MODEL={DEMO_MODEL_ID}\n(Ctrl+C로 종료)", flush=True)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                return 0

        from agno.models.vllm import VLLM

        from agent.agent import build_agent, chat

        agent = build_agent(model=VLLM(id=DEMO_MODEL_ID, base_url=server.base_url, api_key="EMPTY", max_retries=0))
        while True:
            try:
                line = input("나> ").strip()
            except EOFError:
                break
            if line in ("종료", "exit", "quit"):
                break
            if line:
                print(f"도우미> {chat(agent, line)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
