"""가짜 OpenAI 호환 서버 (Plan.md Step 5A): 실제 LLM/API 키 없이 AGNO `VLLM` 연결 경로를 테스트한다.

로컬(127.0.0.1)에서 뜨는 작은 HTTP 서버가 `/v1/chat/completions` 요청에 **미리 넣어 둔 응답**을 차례로 돌려준다.
받은 요청은 `requests`에 기록되므로, 테스트는 (1) LLM이 몇 번 호출되었는지, (2) 요청에 tools/instructions/이전 대화가
실렸는지를 검사할 수 있다. 응답 큐가 비어 있는데 요청이 오면 500 오류를 돌려줘 예상하지 못한 호출이 드러나게 한다.

실제 LLM의 판단(어떤 tool을 고르는가 등)은 이 서버가 대신하지 못한다 — 어떤 응답을 돌려줄지는 테스트가 정한다.
"""

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer


def _completion(message: dict, finish_reason: str) -> dict:
    return {
        "id": "fake-completion", "object": "chat.completion", "created": 0, "model": "fake",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def text_response(content: str) -> dict:
    return _completion({"role": "assistant", "content": content}, "stop")


def tool_calls_response(*calls) -> dict:
    """calls: (tool 이름, 인자 dict) 튜플들. LLM이 tool 호출을 요청하는 응답."""
    tool_calls = [
        {"id": f"call_{index}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}
        for index, (name, arguments) in enumerate(calls)
    ]
    return _completion({"role": "assistant", "content": None, "tool_calls": tool_calls}, "tool_calls")


class FakeOpenAIServer:
    def __init__(self):
        self.requests = []          # 받은 요청 본문(dict) 목록
        self._responses = deque()   # 돌려줄 응답 큐
        self._lock = threading.Lock()
        self._server = None

    # -- 응답 준비 ---------------------------------------------------------------
    def queue_text(self, content: str):
        self._responses.append(text_response(content))

    def queue_tool_calls(self, *calls):
        self._responses.append(tool_calls_response(*calls))

    # -- 수명 --------------------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with owner._lock:
                    owner.requests.append(body)
                    payload = owner._responses.popleft() if owner._responses else None
                if payload is None:
                    data, status = json.dumps({"error": {"message": "응답 큐가 비어 있는데 요청이 왔다", "type": "fake_server"}}).encode(), 500
                else:
                    data, status = json.dumps(payload).encode(), 200
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
