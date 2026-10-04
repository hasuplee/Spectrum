"""규칙 기반 데모 LLM 서버 (Plan.md Step 5C-2). 진짜 LLM이 아니라 키워드 규칙으로 tool 호출과 답변을 흉내 낸다.

`respond(messages)`는 HTTP 없는 순수 함수로, **메시지 이력만 보고** 다음 응답(텍스트 또는 tool 호출)을 정한다(상태 없음).
AGNO는 이전 턴의 assistant(tool_calls)와 tool 결과를 다음 요청에 모두 싣는다. tool 결과(`tool` 메시지)의 content는 JSON이 아니라
Python 표현 문자열(str(dict))이라 `ast.literal_eval`로 읽고, assistant의 tool_calls 인자는 JSON 문자열이다.
모든 텍스트 응답은 "[데모] "로 시작한다.

대화 규칙(위에서부터 우선): 동의/거부(대기 중인 미리보기가 있을 때) → 직전 모델 질문에 대한 답 → 학습 상태 → 예측 → 분자 목록 → 학습된 모델 →
기본값 조회 → 학습 요청 → 도움 → 그 외(이해하지 못함).
"""

import ast
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

DEMO_MODEL_ID = "demo-rule-based"
PREFIX = "[데모] "
_MODELS = ("PaiNN", "Geoformer", "Equiformer")
_MODEL_QUESTION = "PaiNN, Geoformer, Equiformer 중 무엇으로 학습할까요?"
_DEFAULTS_MODEL_QUESTION = "PaiNN, Geoformer, Equiformer 중 어떤 모델의 기본값을 보여 드릴까요?"
_MOLECULE_ID = re.compile(r"[a-z]+\d*_[a-z]+\d*_nn\d+")
_YES_WORDS = {"응", "네", "예", "좋아", "좋아요", "그래", "진행해줘", "시작해줘", "오케이", "ok", "yes", "y"}
_NO_WORDS = {"아니", "아니요", "취소", "취소해줘", "그만", "그만해줘", "멈춰", "멈춰줘", "no", "n"}


# --- 메시지 이력 해석 -----------------------------------------------------------------------


def _content(message) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _parse_result(message) -> dict:
    try:
        parsed = ast.literal_eval(_content(message))
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_calls(message) -> list:
    """assistant 메시지의 (tool 이름, 인자 dict) 목록."""
    return [(call["function"]["name"], json.loads(call["function"]["arguments"] or "{}")) for call in message.get("tool_calls") or []]


def _tokens(text) -> list:
    return [token for token in (re.sub(r"[^\w]", "", part) for part in re.split(r"[\s,]+", text.lower())) if token]


def _is_reply(text, words) -> bool:
    tokens = _tokens(text)
    return bool(tokens) and all(token in words for token in tokens)


def _detect_model(text):
    lowered = text.lower()
    return next((model for model in _MODELS if model.lower() in lowered), None)


def _pending_preview(prior):
    """대기 중인 미리보기: (인자, will_overwrite) 또는 None.

    마지막 preview_training 호출의 결과가 needs_confirmation이고, 그 뒤에 사용자 메시지가 없으면(= 방금 확인 질문을 했으면) 대기 중이다.
    """
    for index in range(len(prior) - 1, -1, -1):
        calls = _tool_calls(prior[index]) if prior[index].get("role") == "assistant" else []
        preview = next((arguments for name, arguments in calls if name == "preview_training"), None)
        if preview is None:
            continue
        results = [message for message in prior[index + 1:] if message.get("role") == "tool"]
        result = _parse_result(results[0]) if results else {}
        later_user_message = any(message.get("role") == "user" for message in prior[index + 1:])
        if result.get("status") == "needs_confirmation" and not later_user_message:
            return preview, bool(result.get("will_overwrite"))
        return None
    return None


def _asked_model_last(prior) -> bool:
    last = prior[-1] if prior else {}
    return last.get("role") == "assistant" and "무엇으로 학습할까요" in _content(last)


def _extract_settings(text) -> dict:
    settings = {}
    step = re.search(r"(\d+)\s*step", text, re.IGNORECASE)
    if step:
        settings["train_steps"] = int(step.group(1))
    batch = re.search(r"배치\s*(?:크기)?\s*(?:는|를|은)?\s*(\d+)", text)
    if batch:
        settings["batch_size"] = int(batch.group(1))
    return settings


# --- 응답 구성 -----------------------------------------------------------------------------


def _completion(message, finish_reason) -> dict:
    return {
        "id": "demo-completion", "object": "chat.completion", "created": 0, "model": DEMO_MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _text(text) -> dict:
    return _completion({"role": "assistant", "content": PREFIX + text}, "stop")


def _call(name, arguments, call_id) -> dict:
    tool_call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}
    return _completion({"role": "assistant", "content": None, "tool_calls": [tool_call]}, "tool_calls")


# --- 대화 규칙 -----------------------------------------------------------------------------


def respond(messages) -> dict:
    """OpenAI chat completion 형식의 다음 응답. 현재 턴(마지막 user 이후)에 tool 호출이 있었으면 그 결과를 이어서 처리한다."""
    last_user = max(index for index, message in enumerate(messages) if message.get("role") == "user")
    user_text = _content(messages[last_user])
    prior, turn = messages[:last_user], messages[last_user + 1:]
    call_id = f"call_demo_{len(messages)}"
    if any(message.get("role") == "tool" for message in turn):
        return _continue(user_text, turn)
    return _start(user_text, prior, call_id)


def _start(user_text, prior, call_id) -> dict:
    lowered = user_text.lower()
    model = _detect_model(user_text)
    pending = _pending_preview(prior)

    if pending and _is_reply(user_text, _YES_WORDS):
        arguments, will_overwrite = pending
        confirmed = {"base_model": arguments["base_model"], "settings": arguments.get("settings") or {}}
        if will_overwrite:
            confirmed["overwrite"] = True
        return _call("start_training_confirmed", confirmed, call_id)
    if pending and _is_reply(user_text, _NO_WORDS):
        return _text("알겠어요. 학습을 시작하지 않고 취소했어요. 다시 하려면 'PaiNN으로 학습해줘'처럼 말씀해 주세요.")
    if _is_reply(user_text, _YES_WORDS):
        return _text("진행할 학습 설정이 아직 없어요. 먼저 'PaiNN으로 학습해줘'처럼 학습할 모델을 알려 주세요.")
    if _asked_model_last(prior) and model:
        return _call("show_training_defaults", {"base_model": model}, call_id)
    if any(word in lowered for word in ("상태", "끝났", "진행 상황", "로그")):
        return _call("check_training_status", {}, call_id)
    if "예측" in lowered or "스펙트럼" in lowered:
        molecule_id = _MOLECULE_ID.search(lowered)
        if molecule_id:
            return _call("predict_molecule_spectrum", {"molecule_id": molecule_id.group(0)}, call_id)
        return _call("list_molecule_ids", {"limit": 5}, call_id)
    if "분자" in lowered and any(word in lowered for word in ("목록", "어떤", "있어", "보여")):
        return _call("list_molecule_ids", {"limit": 10}, call_id)
    if "학습된" in lowered or ("모델" in lowered and any(word in lowered for word in ("있어", "목록", "사용 가능"))):
        return _call("list_trained_models", {}, call_id)
    if _is_defaults_query(lowered):
        if model:
            return _call("show_training_defaults", {"base_model": model}, call_id)
        return _text(_DEFAULTS_MODEL_QUESTION)
    if "학습" in lowered or "train" in lowered or "훈련" in lowered:
        if model:
            return _call("show_training_defaults", {"base_model": model}, call_id)
        return _text(_MODEL_QUESTION)
    if any(word in lowered for word in ("뭘 할 수", "할 수 있", "도움", "사용법")):
        return _text("학습과 예측을 도와드려요. 예: 'PaiNN으로 학습해줘', '학습 기본값 보여줘', '학습 상태 알려줘', '분자 목록 보여줘', 'cn1_cn1_nn1 스펙트럼 예측해줘'.")
    return _text("이해하지 못했어요. 데모 서버는 규칙 기반이라 정해진 표현만 알아듣습니다. 예: 'PaiNN으로 학습해줘', '학습 상태 알려줘', 'cn1_cn1_nn1 스펙트럼 예측해줘'.")


def _is_defaults_query(lowered) -> bool:
    return any(word in lowered for word in ("기본값", "설정")) and any(word in lowered for word in ("보여", "알려"))


def _continue(user_text, turn) -> dict:
    """현재 턴에서 호출한 마지막 tool의 결과를 보고 다음 tool을 호출하거나 최종 텍스트를 만든다."""
    last_call = next(call for message in reversed(turn) if message.get("role") == "assistant" for call in _tool_calls(message))
    name, arguments = last_call
    result = _parse_result(next(message for message in reversed(turn) if message.get("role") == "tool"))
    lowered = user_text.lower()
    status = result.get("status")
    problem = result.get("message")

    if name == "show_training_defaults":
        if status != "ok":
            return _text(problem or "기본값을 조회하지 못했어요.")
        if _is_defaults_query(lowered):
            return _text(f"{result['base_model']}의 기본 설정이에요: step {result['train_steps']}, 배치 크기 {result['batch_size']}, "
                         f"스펙트럼 {result['spectrum_type']}, 장치 {result['device']}. 학습하려면 '{result['base_model']}으로 학습해줘'라고 말씀해 주세요.")
        return _call("preview_training", {"base_model": result["base_model"], "settings": _extract_settings(user_text)}, f"call_demo_{len(turn)}p")
    if name == "preview_training":
        if status != "needs_confirmation":
            return _text(problem or "미리보기를 만들지 못했어요.")
        settings = result["settings"]
        text = (f"{settings['base_model']} 학습 설정이에요: step {settings['train_steps']}, 배치 크기 {settings['batch_size']}, "
                f"스펙트럼 {settings['spectrum_type']}, 저장 위치 {result['output_dir']}.")
        if result.get("will_overwrite"):
            text += " 이미 학습된 결과가 있어 다시 학습하면 기존 결과가 삭제돼요."
        return _text(text + " 진행할까요?")
    if name == "start_training_confirmed":
        if status == "started":
            return _text(f"학습을 시작했어요 (작업 {result['job_id']}). '학습 상태 알려줘'로 진행 상황을 확인할 수 있어요.")
        return _text(problem or "학습을 시작하지 못했어요.")
    if name == "check_training_status":
        return _text(_summarize_status(result))
    if name == "predict_molecule_spectrum":
        if status == "ok":
            samples = ", ".join(f"{sample['wavelength_nm']}nm {sample['intensity']}" for sample in result["samples"])
            return _text(f"{result['molecule_id']}의 예측 결과예요 ({result['base_model']}, {result['spectrum_type']}): 피크 파장은 {result['peak_wavelength_nm']:.1f}nm이고, "
                         f"파장별 상대 강도는 {samples}예요.")
        if status == "needs_training":
            return _text("아직 학습된 모델이 없어요. 먼저 학습이 필요해요. 'PaiNN으로 학습해줘'처럼 말씀해 주세요.")
        return _text(problem or "예측하지 못했어요.")
    if name == "list_molecule_ids":
        ids = ", ".join(result.get("molecule_ids", []))
        if "예측" in lowered or "스펙트럼" in lowered:
            return _text(f"예측할 분자 ID를 알려 주세요. 예: {ids}")
        return _text(f"분자 {result.get('total_matches', '여러')}개 중 일부예요: {ids}")
    if name == "list_trained_models":
        trained = [model for model, info in result.get("models", {}).items() if info["trained"]]
        return _text(f"학습된 모델: {', '.join(trained)}." if trained else "아직 학습된 모델이 없어요. 먼저 학습이 필요해요.")
    return _text(problem or "처리했어요.")


def _summarize_status(result) -> str:
    if result.get("status") == "no_job":
        return result.get("message", "시작된 학습 작업이 없어요.")
    last_log = result["log_tail"][-1] if result.get("log_tail") else ""
    state = result["state"]
    if state == "running":
        return f"학습이 진행 중이에요 (경과 {result['elapsed_seconds']}초). 마지막 로그: {last_log}"
    if state == "finished":
        if result["has_checkpoint"]:
            return f"학습이 끝났어요 ({result['elapsed_seconds']}초). 이제 'cn1_cn1_nn1 스펙트럼 예측해줘'처럼 예측할 수 있어요."
        return f"학습이 끝났어요 ({result['elapsed_seconds']}초). 다만 체크포인트가 아직 없어요."
    return f"학습이 실패했어요 (종료 코드 {result['return_code']}). 마지막 로그: {last_log}"


# --- HTTP 서버 -----------------------------------------------------------------------------


class MockLLMServer:
    """`respond`를 /v1/chat/completions로 서비스하는 로컬 OpenAI 호환 서버 (컨텍스트 매니저). port=0이면 빈 포트를 자동 할당한다."""

    def __init__(self, host="127.0.0.1", port=0):
        self._host, self._port = host, port
        self.requests = []  # 받은 요청 본문 기록
        self._server = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._server.server_port}/v1"

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, payload):
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(body)
                self._send(respond(body["messages"]))

            def do_GET(self):  # /v1/models
                self._send({"object": "list", "data": [{"id": DEMO_MODEL_ID, "object": "model"}]})

            def log_message(self, *args):
                pass

        self._server = HTTPServer((self._host, self._port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
