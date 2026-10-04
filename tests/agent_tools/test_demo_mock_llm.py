"""Plan.md Step 5C-2 [신규 인터페이스], G3, G5: 규칙 기반 데모 LLM `agent.demo.mock_llm.respond`.

agent.demo가 아직 없으므로 실패해야 한다 (RED). `respond(messages)`는 HTTP 없는 순수 함수로, 메시지 이력만 보고 다음 응답(텍스트 또는 tool 호출)을 정한다 (상태 없음).
이 서버는 진짜 LLM이 아니며 모든 텍스트 응답은 "[데모] "로 시작한다.
이력은 AGNO가 실제로 보내는 형식으로 구성한다: tool 결과(`tool` 메시지)의 content는 JSON이 아니라 **Python 표현 문자열**(str(dict)), assistant의 tool_calls 인자는 JSON 문자열.
새 모듈 import는 `_respond()` 안에서 하여 테스트마다 개별 실패로 확인한다.
"""

import json

import pytest

MODELS = ("PaiNN", "Geoformer", "Equiformer")
DEMO_PREFIX = "[데모] "


# --- 메시지/결과 구성 도우미 -------------------------------------------------------------


def _respond(messages):
    from agent.demo.mock_llm import respond

    return respond(messages)


def _system():
    return {"role": "system", "content": "당신은 인광 OLED 스펙트럼 학습/예측 도우미입니다."}


def _user(text):
    return {"role": "user", "content": text}


def _assistant_text(text):
    return {"role": "assistant", "content": text}


def _assistant_calls(*calls):
    """AGNO가 이력에 넣는 assistant(tool_calls) 메시지: arguments는 JSON 문자열."""
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"call_{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}
        for index, (name, arguments) in enumerate(calls)]}


def _tool_result(result):
    """AGNO가 이력에 넣는 tool 메시지: content는 str(dict) — JSON이 아닌 Python 표현 문자열."""
    return {"role": "tool", "tool_call_id": "call_0", "content": str(result)}


def _calls(payload):
    message = payload["choices"][0]["message"]
    return [(call["function"]["name"], json.loads(call["function"]["arguments"])) for call in message.get("tool_calls") or []]


def _text(payload):
    return payload["choices"][0]["message"]["content"]


def _defaults_result(model="PaiNN"):
    return {"status": "ok", "base_model": model, "spectrum_type": "FC", "data_path": "IrDB", "batch_size": 16, "train_steps": 10000,
            "eval_steps": 100, "workers": 0, "learning_rate": 0.001, "model_size": {"embed_dim": 512}, "seed": 0, "fold": 0,
            "device": "cpu", "changeable_parameters": ["train_steps", "batch_size"]}


def _preview_result(model="PaiNN", train_steps=10000, batch_size=16, will_overwrite=False):
    return {"status": "needs_confirmation",
            "settings": {"base_model": model, "spectrum_type": "FC", "data_path": "IrDB", "batch_size": batch_size,
                         "train_steps": train_steps, "eval_steps": 100, "workers": 0, "model_size": {"embed_dim": 512}},
            "output_dir": f"results_{model}/0/0", "will_overwrite": will_overwrite, "message": "확인이 필요합니다."}


def _pending_preview_history(model="PaiNN", settings=None, will_overwrite=False):
    """사용자가 학습을 요청했고 LLM이 미리보기를 보여 주며 진행 여부를 물은 상태의 이력 (마지막이 assistant 질문)."""
    settings = settings if settings is not None else {}
    train_steps = settings.get("train_steps", 10000)
    return [
        _system(), _user(f"{model}으로 학습해줘"),
        _assistant_calls(("show_training_defaults", {"base_model": model})), _tool_result(_defaults_result(model)),
        _assistant_calls(("preview_training", {"base_model": model, "settings": settings})),
        _tool_result(_preview_result(model, train_steps=train_steps, will_overwrite=will_overwrite)),
        _assistant_text(f"{DEMO_PREFIX}{model} 학습 설정을 확인했어요. 진행할까요?"),
    ]


# --- 응답 형식 -----------------------------------------------------------------------------


def test_응답은_OpenAI_chat_completion_형식이다():
    text_payload = _respond([_system(), _user("그냥 학습해줘")])
    call_payload = _respond([_system(), _user("PaiNN으로 학습해줘")])

    assert text_payload["object"] == "chat.completion"
    text_choice = text_payload["choices"][0]
    assert text_choice["finish_reason"] == "stop"
    assert text_choice["message"]["role"] == "assistant" and isinstance(text_choice["message"]["content"], str)
    call_choice = call_payload["choices"][0]
    assert call_choice["finish_reason"] == "tool_calls"
    tool_call = call_choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function" and isinstance(tool_call["id"], str) and tool_call["id"]
    assert tool_call["function"]["name"] == "show_training_defaults"
    assert json.loads(tool_call["function"]["arguments"]) == {"base_model": "PaiNN"}  # arguments는 JSON 문자열


# --- 학습 흐름 -----------------------------------------------------------------------------


def test_모델이_정해지지_않은_학습_요청은_모델을_되묻는다():
    payload = _respond([_system(), _user("그냥 학습해줘")])

    assert _calls(payload) == []
    assert _text(payload).startswith(DEMO_PREFIX)
    assert all(model in _text(payload) for model in MODELS)


def test_모델이_정해진_학습_요청은_기본값_조회_후_미리보기를_호출하고_확인을_묻는다():
    history = [_system(), _user("PaiNN으로 학습해줘")]

    first = _respond(history)
    history += [_assistant_calls(("show_training_defaults", {"base_model": "PaiNN"})), _tool_result(_defaults_result())]
    second = _respond(history)
    history += [_assistant_calls(("preview_training", {"base_model": "PaiNN", "settings": {}})), _tool_result(_preview_result())]
    third = _respond(history)

    assert _calls(first) == [("show_training_defaults", {"base_model": "PaiNN"})]
    (name, arguments), = _calls(second)
    assert name == "preview_training" and arguments["base_model"] == "PaiNN" and arguments.get("settings", {}) == {}
    assert _calls(third) == []
    assert _text(third).startswith(DEMO_PREFIX)
    assert "진행할까요" in _text(third) and "PaiNN" in _text(third) and "10000" in _text(third) and "16" in _text(third)
    assert "삭제" not in _text(third)  # 덮어쓸 결과가 없으면 삭제 경고가 없다


def test_step_수와_배치_크기를_말하면_미리보기_설정에_반영한다():
    history = [_system(), _user("PaiNN으로 20 step, 배치 8로 학습해줘"),
               _assistant_calls(("show_training_defaults", {"base_model": "PaiNN"})), _tool_result(_defaults_result())]

    (name, arguments), = _calls(_respond(history))
    history += [_assistant_calls((name, arguments)), _tool_result(_preview_result(train_steps=20, batch_size=8))]
    confirmation = _text(_respond(history))

    assert name == "preview_training"
    assert arguments["settings"] == {"train_steps": 20, "batch_size": 8}
    assert "20" in confirmation and "8" in confirmation


def test_모델만_답하면_직전_모델_질문의_학습_요청으로_이어간다():
    question = _text(_respond([_system(), _user("그냥 학습해줘")]))

    payload = _respond([_system(), _user("그냥 학습해줘"), _assistant_text(question), _user("Geoformer")])

    assert _calls(payload) == [("show_training_defaults", {"base_model": "Geoformer"})]


def test_사용자가_동의하면_직전_미리보기와_같은_인자로_학습을_시작한다():
    settings = {"train_steps": 20}
    history = _pending_preview_history(settings=settings) + [_user("응")]

    (name, arguments), = _calls(_respond(history))
    history += [_assistant_calls((name, arguments)), _tool_result({
        "status": "started", "job_id": "painn-abc123", "base_model": "PaiNN", "output_dir": "results_PaiNN/0/0",
        "settings": {"train_steps": 20}, "message": "학습을 시작했습니다. check_training_status로 진행 상황을 확인할 수 있습니다."})]
    confirmation = _text(_respond(history))

    assert name == "start_training_confirmed"
    assert arguments["base_model"] == "PaiNN" and arguments["settings"] == settings  # 미리보기와 같은 인자
    assert arguments.get("overwrite", False) is False
    assert "시작" in confirmation and "painn-abc123" in confirmation and confirmation.startswith(DEMO_PREFIX)


def test_덮어쓰기_미리보기에는_삭제를_경고하고_동의하면_overwrite로_시작한다():
    history = _pending_preview_history(will_overwrite=True)

    warning = _respond(history[:-1])  # 미리보기 결과 직후 응답 (질문 문구 생성)
    (name, arguments), = _calls(_respond(history + [_user("응")]))

    assert _calls(warning) == [] and "삭제" in _text(warning) and "진행할까요" in _text(warning)
    assert name == "start_training_confirmed" and arguments["overwrite"] is True


def test_거부하면_학습을_시작하지_않고_취소를_알린다():
    payload = _respond(_pending_preview_history() + [_user("아니요")])

    assert _calls(payload) == []
    assert "취소" in _text(payload) and _text(payload).startswith(DEMO_PREFIX)


def test_대기_중인_미리보기가_없으면_동의_표현에도_학습을_시작하지_않는다():
    no_history = _respond([_system(), _user("응")])
    already_started = _pending_preview_history() + [
        _user("응"), _assistant_calls(("start_training_confirmed", {"base_model": "PaiNN", "settings": {}})),
        _tool_result({"status": "started", "job_id": "j1", "base_model": "PaiNN", "output_dir": "results_PaiNN/0/0", "settings": {}, "message": "시작"}),
        _assistant_text(f"{DEMO_PREFIX}학습을 시작했어요"), _user("응")]  # 미리보기는 이미 소모됨
    after_started = _respond(already_started)

    assert _calls(no_history) == [] and _text(no_history).startswith(DEMO_PREFIX)
    assert _calls(after_started) == []


# --- 상태 / 예측 / 조회 ----------------------------------------------------------------------


@pytest.mark.parametrize("result, expected_phrases", [
    ({"status": "ok", "job_id": "j1", "base_model": "PaiNN", "state": "running", "return_code": None, "elapsed_seconds": 12,
      "log_tail": ["[step 3] train loss=1.0"], "has_checkpoint": False}, ["진행 중", "12"]),
    ({"status": "ok", "job_id": "j1", "base_model": "PaiNN", "state": "finished", "return_code": 0, "elapsed_seconds": 9,
      "log_tail": ["[step 5] test MAE=1.0"], "has_checkpoint": True}, ["끝났", "예측"]),
    ({"status": "ok", "job_id": "j1", "base_model": "PaiNN", "state": "finished", "return_code": 0, "elapsed_seconds": 9,
      "log_tail": [], "has_checkpoint": False}, ["끝났", "체크포인트"]),
    ({"status": "ok", "job_id": "j1", "base_model": "PaiNN", "state": "failed", "return_code": 3, "elapsed_seconds": 2,
      "log_tail": ["boom: 학습 실패"], "has_checkpoint": False}, ["실패", "boom: 학습 실패"]),
    ({"status": "no_job", "message": "시작된 학습 작업이 없습니다."}, ["시작된 학습 작업이 없습니다"]),
])
def test_학습_상태_질문은_상태를_조회하고_running_finished_failed를_요약한다(result, expected_phrases):
    asked = [_system(), _user("끝났어?")]

    first = _respond(asked)
    summary = _text(_respond(asked + [_assistant_calls(("check_training_status", {})), _tool_result(result)]))

    assert _calls(first) == [("check_training_status", {})]
    assert summary.startswith(DEMO_PREFIX)
    for phrase in expected_phrases:
        assert phrase in summary, f"{phrase!r}가 요약에 있어야 한다: {summary}"


def test_예측_요청은_분자_ID로_예측하고_결과_또는_학습_안내를_전달한다():
    asked = [_system(), _user("cn1_cn1_nn1 스펙트럼 예측해줘")]
    called = asked + [_assistant_calls(("predict_molecule_spectrum", {"molecule_id": "cn1_cn1_nn1"}))]
    samples = [{"wavelength_nm": wavelength, "intensity": 0.5} for wavelength in range(400, 800, 50)]

    first = _respond(asked)
    ok = _text(_respond(called + [_tool_result({"status": "ok", "molecule_id": "cn1_cn1_nn1", "base_model": "PaiNN", "checkpoint": "checkpoint_best.ckpt",
                                                "spectrum_type": "FC", "peak_wavelength_nm": 452.0, "samples": samples})]))
    needs_training = _text(_respond(called + [_tool_result({"status": "needs_training", "requested_base_model": None, "message": "학습된 모델이 없습니다."})]))
    unknown = _text(_respond(called + [_tool_result({"status": "error", "error": "unknown_molecule", "molecule_id": "cn1_cn1_nn1",
                                                      "message": "데이터셋에 없는 분자 ID입니다: cn1_cn1_nn1"})]))

    assert _calls(first) == [("predict_molecule_spectrum", {"molecule_id": "cn1_cn1_nn1"})]
    assert "452.0" in ok and "PaiNN" in ok and "cn1_cn1_nn1" in ok  # 결과의 피크 파장을 그대로 전달
    assert "먼저" in needs_training and "학습" in needs_training
    assert "데이터셋에 없는 분자 ID입니다" in unknown


def test_분자_ID가_없는_예측_요청은_분자_목록을_보여준다():
    asked = [_system(), _user("예측해줘")]

    first = _respond(asked)
    (name, arguments), = _calls(first)
    listed = _text(_respond(asked + [_assistant_calls((name, arguments)), _tool_result(
        {"status": "ok", "total_matches": 1024, "molecule_ids": ["cn1_cn1_nn1", "cn1_cn1_nn16"]})]))

    assert name == "list_molecule_ids" and arguments.get("limit") == 5
    assert "cn1_cn1_nn1" in listed and "cn1_cn1_nn16" in listed and "예측" in listed


@pytest.mark.parametrize("user_text, tool_name", [
    ("학습된 모델 있어?", "list_trained_models"),
    ("분자 목록 보여줘", "list_molecule_ids"),
    ("어떤 분자가 있어?", "list_molecule_ids"),
    ("PaiNN 학습 기본값 보여줘", "show_training_defaults"),
])
def test_목록_기본값_학습된_모델_질문은_해당_tool을_호출한다(user_text, tool_name):
    (name, arguments), = _calls(_respond([_system(), _user(user_text)]))

    assert name == tool_name
    if tool_name == "show_training_defaults":
        assert arguments == {"base_model": "PaiNN"}


def test_기본값_조회_질문은_미리보기로_이어가지_않고_기본값만_알려준다():
    asked = [_system(), _user("PaiNN 학습 기본값 보여줘"), _assistant_calls(("show_training_defaults", {"base_model": "PaiNN"})),
             _tool_result(_defaults_result())]

    payload = _respond(asked)

    assert _calls(payload) == []  # 학습 요청이 아니므로 preview_training을 부르지 않는다
    assert "10000" in _text(payload) and "16" in _text(payload) and _text(payload).startswith(DEMO_PREFIX)


# --- 도움 / 모르는 말 / 오류 전달 --------------------------------------------------------------


def test_도움_요청은_가능한_일을_안내한다():
    payload = _respond([_system(), _user("뭘 할 수 있어?")])

    assert _calls(payload) == []
    assert "학습" in _text(payload) and "예측" in _text(payload) and _text(payload).startswith(DEMO_PREFIX)


def test_규칙에_없는_말은_이해하지_못했다고_답한다():
    payload = _respond([_system(), _user("블라블라 아무말")])

    assert _calls(payload) == []
    assert "이해하지 못" in _text(payload) and "규칙 기반" in _text(payload) and _text(payload).startswith(DEMO_PREFIX)


@pytest.mark.parametrize("result", [
    {"status": "busy", "job_id": "j1", "message": "이미 학습이 진행 중입니다: j1 (PaiNN)."},
    {"status": "already_trained", "checkpoint": "results_PaiNN/0/0/checkpoint_best.ckpt", "message": "PaiNN의 학습된 체크포인트가 이미 있습니다. 다시 학습하려면 overwrite=True가 필요합니다."},
    {"status": "error", "error": "not_previewed", "message": "같은 설정으로 preview_training을 먼저 호출해야 합니다."},
])
def test_tool_오류_응답의_message를_사용자에게_전달한다(result):
    history = _pending_preview_history() + [
        _user("응"), _assistant_calls(("start_training_confirmed", {"base_model": "PaiNN", "settings": {}})), _tool_result(result)]

    payload = _respond(history)

    assert _calls(payload) == []
    assert result["message"] in _text(payload) and _text(payload).startswith(DEMO_PREFIX)
