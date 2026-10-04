"""Plan.md Step 5B [신규 인터페이스], G1~G3: Agent용 tool 래퍼 `agent.assistant_tools.build_assistant_tools`.

agent.assistant_tools가 아직 없으므로 실패해야 한다 (RED).
  - 학습: 4B 방식으로 학습 명령을 `[sys.executable, "-c", ...]`로 대체하되 실제 서브프로세스를 쓴다.
  - 예측: tiny PaiNN 체크포인트(임시 results_root)와 실제 IrDB를 쓴다.
  - LLM: 마지막 테스트는 실제 AGNO Agent + VLLM + 가짜 OpenAI 호환 서버(API 키 불필요)로 tool 호출을 확인한다.
새 모듈 import는 `_tools()` 안에서 하여 테스트마다 개별 실패로 확인한다.
"""

import inspect
import json
import time

from tests.agent_tools.test_predict_tool import _save_painn_ckpt
from tests.support.fake_training import use_fake_training

EXPECTED_TOOLS = {
    "list_trained_models", "show_training_defaults", "preview_training", "start_training_confirmed",
    "check_training_status", "list_molecule_ids", "predict_molecule_spectrum",
}


def _tools(project_root=None, results_root=None):
    from agent.assistant_tools import build_assistant_tools

    kwargs = {}
    if project_root is not None:
        kwargs["project_root"] = project_root
    if results_root is not None:
        kwargs["results_root"] = results_root
    return {tool.__name__: tool for tool in build_assistant_tools(**kwargs)}


def _wait_until_done(tools, job_id=None, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = tools["check_training_status"](job_id)
        if status["state"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError(f"작업이 {timeout}초 안에 끝나지 않았다")


def _marker_code(marker):
    return f"open({str(marker)!r}, 'w').write('x')"


# --- tool 목록 / 스키마 -----------------------------------------------------------------


def test_도구_목록은_계획한_7개이고_AGNO_스키마로_변환된다(tmp_path):
    from agno.tools.function import Function

    tools = _tools(tmp_path)

    assert set(tools) == EXPECTED_TOOLS
    for name, tool in tools.items():
        assert (tool.__doc__ or "").strip(), f"{name}: LLM이 읽을 docstring이 필요하다"
        function = Function.from_callable(tool)
        assert function.name == name
        assert set(function.parameters["properties"]) == set(inspect.signature(tool).parameters)
    for name in ("preview_training", "start_training_confirmed"):
        settings_schema = Function.from_callable(tools[name]).parameters["properties"]["settings"]
        # Dict[str, Any]는 값이 object로 변환되어 실제 LLM을 혼동시키므로, 값은 정수 또는 문자열로 선언한다.
        assert settings_schema["additionalProperties"] == {"anyOf": [{"type": "integer"}, {"type": "string"}]}


# --- 조회 --------------------------------------------------------------------------------


def test_학습된_모델_목록은_모델별_체크포인트_유무를_알려준다(tmp_path):
    tools = _tools(tmp_path)

    empty = tools["list_trained_models"]()
    _save_painn_ckpt(tmp_path)
    trained = tools["list_trained_models"]()

    assert empty["status"] == "ok" and empty["any_trained"] is False
    assert empty["models"] == {name: {"trained": False, "checkpoint": None} for name in ("PaiNN", "Equiformer", "Geoformer")}
    assert trained["any_trained"] is True
    assert trained["models"]["PaiNN"] == {"trained": True, "checkpoint": "checkpoint_best.ckpt"}
    assert trained["models"]["Geoformer"]["trained"] is False


def test_학습_기본값에는_바꿀_수_있는_파라미터_목록이_포함된다(tmp_path):
    tools = _tools(tmp_path)

    painn = tools["show_training_defaults"]("PaiNN")
    equiformer = tools["show_training_defaults"]("Equiformer")
    unsupported = tools["show_training_defaults"]("GPT")

    assert painn["status"] == "ok" and painn["train_steps"] == 10000
    assert {"spectrum_type", "data_path", "batch_size", "train_steps", "eval_steps", "workers",
            "embed_dim", "num_layers", "num_basis"} <= set(painn["changeable_parameters"])
    assert "lr" not in painn["changeable_parameters"] and "seed" not in painn["changeable_parameters"]
    assert "num_basis" in equiformer["changeable_parameters"]
    assert "embed_dim" not in equiformer["changeable_parameters"]  # Equiformer에는 없는 파라미터
    assert (unsupported["status"], unsupported["error"]) == ("error", "unsupported_base_model")


def test_분자_ID_목록은_그대로_전달한다():
    tools = _tools()

    listing = tools["list_molecule_ids"](limit=3)
    searched = tools["list_molecule_ids"](query="nn1", limit=5)

    assert listing["status"] == "ok" and len(listing["molecule_ids"]) == 3 and listing["total_matches"] == 1024
    assert searched["status"] == "ok" and all("nn1" in molecule_id for molecule_id in searched["molecule_ids"])


# --- 미리보기 / 확인 실행 -----------------------------------------------------------------


def test_미리보기는_실행하지_않고_최종_설정을_보여_준다(tmp_path, monkeypatch):
    marker = tmp_path / "ran.txt"
    use_fake_training(monkeypatch, _marker_code(marker))
    tools = _tools(tmp_path)

    preview = tools["preview_training"]("PaiNN", {"train_steps": 5})
    time.sleep(1.0)  # 프로세스가 (잘못) 시작되었다면 흔적을 남길 시간

    assert preview["status"] == "needs_confirmation"
    assert preview["settings"]["train_steps"] == 5
    assert preview["will_overwrite"] is False
    assert not marker.exists()


def test_미리보기는_잘못된_설정을_검증_오류로_돌려준다(tmp_path):
    tools = _tools(tmp_path)

    unknown = tools["preview_training"]("PaiNN", {"lr": 0.1})
    invalid = tools["preview_training"]("PaiNN", {"train_steps": 0})

    assert (unknown["status"], unknown["error"]) == ("error", "unknown_parameter")
    assert (invalid["status"], invalid["error"]) == ("error", "invalid_value")


def test_미리보기_없이_확인_실행하면_not_previewed를_반환하고_실행하지_않는다(tmp_path, monkeypatch):
    marker = tmp_path / "ran.txt"
    use_fake_training(monkeypatch, _marker_code(marker))
    tools = _tools(tmp_path)

    result = tools["start_training_confirmed"]("PaiNN")
    time.sleep(1.0)

    assert (result["status"], result["error"]) == ("error", "not_previewed")
    assert isinstance(result["message"], str) and result["message"]
    assert not marker.exists()


def test_미리보기와_다른_모델이나_설정으로_확인_실행하면_not_previewed를_반환한다(tmp_path, monkeypatch):
    marker = tmp_path / "ran.txt"
    use_fake_training(monkeypatch, _marker_code(marker))
    tools = _tools(tmp_path)
    tools["preview_training"]("PaiNN", {"train_steps": 5})

    other_settings = tools["start_training_confirmed"]("PaiNN", {"train_steps": 6})
    other_model = tools["start_training_confirmed"]("Geoformer", {"train_steps": 5})
    time.sleep(1.0)

    assert other_settings["error"] == "not_previewed"
    assert other_model["error"] == "not_previewed"
    assert not marker.exists()


def test_미리보기와_같은_최종_설정이면_표현이_달라도_실행된다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "print('학습')")
    tools = _tools(tmp_path)
    default_eval_steps = tools["show_training_defaults"]("PaiNN")["eval_steps"]
    tools["preview_training"]("PaiNN", {"train_steps": 5})

    # eval_steps를 기본값으로 명시해도 최종 설정은 같다
    started = tools["start_training_confirmed"]("PaiNN", {"train_steps": 5, "eval_steps": default_eval_steps})
    _wait_until_done(tools, started["job_id"])

    assert started["status"] == "started"


def test_미리보기_후_확인_실행하면_요약된_started를_반환하고_미리보기를_소모한다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "print('학습 로그')")
    tools = _tools(tmp_path)
    tools["preview_training"]("PaiNN", {"train_steps": 5})

    started = tools["start_training_confirmed"]("PaiNN", {"train_steps": 5})
    status = _wait_until_done(tools, started["job_id"])
    again = tools["start_training_confirmed"]("PaiNN", {"train_steps": 5})

    assert started["status"] == "started"
    assert started["base_model"] == "PaiNN" and started["output_dir"] == "results_PaiNN/0/0"
    assert started["settings"]["train_steps"] == 5 and started["job_id"]
    assert "command" not in started and "log_path" not in started  # LLM에 불필요한 정보는 제외
    assert status["state"] == "finished" and status["return_code"] == 0
    assert isinstance(status["elapsed_seconds"], int)
    assert status["has_checkpoint"] is False
    assert again["error"] == "not_previewed"  # 시작에 성공하면 미리보기가 소모된다


def test_이미_학습된_모델은_overwrite_확인_후에만_다시_학습한다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "import os; print('old_exists=' + str(os.path.exists('results_PaiNN/0/0/old.txt')))")
    old_output = tmp_path / "results_PaiNN" / "0" / "0"
    old_output.mkdir(parents=True)
    (old_output / "checkpoint_best.ckpt").write_bytes(b"")
    (old_output / "old.txt").write_text("이전 학습 산출물", encoding="utf-8")
    tools = _tools(tmp_path)

    preview = tools["preview_training"]("PaiNN")
    refused = tools["start_training_confirmed"]("PaiNN")
    kept_old_output = (old_output / "old.txt").exists()
    started = tools["start_training_confirmed"]("PaiNN", overwrite=True)  # 미리보기가 유지되어 같은 설정으로 다시 호출 가능
    status = _wait_until_done(tools, started["job_id"])

    assert preview["will_overwrite"] is True
    assert refused["status"] == "already_trained" and kept_old_output
    assert started["status"] == "started"
    assert "old_exists=False" in status["log_tail"]


def test_실행_중에_다시_시작하면_busy를_반환하고_미리보기를_유지한다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "import time; time.sleep(3)")
    first_tools, second_tools = _tools(tmp_path), _tools(tmp_path)
    second_tools["preview_training"]("Geoformer")  # 두 번째 도구 인스턴스가 먼저 미리보기를 해 둔다

    first_tools["preview_training"]("PaiNN")
    first = first_tools["start_training_confirmed"]("PaiNN")
    during_preview = first_tools["preview_training"]("PaiNN")
    busy = second_tools["start_training_confirmed"]("Geoformer")
    _wait_until_done(first_tools, first["job_id"])
    retried = second_tools["start_training_confirmed"]("Geoformer")  # busy였어도 미리보기는 남아 있다
    _wait_until_done(second_tools, retried["job_id"])

    assert first["status"] == "started"
    assert during_preview["status"] == "busy"
    assert busy["status"] == "busy" and busy["job_id"] == first["job_id"]
    assert retried["status"] == "started"


def test_도구_상태는_build_assistant_tools_호출마다_독립적이다(tmp_path, monkeypatch):
    marker = tmp_path / "ran.txt"
    use_fake_training(monkeypatch, _marker_code(marker))
    tools_a, tools_b = _tools(tmp_path), _tools(tmp_path)

    tools_a["preview_training"]("PaiNN")
    result = tools_b["start_training_confirmed"]("PaiNN")
    time.sleep(1.0)

    assert result["error"] == "not_previewed"
    assert not marker.exists()


# --- 학습 상태 ---------------------------------------------------------------------------


def test_학습_상태는_로그를_5줄_200자로_요약한다(tmp_path, monkeypatch):
    tools = _tools(tmp_path)
    no_job = tools["check_training_status"]()
    unknown = tools["check_training_status"]("없는-작업")

    use_fake_training(monkeypatch, "for i in range(50): print(str(i) + 'x' * 500)")
    tools["preview_training"]("PaiNN")
    long_job = tools["start_training_confirmed"]("PaiNN")
    summary = _wait_until_done(tools, long_job["job_id"])

    use_fake_training(monkeypatch, "import sys; print('boom: 학습 실패'); sys.exit(3)")
    tools["preview_training"]("PaiNN")
    failing_job = tools["start_training_confirmed"]("PaiNN")
    failed = _wait_until_done(tools, failing_job["job_id"])

    assert no_job["status"] == "no_job"
    assert (unknown["status"], unknown["error"]) == ("error", "unknown_job")
    assert len(summary["log_tail"]) == 5 and all(len(line) <= 200 for line in summary["log_tail"])
    assert summary["log_tail"][-1].startswith("49")
    assert failed["state"] == "failed" and failed["return_code"] == 3
    assert "boom: 학습 실패" in failed["log_tail"]


# --- 예측 --------------------------------------------------------------------------------


def test_예측_결과는_요약만_돌려주고_전체_곡선과_일치한다(tmp_path):
    from agent.tools.predict_tool import predict_spectrum

    _save_painn_ckpt(tmp_path)
    tools = _tools(results_root=tmp_path)  # 체크포인트는 임시 results_root에, 데이터셋은 실제 저장소 루트에서 연다

    result = tools["predict_molecule_spectrum"]("cn1_cn1_nn1")
    full = predict_spectrum("cn1_cn1_nn1", results_root=tmp_path)

    assert result["status"] == "ok"
    assert (result["molecule_id"], result["base_model"], result["spectrum_type"]) == ("cn1_cn1_nn1", "PaiNN", "FC")
    assert result["checkpoint"] == "checkpoint_best.ckpt"
    assert result["peak_wavelength_nm"] == full["peak_wavelength_nm"]
    samples = result["samples"]
    assert [sample["wavelength_nm"] for sample in samples] == list(range(400, 800, 50))
    for sample in samples:
        full_curve_value = full["intensity"][int((sample["wavelength_nm"] - 400) * 2)]  # 0.5nm 간격 격자
        assert sample["intensity"] == round(full_curve_value, 3)
    assert "intensity" not in result and "wavelength_nm" not in result  # 전체 곡선은 포함하지 않는다
    assert len(json.dumps(result)) < 2000  # 전체 결과(약 20KB) 대신 요약


def test_학습된_모델이_없으면_예측은_needs_training을_전달한다(tmp_path):
    tools = _tools(results_root=tmp_path)

    result = tools["predict_molecule_spectrum"]("cn1_cn1_nn1")

    assert result["status"] == "needs_training"
    assert isinstance(result["message"], str) and result["message"]


# --- AGNO Agent + 가짜 LLM 서버 ------------------------------------------------------------


def test_AGNO_Agent가_가짜_LLM_서버를_통해_미리보기와_확인_실행을_호출할_수_있다(tmp_path, monkeypatch, fake_llm):
    from agno.agent import Agent
    from agno.models.vllm import VLLM

    marker = tmp_path / "ran.txt"
    use_fake_training(monkeypatch, _marker_code(marker) + "; print('학습 실행')")
    tools = _tools(tmp_path)
    settings = {"train_steps": 5}
    fake_llm.queue_tool_calls(("preview_training", {"base_model": "PaiNN", "settings": settings}))
    fake_llm.queue_tool_calls(("start_training_confirmed", {"base_model": "PaiNN", "settings": settings}))
    fake_llm.queue_text("학습을 시작했어요")
    agent = Agent(model=VLLM(id="fake", base_url=fake_llm.base_url, api_key="EMPTY"),
                  tools=list(tools.values()), telemetry=False)

    result = agent.run("PaiNN으로 학습해줘")
    status = _wait_until_done(tools)  # 가장 최근 작업

    assert result.content == "학습을 시작했어요"
    assert len(fake_llm.requests) == 3
    assert {tool["function"]["name"] for tool in fake_llm.requests[0]["tools"]} == EXPECTED_TOOLS
    first_tool_results = [m["content"] for m in fake_llm.requests[1]["messages"] if m["role"] == "tool"]
    second_tool_results = [m["content"] for m in fake_llm.requests[2]["messages"] if m["role"] == "tool"]
    assert any("needs_confirmation" in content for content in first_tool_results)  # dict 인자가 전달되어 미리보기가 실행됨
    assert any("started" in content for content in second_tool_results)
    assert status["state"] == "finished" and marker.exists()  # 학습 프로세스가 실제로 실행되었다
