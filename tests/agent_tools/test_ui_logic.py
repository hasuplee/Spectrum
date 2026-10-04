"""Plan.md Step 6A [신규 인터페이스], G4: UI 로직 `agent.ui_logic` (Gradio 비의존 핸들러).

agent.ui_logic이 아직 없으므로 실패해야 한다 (RED). 화면(6B)은 이 함수들을 이벤트에 연결할 뿐이다.
LLM은 가짜 OpenAI 호환 서버(`fake_llm`)/데모 서버(키 불필요), 학습은 4B 방식의 가짜 명령(실제 서브프로세스), 예측은 tiny PaiNN 체크포인트와 실제 IrDB를 쓴다.
새 모듈 import는 각 테스트(또는 도우미) 안에서 하여 개별 실패로 확인한다.
"""

import json
import re
import time
import urllib.request

import pytest

from tests.agent_tools.test_predict_tool import _save_painn_ckpt
from tests.support.fake_training import use_fake_training

QUICK_MODELS = ("PaiNN", "Equiformer", "Geoformer")


def _llm_status(fake_llm):
    from agent.ui_logic import LlmStatus

    return LlmStatus(mode="real", message="", base_url=fake_llm.base_url, model_id="fake")


def _wait_for_training(project_root, timeout=60):
    from agent.tools.train_tool import get_training_status

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = get_training_status(project_root=project_root)
        if status["status"] == "ok" and status["state"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError("학습 작업이 끝나지 않았다")


def _marker_code(marker):
    return f"open({str(marker)!r}, 'w').write('x')"


# --- LLM 상태 ------------------------------------------------------------------------------


def test_demo_모드는_데모_서버를_한_번만_띄우고_진짜_LLM이_아님을_알린다():
    from agent import ui_logic

    try:
        first = ui_logic.resolve_llm({}, demo=True)
        second = ui_logic.resolve_llm({}, demo=True)
        with urllib.request.urlopen(f"{first.base_url}/models", timeout=10) as response:  # 서버가 실제로 떠 있다
            models = json.loads(response.read().decode("utf-8"))

        assert first.mode == "demo" and first.model_id == "demo-rule-based"
        assert first.base_url == second.base_url  # 한 번만 띄운다
        assert "진짜 LLM" in first.message and "규칙 기반" in first.message
        assert models["data"][0]["id"] == "demo-rule-based"
    finally:
        ui_logic.shutdown_demo_server()


def test_실제_모드는_필수_환경변수가_모두_있어야_하고_없으면_누락된_이름과_데모_실행법을_안내한다():
    from agent.ui_logic import resolve_llm

    real = resolve_llm({"VLLM_BASE_URL": "http://internal:8000/v1", "VLLM_MODEL": "my-model"})
    nothing = resolve_llm({})
    only_url = resolve_llm({"VLLM_BASE_URL": "http://internal:8000/v1"})

    assert (real.mode, real.base_url, real.model_id) == ("real", "http://internal:8000/v1", "my-model")
    assert nothing.mode == "disconnected" and "VLLM_BASE_URL" in nothing.message and "VLLM_MODEL" in nothing.message
    assert only_url.mode == "disconnected" and "VLLM_MODEL" in only_url.message
    assert "--demo" in nothing.message  # 데모로 시험하는 방법
    assert nothing.base_url is None and nothing.model_id is None


# --- 채팅 ----------------------------------------------------------------------------------


def test_연결되지_않은_상태의_채팅은_LLM이_필요하다는_안내를_돌려주고_Agent를_만들지_않는다(tmp_path):
    from agent.ui_logic import chat_turn, resolve_llm

    status = resolve_llm({})

    history, agent = chat_turn("학습 상태 알려줘", [], None, status, project_root=tmp_path)

    assert agent is None
    assert history[0] == {"role": "user", "content": "학습 상태 알려줘"}
    reply = history[-1]
    assert reply["role"] == "assistant" and "LLM" in reply["content"]
    assert "학습" in reply["content"] and "예측" in reply["content"]  # 학습/예측 탭은 쓸 수 있다는 안내


def test_채팅은_첫_메시지에서_Agent를_만들고_세션_동안_재사용하며_대화를_누적한다(fake_llm, tmp_path):
    from agent.ui_logic import chat_turn

    status = _llm_status(fake_llm)
    fake_llm.queue_text("PaiNN, Geoformer, Equiformer 중 무엇으로 학습할까요?")
    fake_llm.queue_text("PaiNN으로 진행할게요")

    first_history, first_agent = chat_turn("그냥 학습해줘", [], None, status, session_id="s", project_root=tmp_path)
    second_history, second_agent = chat_turn("PaiNN", first_history, first_agent, status, session_id="s", project_root=tmp_path)

    assert first_agent is not None and second_agent is first_agent  # 세션 동안 같은 Agent
    assert [m["role"] for m in second_history] == ["user", "assistant", "user", "assistant"]
    assert second_history[1]["content"] == "PaiNN, Geoformer, Equiformer 중 무엇으로 학습할까요?"
    assert second_history[3]["content"] == "PaiNN으로 진행할게요"
    assert len(first_history) == 2  # 입력으로 받은 history를 변경하지 않고 새 목록을 돌려준다
    assert "그냥 학습해줘" in str(fake_llm.requests[1]["messages"])  # Agent의 대화 기억이 다음 요청에 실린다


def test_실제_모드의_API_키는_상태에_담겨_Agent에_전달되고_출력에는_드러나지_않는다(tmp_path):
    # REVIEW에서 추가: resolve_llm(environ)에 주입한 환경의 토큰이 new_agent에서 무시되던 불일치(os.environ을 따로 읽음)를 막는다.
    from agent.ui_logic import new_agent, resolve_llm

    base = {"VLLM_BASE_URL": "http://internal:8000/v1", "VLLM_MODEL": "my-model"}
    status = resolve_llm({**base, "VLLM_API_KEY": "secret-token"})

    with_key = new_agent(status, project_root=tmp_path)
    without_key = new_agent(resolve_llm(base), project_root=tmp_path)

    assert with_key.model.api_key == "secret-token"
    assert without_key.model.api_key == "EMPTY"  # 키가 없으면 인증 없는 서버용 더미
    assert "secret-token" not in repr(status) and "secret-token" not in status.message  # 토큰이 로그/출력에 드러나지 않는다


def test_빈_메시지는_무시한다(fake_llm, tmp_path):
    from agent.ui_logic import chat_turn

    history = [{"role": "user", "content": "이전"}, {"role": "assistant", "content": "답변"}]

    new_history, agent = chat_turn("   ", history, None, _llm_status(fake_llm), project_root=tmp_path)

    assert new_history == history and agent is None
    assert len(fake_llm.requests) == 0


def test_범위_밖_질문은_채팅에서도_거절_문구로_답한다(fake_llm, tmp_path):
    from agent.guard import REFUSAL_MESSAGE
    from agent.ui_logic import chat_turn

    fake_llm.queue_text("이 응답은 사용되면 안 된다")

    history, agent = chat_turn("오늘 날씨가 뭐야?", [], None, _llm_status(fake_llm), project_root=tmp_path)

    assert history[-1] == {"role": "assistant", "content": REFUSAL_MESSAGE}
    assert agent is not None and len(fake_llm.requests) == 0


# --- 학습: 설정 ----------------------------------------------------------------------------


@pytest.mark.parametrize("base_model", QUICK_MODELS)
def test_빠른_설정은_세_모델_모두_검증을_통과하고_기본값보다_작다(base_model):
    from agent.tools.train_tool import get_training_defaults, validate_training_request
    from agent.ui_logic import quick_settings

    quick = quick_settings(base_model)
    defaults = get_training_defaults(base_model)

    assert validate_training_request(base_model, quick)["status"] == "ok"
    assert quick["train_steps"] < defaults["train_steps"] and quick["batch_size"] <= defaults["batch_size"]
    sizes = {key: value for key, value in quick.items() if key in defaults["model_size"]}
    assert all(value <= defaults["model_size"][key] for key, value in sizes.items())  # 모델 크기도 줄인다


def test_사용자_JSON은_객체만_허용하고_잘못된_입력은_오류_문구를_돌려준다():
    from agent.ui_logic import parse_custom_settings

    assert parse_custom_settings("") == ({}, None)
    assert parse_custom_settings("   ") == ({}, None)
    assert parse_custom_settings('{"train_steps": 20, "spectrum_type": "FC"}') == ({"train_steps": 20, "spectrum_type": "FC"}, None)
    for bad_text, expected_phrase in (("이건 JSON이 아님", "JSON"), ("[1, 2]", "객체"), ('{"a": {"b": 1}}', "정수"), ('{"x": 1.5}', "정수")):
        settings, error = parse_custom_settings(bad_text)
        assert settings == {} and isinstance(error, str) and expected_phrase in error, (bad_text, error)


def test_학습_기본값_표시에는_기본값과_바꿀_수_있는_파라미터가_담긴다():
    from agent.ui_logic import training_defaults_markdown

    painn = training_defaults_markdown("PaiNN")
    geoformer = training_defaults_markdown("Geoformer")
    unsupported = training_defaults_markdown("GPT")

    assert "10000" in painn and "16" in painn and "cpu" in painn  # 기본 step 수, 배치 크기, 장치
    assert "train_steps" in painn and "embed_dim" in painn
    assert "embedding_dim" in geoformer
    assert "지원하지 않는" in unsupported


# --- 학습: 미리보기 / 시작 -------------------------------------------------------------------


def test_학습_미리보기는_실행하지_않고_최종_설정과_덮어쓰기_경고를_보여_준다(tmp_path, monkeypatch):
    from agent.ui_logic import quick_settings, training_preview

    marker = tmp_path / "ran.txt"
    use_fake_training(monkeypatch, _marker_code(marker))
    quick = quick_settings("PaiNN")

    merged = training_preview("PaiNN", True, '{"train_steps": 7}', project_root=tmp_path)  # 사용자 JSON이 빠른 설정을 덮는다
    plain = training_preview("PaiNN", False, "", project_root=tmp_path)
    old_output = tmp_path / "results_PaiNN" / "0" / "0"
    old_output.mkdir(parents=True)
    (old_output / "checkpoint_best.ckpt").write_bytes(b"")
    with_old_result = training_preview("PaiNN", False, "", project_root=tmp_path)
    bad_json = training_preview("PaiNN", False, "이건 JSON이 아님", project_root=tmp_path)
    bad_parameter = training_preview("PaiNN", False, '{"seed": 1}', project_root=tmp_path)  # 정수 값이라 파싱은 통과하고, tool이 변경 불가를 알린다
    time.sleep(1.0)  # 프로세스가 (잘못) 시작되었다면 흔적을 남길 시간

    assert re.search(r"train_steps\D*7\b", merged) and str(quick["batch_size"]) in merged
    assert re.search(r"train_steps\D*10000\b", plain)  # 빠른 설정을 끄면 기본값
    assert "삭제" not in plain and "삭제" in with_old_result  # 기존 학습 결과가 있으면 덮어쓰기 경고
    assert "JSON" in bad_json
    assert "seed" in bad_parameter and "변경할 수 없는" in bad_parameter
    assert not marker.exists()


def test_확인_체크_없이는_학습을_시작하지_않는다(tmp_path, monkeypatch):
    from agent.ui_logic import training_start

    marker = tmp_path / "ran.txt"
    use_fake_training(monkeypatch, _marker_code(marker))

    message = training_start("PaiNN", False, "", False, False, project_root=tmp_path)
    time.sleep(1.0)

    assert "확인" in message and "체크" in message
    assert not marker.exists()


def test_확인하면_학습이_시작되고_기존_결과가_있으면_덮어쓰기_체크가_필요하다(tmp_path, monkeypatch):
    from agent.ui_logic import training_start

    use_fake_training(monkeypatch, "import os; print('old_exists=' + str(os.path.exists('results_PaiNN/0/0/old.txt')))")
    started = training_start("PaiNN", False, "", True, False, project_root=tmp_path)
    _wait_for_training(tmp_path)

    old_output = tmp_path / "results_PaiNN" / "0" / "0"
    old_output.mkdir(parents=True, exist_ok=True)
    (old_output / "checkpoint_best.ckpt").write_bytes(b"")
    (old_output / "old.txt").write_text("이전 학습 산출물", encoding="utf-8")
    refused = training_start("PaiNN", False, "", True, False, project_root=tmp_path)
    kept_old_output = (old_output / "old.txt").exists()
    overwritten = training_start("PaiNN", False, "", True, True, project_root=tmp_path)
    status = _wait_for_training(tmp_path)

    assert "시작했습니다" in started
    assert "덮어쓰기" in refused and "시작했습니다" not in refused and kept_old_output  # 덮어쓰기 체크 없이는 기존 결과를 지우지 않는다
    assert "시작했습니다" in overwritten
    assert "old_exists=False" in status["log_tail"]


# --- 학습: 상태 ----------------------------------------------------------------------------


def test_학습_상태_보기는_상태와_진행_막대를_걸러낸_로그를_돌려준다(tmp_path, monkeypatch):
    from agent.ui_logic import training_start, training_status_view

    no_job_status, no_job_log, no_job_running = training_status_view(project_root=tmp_path)

    use_fake_training(monkeypatch, "import time; print('시작'); time.sleep(3)")
    training_start("PaiNN", False, "", True, False, project_root=tmp_path)
    running_status, _, running = training_status_view(project_root=tmp_path)
    _wait_for_training(tmp_path)

    use_fake_training(monkeypatch, (
        "for i in range(30):\n"
        "    print(f'n{i}')\n"
        "    print(f'Epoch {i}:  50%|#####     | 2/4 [00:01<00:01,  1.20it/s]')"))
    training_start("PaiNN", False, "", True, False, project_root=tmp_path)
    _wait_for_training(tmp_path)
    finished_status, log_text, finished_running = training_status_view(project_root=tmp_path)

    assert "작업 없음" in no_job_status and no_job_log == "" and no_job_running is False
    assert running is True and "진행 중" in running_status
    assert finished_running is False and "완료" in finished_status
    assert log_text.splitlines() == [f"n{i}" for i in range(10, 30)]  # 진행 막대 프레임은 걸러내고 최근 20줄만 (프레임 때문에 줄어들지 않는다)


def test_학습_로그에서_빈_줄과_ANSI_제어_문자와_진행_막대를_걸러낸다(tmp_path, monkeypatch):
    # REVIEW에서 추가: 실제 Geoformer(Lightning) 로그 117줄 중 의미 있는 줄은 36줄뿐이었다
    # (진행 막대 프레임 65줄, 빈 줄 15줄, 커서 이동 같은 ANSI 코드가 있는 줄 6줄).
    from agent.ui_logic import training_start, training_status_view

    use_fake_training(monkeypatch, (
        "print('시작'); print(); print('   \x1b[A'); print('\x1b[2K실제 로그 줄'); "
        "print('\x1b[A Epoch 0:  50%|#####     | 2/4 [00:01<00:01,  1.20it/s]'); print('마지막 줄')"))
    training_start("PaiNN", False, "", True, False, project_root=tmp_path)
    _wait_for_training(tmp_path)

    _, log_text, _ = training_status_view(project_root=tmp_path)

    assert log_text.splitlines() == ["시작", "실제 로그 줄", "마지막 줄"]  # 빈 줄, ANSI만 있는 줄, 진행 막대는 제거하고 ANSI는 지운 본문만 남는다


def test_실패한_학습은_실패로_표시한다(tmp_path, monkeypatch):
    from agent.ui_logic import training_start, training_status_view

    use_fake_training(monkeypatch, "import sys; print('boom: 학습 실패'); sys.exit(3)")
    training_start("PaiNN", False, "", True, False, project_root=tmp_path)
    _wait_for_training(tmp_path)

    status, log_text, running = training_status_view(project_root=tmp_path)

    assert "실패" in status and "3" in status and running is False
    assert "boom: 학습 실패" in log_text


# --- 예측 ----------------------------------------------------------------------------------


def test_분자_선택_목록은_검색어로_걸러진다():
    from agent.ui_logic import molecule_choices

    all_choices = molecule_choices()
    searched = molecule_choices("NN1", limit=5)

    assert len(all_choices) == 50 and all_choices[0] == "cn1_cn1_nn1"
    assert 1 <= len(searched) <= 5 and all("nn1" in molecule_id for molecule_id in searched)
    assert molecule_choices("no_such_molecule") == []


def test_예측_보기는_예측_곡선_표와_요약을_돌려준다(tmp_path):
    from agent.tools.predict_tool import predict_spectrum
    from agent.ui_logic import prediction_view

    _save_painn_ckpt(tmp_path)

    frame, summary = prediction_view("cn1_cn1_nn1", None, False, results_root=tmp_path)
    auto_frame, _ = prediction_view("cn1_cn1_nn1", "자동", False, results_root=tmp_path)
    explicit_frame, _ = prediction_view("cn1_cn1_nn1", "PaiNN", False, results_root=tmp_path)
    peak = predict_spectrum("cn1_cn1_nn1", results_root=tmp_path)["peak_wavelength_nm"]

    assert list(frame.columns) == ["wavelength_nm", "intensity", "종류"]
    assert len(frame) == 800 and set(frame["종류"]) == {"예측"}
    assert frame["wavelength_nm"].iloc[0] == 400.0 and frame["wavelength_nm"].iloc[-1] == 799.5
    assert f"{peak:.1f}" in summary and "PaiNN" in summary and "cn1_cn1_nn1" in summary
    assert len(auto_frame) == len(explicit_frame) == 800  # "자동"과 모델 이름 모두 받는다


def test_실험_스펙트럼을_함께_보면_파장으로_변환한_실험_곡선이_추가된다(tmp_path):
    from agent.ui_logic import prediction_view

    _save_painn_ckpt(tmp_path)

    frame, summary = prediction_view("cn1_cn1_nn1", None, True, results_root=tmp_path)

    predicted = frame[frame["종류"] == "예측"]
    experimental = frame[frame["종류"] == "실험"]
    assert len(predicted) == 800 and len(experimental) == 800
    wavelengths = experimental["wavelength_nm"].tolist()
    assert wavelengths == sorted(wavelengths)  # 파장 오름차순 (eV 오름차순 → nm 내림차순을 뒤집어 정렬)
    assert 399.0 <= wavelengths[0] <= 401.0 and 799.0 <= wavelengths[-1] <= 801.0  # 1240/eV: 3.100eV ≈ 400nm, 1.551eV ≈ 799.5nm
    assert 0.0 <= experimental["intensity"].min() and experimental["intensity"].max() <= 1.0
    assert "실험" in summary


def test_정상_곡선에는_경고가_없고_비정상_곡선에는_경고가_표시된다(tmp_path, monkeypatch):
    from agent import ui_logic

    wavelengths = [400 + 0.5 * i for i in range(800)]

    def fake_result(intensity):
        return {"status": "ok", "molecule_id": "cn1_cn1_nn1", "base_model": "PaiNN", "checkpoint": "checkpoint_best.ckpt",
                "spectrum_type": "FC", "wavelength_nm": wavelengths, "intensity": intensity,
                "peak_wavelength_nm": wavelengths[int(max(range(800), key=lambda i: intensity[i]))]}

    normal = [i / 799 for i in range(800)]                # 0~1
    huge = [1.0 + i for i in range(800)]                  # 모든 값 >= 1, 최댓값 800 (음수 최댓값으로 정규화된 비정상 곡선)
    negative = [(i / 799) - 0.5 for i in range(800)]      # 최솟값 -0.5

    summaries = {}
    for label, intensity in (("normal", normal), ("huge", huge), ("negative", negative)):
        monkeypatch.setattr(ui_logic, "predict_spectrum", lambda *args, _result=fake_result(intensity), **kwargs: _result)
        summaries[label] = ui_logic.prediction_view("cn1_cn1_nn1", None, False, results_root=tmp_path)[1]

    assert "비정상" not in summaries["normal"]
    assert "비정상" in summaries["huge"] and "학습" in summaries["huge"]
    assert "비정상" in summaries["negative"]


def test_학습된_모델이_없으면_학습_탭을_안내한다(tmp_path):
    from agent.ui_logic import prediction_view

    frame, summary = prediction_view("cn1_cn1_nn1", None, False, results_root=tmp_path)

    assert len(frame) == 0 and list(frame.columns) == ["wavelength_nm", "intensity", "종류"]
    assert "학습 탭" in summary


def test_알_수_없는_분자는_오류_메시지를_돌려준다(tmp_path):
    from agent.ui_logic import prediction_view

    _save_painn_ckpt(tmp_path)

    frame, summary = prediction_view("no_such_molecule", None, False, results_root=tmp_path)

    assert len(frame) == 0 and "데이터셋에 없는" in summary
