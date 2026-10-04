"""Plan.md Step 4B [신규 인터페이스], G1: 학습 tool의 확인 절차 / 백그라운드 실행 / 상태 조회.

start_training/get_training_status가 아직 없으므로 실패해야 한다 (RED).
실제 학습 명령은 오래 걸리므로, 대부분의 테스트는 build_training_command를 `[sys.executable, "-c", 코드]`로
대체하되 **실제 서브프로세스**(Popen, 로그 파일, 종료 코드)를 사용한다. 마지막 slow 테스트만 실제 tiny 학습을 돌린다.
작업 목록(`train_tool._jobs`)은 테스트마다 초기화한다 (모듈 전역 상태 격리).
새 함수 import는 각 테스트 안에서 하여 개별 실패로 확인한다.
"""

import shutil
import sys
import time
from pathlib import Path

import pytest
import torch

from tests.support.fake_training import use_fake_training

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TINY_PAINN = {"train_steps": 5, "eval_steps": 5, "batch_size": 4, "embed_dim": 8, "num_layers": 1, "num_basis": 8}


def _wait_until_done(job_id, project_root, timeout=60):
    from agent.tools.train_tool import get_training_status

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = get_training_status(job_id, project_root=project_root)
        if status["state"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError(f"작업이 {timeout}초 안에 끝나지 않았다: {job_id}")


# --- 확인 절차 / 검증 -----------------------------------------------------------------


def test_확인하지_않으면_실행하지_않고_needs_confirmation을_반환한다(tmp_path, monkeypatch):
    marker = tmp_path / "ran.txt"
    use_fake_training(monkeypatch, f"open({str(marker)!r}, 'w').write('x')")
    from agent.tools.train_tool import start_training

    result = start_training("PaiNN", {"train_steps": 5}, project_root=tmp_path)
    time.sleep(1.0)  # 프로세스가 (잘못) 시작되었다면 흔적을 남길 시간

    assert result["status"] == "needs_confirmation"
    assert result["settings"]["base_model"] == "PaiNN"
    assert result["settings"]["train_steps"] == 5
    assert result["output_dir"] == "results_PaiNN/0/0"
    assert result["will_overwrite"] is False
    assert not marker.exists()


def test_잘못된_요청은_검증_오류를_그대로_반환하고_실행하지_않는다(tmp_path, monkeypatch):
    marker = tmp_path / "ran.txt"
    use_fake_training(monkeypatch, f"open({str(marker)!r}, 'w').write('x')")
    from agent.tools.train_tool import start_training

    unknown_parameter = start_training("PaiNN", {"lr": 0.1}, confirmed=True, project_root=tmp_path)
    unsupported_model = start_training("GPT", confirmed=True, project_root=tmp_path)
    time.sleep(1.0)

    assert (unknown_parameter["status"], unknown_parameter["error"]) == ("error", "unknown_parameter")
    assert (unsupported_model["status"], unsupported_model["error"]) == ("error", "unsupported_base_model")
    assert not marker.exists()


# --- 시작 / 상태 -----------------------------------------------------------------------


def test_확인하면_백그라운드로_시작하고_작업_디렉터리와_로그를_남긴다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "import os; print('cwd=' + os.getcwd(), flush=True)")
    from agent.tools.train_tool import start_training

    started = start_training("PaiNN", confirmed=True, project_root=tmp_path)
    status = _wait_until_done(started["job_id"], tmp_path)

    assert started["status"] == "started"
    assert started["base_model"] == "PaiNN"
    assert started["output_dir"] == "results_PaiNN/0/0"
    assert started["command"][:2] == [sys.executable, "-c"]
    assert started["settings"]["base_model"] == "PaiNN"
    log_path = Path(started["log_path"])
    assert log_path.exists() and log_path.parent == tmp_path / "results_agent_logs"
    logged_cwd = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.startswith("cwd=")][0]
    assert Path(logged_cwd[len("cwd="):]).resolve() == tmp_path.resolve()  # 프로세스는 project_root에서 실행된다
    assert status["state"] == "finished"


def test_작업이_끝나면_finished_상태와_로그_tail을_알려준다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "print('시작'); print('학습 완료 마커')")
    from agent.tools.train_tool import start_training

    started = start_training("Equiformer", confirmed=True, project_root=tmp_path)
    status = _wait_until_done(started["job_id"], tmp_path)

    assert status["status"] == "ok"
    assert status["job_id"] == started["job_id"]
    assert status["base_model"] == "Equiformer"
    assert status["state"] == "finished"
    assert status["return_code"] == 0
    assert status["elapsed_seconds"] >= 0
    assert status["log_tail"][-1] == "학습 완료 마커"
    assert status["output_dir"] == "results_Equiformer/0/0"
    assert status["settings"]["base_model"] == "Equiformer"
    assert status["has_checkpoint"] is False  # 가짜 학습은 체크포인트를 만들지 않았다


def test_작업이_실패하면_failed_상태와_return_code와_로그를_알려준다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "import sys; print('boom: 학습 실패', file=sys.stderr); sys.exit(3)")
    from agent.tools.train_tool import start_training

    started = start_training("PaiNN", confirmed=True, project_root=tmp_path)
    status = _wait_until_done(started["job_id"], tmp_path)

    assert status["state"] == "failed"
    assert status["return_code"] == 3
    assert any("boom" in line for line in status["log_tail"])  # stderr도 같은 로그에 남는다


def test_실행_중에_다시_시작하면_busy를_반환한다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "import time; time.sleep(3)")
    from agent.tools.train_tool import get_training_status, start_training

    first = start_training("PaiNN", confirmed=True, project_root=tmp_path)
    second = start_training("Geoformer", confirmed=True, project_root=tmp_path)
    running = get_training_status(first["job_id"], project_root=tmp_path)
    _wait_until_done(first["job_id"], tmp_path)  # 임시 디렉터리를 정리할 수 있도록 끝까지 기다린다

    assert first["status"] == "started"
    assert running["state"] == "running" and running["return_code"] is None
    assert second["status"] == "busy"
    assert second["job_id"] == first["job_id"]


def test_log_tail은_tail_lines만큼만_반환한다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "for i in range(50): print('line', i)")
    from agent.tools.train_tool import get_training_status, start_training

    started = start_training("PaiNN", confirmed=True, project_root=tmp_path)
    _wait_until_done(started["job_id"], tmp_path)
    status = get_training_status(started["job_id"], tail_lines=5, project_root=tmp_path)

    assert status["log_tail"] == [f"line {i}" for i in range(45, 50)]


def test_작업이_없으면_no_job을_모르는_job_id는_unknown_job_error를_반환한다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "pass")
    from agent.tools.train_tool import get_training_status, start_training

    assert get_training_status(project_root=tmp_path)["status"] == "no_job"

    started = start_training("PaiNN", confirmed=True, project_root=tmp_path)
    _wait_until_done(started["job_id"], tmp_path)
    unknown = get_training_status("없는-작업", project_root=tmp_path)
    latest = get_training_status(project_root=tmp_path)  # job_id를 생략하면 가장 최근 작업

    assert (unknown["status"], unknown["error"]) == ("error", "unknown_job")
    assert latest["job_id"] == started["job_id"]


def test_실행_중에도_로그_tail에서_진행_상황을_볼_수_있다(tmp_path, monkeypatch):
    # 학습 로그는 flush 없이 print되므로, 자식 프로세스를 버퍼링 없이(PYTHONUNBUFFERED) 실행해야 실행 중에도 보인다.
    # 한글 출력은 자식 프로세스가 UTF-8(PYTHONUTF8)로 기록해야 로그를 올바르게 읽을 수 있다 (Windows 기본은 cp949).
    use_fake_training(monkeypatch, "print('진행 중 마커'); import time; time.sleep(8)")
    from agent.tools.train_tool import get_training_status, start_training

    started = start_training("PaiNN", confirmed=True, project_root=tmp_path)
    seen = None
    deadline = time.time() + 6
    while time.time() < deadline:
        seen = get_training_status(started["job_id"], project_root=tmp_path)
        if "진행 중 마커" in seen["log_tail"]:
            break
        time.sleep(0.2)
    still_running = get_training_status(started["job_id"], project_root=tmp_path)["state"] == "running"
    _wait_until_done(started["job_id"], tmp_path)

    assert "진행 중 마커" in seen["log_tail"]
    assert seen["state"] == "running" and still_running


# --- 기존 체크포인트 / overwrite ---------------------------------------------------------


def test_이미_학습된_체크포인트가_있으면_overwrite_없이는_already_trained를_반환한다(tmp_path, monkeypatch):
    marker = tmp_path / "ran.txt"
    use_fake_training(monkeypatch, f"open({str(marker)!r}, 'w').write('x')")
    checkpoint = tmp_path / "results_PaiNN" / "0" / "0" / "checkpoint_best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    from agent.tools.train_tool import start_training

    preview = start_training("PaiNN", project_root=tmp_path)
    result = start_training("PaiNN", confirmed=True, project_root=tmp_path)
    time.sleep(1.0)

    assert preview["status"] == "needs_confirmation"
    assert preview["will_overwrite"] is True
    assert result["status"] == "already_trained"
    assert Path(result["checkpoint"]) == checkpoint
    assert not marker.exists() and checkpoint.exists()


def test_overwrite를_주면_기존_출력_디렉터리를_지우고_시작한다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, "import os; print('old_exists=' + str(os.path.exists('results_PaiNN/0/0/old.txt')))")
    old_output = tmp_path / "results_PaiNN" / "0" / "0"
    old_output.mkdir(parents=True)
    (old_output / "checkpoint_best.ckpt").write_bytes(b"")
    (old_output / "old.txt").write_text("이전 학습 산출물", encoding="utf-8")
    other_model = tmp_path / "results_Geoformer" / "0" / "0" / "keep.txt"
    other_model.parent.mkdir(parents=True)
    other_model.write_text("다른 모델의 결과는 건드리지 않는다", encoding="utf-8")
    from agent.tools.train_tool import start_training

    started = start_training("PaiNN", confirmed=True, overwrite=True, project_root=tmp_path)
    status = _wait_until_done(started["job_id"], tmp_path)

    assert started["status"] == "started"
    assert "old_exists=False" in status["log_tail"]  # 학습 프로세스가 시작될 때 이전 산출물이 이미 없었다
    assert other_model.exists()


def test_학습이_체크포인트를_만들면_has_checkpoint가_True이다(tmp_path, monkeypatch):
    use_fake_training(monkeypatch, (
        "import os; os.makedirs('results_PaiNN/0/0', exist_ok=True); "
        "open('results_PaiNN/0/0/checkpoint_best.ckpt', 'wb').close()"))
    from agent.tools.train_tool import start_training

    started = start_training("PaiNN", confirmed=True, project_root=tmp_path)
    status = _wait_until_done(started["job_id"], tmp_path)

    assert status["state"] == "finished"
    assert status["has_checkpoint"] is True  # registry가 학습 산출물을 찾는다


# --- 실제 학습 (slow) --------------------------------------------------------------------


@pytest.mark.slow
def test_실제_PaiNN_tiny_학습이_체크포인트를_만들고_예측_tool로_이어진다():
    outputs = [PROJECT_ROOT / "results_PaiNN", PROJECT_ROOT / "results_agent_logs"]
    if any(path.exists() for path in outputs):
        pytest.skip("저장소 루트에 results_PaiNN 또는 results_agent_logs가 이미 있어 덮어쓰지 않도록 건너뜀")
    from agent.tools.predict_tool import predict_spectrum
    from agent.tools.train_tool import start_training

    try:
        started = start_training("PaiNN", TINY_PAINN, confirmed=True, project_root=PROJECT_ROOT)
        assert started["status"] == "started", started
        if not torch.cuda.is_available():
            assert started["command"][started["command"].index("--workers") + 1] == "0"  # CPU 기본 워커 수 (Step 4B 정책)
        status = _wait_until_done(started["job_id"], PROJECT_ROOT, timeout=300)

        assert status["state"] == "finished", status["log_tail"]
        assert status["has_checkpoint"] is True
        prediction = predict_spectrum("cn1_cn1_nn1", base_model="PaiNN", results_root=PROJECT_ROOT, project_root=PROJECT_ROOT)
        assert prediction["status"] == "ok"
        assert len(prediction["intensity"]) == 800
    finally:
        for path in outputs:
            shutil.rmtree(path, ignore_errors=True)  # 이 테스트가 만든 산출물만 정리 (시작 전에 없음을 확인함)
