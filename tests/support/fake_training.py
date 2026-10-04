"""가짜 학습 명령 (Step 4B/5B/5C 테스트 공용): 실제 서브프로세스(Popen, 로그 파일, 종료 코드)는 그대로 쓰되 명령만 바꾼다."""

import sys


def use_fake_training(monkeypatch, code):
    """학습 명령을 `python -c 코드`로 대체한다 (프로세스 실행/로그/종료 코드는 실제 그대로)."""
    from agent.tools import train_tool

    monkeypatch.setattr(train_tool, "build_training_command", lambda request: [sys.executable, "-c", code])
