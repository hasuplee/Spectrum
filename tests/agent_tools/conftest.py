"""agent_tools 테스트 공용 fixture."""

import pytest


@pytest.fixture(autouse=True)
def _fresh_jobs(monkeypatch):
    """학습 작업 목록(`train_tool._jobs`, 모듈 전역 상태)을 테스트마다 초기화한다."""
    from agent.tools import train_tool

    monkeypatch.setattr(train_tool, "_jobs", {}, raising=False)
