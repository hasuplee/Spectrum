"""Plan.md Step 5C-2 [신규 인터페이스]: `python -m agent.demo` 콘솔 채팅 (실제 서브프로세스).

agent.demo가 아직 없으므로 실패해야 한다 (RED). 포트는 0(자동 할당)으로 주어 다른 프로세스와 충돌하지 않게 한다.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_python_m_agent_demo는_데모_배너를_출력하고_질문에_답한다():
    result = subprocess.run(
        [sys.executable, "-m", "agent.demo", "--port", "0"],
        input="PaiNN 학습 기본값 보여줘\n종료\n", capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}, cwd=PROJECT_ROOT, timeout=180,
    )

    assert result.returncode == 0, result.stderr[-500:]
    assert "진짜 LLM" in result.stdout and "규칙 기반" in result.stdout  # 시작 배너: 가짜 서버임을 분명히 알린다
    assert "[데모]" in result.stdout  # 질문에 대한 응답
    assert "10000" in result.stdout  # PaiNN 기본 train_steps (show_training_defaults 결과가 응답에 반영됨)
