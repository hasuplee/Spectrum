"""predict_tool/train_tool이 함께 쓰는 상수와 응답 (Plan.md Step 4B REVIEW 리팩토링)."""

from pathlib import Path

from agent.tools.registry import SUPPORTED_BASE_MODELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def unsupported_base_model_error(base_model) -> dict:
    return {
        "status": "error",
        "error": "unsupported_base_model",
        "supported": list(SUPPORTED_BASE_MODELS),
        "message": f"지원하지 않는 모델입니다: {base_model}. 사용 가능: {', '.join(SUPPORTED_BASE_MODELS)}",
    }
