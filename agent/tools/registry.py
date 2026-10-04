"""학습된 체크포인트 탐색 (Plan.md Step 1, G2 전제).

기존 학습 스크립트의 저장 경로를 읽기만 한다:
  - PaiNN/Equiformer: {results_root}/results_{모델}/{seed}/{fold}/checkpoint_best.ckpt
  - Geoformer:        {results_root}/results_Geoformer/{seed}/{fold}/checkpoints/*.ckpt
"""

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_BASE_MODELS = ("PaiNN", "Equiformer", "Geoformer")

_CHECKPOINT_PATTERNS = {
    "PaiNN": "*/*/checkpoint_best.ckpt",
    "Equiformer": "*/*/checkpoint_best.ckpt",
    "Geoformer": "*/*/checkpoints/*.ckpt",
}


@dataclass(frozen=True)
class CheckpointInfo:
    base_model: str
    path: Path
    modified_time: float


def find_checkpoints(results_root=".", base_model=None) -> list[CheckpointInfo]:
    """수정 시각 최신순으로 정렬된 체크포인트 목록. base_model이 None이면 전 모델."""
    if base_model is not None and base_model not in SUPPORTED_BASE_MODELS:
        raise ValueError(
            f"Unsupported base_model: {base_model!r}. Available: {', '.join(SUPPORTED_BASE_MODELS)}"
        )
    models = SUPPORTED_BASE_MODELS if base_model is None else (base_model,)

    found = []
    for model in models:
        model_dir = Path(results_root) / f"results_{model}"
        for path in model_dir.glob(_CHECKPOINT_PATTERNS[model]):
            found.append(CheckpointInfo(model, path, path.stat().st_mtime))
    return sorted(found, key=lambda c: c.modified_time, reverse=True)


def get_latest_checkpoint(results_root=".", base_model=None):
    found = find_checkpoints(results_root, base_model)
    return found[0] if found else None


def has_trained_model(results_root=".", base_model=None) -> bool:
    return bool(find_checkpoints(results_root, base_model))
