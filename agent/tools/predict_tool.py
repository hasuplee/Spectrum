"""스펙트럼 예측 tool과 분자 목록 tool (Plan.md Step 3B, G2).

Agent/UI가 호출하는 함수들이며, 결과는 예외 대신 status가 담긴 JSON 직렬화 가능한 dict로 돌려준다.
곡선 계산은 common.inference(load_checkpoint/build_batch/predict_curves)를 거치며, 이 모듈이 직접
스펙트럼 물리 로직을 갖지는 않는다.

주의: predict_spectrum은 호출 동안 프로세스의 작업 디렉터리를 바꾼다(체크포인트 args의 상대 경로와
평균/표준편차 재계산이 저장소 루트 기준이기 때문). 전역 상태이므로 동시 호출은 지원하지 않는다.
"""

import contextlib
from pathlib import Path

from agent.tools._shared import PROJECT_ROOT, unsupported_base_model_error
from agent.tools.registry import SUPPORTED_BASE_MODELS, get_latest_checkpoint
from common.data import build_dataset
from common.inference import build_batch, load_checkpoint, predict_curves
from spectrum.reconstruct import wavelength_grid_nm

MAX_LIST_LIMIT = 100

# base_model -> ((체크포인트 경로, 수정 시각), LoadedCheckpoint). 모델 로드(특히 Equiformer 약 7초)를 반복하지 않기 위함.
# 모델 종류마다 최신 1개만 유지한다: 재학습으로 체크포인트가 바뀌면 이전 모델은 교체되어 메모리에서 해제된다.
_loaded_checkpoint_cache = {}


def predict_spectrum(molecule_id, base_model=None, *, results_root=PROJECT_ROOT, project_root=PROJECT_ROOT) -> dict:
    """IrDB 분자 ID의 스펙트럼을 학습된 체크포인트로 예측한다.

    base_model: "PaiNN" | "Equiformer" | "Geoformer". 생략하면 학습된 체크포인트 중 가장 최근 것을 쓴다.
    반환: status가 "ok"(곡선 포함), "needs_training", "error" 중 하나인 dict.
    """
    if base_model is not None and base_model not in SUPPORTED_BASE_MODELS:
        return unsupported_base_model_error(base_model)

    # 작업 디렉터리를 바꾸기 전에 절대 경로로 확정한다.
    checkpoint = get_latest_checkpoint(Path(results_root).resolve(), base_model)
    if checkpoint is None:
        target = base_model if base_model is not None else "어떤 모델"
        return {
            "status": "needs_training",
            "requested_base_model": base_model,
            "message": f"학습된 {target}의 체크포인트가 없습니다. 먼저 학습을 진행해 주세요.",
        }

    with contextlib.chdir(project_root):
        loaded = _load_cached(checkpoint)
        dataset = _open_dataset(loaded)
        names = _molecule_names(dataset)
        if molecule_id not in names:
            return {
                "status": "error",
                "error": "unknown_molecule",
                "molecule_id": molecule_id,
                "message": f"데이터셋에 없는 분자 ID입니다: {molecule_id}",
            }
        batch = build_batch(checkpoint.base_model, [dataset[names.index(molecule_id)]])
        curve = predict_curves(loaded, batch)[0]

    wavelength = wavelength_grid_nm().tolist()
    return {
        "status": "ok",
        "molecule_id": molecule_id,
        "base_model": checkpoint.base_model,
        "checkpoint": str(checkpoint.path),
        "spectrum_type": loaded.args.spectrum_type,
        "wavelength_nm": wavelength,
        "intensity": curve.tolist(),
        "peak_wavelength_nm": wavelength[int(curve.argmax())],
    }


def list_molecules(query="", limit=20, *, project_root=PROJECT_ROOT) -> dict:
    """IrDB 분자 ID 목록. query는 대소문자를 무시한 부분 일치, limit은 1~100."""
    if not isinstance(limit, int) or not 1 <= limit <= MAX_LIST_LIMIT:
        return {
            "status": "error",
            "error": "invalid_limit",
            "message": f"limit은 1~{MAX_LIST_LIMIT} 사이의 정수여야 합니다: {limit}",
        }
    with contextlib.chdir(project_root):  # build_dataset은 'IrDB'로 시작하는 상대 경로를 기대한다 (predict_spectrum과 동일)
        dataset = build_dataset("IrDB", ["S1"])
    needle = query.lower()
    matches = [name for name in _molecule_names(dataset) if needle in name.lower()]
    return {"status": "ok", "total_matches": len(matches), "molecule_ids": matches[:limit]}


def _load_cached(checkpoint):
    key = (str(checkpoint.path), checkpoint.modified_time)
    cached = _loaded_checkpoint_cache.get(checkpoint.base_model)
    if cached is not None and cached[0] == key:
        return cached[1]
    loaded = load_checkpoint(checkpoint.path, checkpoint.base_model)
    _loaded_checkpoint_cache[checkpoint.base_model] = (key, loaded)
    return loaded


def _open_dataset(loaded):
    """체크포인트를 학습할 때 쓴 데이터셋을 연다 (경로는 저장소 루트 기준)."""
    args = loaded.args
    if loaded.base_model == "Geoformer":
        root, targets = args.dataset_root, args.dataset_arg
    else:
        root, targets = args.data_path, args.targets
    return build_dataset(root, targets)


def _molecule_names(dataset) -> list:
    return [data.name for data in dataset]
