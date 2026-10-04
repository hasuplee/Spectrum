"""Agent(LLM)가 쓰는 tool 래퍼 (Plan.md Step 5B, G1~G3).

Step 1~4의 tool을 LLM이 쓰기 좋게 감싼다: 결과를 간결하게 요약하고(예측 곡선 약 20KB, 학습 명령의 절대 경로 등
LLM에 불필요한 정보 제외), 학습은 "미리보기 → 확인 실행" 두 단계로만 시작하게 한다. AGNO는 함수의 시그니처와
docstring(Args)으로 tool 스키마를 만들므로 docstring은 LLM이 읽는 설명이다. `settings`의 값은 `Dict[str, Any]`가
아니라 `Union[int, str]`로 선언해야 스키마가 `object`가 아니라 정수/문자열로 변환된다.
"""

from pathlib import Path
from typing import Dict, Optional, Union

from agent.tools._shared import PROJECT_ROOT
from agent.tools.predict_tool import list_molecules, predict_spectrum
from agent.tools.registry import SUPPORTED_BASE_MODELS, get_latest_checkpoint
from agent.tools.train_tool import (
    changeable_parameters,
    get_training_defaults,
    get_training_status,
    start_training,
    validate_training_request,
)

_LOG_TAIL_LINES = 5
_LOG_LINE_LIMIT = 200
_SAMPLE_WAVELENGTHS_NM = range(400, 800, 50)  # 예측 요약의 샘플 파장 (8개)


def build_assistant_tools(project_root=PROJECT_ROOT, results_root=None) -> list:
    """Agent에 넘길 tool 함수 7개. 호출마다 독립적인 상태(마지막 학습 미리보기)를 가진다.

    학습은 project_root(작업 디렉터리와 results_* 위치)에서, 예측은 results_root(기본값 project_root)에서
    체크포인트를 찾는다. 실제 사용에서는 둘 다 저장소 루트이고 다르게 주는 것은 테스트뿐이다.
    """
    project_root = Path(project_root)
    results_root = Path(results_root) if results_root is not None else project_root
    state = {"preview": None}  # (base_model, 최종 설정) — start_training_confirmed가 같은 설정인지 확인한다

    def list_trained_models() -> dict:
        """모델(PaiNN, Geoformer, Equiformer)별로 학습된 체크포인트가 있는지 알려준다. 예측 전에 학습이 필요한지 판단할 때 쓴다."""
        models = {}
        for name in SUPPORTED_BASE_MODELS:
            latest = get_latest_checkpoint(results_root, name)
            models[name] = {"trained": latest is not None, "checkpoint": latest.path.name if latest else None}
        return {"status": "ok", "models": models, "any_trained": any(model["trained"] for model in models.values())}

    def show_training_defaults(base_model: str) -> dict:
        """모델의 기본 학습 설정과, 사용자가 바꿀 수 있는 파라미터 이름 목록을 알려준다.

        Args:
            base_model: 학습할 모델. PaiNN, Geoformer, Equiformer 중 하나.
        """
        defaults = get_training_defaults(base_model)
        if defaults["status"] != "ok":
            return defaults
        return {**defaults, "changeable_parameters": changeable_parameters(base_model)}

    def preview_training(base_model: str, settings: Optional[Dict[str, Union[int, str]]] = None) -> dict:
        """학습을 시작하지 않고 최종 설정을 보여 준다. 학습 전에 반드시 먼저 호출하고, 결과를 사용자에게 보여 준 뒤 확인을 받아야 한다.

        Args:
            base_model: 학습할 모델. PaiNN, Geoformer, Equiformer 중 하나.
            settings: 기본값에서 바꿀 파라미터 이름과 값 (예: {"train_steps": 20, "batch_size": 8}). 생략하면 기본값 그대로.
        """
        result = start_training(base_model, settings, confirmed=False, project_root=project_root)
        if result["status"] != "needs_confirmation":
            return result
        state["preview"] = (base_model, result["settings"])
        message = "이 설정을 사용자에게 보여 주고 확인을 받은 뒤에만 start_training_confirmed를 같은 인자로 호출하세요."
        if result["will_overwrite"]:
            message += " 이미 학습된 결과가 있어 다시 학습하면 기존 결과가 삭제되므로 그 점도 알리고, 확인되면 overwrite=True로 호출하세요."
        return {**result, "message": message}

    def start_training_confirmed(
        base_model: str,
        settings: Optional[Dict[str, Union[int, str]]] = None,
        overwrite: bool = False,
    ) -> dict:
        """사용자가 확인한 학습을 시작한다. 직전에 preview_training으로 보여 준 것과 같은 모델·설정일 때만 실행된다.

        Args:
            base_model: preview_training에서 보여 준 모델.
            settings: preview_training에서 보여 준 것과 같은 설정.
            overwrite: 이미 학습된 결과를 삭제하고 다시 학습할 때만 True (사용자가 삭제를 확인한 경우).
        """
        validated = validate_training_request(base_model, settings)
        if validated["status"] != "ok":
            return validated
        if state["preview"] != (base_model, validated["request"]):
            return {
                "status": "error",
                "error": "not_previewed",
                "message": "같은 설정으로 preview_training을 먼저 호출해 사용자에게 보여 주고 확인을 받아야 합니다.",
            }
        result = start_training(base_model, settings, confirmed=True, overwrite=overwrite, project_root=project_root)
        if result["status"] != "started":
            return result  # already_trained / busy: 미리보기를 유지해 사용자 확인 후 다시 호출할 수 있게 한다
        state["preview"] = None  # 시작에 성공하면 소모한다 (다시 학습하려면 새 미리보기가 필요)
        return {
            "status": "started",
            "job_id": result["job_id"],
            "base_model": result["base_model"],
            "output_dir": result["output_dir"],
            "settings": result["settings"],
            "message": "학습을 시작했습니다. check_training_status로 진행 상황을 확인할 수 있습니다.",
        }

    def check_training_status(job_id: Optional[str] = None) -> dict:
        """학습 작업의 상태(running/finished/failed), 경과 시간, 로그 마지막 몇 줄, 체크포인트 생성 여부를 알려준다.

        Args:
            job_id: 조회할 작업 ID. 생략하면 가장 최근 작업.
        """
        status = get_training_status(job_id, tail_lines=_LOG_TAIL_LINES, project_root=project_root)
        if status["status"] != "ok":
            return status
        return {
            "status": "ok",
            "job_id": status["job_id"],
            "base_model": status["base_model"],
            "state": status["state"],
            "return_code": status["return_code"],
            "elapsed_seconds": int(status["elapsed_seconds"]),
            "log_tail": [line[:_LOG_LINE_LIMIT] for line in status["log_tail"]],
            "has_checkpoint": status["has_checkpoint"],
        }

    def list_molecule_ids(query: str = "", limit: int = 10) -> dict:
        """예측할 수 있는 IrDB 분자 ID 목록을 알려준다.

        Args:
            query: ID에 포함된 문자열로 검색 (대소문자 무시). 생략하면 전체.
            limit: 돌려줄 최대 개수 (1~100).
        """
        return list_molecules(query, limit, project_root=project_root)

    def predict_molecule_spectrum(molecule_id: str, base_model: Optional[str] = None) -> dict:
        """IrDB 분자의 발광 스펙트럼을 학습된 모델로 예측해 요약(피크 파장과 파장별 상대 강도)을 돌려준다.
        학습된 모델이 없으면 학습이 먼저 필요하다는 결과를 돌려준다.

        Args:
            molecule_id: 분자 ID (예: cn1_cn1_nn1). 모르면 list_molecule_ids로 찾는다.
            base_model: 사용할 모델. 생략하면 가장 최근에 학습된 모델.
        """
        result = predict_spectrum(molecule_id, base_model, results_root=results_root, project_root=project_root)
        if result["status"] != "ok":
            return result
        wavelengths = result["wavelength_nm"]
        samples = [
            {"wavelength_nm": wavelength, "intensity": round(result["intensity"][wavelengths.index(float(wavelength))], 3)}
            for wavelength in _SAMPLE_WAVELENGTHS_NM
        ]
        return {
            "status": "ok",
            "molecule_id": result["molecule_id"],
            "base_model": result["base_model"],
            "checkpoint": Path(result["checkpoint"]).name,
            "spectrum_type": result["spectrum_type"],
            "peak_wavelength_nm": result["peak_wavelength_nm"],
            "samples": samples,
        }

    return [
        list_trained_models,
        show_training_defaults,
        preview_training,
        start_training_confirmed,
        check_training_status,
        list_molecule_ids,
        predict_molecule_spectrum,
    ]
