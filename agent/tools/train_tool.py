"""학습 tool: 기본값 조회 / 요청 검증 / 명령 조립 (Plan.md Step 4A, G1).

프로세스를 띄우지 않는 순수 함수만 둔다(백그라운드 실행과 상태 조회는 Step 4B).
기본값은 코드에 복사하지 않고 학습 스크립트(argparse 파서, train.py)와 geoformer/examples/*.yml에서
읽어 온다 — 스크립트의 기본값은 바꾸지 않고(CLAUDE.md 제약 2), 사용자가 명시적으로 지정한 값만
CLI 인자로 덮어쓴다. 학습률 등 최적화 하이퍼파라미터는 읽기 전용이다.
결과는 예외 대신 status가 담긴 dict로 돌려준다(LLM tool 결과로 쓰기 위함).
"""

import shlex
import sys
from argparse import Namespace
from pathlib import Path

import torch
import yaml

import train
from agent.tools.registry import SUPPORTED_BASE_MODELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED, FOLD = 0, 0  # train.py와 같은 고정값 (체크포인트 경로 규칙: registry와 일치)

# 덮어쓸 수 있는 공통 파라미터와 모델 크기 파라미터 (허용 목록)
_MODEL_SIZE_PARAMETERS = {
    "PaiNN": ("embed_dim", "num_layers", "num_basis"),
    "Equiformer": ("num_basis",),
    "Geoformer": ("embedding_dim", "ffn_embedding_dim", "num_layers", "num_heads", "num_rbf"),
}
_COMMON_INT_MINIMUMS = {"batch_size": 1, "train_steps": 1, "eval_steps": 1, "workers": 0}

# 파라미터 이름 -> 학습 스크립트의 CLI 플래그 (Geoformer만 이름이 다르다)
_CLI_FLAGS = {
    "PaiNN": {"train_steps": "--train-steps", "eval_steps": "--eval-steps", "workers": "--workers",
              "embed_dim": "--embed-dim", "num_layers": "--num-layers", "num_basis": "--num-basis"},
    "Equiformer": {"train_steps": "--train-steps", "eval_steps": "--eval-steps", "workers": "--workers",
                   "num_basis": "--num-basis"},
    "Geoformer": {"train_steps": "--num-steps", "eval_steps": "--eval-every", "workers": "--num-workers",
                  "embedding_dim": "--embedding-dim", "ffn_embedding_dim": "--ffn-embedding-dim",
                  "num_layers": "--num-layers", "num_heads": "--num-heads", "num_rbf": "--num-rbf"},
}


def get_training_defaults(base_model) -> dict:
    """학습 스크립트가 실제로 쓰는 기본 설정 (train.py가 넘기는 값 + 각 스크립트/yml의 기본값)."""
    if base_model not in SUPPORTED_BASE_MODELS:
        return _unsupported_base_model(base_model)

    train_py = _train_py_defaults(base_model)
    script = _script_defaults(base_model)
    return {
        "status": "ok",
        "base_model": base_model,
        "spectrum_type": train_py.spectrum_type,
        "data_path": train_py.data_path,
        "batch_size": train_py.batch_size,
        "train_steps": script["train_steps"],
        "eval_steps": script["eval_steps"],
        "workers": script["workers"],
        "learning_rate": script["learning_rate"],
        "model_size": script["model_size"],
        "seed": SEED,
        "fold": FOLD,
        "device": "gpu" if torch.cuda.is_available() else "cpu",
    }


def validate_training_request(base_model, overrides=None) -> dict:
    """기본값 위에 overrides를 적용한 최종 설정을 돌려준다. 허용 목록 밖이거나 값이 잘못되면 error."""
    defaults = get_training_defaults(base_model)
    if defaults["status"] != "ok":
        return defaults

    request = {key: value for key, value in defaults.items() if key != "status"}
    request["model_size"] = dict(request["model_size"])
    size_parameters = _MODEL_SIZE_PARAMETERS[base_model]
    allowed = ["spectrum_type", "data_path", *_COMMON_INT_MINIMUMS, *size_parameters]

    for parameter, value in (overrides or {}).items():
        if parameter not in allowed:
            return _error("unknown_parameter", parameter,
                          f"변경할 수 없는 파라미터입니다: {parameter}. 변경 가능: {', '.join(allowed)}")
        problem = _value_problem(parameter, value, size_parameters)
        if problem:
            return _error("invalid_value", parameter, f"{parameter}={value!r}: {problem}")
        if parameter in size_parameters:
            request["model_size"][parameter] = value
        else:
            request[parameter] = value
    return {"status": "ok", "request": request}


def build_training_command(request) -> list:
    """request(validate_training_request의 결과)를 실행 가능한 명령(list)으로 만든다.

    핵심 인자는 train.build_command를 그대로 재사용하고(첫 토큰 python만 현재 인터프리터로 교체),
    기본값과 다른 값만 모델별 CLI 플래그로 덧붙인다.
    """
    base_model = request["base_model"]
    command = _core_command(base_model, request["spectrum_type"], request["batch_size"], request["data_path"])

    defaults = get_training_defaults(base_model)
    flags = _CLI_FLAGS[base_model]
    for parameter, flag in flags.items():
        if parameter in _COMMON_INT_MINIMUMS:
            value, default = request[parameter], defaults[parameter]
        else:
            value, default = request["model_size"][parameter], defaults["model_size"][parameter]
        if value != default:
            command += [flag, str(value)]
    return command


def training_output_dir(base_model) -> str:
    """학습 결과가 저장되는 디렉터리 (저장소 루트 기준). registry의 탐색 규칙과 같다."""
    if base_model not in SUPPORTED_BASE_MODELS:
        raise ValueError(f"Unsupported base_model: {base_model!r}. Available: {', '.join(SUPPORTED_BASE_MODELS)}")
    defaults = get_training_defaults(base_model)
    command = _core_command(base_model, defaults["spectrum_type"], defaults["batch_size"], defaults["data_path"])
    flag = "--log-dir" if base_model == "Geoformer" else "--output-dir"
    return command[command.index(flag) + 1]


def _core_command(base_model, spectrum_type, batch_size, data_path) -> list:
    args = Namespace(base_model=base_model, spectrum_type=spectrum_type, batch_size=batch_size, data_path=data_path)
    split_npz = train.resolve_split_npz(data_path, SEED, FOLD)
    tokens = shlex.split(train.build_command(args, SEED, FOLD, split_npz))
    return [sys.executable] + tokens[1:]


def _train_py_defaults(base_model):
    """train.py의 get_args()가 돌려주는 기본값 (spectrum-type, batch-size, data-path)."""
    saved_argv = sys.argv
    sys.argv = ["train.py", "--base-model", base_model]
    try:
        return train.get_args()
    finally:
        sys.argv = saved_argv


def _script_defaults(base_model) -> dict:
    if base_model == "Geoformer":
        # get_args()는 sys.argv를 파싱하고 log_dir에 파일을 쓰므로, train.py가 쓰는 yml을 직접 읽는다.
        config = yaml.safe_load((PROJECT_ROOT / "geoformer/examples/FC.yml").read_text(encoding="utf-8"))
        return {
            "train_steps": config["num_steps"], "eval_steps": config["eval_every"],
            "workers": config["num_workers"], "learning_rate": config["lr"],
            "model_size": {key: config[key] for key in _MODEL_SIZE_PARAMETERS["Geoformer"]},
        }
    if base_model == "PaiNN":
        import train_PaiNN as script_module
    else:
        import train_Equiformer as script_module
    script = script_module.get_args_parser().parse_args([])
    return {
        "train_steps": script.train_steps, "eval_steps": script.eval_steps,
        "workers": script.workers, "learning_rate": script.lr,
        "model_size": {key: getattr(script, key) for key in _MODEL_SIZE_PARAMETERS[base_model]},
    }


def _value_problem(parameter, value, size_parameters):
    """값이 올바르면 None, 아니면 사유 문자열."""
    if parameter == "spectrum_type":
        return None if value in train.spectrum_types else f"가능한 값: {', '.join(train.spectrum_types)}"
    if parameter == "data_path":
        return None if value in train.data_path_list else f"가능한 값: {', '.join(train.data_path_list)}"
    minimum = _COMMON_INT_MINIMUMS.get(parameter, 1)  # 모델 크기 파라미터는 1 이상
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        return f"{minimum} 이상의 정수여야 합니다"
    return None


def _unsupported_base_model(base_model) -> dict:
    return {
        "status": "error",
        "error": "unsupported_base_model",
        "supported": list(SUPPORTED_BASE_MODELS),
        "message": f"지원하지 않는 모델입니다: {base_model}. 사용 가능: {', '.join(SUPPORTED_BASE_MODELS)}",
    }


def _error(error, parameter, message) -> dict:
    return {"status": "error", "error": error, "parameter": parameter, "message": message}
