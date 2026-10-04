"""학습 tool: 기본값 조회 / 요청 검증 / 명령 조립 (Plan.md Step 4A) + 확인 절차 / 백그라운드 실행 / 상태 조회 (Step 4B), G1.

기본값은 코드에 복사하지 않고 학습 스크립트(argparse 파서, train.py)와 geoformer/examples/*.yml에서
읽어 온다 — 스크립트의 기본값은 바꾸지 않고(CLAUDE.md 제약 2), 사용자가 명시적으로 지정한 값만
CLI 인자로 덮어쓴다. 학습률 등 최적화 하이퍼파라미터는 읽기 전용이다.
결과는 예외 대신 status가 담긴 dict로 돌려준다(LLM tool 결과로 쓰기 위함).
"""

import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from argparse import Namespace
from pathlib import Path

import torch
import yaml

import train
from agent.tools._shared import PROJECT_ROOT, unsupported_base_model_error
from agent.tools.registry import SUPPORTED_BASE_MODELS, get_latest_checkpoint, has_trained_model

SEED, FOLD = 0, 0  # train.py와 같은 고정값 (체크포인트 경로 규칙: registry와 일치)

# job_id -> 작업 기록. 프로세스 메모리에만 있다(재시작 후 복구하지 않음). 삽입 순서가 곧 시작 순서다.
_jobs = {}

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
        return unsupported_base_model_error(base_model)

    train_py = _train_py_defaults(base_model)
    script = _script_defaults(base_model)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    return {
        "status": "ok",
        "base_model": base_model,
        "spectrum_type": train_py.spectrum_type,
        "data_path": train_py.data_path,
        "batch_size": train_py.batch_size,
        "train_steps": script["train_steps"],
        "eval_steps": script["eval_steps"],
        # CPU(Windows)에서는 기본 워커 수로 돌리면 크게 느려지므로(Geoformer tiny 217.8초 vs 12초) CPU일 때만 0으로 한다.
        # 워커 수는 학습 수치에 영향이 없는 런타임 설정이며, GPU에서는 스크립트 기본값 그대로다 (Plan.md Step 4B).
        "workers": script["workers"] if device == "gpu" else 0,
        "learning_rate": script["learning_rate"],
        "model_size": script["model_size"],
        "seed": SEED,
        "fold": FOLD,
        "device": device,
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
        problem = _value_problem(parameter, value)
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
    **스크립트 기본값**과 다른 값만 모델별 CLI 플래그로 덧붙인다(CPU의 tool 기본 workers=0도 스크립트 기본값과
    다르므로 플래그가 붙는다 — 그렇지 않으면 스크립트 기본값 4/6으로 조용히 되돌아간다).
    """
    base_model = request["base_model"]
    command = _core_command(base_model, request["spectrum_type"], request["batch_size"], request["data_path"])

    script = _script_defaults(base_model)
    for parameter, flag in _CLI_FLAGS[base_model].items():
        value = _setting(request, parameter)
        if value != _setting(script, parameter):
            command += [flag, str(value)]
    return command


def start_training(base_model, overrides=None, *, confirmed=False, overwrite=False, project_root=PROJECT_ROOT) -> dict:
    """사용자 확인을 거친 학습을 백그라운드 프로세스로 시작한다.

    confirmed=False이면 실행하지 않고 최종 설정을 돌려준다(needs_confirmation). 기존 체크포인트가 있으면
    overwrite=True가 있어야 시작하며, 그때는 기존 출력 디렉터리를 **삭제**한다(Geoformer의 이어 학습과 이전
    산출물 혼입을 막기 위함). 학습 프로세스는 project_root 기준 상대 경로를 쓰므로 실제 학습의 project_root는
    저장소 루트여야 한다. 작업은 한 번에 하나만 실행한다.
    """
    validated = validate_training_request(base_model, overrides)
    if validated["status"] != "ok":
        return validated
    request = validated["request"]
    project_root = Path(project_root)

    running = _running_job()
    if running is not None:
        return {
            "status": "busy",
            "job_id": running["job_id"],
            "message": f"이미 학습이 진행 중입니다: {running['job_id']} ({running['base_model']}). 끝난 뒤 다시 시작해 주세요.",
        }

    output_dir = training_output_dir(base_model)
    existing = get_latest_checkpoint(project_root, base_model)
    if not confirmed:
        return _needs_confirmation(request, output_dir, will_overwrite=existing is not None)
    if existing is not None and not overwrite:
        return {
            "status": "already_trained",
            "checkpoint": str(existing.path),
            "message": f"{base_model}의 학습된 체크포인트가 이미 있습니다. 다시 학습하려면 overwrite=True가 필요합니다(기존 결과가 삭제됩니다).",
        }

    if overwrite and (project_root / output_dir).exists():
        shutil.rmtree(project_root / output_dir)
    return _launch_job(base_model, request, output_dir, project_root)


def _needs_confirmation(request, output_dir, will_overwrite) -> dict:
    message = "아래 설정으로 학습을 시작합니다. 진행하려면 사용자에게 확인을 받은 뒤 confirmed=True로 다시 호출하세요."
    if will_overwrite:
        message += " 이미 학습된 체크포인트가 있어 덮어쓰려면 overwrite=True도 필요합니다(기존 결과가 삭제됩니다)."
    return {
        "status": "needs_confirmation",
        "settings": request,
        "output_dir": output_dir,
        "will_overwrite": will_overwrite,
        "message": message,
    }


def _launch_job(base_model, request, output_dir, project_root) -> dict:
    """학습 프로세스를 백그라운드로 시작하고 작업 기록을 남긴다."""
    job_id = f"{base_model.lower()}-{uuid.uuid4().hex[:8]}"
    log_path = project_root / "results_agent_logs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_training_command(request)
    # Windows 기본(cp949)으로 기록되면 UTF-8로 읽을 수 없고, 파일로 리다이렉트된 stdout은 버퍼링되어 실행 중 로그가 보이지 않는다.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    with open(log_path, "wb") as log_file:
        process = subprocess.Popen(command, cwd=project_root, stdout=log_file, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env)
    _jobs[job_id] = {
        "job_id": job_id, "base_model": base_model, "command": command, "log_path": log_path,
        "output_dir": output_dir, "settings": request, "process": process, "started_at": time.time(), "ended_at": None,
    }
    return {
        "status": "started",
        "job_id": job_id,
        "base_model": base_model,
        "command": command,
        "log_path": str(log_path),
        "output_dir": output_dir,
        "settings": request,
    }


def get_training_status(job_id=None, *, tail_lines=20, project_root=PROJECT_ROOT) -> dict:
    """학습 작업의 상태(running/finished/failed)와 로그 마지막 줄들. job_id를 생략하면 가장 최근 작업."""
    if job_id is None:
        if not _jobs:
            return {"status": "no_job", "message": "시작된 학습 작업이 없습니다."}
        job = list(_jobs.values())[-1]
    elif job_id in _jobs:
        job = _jobs[job_id]
    else:
        return {"status": "error", "error": "unknown_job", "job_id": job_id, "message": f"알 수 없는 작업입니다: {job_id}"}

    return_code = job["process"].poll()
    if return_code is not None and job["ended_at"] is None:
        job["ended_at"] = time.time()  # 종료를 처음 관찰한 시각(근사값)
    state = "running" if return_code is None else ("finished" if return_code == 0 else "failed")
    log_lines = Path(job["log_path"]).read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "status": "ok",
        "job_id": job["job_id"],
        "base_model": job["base_model"],
        "state": state,
        "return_code": return_code,
        "elapsed_seconds": (job["ended_at"] or time.time()) - job["started_at"],
        "log_tail": log_lines[-tail_lines:] if tail_lines > 0 else [],
        "log_path": str(job["log_path"]),
        "output_dir": job["output_dir"],
        "has_checkpoint": has_trained_model(Path(project_root), job["base_model"]),
        "settings": job["settings"],
    }


def training_output_dir(base_model) -> str:
    """학습 결과가 저장되는 디렉터리 (저장소 루트 기준). registry의 탐색 규칙과 같다."""
    if base_model not in SUPPORTED_BASE_MODELS:
        raise ValueError(f"Unsupported base_model: {base_model!r}. Available: {', '.join(SUPPORTED_BASE_MODELS)}")
    train_py = _train_py_defaults(base_model)
    command = _core_command(base_model, train_py.spectrum_type, train_py.batch_size, train_py.data_path)
    flag = "--log-dir" if base_model == "Geoformer" else "--output-dir"
    return command[command.index(flag) + 1]


def _setting(settings, parameter):
    """공통 파라미터는 최상위에, 모델 크기 파라미터는 `model_size` 안에 있다."""
    if parameter in _COMMON_INT_MINIMUMS:
        return settings[parameter]
    return settings["model_size"][parameter]


def _running_job():
    for job in _jobs.values():
        if job["process"].poll() is None:
            return job
    return None


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


def _value_problem(parameter, value):
    """값이 올바르면 None, 아니면 사유 문자열."""
    if parameter == "spectrum_type":
        return None if value in train.spectrum_types else f"가능한 값: {', '.join(train.spectrum_types)}"
    if parameter == "data_path":
        return None if value in train.data_path_list else f"가능한 값: {', '.join(train.data_path_list)}"
    minimum = _COMMON_INT_MINIMUMS.get(parameter, 1)  # 모델 크기 파라미터는 1 이상
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        return f"{minimum} 이상의 정수여야 합니다"
    return None


def _error(error, parameter, message) -> dict:
    return {"status": "error", "error": error, "parameter": parameter, "message": message}
