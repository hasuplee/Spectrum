"""Plan.md Step 4A [신규 인터페이스], G1: 학습 tool의 기본값 조회 / 요청 검증 / 명령 조립.

agent.tools.train_tool이 아직 없으므로 실패해야 한다 (RED). 프로세스는 띄우지 않는 순수 함수만 다룬다.
기본값은 코드에 복사하지 않고 학습 스크립트(argparse 파서, train.py)와 geoformer/examples/*.yml에서 읽어
오므로, 테스트도 같은 출처와 독립적으로 비교한다 (스크립트 기본값 불변: CLAUDE.md 제약 2).
새 함수 import는 각 테스트 안에서 하여 개별 실패로 확인한다.
"""

import shlex
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED = ("PaiNN", "Equiformer", "Geoformer")


def _train_py_defaults(monkeypatch, base_model):
    """train.py의 get_args()가 돌려주는 기본값 (batch-size, spectrum-type, data-path)."""
    import train

    monkeypatch.setattr(sys, "argv", ["train.py", "--base-model", base_model])
    return train.get_args()


def _flag_value(command, flag):
    return command[command.index(flag) + 1]


def _default_request(base_model):
    from agent.tools.train_tool import validate_training_request

    return validate_training_request(base_model)["request"]


# --- 기본값 조회 ---------------------------------------------------------------------


def test_PaiNN_기본값은_학습_스크립트의_기본값과_같다(monkeypatch):
    from agent.tools.train_tool import get_training_defaults
    import train_PaiNN

    script = train_PaiNN.get_args_parser().parse_args([])
    train_py = _train_py_defaults(monkeypatch, "PaiNN")

    defaults = get_training_defaults("PaiNN")

    assert defaults["status"] == "ok"
    assert defaults["base_model"] == "PaiNN"
    assert (defaults["spectrum_type"], defaults["data_path"], defaults["batch_size"]) == (
        train_py.spectrum_type, train_py.data_path, train_py.batch_size)
    assert defaults["train_steps"] == script.train_steps
    assert defaults["eval_steps"] == script.eval_steps
    assert defaults["workers"] == script.workers
    assert defaults["learning_rate"] == script.lr
    assert defaults["model_size"] == {
        "embed_dim": script.embed_dim, "num_layers": script.num_layers, "num_basis": script.num_basis}
    assert (defaults["seed"], defaults["fold"]) == (0, 0)
    assert defaults["device"] == ("gpu" if torch.cuda.is_available() else "cpu")


def test_Equiformer_기본값은_학습_스크립트의_기본값과_같다(monkeypatch):
    from agent.tools.train_tool import get_training_defaults
    import train_Equiformer

    script = train_Equiformer.get_args_parser().parse_args([])
    train_py = _train_py_defaults(monkeypatch, "Equiformer")

    defaults = get_training_defaults("Equiformer")

    assert defaults["status"] == "ok"
    assert (defaults["spectrum_type"], defaults["data_path"], defaults["batch_size"]) == (
        train_py.spectrum_type, train_py.data_path, train_py.batch_size)
    assert defaults["train_steps"] == script.train_steps
    assert defaults["eval_steps"] == script.eval_steps
    assert defaults["workers"] == script.workers
    assert defaults["learning_rate"] == script.lr
    assert defaults["model_size"] == {"num_basis": script.num_basis}


def test_Geoformer_기본값은_yml_설정과_같다(monkeypatch):
    from agent.tools.train_tool import get_training_defaults

    config = yaml.safe_load((PROJECT_ROOT / "geoformer/examples/FC.yml").read_text(encoding="utf-8"))
    train_py = _train_py_defaults(monkeypatch, "Geoformer")

    defaults = get_training_defaults("Geoformer")

    assert defaults["status"] == "ok"
    # train.py는 --batch-size를 항상 넘기므로 yml의 batch_size(64)가 아니라 train.py 기본값이 실제 값이다.
    assert (defaults["spectrum_type"], defaults["data_path"], defaults["batch_size"]) == (
        train_py.spectrum_type, train_py.data_path, train_py.batch_size)
    assert defaults["train_steps"] == config["num_steps"]
    assert defaults["eval_steps"] == config["eval_every"]
    assert defaults["workers"] == config["num_workers"]
    assert defaults["learning_rate"] == config["lr"]
    assert defaults["model_size"] == {
        key: config[key] for key in ("embedding_dim", "ffn_embedding_dim", "num_layers", "num_heads", "num_rbf")}


def test_지원하지_않는_모델은_기본값_조회와_검증에서_error를_반환한다():
    from agent.tools.train_tool import get_training_defaults, validate_training_request

    for result in (get_training_defaults("GPT"), validate_training_request("GPT")):
        assert result["status"] == "error"
        assert result["error"] == "unsupported_base_model"
        assert set(result["supported"]) == set(SUPPORTED)


# --- 요청 검증 -----------------------------------------------------------------------


@pytest.mark.parametrize("base_model", SUPPORTED)
def test_override가_없으면_검증_결과는_기본값과_같다(base_model):
    from agent.tools.train_tool import get_training_defaults, validate_training_request

    defaults = get_training_defaults(base_model)

    result = validate_training_request(base_model)

    assert result["status"] == "ok"
    assert result["request"] == {key: value for key, value in defaults.items() if key != "status"}


def test_override는_기본값_위에_적용된다():
    from agent.tools.train_tool import get_training_defaults, validate_training_request

    painn = validate_training_request("PaiNN", {
        "train_steps": 5, "eval_steps": 5, "workers": 0, "batch_size": 4, "spectrum_type": "GMM",
        "embed_dim": 8, "num_layers": 1})
    geoformer = validate_training_request("Geoformer", {"embedding_dim": 8, "num_heads": 2})

    request = painn["request"]
    defaults = get_training_defaults("PaiNN")
    assert painn["status"] == "ok"
    assert (request["train_steps"], request["eval_steps"], request["workers"], request["batch_size"],
            request["spectrum_type"]) == (5, 5, 0, 4, "GMM")
    assert request["model_size"] == {**defaults["model_size"], "embed_dim": 8, "num_layers": 1}  # 나머지는 기본값 유지
    assert request["data_path"] == defaults["data_path"]
    assert request["learning_rate"] == defaults["learning_rate"]
    assert geoformer["request"]["model_size"] == {
        **get_training_defaults("Geoformer")["model_size"], "embedding_dim": 8, "num_heads": 2}


@pytest.mark.parametrize("base_model, overrides, parameter", [
    ("PaiNN", {"lr": 0.1}, "lr"),
    ("PaiNN", {"seed": 1}, "seed"),
    ("Equiformer", {"embed_dim": 8}, "embed_dim"),  # Equiformer에는 없는 모델 크기 파라미터
    ("Geoformer", {"embed_dim": 8}, "embed_dim"),   # Geoformer의 이름은 embedding_dim
])
def test_허용되지_않는_파라미터는_unknown_parameter_error를_반환한다(base_model, overrides, parameter):
    from agent.tools.train_tool import validate_training_request

    result = validate_training_request(base_model, overrides)

    assert result["status"] == "error"
    assert result["error"] == "unknown_parameter"
    assert result["parameter"] == parameter


@pytest.mark.parametrize("overrides, parameter", [
    ({"batch_size": 0}, "batch_size"),
    ({"train_steps": "abc"}, "train_steps"),
    ({"eval_steps": 1.5}, "eval_steps"),
    ({"spectrum_type": "XYZ"}, "spectrum_type"),
    ({"data_path": "Foo"}, "data_path"),
    ({"workers": -1}, "workers"),
])
def test_잘못된_값은_invalid_value_error를_반환한다(overrides, parameter):
    from agent.tools.train_tool import validate_training_request

    result = validate_training_request("PaiNN", overrides)

    assert result["status"] == "error"
    assert result["error"] == "invalid_value"
    assert result["parameter"] == parameter
    assert isinstance(result["message"], str) and result["message"]


# --- 명령 조립 -----------------------------------------------------------------------


def test_PaiNN_기본_명령은_학습_스크립트_모듈과_train_py_인자로_조립된다():
    from agent.tools.train_tool import build_training_command

    command = build_training_command(_default_request("PaiNN"))

    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "train_PaiNN"]
    assert _flag_value(command, "--spectrum-type") == "FC"
    assert _flag_value(command, "--batch-size") == "16"
    assert _flag_value(command, "--data-path") == "IrDB"
    assert _flag_value(command, "--output-dir") == "results_PaiNN/0/0"
    assert _flag_value(command, "--split-index-npz") == "IrDB/raw/CV811/splits.0.0.npz"
    assert _flag_value(command, "--seed") == "0"
    # 기본값과 같은 값은 덮어쓰기 플래그를 만들지 않는다 (스크립트 기본값에 맡긴다)
    for flag in ("--train-steps", "--eval-steps", "--workers", "--embed-dim", "--num-layers", "--num-basis"):
        assert flag not in command


def test_Equiformer_기본_명령은_학습_스크립트_모듈과_train_py_인자로_조립된다():
    from agent.tools.train_tool import build_training_command

    command = build_training_command(_default_request("Equiformer"))

    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "train_Equiformer"]
    assert _flag_value(command, "--output-dir") == "results_Equiformer/0/0"
    assert _flag_value(command, "--batch-size") == "16"
    for flag in ("--train-steps", "--eval-steps", "--workers", "--num-basis"):
        assert flag not in command


def test_Geoformer_기본_명령은_yml과_CPU_플래그를_포함한다():
    from agent.tools.train_tool import build_training_command

    command = build_training_command(_default_request("Geoformer"))

    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "train_Geoformer"]
    assert _flag_value(command, "--conf") == "geoformer/examples/FC.yml"
    assert _flag_value(command, "--log-dir") == "results_Geoformer/0/0"
    assert _flag_value(command, "--splits") == "IrDB/raw/CV811/splits.0.0.npz"
    assert _flag_value(command, "--dataset-root") == "IrDB"
    assert _flag_value(command, "--batch-size") == "16"
    if not torch.cuda.is_available():
        assert _flag_value(command, "--accelerator") == "cpu"
        assert _flag_value(command, "--ndevices") == "1"
    for flag in ("--num-steps", "--eval-every", "--num-workers", "--embedding-dim"):
        assert flag not in command


@pytest.mark.parametrize("base_model", SUPPORTED)
def test_기본_명령의_핵심_인자는_train_py의_build_command와_같다(base_model):
    # 드리프트 방지: 인자 조립 규칙을 복사하지 않고 train.build_command를 재사용하는지 확인한다.
    import train
    from agent.tools.train_tool import build_training_command

    legacy = shlex.split(train.build_command(
        Namespace(base_model=base_model, spectrum_type="FC", batch_size=16, data_path="IrDB"),
        0, 0, train.resolve_split_npz("IrDB", 0, 0)))

    command = build_training_command(_default_request(base_model))

    assert command == [sys.executable] + legacy[1:]  # 첫 토큰 'python'만 가상환경 인터프리터로 교체


def test_override는_모델별_CLI_플래그로_변환된다():
    from agent.tools.train_tool import build_training_command, validate_training_request

    def command_for(base_model, overrides):
        return build_training_command(validate_training_request(base_model, overrides)["request"])

    painn = command_for("PaiNN", {
        "train_steps": 5, "eval_steps": 6, "workers": 0, "embed_dim": 8, "num_layers": 1, "num_basis": 8, "batch_size": 4})
    equiformer = command_for("Equiformer", {"train_steps": 5, "num_basis": 8})
    geoformer = command_for("Geoformer", {
        "train_steps": 5, "eval_steps": 6, "workers": 0, "embedding_dim": 8, "ffn_embedding_dim": 16,
        "num_layers": 1, "num_heads": 2, "num_rbf": 8, "spectrum_type": "GMM"})

    for flag, value in (("--train-steps", "5"), ("--eval-steps", "6"), ("--workers", "0"), ("--embed-dim", "8"),
                        ("--num-layers", "1"), ("--num-basis", "8"), ("--batch-size", "4")):
        assert _flag_value(painn, flag) == value
    assert _flag_value(equiformer, "--train-steps") == "5"
    assert _flag_value(equiformer, "--num-basis") == "8"
    for flag, value in (("--num-steps", "5"), ("--eval-every", "6"), ("--num-workers", "0"), ("--embedding-dim", "8"),
                        ("--ffn-embedding-dim", "16"), ("--num-layers", "1"), ("--num-heads", "2"), ("--num-rbf", "8"),
                        ("--conf", "geoformer/examples/GMM.yml")):
        assert _flag_value(geoformer, flag) == value


# --- 출력 경로 / registry 일관성 ---------------------------------------------------------


def test_학습_출력_경로는_registry가_찾는_경로와_같다(tmp_path):
    from agent.tools.registry import find_checkpoints
    from agent.tools.train_tool import training_output_dir

    paths = {
        "PaiNN": "checkpoint_best.ckpt",
        "Equiformer": "checkpoint_best.ckpt",
        "Geoformer": "checkpoints/last.ckpt",  # Lightning이 log_dir/checkpoints에 저장
    }
    for base_model, ckpt_name in paths.items():
        output_dir = training_output_dir(base_model)
        assert output_dir == f"results_{base_model}/0/0"
        ckpt = tmp_path / output_dir / ckpt_name
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"")

        assert [c.path for c in find_checkpoints(tmp_path, base_model)] == [ckpt]
