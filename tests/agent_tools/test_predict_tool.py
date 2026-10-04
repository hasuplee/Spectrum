"""Plan.md Step 3B [신규 인터페이스], G2: 예측 tool(predict_spectrum)과 분자 목록 tool(list_molecules).

agent.tools.predict_tool이 아직 없으므로 실패해야 한다 (RED).
tiny PaiNN/Geoformer 체크포인트를 tmp_path/results_*에 저장해 사용한다.
  - PaiNN: ckpt args를 standardize=True + 실제 split으로 저장해 평균/표준편차를 학습 split에서 재계산하게 한다
    (임의 초기화 모델의 곡선이 물리적으로 말이 되도록).
  - Geoformer: 모델 mean/std 버퍼에 물리적인 값을 준다.
새 모듈 import는 체크포인트 준비 이후 각 테스트 안에서 하여, 준비 코드가 정상임을 확인하면서 개별 실패로 본다.
"""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from common.adapters import painn_adapter, geoformer_adapter
from common.inference import build_batch, load_checkpoint, predict_curves
from dataset.IrDB import IrDB
from tests.refactor.test_inference_checkpoint import FC_TARGETS, _tiny_args
from tests.refactor.test_inference_checkpoint_geoformer import _hyper_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _save_painn_ckpt(results_root: Path, mtime=None) -> Path:
    args = _tiny_args(standardize=True)  # data_path='IrDB', 실제 split → 평균/표준편차를 학습 split에서 재계산
    torch.manual_seed(7)
    model = painn_adapter.build(args)
    path = results_root / "results_PaiNN" / "0" / "0" / "checkpoint_best.ckpt"
    path.parent.mkdir(parents=True)
    torch.save({'model': model.state_dict(), 'optimizer': {}, 'args': args}, path)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _save_geoformer_ckpt(results_root: Path, mtime=None) -> Path:
    hparams = _hyper_parameters()
    hparams["dataset_root"] = "IrDB"  # 분자 조회에 쓰인다 (모델 구성에는 영향 없음)
    torch.manual_seed(7)
    model = geoformer_adapter.build(SimpleNamespace(**hparams))
    path = results_root / "results_Geoformer" / "0" / "0" / "checkpoints" / "last.ckpt"
    path.parent.mkdir(parents=True)
    torch.save({
        'epoch': 0,
        'state_dict': {f'model.{key}': value for key, value in model.state_dict().items()},
        'hyper_parameters': hparams,
    }, path)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _irdb():
    return IrDB(root=str(PROJECT_ROOT / "IrDB"), dataset_arg=FC_TARGETS)


def _molecule_id(index: int) -> str:
    return _irdb()[index].name


# --- needs_training / error 응답 ------------------------------------------------------


def test_학습된_모델이_없으면_needs_training을_반환한다(tmp_path):
    from agent.tools.predict_tool import predict_spectrum

    result = predict_spectrum("cn1_cn1_nn1", results_root=tmp_path)

    assert result["status"] == "needs_training"
    assert result["requested_base_model"] is None
    assert isinstance(result["message"], str) and result["message"]


def test_모델을_지정했는데_해당_모델만_없으면_needs_training을_반환한다(tmp_path):
    _save_painn_ckpt(tmp_path)
    from agent.tools.predict_tool import predict_spectrum

    result = predict_spectrum("cn1_cn1_nn1", base_model="Geoformer", results_root=tmp_path)

    assert result["status"] == "needs_training"
    assert result["requested_base_model"] == "Geoformer"


def test_지원하지_않는_모델을_지정하면_error를_반환한다(tmp_path):
    from agent.tools.predict_tool import predict_spectrum

    result = predict_spectrum("cn1_cn1_nn1", base_model="GPT", results_root=tmp_path)

    assert result["status"] == "error"
    assert result["error"] == "unsupported_base_model"
    assert set(result["supported"]) == {"PaiNN", "Equiformer", "Geoformer"}


def test_존재하지_않는_분자_ID는_error를_반환한다(tmp_path):
    _save_painn_ckpt(tmp_path)
    from agent.tools.predict_tool import predict_spectrum

    result = predict_spectrum("no_such_molecule", results_root=tmp_path)

    assert result["status"] == "error"
    assert result["error"] == "unknown_molecule"
    assert result["molecule_id"] == "no_such_molecule"


# --- 정상 예측 ----------------------------------------------------------------------


def test_PaiNN_체크포인트로_IrDB_분자의_스펙트럼을_예측한다(tmp_path):
    ckpt = _save_painn_ckpt(tmp_path)
    from agent.tools.predict_tool import predict_spectrum

    result = predict_spectrum("cn1_cn1_nn1", results_root=tmp_path)

    assert result["status"] == "ok"
    assert result["molecule_id"] == "cn1_cn1_nn1"
    assert result["base_model"] == "PaiNN"
    assert result["spectrum_type"] == "FC"
    assert Path(result["checkpoint"]).resolve() == ckpt.resolve()
    wavelength, intensity = result["wavelength_nm"], result["intensity"]
    assert len(wavelength) == 800 and len(intensity) == 800
    assert wavelength[0] == 400.0 and wavelength[-1] == 799.5
    assert all(math.isfinite(v) for v in intensity)
    assert math.isclose(max(intensity), 1.0, abs_tol=1e-5)
    assert result["peak_wavelength_nm"] == wavelength[intensity.index(max(intensity))]
    json.dumps(result)  # LLM tool 결과로 쓰이므로 JSON 직렬화가 가능해야 한다


def test_예측_곡선은_직접_계산한_값과_일치한다(tmp_path):
    ckpt = _save_painn_ckpt(tmp_path)
    from agent.tools.predict_tool import predict_spectrum

    first_id, other_id = _molecule_id(0), _molecule_id(10)
    loaded = load_checkpoint(ckpt, "PaiNN")
    dataset = _irdb()
    first = predict_spectrum(first_id, results_root=tmp_path)
    other = predict_spectrum(other_id, results_root=tmp_path)

    for result, index in ((first, 0), (other, 10)):
        expected = predict_curves(loaded, build_batch("PaiNN", [dataset[index]]))[0]
        assert torch.allclose(torch.tensor(result["intensity"]), expected, atol=1e-6)
    # 분자가 실제로 구분되어 선택되는지: 서로 다른 분자의 곡선은 달라야 한다
    assert not torch.allclose(torch.tensor(first["intensity"]), torch.tensor(other["intensity"]), atol=1e-6)


def test_Geoformer_체크포인트로도_예측한다(tmp_path):
    _save_geoformer_ckpt(tmp_path)
    from agent.tools.predict_tool import predict_spectrum

    result = predict_spectrum("cn1_cn1_nn1", base_model="Geoformer", results_root=tmp_path)

    assert result["status"] == "ok"
    assert result["base_model"] == "Geoformer"
    assert len(result["intensity"]) == 800
    assert all(math.isfinite(v) for v in result["intensity"])


def test_모델을_지정하지_않으면_가장_최근_체크포인트의_모델을_쓴다(tmp_path):
    older, newer = 1_000_000, 2_000_000
    _save_painn_ckpt(tmp_path / "a", mtime=newer)
    _save_geoformer_ckpt(tmp_path / "a", mtime=older)
    _save_painn_ckpt(tmp_path / "b", mtime=older)
    _save_geoformer_ckpt(tmp_path / "b", mtime=newer)
    from agent.tools.predict_tool import predict_spectrum

    painn_newer = predict_spectrum("cn1_cn1_nn1", results_root=tmp_path / "a")
    geoformer_newer = predict_spectrum("cn1_cn1_nn1", results_root=tmp_path / "b")

    assert painn_newer["base_model"] == "PaiNN"
    assert geoformer_newer["base_model"] == "Geoformer"


def test_같은_체크포인트로_다시_예측하면_모델을_다시_로드하지_않는다(tmp_path, monkeypatch):
    _save_painn_ckpt(tmp_path)
    from agent.tools import predict_tool

    calls = []

    def counting_load_checkpoint(*args, **kwargs):
        calls.append(args)
        return load_checkpoint(*args, **kwargs)

    monkeypatch.setattr(predict_tool, "load_checkpoint", counting_load_checkpoint)

    first = predict_tool.predict_spectrum("cn1_cn1_nn1", results_root=tmp_path)
    second = predict_tool.predict_spectrum(_molecule_id(10), results_root=tmp_path)

    assert first["status"] == "ok" and second["status"] == "ok"
    assert len(calls) == 1


def test_작업_디렉터리가_달라도_예측하고_원래_디렉터리로_복원한다(tmp_path, monkeypatch):
    _save_painn_ckpt(tmp_path)
    from agent.tools.predict_tool import predict_spectrum

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = predict_spectrum("cn1_cn1_nn1", results_root=tmp_path)

    assert result["status"] == "ok"
    assert Path(os.getcwd()).resolve() == elsewhere.resolve()


# --- 분자 목록 ----------------------------------------------------------------------


def test_분자_ID_목록을_limit만큼_반환한다():
    from agent.tools.predict_tool import list_molecules

    result = list_molecules(limit=5)

    assert result["status"] == "ok"
    assert len(result["molecule_ids"]) == 5
    assert result["total_matches"] == 1024
    assert result["molecule_ids"][0] == _molecule_id(0)


def test_query로_분자_ID를_대소문자_무시하고_부분일치_검색한다():
    from agent.tools.predict_tool import list_molecules

    result = list_molecules(query="CN1_CN1_NN1", limit=100)

    assert result["status"] == "ok"
    assert "cn1_cn1_nn1" in result["molecule_ids"]
    assert all("cn1_cn1_nn1" in molecule_id.lower() for molecule_id in result["molecule_ids"])
    assert result["total_matches"] >= len(result["molecule_ids"]) >= 1
    assert list_molecules(query="no_such_molecule")["total_matches"] == 0


def test_limit이_범위를_벗어나면_error를_반환한다():
    from agent.tools.predict_tool import list_molecules

    for invalid in (0, 101, -5):
        result = list_molecules(limit=invalid)
        assert result["status"] == "error"
        assert result["error"] == "invalid_limit"
