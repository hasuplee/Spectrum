"""Plan.md Step 1 [신규 인터페이스], G2 전제: results_*/ 아래 체크포인트를 찾아
"예측 가능 여부"를 판정하는 agent.tools.registry.

실제 모델/torch 없이 tmp_path에 빈 파일로 디렉터리 구조만 만들어 검증한다.
"""

import os

import pytest

from agent.tools.registry import (
    SUPPORTED_BASE_MODELS,
    find_checkpoints,
    get_latest_checkpoint,
    has_trained_model,
)


def _touch(path, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _painn_ckpt(root, seed=0, fold=0, mtime=None):
    return _touch(root / "results_PaiNN" / str(seed) / str(fold) / "checkpoint_best.ckpt", mtime)


def _equiformer_ckpt(root, seed=0, fold=0, mtime=None):
    return _touch(root / "results_Equiformer" / str(seed) / str(fold) / "checkpoint_best.ckpt", mtime)


def _geoformer_ckpt(root, name, seed=0, fold=0, mtime=None):
    return _touch(root / "results_Geoformer" / str(seed) / str(fold) / "checkpoints" / name, mtime)


def test_results_폴더가_없으면_체크포인트_목록이_비어있다(tmp_path):
    assert find_checkpoints(tmp_path) == []


def test_PaiNN_checkpoint_best를_찾는다(tmp_path):
    expected = _painn_ckpt(tmp_path)

    found = find_checkpoints(tmp_path)

    assert [(c.base_model, c.path) for c in found] == [("PaiNN", expected)]


def test_Equiformer_checkpoint_best를_찾는다(tmp_path):
    expected = _equiformer_ckpt(tmp_path)

    found = find_checkpoints(tmp_path)

    assert [(c.base_model, c.path) for c in found] == [("Equiformer", expected)]


def test_Geoformer_checkpoints_폴더의_ckpt를_모두_찾는다(tmp_path):
    best = _geoformer_ckpt(tmp_path, "epoch=3-val_loss=0.1000.ckpt")
    last = _geoformer_ckpt(tmp_path, "last.ckpt")

    found = find_checkpoints(tmp_path, base_model="Geoformer")

    assert {c.path for c in found} == {best, last}
    assert {c.base_model for c in found} == {"Geoformer"}


def test_모델을_지정하면_해당_모델의_체크포인트만_반환한다(tmp_path):
    painn = _painn_ckpt(tmp_path)
    _equiformer_ckpt(tmp_path)
    _geoformer_ckpt(tmp_path, "last.ckpt")

    found = find_checkpoints(tmp_path, base_model="PaiNN")

    assert [c.path for c in found] == [painn]


def test_여러_체크포인트는_수정_시각_최신순으로_정렬된다(tmp_path):
    oldest = _painn_ckpt(tmp_path, seed=0, mtime=1_000_000)
    newest = _painn_ckpt(tmp_path, seed=1, mtime=3_000_000)
    middle = _painn_ckpt(tmp_path, seed=2, mtime=2_000_000)

    found = find_checkpoints(tmp_path, base_model="PaiNN")

    assert [c.path for c in found] == [newest, middle, oldest]
    assert [c.modified_time for c in found] == [3_000_000, 2_000_000, 1_000_000]


def test_최신_체크포인트를_반환한다(tmp_path):
    _painn_ckpt(tmp_path, mtime=1_000_000)
    newest = _equiformer_ckpt(tmp_path, mtime=2_000_000)

    latest = get_latest_checkpoint(tmp_path)

    assert latest.path == newest
    assert latest.base_model == "Equiformer"


def test_체크포인트가_없으면_최신_체크포인트는_None이다(tmp_path):
    assert get_latest_checkpoint(tmp_path) is None
    assert get_latest_checkpoint(tmp_path, base_model="PaiNN") is None


def test_학습된_모델이_있으면_True이고_없으면_False이다(tmp_path):
    assert has_trained_model(tmp_path) is False

    _painn_ckpt(tmp_path)

    assert has_trained_model(tmp_path) is True
    assert has_trained_model(tmp_path, base_model="PaiNN") is True
    assert has_trained_model(tmp_path, base_model="Equiformer") is False


def test_지원하지_않는_모델을_지정하면_ValueError가_발생한다(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        find_checkpoints(tmp_path, base_model="GPT")

    for name in SUPPORTED_BASE_MODELS:
        assert name in str(excinfo.value)


def test_다른_모델의_폴더에_있는_ckpt는_무시한다(tmp_path):
    # PaiNN 폴더에 Geoformer 형식(checkpoints/*.ckpt)만 있으면 PaiNN 체크포인트가 아니다.
    _touch(tmp_path / "results_PaiNN" / "0" / "0" / "checkpoints" / "last.ckpt")

    assert has_trained_model(tmp_path, base_model="PaiNN") is False
    assert find_checkpoints(tmp_path) == []
