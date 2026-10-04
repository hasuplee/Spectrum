"""Plan.md Step 2C [신규 인터페이스], G2: Equiformer 체크포인트 로드 + 곡선 복원.

load_checkpoint()가 아직 Equiformer를 지원하지 않으므로 실패해야 한다 (RED).
Equiformer는 모델 1개 생성에 약 8초가 걸리므로, 체크포인트는 모듈 범위에서 한 번만 만들고
load_checkpoint 결과도 캐시해서 테스트 간에 재사용한다.
새 동작 호출(load_checkpoint)은 각 테스트 안에서 하여 개별 실패로 확인한다.
"""

from argparse import Namespace

import pytest
import torch

from common.adapters import equiformer_adapter
from common.adapters._shared import engine_style_forward
from common.inference import predict
from spectrum.reconstruct import reconstruct_spectrum
from tests.support.tiny_batches import make_tiny_pyg_batch

FC_TARGETS = ['S1', 'S2', 'S3', 'C', 'E0', 'h1', 'h2', 'h3']
# 임의 초기화 모델의 출력이 물리적으로 말이 되도록(NaN 방지) FC 파라미터 근처의 평균/작은 표준편차를 ckpt args에 저장한다.
TASK_MEAN = torch.tensor([1.0, 0.6, 0.3, 0.08, 2.30, 0.18, 0.15, 0.10])
TASK_STD = torch.full((8,), 0.01)

_loaded_cache = {}


@pytest.fixture(scope="module")
def equiformer_ckpt(tmp_path_factory):
    """(ckpt 경로, 원본 모델). split 경로는 일부러 존재하지 않는 값으로 두고 standardize=True로 저장한다."""
    args = Namespace(
        model_name="graph_attention_transformer_nonlinear_l2_e3", input_irreps=None,
        radius=5.0, num_basis=8, out_channels=len(FC_TARGETS), drop_path=0.0,
        task_mean=TASK_MEAN, task_std=TASK_STD,
        targets=list(FC_TARGETS), standardize=True, spectrum_type='FC', lineshape='gaussian',
        beta=2.0, n_mode=3, data_path='IrDB', split_index_npz="no/such/splits.npz",
        seed=0, distributed=False,
    )
    torch.manual_seed(7)
    model = equiformer_adapter.build(args)
    model.eval()
    path = tmp_path_factory.mktemp("equiformer_ckpt") / "checkpoint_best.ckpt"
    torch.save({'model': model.state_dict(), 'optimizer': {}, 'args': args}, path)
    return path, model


def _load(path):
    """load_checkpoint 결과를 캐시한다 (Equiformer 모델 생성이 느리므로)."""
    from common.inference import load_checkpoint

    if path not in _loaded_cache:
        _loaded_cache[path] = load_checkpoint(path, "Equiformer")
    return _loaded_cache[path]


def test_Equiformer_체크포인트를_로드하면_원본_모델과_같은_출력을_낸다(equiformer_ckpt):
    path, original = equiformer_ckpt
    batch = make_tiny_pyg_batch()
    norm_factor = [torch.zeros(8), torch.ones(8)]

    loaded = _load(path)

    assert loaded.base_model == "Equiformer"
    assert loaded.model.training is False
    expected = predict(original, batch, norm_factor, base_model="Equiformer")
    actual = predict(loaded.model, batch, norm_factor, base_model="Equiformer")
    assert torch.equal(actual, expected)


def test_Equiformer는_체크포인트_args의_task_mean_std를_norm_factor로_쓴다(equiformer_ckpt):
    # standardize=True인데 split 경로가 존재하지 않는다: 재계산을 시도하면 예외가 나므로,
    # 통과한다면 데이터셋을 읽지 않고 ckpt에 저장된 값을 쓴 것이다.
    path, _ = equiformer_ckpt

    loaded = _load(path)

    mean, std = loaded.norm_factor
    assert torch.allclose(mean, TASK_MEAN)
    assert torch.allclose(std, TASK_STD)


def test_Equiformer_predict_curves는_역정규화한_파라미터로_곡선을_복원한다(equiformer_ckpt):
    from common.inference import predict_curves

    path, original = equiformer_ckpt
    batch = make_tiny_pyg_batch()
    loaded = _load(path)

    curves = predict_curves(loaded, batch)

    with torch.no_grad():
        raw = engine_style_forward(original, batch).view(2, -1)
    expected = reconstruct_spectrum(raw * TASK_STD + TASK_MEAN, 'FC', kernel_kind='gaussian', beta=2.0)
    assert curves.shape == (2, 800)
    assert torch.isfinite(curves).all()
    assert torch.allclose(curves, expected, atol=1e-6)
