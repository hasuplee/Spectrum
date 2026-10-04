"""Plan.md Step 2B [신규 인터페이스], G2: PaiNN 체크포인트 로드 + 예측 + 곡선 복원.

common.inference의 load_checkpoint()/predict_curves()가 아직 없으므로 실패해야 한다 (RED).
새 함수 import는 각 테스트 안에서 하여, 테스트마다 개별 실패로 확인한다.
tiny PaiNN(embed_dim=8, num_layers=1, num_basis=8)을 tmp_path에 저장해 사용한다.
"""

from argparse import Namespace

import pytest
import torch

from common.adapters import painn_adapter
from common.adapters._shared import engine_style_forward
from common.data import load_dataset_splits
from common.inference import predict
from spectrum.reconstruct import reconstruct_spectrum
from tests.support.tiny_batches import make_tiny_pyg_batch

FC_TARGETS = ['S1', 'S2', 'S3', 'C', 'E0', 'h1', 'h2', 'h3']
REAL_SPLIT_NPZ = 'IrDB/raw/CV811/splits.0.0.npz'


def _tiny_args(standardize=False, split_index_npz=REAL_SPLIT_NPZ) -> Namespace:
    """실제 train_PaiNN.py가 ckpt에 저장하는 args와 같은 필드를 가진 Namespace."""
    return Namespace(
        out_channels=len(FC_TARGETS), radius=5.0, num_basis=8, embed_dim=8, num_layers=1,
        targets=list(FC_TARGETS), standardize=standardize, spectrum_type='FC',
        lineshape='gaussian', beta=2.0, n_mode=3, data_path='IrDB',
        split_index_npz=split_index_npz, seed=0, distributed=False,
    )


def _save_tiny_painn_ckpt(tmp_path, standardize=False, split_index_npz=REAL_SPLIT_NPZ):
    args = _tiny_args(standardize, split_index_npz)
    torch.manual_seed(7)
    model = painn_adapter.build(args)
    model.eval()
    path = tmp_path / "checkpoint_best.ckpt"
    torch.save({'model': model.state_dict(), 'optimizer': {}, 'args': args}, path)
    return path, model, args


def test_PaiNN_체크포인트를_로드하면_원본_모델과_같은_출력을_낸다(tmp_path):
    from common.inference import load_checkpoint

    path, original, _ = _save_tiny_painn_ckpt(tmp_path)
    batch = make_tiny_pyg_batch()
    norm_factor = [torch.zeros(8), torch.ones(8)]

    loaded = load_checkpoint(path, "PaiNN")

    assert loaded.base_model == "PaiNN"
    assert loaded.model.training is False
    expected = predict(original, batch, norm_factor, base_model="PaiNN")
    actual = predict(loaded.model, batch, norm_factor, base_model="PaiNN")
    assert torch.equal(actual, expected)


def test_standardize가_False이면_norm_factor는_0과_1이다(tmp_path):
    from common.inference import load_checkpoint

    path, _, _ = _save_tiny_painn_ckpt(tmp_path, standardize=False)

    loaded = load_checkpoint(path, "PaiNN")

    mean, std = loaded.norm_factor
    assert torch.equal(mean, torch.zeros(8))
    assert torch.equal(std, torch.ones(8))


def test_standardize가_True이면_학습_split에서_평균_표준편차를_다시_계산한다(tmp_path):
    from common.inference import load_checkpoint

    path, _, args = _save_tiny_painn_ckpt(tmp_path, standardize=True)
    _, _, _, task_mean, task_std = load_dataset_splits(args)

    loaded = load_checkpoint(path, "PaiNN")

    mean, std = loaded.norm_factor
    assert torch.allclose(mean, torch.tensor(task_mean, dtype=torch.float32))
    assert torch.allclose(std, torch.tensor(task_std, dtype=torch.float32))


def test_norm_factor를_직접_주면_데이터셋을_읽지_않는다(tmp_path):
    from common.inference import load_checkpoint

    # split 파일이 존재하지 않으므로, 재계산을 시도하면 예외가 난다.
    path, _, _ = _save_tiny_painn_ckpt(tmp_path, standardize=True, split_index_npz="no/such/splits.npz")
    given = [torch.full((8,), 0.5), torch.full((8,), 2.0)]

    loaded = load_checkpoint(path, "PaiNN", norm_factor=given)

    assert torch.equal(loaded.norm_factor[0], given[0])
    assert torch.equal(loaded.norm_factor[1], given[1])


def test_predict_curves는_역정규화한_파라미터로_곡선을_복원한다(tmp_path):
    from common.inference import load_checkpoint, predict_curves

    path, original, _ = _save_tiny_painn_ckpt(tmp_path)
    batch = make_tiny_pyg_batch()
    # 임의 초기화 모델의 출력이 물리적으로 말이 되도록(NaN 방지) FC 파라미터 근처의 평균/작은 표준편차를 주입한다.
    mean = torch.tensor([1.0, 0.6, 0.3, 0.08, 2.30, 0.18, 0.15, 0.10])
    std = torch.full((8,), 0.01)
    loaded = load_checkpoint(path, "PaiNN", norm_factor=[mean, std])

    curves = predict_curves(loaded, batch)

    with torch.no_grad():
        raw = engine_style_forward(original, batch).view(2, -1)
    expected = reconstruct_spectrum(raw * std + mean, 'FC', kernel_kind='gaussian', beta=2.0)
    assert curves.shape == (2, 800)
    assert torch.isfinite(curves).all()
    assert torch.allclose(curves, expected, atol=1e-6)


def test_지원하지_않는_모델이면_ValueError가_발생한다(tmp_path):
    from common.inference import load_checkpoint

    path, _, _ = _save_tiny_painn_ckpt(tmp_path)

    with pytest.raises(ValueError, match="GPT"):
        load_checkpoint(path, "GPT")


def test_체크포인트_파일이_없으면_FileNotFoundError가_발생한다(tmp_path):
    from common.inference import load_checkpoint

    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "없는파일.ckpt", "PaiNN")
