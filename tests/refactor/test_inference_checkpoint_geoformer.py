"""Plan.md Step 2D [신규 인터페이스], G2: Geoformer(Lightning) 체크포인트 로드 + 곡선 복원.

load_checkpoint()가 아직 Geoformer를 지원하지 않으므로 실패해야 한다 (RED).
체크포인트는 실제 train_Geoformer가 만든 Lightning ckpt와 같은 구조로 만든다:
  - state_dict: 모든 키가 'model.' 접두사 (LNNP의 self.model), mean/std 버퍼 포함
  - hyper_parameters: Namespace가 아닌 일반 dict, spectrum_type 대신 spec_loss_type
Geoformer 모델은 logits * std + mean을 내부에서 적용하므로 출력이 이미 역정규화되어 있다.
새 동작 호출은 각 테스트 안에서 하여 개별 실패로 확인한다.
"""

from types import SimpleNamespace

import torch

from common.adapters import geoformer_adapter
from common.inference import predict
from spectrum.reconstruct import reconstruct_spectrum
from tests.support.tiny_batches import make_tiny_geoformer_batch

FC_TARGETS = ['S1', 'S2', 'S3', 'C', 'E0', 'h1', 'h2', 'h3']
# 모델 내부 mean/std 버퍼: 임의 초기화 모델의 출력이 물리적으로 말이 되도록(NaN 방지) FC 파라미터 근처의 값을 쓴다.
MODEL_MEAN = torch.tensor([1.0, 0.6, 0.3, 0.08, 2.30, 0.18, 0.15, 0.10])
MODEL_STD = torch.full((8,), 0.01)


def _hyper_parameters() -> dict:
    return dict(
        max_z=100, embedding_dim=8, ffn_embedding_dim=16, num_layers=1, num_heads=2,
        cutoff=5.0, num_rbf=8, trainable_rbf=False, norm_type="none", dropout=0.0,
        attention_dropout=0.0, activation_dropout=0.0, activation_function="silu",
        decoder_type="scalar", aggr="sum", dataset_root=None, dataset_arg=list(FC_TARGETS),
        mean=MODEL_MEAN, std=MODEL_STD, prior_model=None, num_classes=len(FC_TARGETS), pad_token_id=0,
        spec_loss_type='FC', lineshape='gaussian', beta=2.0, n_mode=3,
        standardize=True, splits="no/such/splits.npz",
    )


def _save_tiny_geoformer_ckpt(tmp_path):
    """(ckpt 경로, 원본 모델). standardize=True + 존재하지 않는 split 경로로 저장한다."""
    hparams = _hyper_parameters()
    torch.manual_seed(7)
    model = geoformer_adapter.build(SimpleNamespace(**hparams))
    model.eval()
    lightning_style = {
        'epoch': 0,
        'state_dict': {f'model.{key}': value for key, value in model.state_dict().items()},
        'hyper_parameters': hparams,
    }
    path = tmp_path / "last.ckpt"
    torch.save(lightning_style, path)
    return path, model


def test_Geoformer_Lightning_체크포인트를_로드하면_원본_모델과_같은_출력을_낸다(tmp_path):
    from common.inference import load_checkpoint

    path, original = _save_tiny_geoformer_ckpt(tmp_path)
    batch = make_tiny_geoformer_batch()
    norm_factor = [torch.zeros(8), torch.ones(8)]

    loaded = load_checkpoint(path, "Geoformer")

    assert loaded.base_model == "Geoformer"
    assert loaded.model.training is False
    expected = predict(original, batch, norm_factor, base_model="Geoformer")
    actual = predict(loaded.model, batch, norm_factor, base_model="Geoformer")
    assert torch.equal(actual, expected)


def test_Geoformer는_모델이_이미_역정규화하므로_norm_factor가_0과_1이다(tmp_path):
    # standardize=True이고 split 경로가 존재하지 않는다: 재계산을 시도하면 예외가 나므로,
    # 통과한다면 데이터셋을 읽지 않은 것이다. 또 hyper_parameters의 mean/std(0이 아닌 값)를
    # norm_factor로 쓰면 이중 역정규화가 되므로 [0, 1]이어야 한다.
    from common.inference import load_checkpoint

    path, _ = _save_tiny_geoformer_ckpt(tmp_path)

    loaded = load_checkpoint(path, "Geoformer")

    mean, std = loaded.norm_factor
    assert torch.equal(mean, torch.zeros(8))
    assert torch.equal(std, torch.ones(8))


def test_Geoformer_hyper_parameters는_속성_접근이_가능한_args로_복원되고_spectrum_type은_spec_loss_type에서_온다(tmp_path):
    from common.inference import load_checkpoint

    path, _ = _save_tiny_geoformer_ckpt(tmp_path)

    loaded = load_checkpoint(path, "Geoformer")

    assert loaded.args.spectrum_type == 'FC'
    assert loaded.args.lineshape == 'gaussian'
    assert loaded.args.beta == 2.0
    assert loaded.args.embedding_dim == 8


def test_Geoformer_predict_curves는_모델이_출력한_파라미터로_곡선을_복원한다(tmp_path):
    from common.inference import load_checkpoint, predict_curves

    path, original = _save_tiny_geoformer_ckpt(tmp_path)
    batch = make_tiny_geoformer_batch()
    loaded = load_checkpoint(path, "Geoformer")

    curves = predict_curves(loaded, batch)

    with torch.no_grad():
        params = geoformer_adapter.forward(original, batch)  # 이미 역정규화된 파라미터
    expected = reconstruct_spectrum(params, 'FC', kernel_kind='gaussian', beta=2.0)
    assert curves.shape == (2, 800)
    assert torch.isfinite(curves).all()
    assert torch.allclose(curves, expected, atol=1e-6)


def test_Geoformer_norm_factor를_직접_주면_그대로_사용한다(tmp_path):
    from common.inference import load_checkpoint

    path, _ = _save_tiny_geoformer_ckpt(tmp_path)
    given = [torch.full((8,), 0.5), torch.full((8,), 2.0)]

    loaded = load_checkpoint(path, "Geoformer", norm_factor=given)

    assert torch.equal(loaded.norm_factor[0], given[0])
    assert torch.equal(loaded.norm_factor[1], given[1])
