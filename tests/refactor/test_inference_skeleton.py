"""Plan.md Step 9 [신규 인터페이스], G4: 학습 루프(optimizer/scheduler)에 의존하지
않는 순수 예측 함수 common.inference.predict(model, batch, norm_factor, base_model).

세 모델 모두 task_mean=0, task_std=1(무정규화)로 호출하면 Step 1의 raw forward
golden과 정확히 같아야 한다 — 하나의 통일된 함수로 세 모델을 다 다룰 수 있음을
보여주는 것이 이 테스트의 목적이다.
"""

from types import SimpleNamespace

import torch

from common.adapters import painn_adapter, equiformer_adapter, geoformer_adapter
from common.inference import predict
from tests.support.golden import assert_matches_golden
from tests.support.tiny_batches import make_tiny_pyg_batch, make_tiny_geoformer_batch, n_mode_target_count


def test_predict_PaiNN는_Step1_forward_오라클과_동일하다():
    num_classes = n_mode_target_count()
    args = SimpleNamespace(out_channels=num_classes, radius=5.0, num_basis=8, embed_dim=8, num_layers=1)
    torch.manual_seed(7)
    model = painn_adapter.build(args)
    model.eval()
    batch = make_tiny_pyg_batch()
    norm_factor = [torch.zeros(num_classes), torch.ones(num_classes)]

    output = predict(model, batch, norm_factor, base_model="PaiNN")

    assert_matches_golden("painn_forward_output", output)


def test_predict_Equiformer는_Step1_forward_오라클과_동일하다():
    num_classes = n_mode_target_count()
    args = SimpleNamespace(
        model_name="graph_attention_transformer_nonlinear_l2_e3",
        input_irreps=None, radius=5.0, num_basis=8, out_channels=num_classes, drop_path=0.0,
    )
    torch.manual_seed(7)
    model = equiformer_adapter.build(args)
    model.eval()
    batch = make_tiny_pyg_batch()
    norm_factor = [torch.zeros(num_classes), torch.ones(num_classes)]

    output = predict(model, batch, norm_factor, base_model="Equiformer")

    assert_matches_golden("equiformer_forward_output", output)


def test_predict_Geoformer는_Step1_forward_오라클과_동일하다():
    num_classes = n_mode_target_count()
    args = SimpleNamespace(
        max_z=100, embedding_dim=8, ffn_embedding_dim=16, num_layers=1, num_heads=2,
        cutoff=5.0, num_rbf=8, trainable_rbf=False, norm_type="none", dropout=0.0,
        attention_dropout=0.0, activation_dropout=0.0, activation_function="silu",
        decoder_type="scalar", aggr="sum", dataset_root=None, dataset_arg=None,
        mean=0.0, std=1.0, prior_model=None, num_classes=num_classes, pad_token_id=0,
    )
    torch.manual_seed(7)
    model = geoformer_adapter.build(args)
    model.eval()
    batch = make_tiny_geoformer_batch()
    norm_factor = [torch.zeros(num_classes), torch.ones(num_classes)]

    output = predict(model, batch, norm_factor, base_model="Geoformer")

    assert_matches_golden("geoformer_forward_output", output)


def test_predict_알수없는_base_model은_예외를_낸다():
    try:
        predict(None, None, [0.0, 1.0], base_model="NoSuchModel")
        assert False, "예외가 발생해야 한다"
    except KeyError:
        pass
