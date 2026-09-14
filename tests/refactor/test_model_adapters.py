"""Plan.md Step 3 [신규 인터페이스]: common/adapters 의 model registry/adapter.

각 adapter의 build(args)/forward(model, batch)가, Step 1에서 고정해둔
legacy 오라클(golden fixture)과 정확히 같은 forward 출력을 내는지 확인한다.
같은 seed(7)와 같은 tiny 설정으로 구성하면, 어댑터를 거치더라도 수치가
달라져서는 안 된다 (G1 behavior preservation).
"""

from types import SimpleNamespace

import torch

from common.adapters import ADAPTERS, get_adapter, painn_adapter, equiformer_adapter, geoformer_adapter
from tests.support.golden import assert_matches_golden
from tests.support.tiny_batches import make_tiny_pyg_batch, make_tiny_geoformer_batch, n_mode_target_count


def test_registry에_세_모델_모두_등록되어_있다():
    assert set(ADAPTERS) == {"PaiNN", "Equiformer", "Geoformer"}
    assert get_adapter("PaiNN") is painn_adapter
    assert get_adapter("Equiformer") is equiformer_adapter
    assert get_adapter("Geoformer") is geoformer_adapter


def test_registry에_없는_이름은_예외를_낸다():
    try:
        get_adapter("NoSuchModel")
        assert False, "예외가 발생해야 한다"
    except KeyError:
        pass


def test_PaiNN_adapter_forward는_Step1_오라클과_동일하다():
    num_classes = n_mode_target_count()
    args = SimpleNamespace(out_channels=num_classes, radius=5.0, num_basis=8, embed_dim=8, num_layers=1)
    torch.manual_seed(7)
    model = painn_adapter.build(args)
    model.eval()
    batch = make_tiny_pyg_batch()

    with torch.no_grad():
        output = painn_adapter.forward(model, batch)

    assert_matches_golden("painn_forward_output", output)


def test_Equiformer_adapter_forward는_Step1_오라클과_동일하다():
    num_classes = n_mode_target_count()
    args = SimpleNamespace(
        model_name="graph_attention_transformer_nonlinear_l2_e3",
        input_irreps=None,
        radius=5.0,
        num_basis=8,
        out_channels=num_classes,
        drop_path=0.0,
    )
    torch.manual_seed(7)
    model = equiformer_adapter.build(args)
    model.eval()
    batch = make_tiny_pyg_batch()

    with torch.no_grad():
        output = equiformer_adapter.forward(model, batch)

    assert_matches_golden("equiformer_forward_output", output)


def test_Geoformer_adapter_forward는_Step1_오라클과_동일하다():
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

    with torch.no_grad():
        output = geoformer_adapter.forward(model, batch)

    assert_matches_golden("geoformer_forward_output", output)
