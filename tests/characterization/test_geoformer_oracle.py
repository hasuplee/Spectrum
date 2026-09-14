"""Plan.md Step 1: characterization tests for the Geoformer training path.

Geoformer is driven through PyTorch Lightning (geoformer/module.py's LNNP) in
production, but its actual numerics live in
geoformer.model.modeling_geoformer.create_model + spectrum.physics.fc_loss,
exactly like LNNP.forward / LNNP.spectrum_step call them (geoformer/module.py:87-164).
These tests call that same core computation directly against a tiny synthetic
batch, without going through Lightning's Trainer, and pin the numeric output
as a regression oracle.
"""

from types import SimpleNamespace

import torch

from geoformer.model.modeling_geoformer import create_model
from spectrum.physics import fc_loss
from tests.support.golden import assert_matches_golden, assert_matches_golden_json
from tests.support.tiny_batches import make_tiny_geoformer_batch, n_mode_target_count


def _build_config(num_classes: int) -> SimpleNamespace:
    return SimpleNamespace(
        max_z=100,
        embedding_dim=8,
        ffn_embedding_dim=16,
        num_layers=1,
        num_heads=2,
        cutoff=5.0,
        num_rbf=8,
        trainable_rbf=False,
        norm_type="none",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_function="silu",
        decoder_type="scalar",
        aggr="sum",
        dataset_root=None,
        dataset_arg=None,
        mean=0.0,
        std=1.0,
        prior_model=None,
        num_classes=num_classes,
        pad_token_id=0,
    )


def _build_model(num_classes: int):
    torch.manual_seed(7)
    return create_model(_build_config(num_classes))


def test_Geoformer_모델_파라미터_이름과_shape가_고정된다():
    model = _build_model(num_classes=n_mode_target_count())

    structure = [[name, list(p.shape)] for name, p in model.named_parameters()]

    assert_matches_golden_json("geoformer_param_structure", structure)


def test_Geoformer_forward_출력이_고정된다():
    batch = make_tiny_geoformer_batch()
    model = _build_model(num_classes=n_mode_target_count())
    model.eval()

    with torch.no_grad():
        output = model(z=batch["z"], pos=batch["pos"])
    output = output.view(output.shape[0], -1)

    assert_matches_golden("geoformer_forward_output", output)


def test_Geoformer_1스텝_학습후_gradient_norm과_파라미터_norm이_고정된다():
    batch = make_tiny_geoformer_batch()
    model = _build_model(num_classes=n_mode_target_count())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    pred = model(z=batch["z"], pos=batch["pos"])
    pred = pred.view(pred.shape[0], -1)
    loss = fc_loss(batch["spec_x"], batch["spec_y"], pred, loss_type="MAE", line_shape="gaussian", beta=2.0)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    grad_norm = torch.linalg.vector_norm(
        torch.stack([p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None])
    )
    param_norm = torch.linalg.vector_norm(
        torch.stack([p.detach().norm(2) for p in model.parameters()])
    )

    assert_matches_golden("geoformer_loss_after_1_step", loss.detach().reshape(1))
    assert_matches_golden("geoformer_grad_norm_after_1_step", grad_norm.reshape(1))
    assert_matches_golden("geoformer_param_norm_after_1_step", param_norm.reshape(1))
