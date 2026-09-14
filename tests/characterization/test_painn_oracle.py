"""Plan.md Step 1: characterization tests for the PaiNN training path.

These call the same functions production training uses (engine.train_one_step
/ engine.evaluate, and the PaiNN wrapper in train_PaiNN.py) against a tiny
synthetic batch, and pin their numeric output as a regression oracle.
"""

import torch

from engine import evaluate, train_one_step
from train_PaiNN import PaiNN
from tests.support.golden import assert_matches_golden, assert_matches_golden_json
from tests.support.tiny_batches import make_tiny_pyg_batch, n_mode_target_count


class _DummyLogger:
    def info(self, msg):
        pass


def _build_model(num_classes: int) -> PaiNN:
    torch.manual_seed(7)
    return PaiNN(out_channels=num_classes, cutoff=5.0, hidden_channels=8, num_layers=1, num_rbf=8)


def test_PaiNN_모델_파라미터_이름과_shape가_고정된다():
    model = _build_model(num_classes=n_mode_target_count())

    structure = [[name, list(p.shape)] for name, p in model.named_parameters()]

    assert_matches_golden_json("painn_param_structure", structure)


def test_PaiNN_forward_출력이_고정된다():
    batch = make_tiny_pyg_batch()
    model = _build_model(num_classes=n_mode_target_count())
    model.eval()

    with torch.no_grad():
        output = model(f_in=batch.x, pos=batch.pos, batch=batch.batch, node_atom=batch.z)

    assert_matches_golden("painn_forward_output", output)


def test_PaiNN_1스텝_학습후_gradient_norm과_파라미터_norm이_고정된다():
    batch = make_tiny_pyg_batch()
    num_classes = n_mode_target_count()
    model = _build_model(num_classes=num_classes)
    task_mean = torch.zeros(num_classes)
    task_std = torch.ones(num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss()

    train_mae = train_one_step(
        model=model,
        criterion=criterion,
        norm_factor=[task_mean, task_std],
        data_loader=[batch],
        optimizer=optimizer,
        device=torch.device("cpu"),
        clip_grad=1.0,
        input_step=1,
        print_freq=1_000_000,
        loss_type="MAE",
        spec_type="FC",
        line_shape="gaussian",
        beta=2.0,
        logger=_DummyLogger(),
    )

    grad_norm = torch.linalg.vector_norm(
        torch.stack([p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None])
    )
    param_norm = torch.linalg.vector_norm(
        torch.stack([p.detach().norm(2) for p in model.parameters()])
    )

    assert_matches_golden("painn_train_mae", torch.tensor([train_mae]))
    assert_matches_golden("painn_grad_norm_after_1_step", grad_norm.reshape(1))
    assert_matches_golden("painn_param_norm_after_1_step", param_norm.reshape(1))


def test_PaiNN_evaluate가_고정된다():
    batch = make_tiny_pyg_batch()
    num_classes = n_mode_target_count()
    model = _build_model(num_classes=num_classes)
    task_mean = torch.zeros(num_classes)
    task_std = torch.ones(num_classes)

    mae, loss, preds, ids = evaluate(
        model,
        norm_factor=[task_mean, task_std],
        data_loader=[batch],
        device=torch.device("cpu"),
        loss_type="MAE",
        spec_type="FC",
        line_shape="gaussian",
        beta=2.0,
        logger=_DummyLogger(),
    )

    assert ids == ["tiny_mol_0", "tiny_mol_1"]
    assert_matches_golden("painn_evaluate_mae_loss", torch.tensor([mae, loss]))
    assert_matches_golden("painn_evaluate_preds", preds)
