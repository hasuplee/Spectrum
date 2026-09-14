"""Plan.md Step 4 [구조 이동]: spectrum/loss.py -> spectrum/physics/{fc,gmm,common}.py.

Reuses the exact same fixed inputs as
tests/characterization/test_spectrum_physics_oracle.py, but imports from the
new spectrum.physics location, and compares against the same Step 1 golden
fixtures. If the move preserved the logic exactly, these must match.
"""

import torch

from spectrum.physics import fc_loss, gmm_loss
from tests.support.golden import assert_matches_golden

NUM_X = 20
BATCH = 2


def _fixed_shape_x():
    return torch.linspace(1.7, 2.9, NUM_X).unsqueeze(0).repeat(BATCH, 1)


def _fixed_shape_y():
    generator = torch.Generator().manual_seed(4242)
    return torch.rand(BATCH, NUM_X, generator=generator)


def _fixed_fc_params():
    generator = torch.Generator().manual_seed(4243)
    return torch.rand(BATCH, 8, generator=generator) * 0.4 + 0.3


def _fixed_gmm_params():
    generator = torch.Generator().manual_seed(4244)
    return torch.rand(BATCH, 8, generator=generator) * 0.4 + 0.3


def test_spectrum_physics_새_위치의_FC_loss가_Step1_오라클과_동일하다():
    shape_x, shape_y, params = _fixed_shape_x(), _fixed_shape_y(), _fixed_fc_params()

    loss = fc_loss(shape_x, shape_y, params, loss_type="MAE", line_shape="gaussian", beta=2.0)

    assert_matches_golden("fc_loss_value", loss.reshape(1))


def test_spectrum_physics_새_위치의_GMM_loss가_Step1_오라클과_동일하다():
    shape_x, shape_y, params = _fixed_shape_x(), _fixed_shape_y(), _fixed_gmm_params()

    loss = gmm_loss(shape_x, shape_y, params, loss_type="MAE")

    assert_matches_golden("gmm_loss_value", loss.reshape(1))


def test_spectrum_physics_새_위치의_FC_loss_MSE_lorentzian도_동일하다():
    shape_x, shape_y, params = _fixed_shape_x(), _fixed_shape_y(), _fixed_fc_params()

    loss = fc_loss(shape_x, shape_y, params, loss_type="MSE", line_shape="lorentzian", beta=2.0)

    assert_matches_golden("fc_loss_value_mse_lorentzian", loss.reshape(1))
