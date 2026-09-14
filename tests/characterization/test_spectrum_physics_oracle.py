"""Plan.md Step 1: characterization tests for spectrum/loss.py (FC/GMM physics).

These tests pin the legacy spectrum_fc/fc_loss/spectrum_gmm/gmm_loss outputs
for a fixed synthetic input so that Step 4 (spectrum/ internal reorganization)
can be verified to be behavior-preserving.
"""

import torch

from spectrum.loss import fc_loss, gmm_loss
from tests.support.golden import assert_matches_golden

NUM_X = 20
BATCH = 2


def _fixed_shape_x():
    return torch.linspace(1.7, 2.9, NUM_X).unsqueeze(0).repeat(BATCH, 1)


def _fixed_shape_y():
    generator = torch.Generator().manual_seed(4242)
    return torch.rand(BATCH, NUM_X, generator=generator)


def _fixed_fc_params():
    # [S1, S2, S3, cc, E0, h1, h2, h3] per row
    generator = torch.Generator().manual_seed(4243)
    return torch.rand(BATCH, 8, generator=generator) * 0.4 + 0.3


def _fixed_gmm_params():
    # [a2, a3, b1, b2, b3, c1, c2, c3] per row
    generator = torch.Generator().manual_seed(4244)
    return torch.rand(BATCH, 8, generator=generator) * 0.4 + 0.3


def test_FC_loss_값이_고정된다():
    shape_x, shape_y, params = _fixed_shape_x(), _fixed_shape_y(), _fixed_fc_params()

    loss = fc_loss(shape_x, shape_y, params, loss_type="MAE", line_shape="gaussian", beta=2.0)

    assert_matches_golden("fc_loss_value", loss.reshape(1))


def test_GMM_loss_값이_고정된다():
    shape_x, shape_y, params = _fixed_shape_x(), _fixed_shape_y(), _fixed_gmm_params()

    loss = gmm_loss(shape_x, shape_y, params, loss_type="MAE")

    assert_matches_golden("gmm_loss_value", loss.reshape(1))


def test_FC_loss는_MSE_옵션에서도_고정된다():
    shape_x, shape_y, params = _fixed_shape_x(), _fixed_shape_y(), _fixed_fc_params()

    loss = fc_loss(shape_x, shape_y, params, loss_type="MSE", line_shape="lorentzian", beta=2.0)

    assert_matches_golden("fc_loss_value_mse_lorentzian", loss.reshape(1))
