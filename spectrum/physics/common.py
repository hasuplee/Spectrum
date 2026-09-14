# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2025 Hasup Lee. All rights reserved.

import torch

coeff_hinge = 1e6
eps = 1e-8
min_cut = 0.01
max_cut = 1.0
min_ev = 1.6
max_ev = 3.2

def fn_spec_loss(y_pred, y_true, loss_type='MAE'):
    if loss_type == 'MAE':
        val_loss = torch.mean(torch.abs(y_pred-y_true), dim=1)
    elif loss_type == 'MSE':
        val_loss = torch.mean(torch.square(y_pred-y_true), dim=1)
    else:
        raise Exception('Unknown loss type')
    return val_loss
