# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2025 Hasup Lee. All rights reserved.

import torch
from spectrum.physics.common import coeff_hinge, eps, min_cut, max_cut, min_ev, max_ev, fn_spec_loss

def spectrum_gmm(shape_x, a2, a3, b1, b2, b3, c1, c2, c3):
    y1 = torch.exp(-((shape_x - b1) ** 2) / (2.0 * (c1**2)))
    y2 = a2 * torch.exp(-((shape_x - b2) ** 2) / (2.0 * (c2**2)))
    y3 = a3 * torch.exp(-((shape_x - b3) ** 2) / (2.0 * (c3**2)))

    y = y1 + y2 + y3
    ymax = torch.amax(y, dim=1, keepdim=True)
    y = y / (ymax+eps)
    return y

def gmm_loss(shape_x, shape_y, params, loss_type='MAE', line_shape='none', beta='none'):
    a2, a3 = (params[:, 0:1], params[:, 1:2])
    b1, b2, b3 = (params[:, 2:3], params[:, 3:4], params[:, 4:5])
    c1, c2, c3 = (params[:, 5:6], params[:, 6:7], params[:, 7:8])

    a_tensor = torch.concat([a2, a3], dim=1)
    b_tensor = torch.concat([b1, b2, b3], dim=1)
    c_tensor = torch.concat([c1, c2, c3], dim=1)

    hinge_loss1 = torch.sum(
        torch.clamp(min_cut - a_tensor, min=0.0)**2+
        torch.clamp(a_tensor - max_cut, min=0.0)**2,
        dim=1,
    )  # minimum 0.01
    hinge_loss2 = torch.sum(
        torch.clamp(min_cut - c_tensor, min=0.0)**2+
        torch.clamp(c_tensor - max_cut, min=0.0)**2,
        dim=1,
    )  # minimum 0.01
    hinge_loss3 = torch.sum(
        torch.clamp(min_ev - b_tensor, min=0.0)**2
        + torch.clamp(b_tensor - max_ev, min=0.0)**2,
        dim=1,
    )  # 1.6~3.2
    hinge_loss = coeff_hinge * (hinge_loss1 + hinge_loss2 + hinge_loss3)

    for var in [a2, a3, c1, c2, c3]:
        var.clamp_(min=min_cut, max=max_cut)
    for var in [b1, b2, b3]:
        var.clamp_(min=min_ev, max=max_ev)

    y = spectrum_gmm(shape_x, a2, a3, b1, b2, b3, c1, c2, c3)
    val_loss = fn_spec_loss(y, shape_y, loss_type=loss_type)
    return torch.sum(val_loss + hinge_loss)
