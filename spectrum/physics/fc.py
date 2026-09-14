# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2025 Hasup Lee. All rights reserved.

import torch
from spectrum.lineshapes import make_kernel, sigma_to_fwhm
from spectrum.physics.common import coeff_hinge, eps, min_cut, max_cut, min_ev, max_ev, fn_spec_loss

def spectrum_fc(shape_x, S_list, hn_list, cc, E0, kernel_kind='gaussian', beta=2.0):
    kernel = make_kernel(kernel_kind, beta=beta)

    batch_size = E0.shape[0]
    num_x = shape_x.shape[1]
    device = E0.device
    v_range = torch.arange(1, 6, dtype=torch.float32, device=device)  # (5,)
    V = v_range.numel()
    N = S_list.shape[1]

    # base term
    delta0 = shape_x - E0
    fwhm = sigma_to_fwhm(cc)
    L0 = kernel(delta0, fwhm)
    y_base = (E0**3) * L0                              # (B, num_x)

    E0 = E0.squeeze(1)
    fwhm = fwhm.squeeze(1)

    # 1D mode
    v1 = v_range.view(1, V).expand(batch_size, V)      # (B,V)
    v1_exp = v1[:, None, :].expand(-1, N, -1)          # (B,N,V)

    S1  = S_list[:, :, None].expand(-1, -1, V)         # (B,N,V)
    hn1 = hn_list[:, :, None].expand(-1, -1, V)        # (B,N,V)

    bb1 = E0[:, None, None] - v1_exp * hn1             # (B,N,V)
    aa1 = (bb1**3) * (S1**v1_exp) / torch.exp(torch.lgamma(v1_exp + 1))

    # kernel evaluation
    # delta: (B,N,V,num_x), fwhm: (B,1,1,1)
    delta1 = (shape_x[:, None, None, :] - bb1[..., None])
    L1 = kernel(delta1, fwhm.view(batch_size, 1, 1, 1))
    y_1D = (aa1[..., None] * L1).sum(dim=(1, 2))       # (B,num_x)

    # 2D mode
    v1_grid, v2_grid = torch.meshgrid(v_range, v_range, indexing="ij")   # (V,V)
    v1_grid = v1_grid.view(1, 1, V, V).expand(batch_size, -1, -1, -1)    # (B,1,V,V)
    v2_grid = v2_grid.view(1, 1, V, V).expand(batch_size, -1, -1, -1)    # (B,1,V,V)

    S1 = S_list[:, 0].view(batch_size, 1, 1, 1)       # (B,1,1,1)
    S2 = S_list[:, 1:N].view(batch_size, N-1, 1, 1)   # (B,N-1,1,1)
    hn1 = hn_list[:, 0].view(batch_size, 1, 1, 1)     # (B,1,1,1)
    hn2 = hn_list[:, 1:N].view(batch_size, N-1, 1, 1) # (B,N-1,1,1)

    bb2 = E0.view(batch_size, 1, 1, 1) - v1_grid * hn1 - v2_grid * hn2   # (B,N-1,V,V)
    fac_v1 = torch.exp(torch.lgamma(v1_grid + 1))
    fac_v2 = torch.exp(torch.lgamma(v2_grid + 1))

    aa2 = (bb2**3) * (S1**v1_grid) * (S2**v2_grid) / (fac_v1 * fac_v2)   # (B,N-1,V,V)

    delta2 = shape_x.view(batch_size, 1, 1, 1, num_x) - bb2[..., None]    # (B,N-1,V,V,num_x)
    L2 = kernel(delta2, fwhm.view(batch_size, 1, 1, 1, 1))
    y_2D = (aa2[..., None] * L2).sum(dim=(1, 2, 3))                       # (B,num_x)

    #Sum 1D and 2D
    y = y_base + y_1D + y_2D  # (B,num_x)
    ymax = torch.amax(y, dim=1, keepdim=True)
    y = y / (ymax+eps)

    return y

def fc_loss(shape_x, shape_y, params, loss_type='MAE', line_shape='gaussian', beta=2.0):
    n_S = (params.shape[1]-2)//2
    S_list = params[:, 0:n_S]
    hn_list = params[:, n_S+2:2*n_S+2]
    cc = params[:, n_S:n_S+1]
    E0 = params[:, n_S+1:n_S+2]

    hinge_loss1 = torch.sum(
        torch.clamp(min_cut - S_list, min=0.0)**2+
        torch.clamp(S_list - max_cut, min=0.0)**2,
        dim=1,
    )  # minimum 0.01 of S_list
    hinge_loss2 = torch.sum(
        torch.clamp(min_cut - hn_list, min=0.0)**2+
        torch.clamp(hn_list - max_cut, min=0.0)**2,
        dim=1,
    )  # minimum 0.01 of hn_list
    hinge_loss3 = torch.sum(
        torch.clamp(min_ev - E0, min=0.0)**2 + torch.clamp(E0 - max_ev, min=0.0)**2,
        dim=1,
    )  # E0 is in 1.6~3.2
    hinge_loss4 = torch.sum(
        torch.clamp(min_cut - cc, min=0.0)**2+
        torch.clamp(cc - max_cut, min=0.0)**2,
        dim=1,
    )  # minimum 0.01 of cc
    hinge_loss = coeff_hinge * (hinge_loss1 + hinge_loss2 + hinge_loss3 + hinge_loss4)

    S_list = torch.clamp(S_list, min=min_cut, max=max_cut)
    hn_list = torch.clamp(hn_list, min=min_cut, max=max_cut)
    cc = torch.clamp(cc, min=min_cut, max=max_cut)
    E0 = torch.clamp(E0, min=min_ev, max=max_ev)

    y = spectrum_fc(shape_x, S_list, hn_list, cc, E0, kernel_kind=line_shape, beta=beta)
    val_loss = fn_spec_loss(y, shape_y, loss_type=loss_type)
    return torch.sum(val_loss + hinge_loss)
