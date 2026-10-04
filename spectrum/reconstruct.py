import torch
from spectrum.physics import spectrum_gmm, spectrum_fc

# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2025 Hasup Lee. All rights reserved.

def wavelength_grid_nm():
    return torch.arange(400.0, 800.0, 0.5)

def reconstruct_spectrum(preds, spectrum_type='FC', kernel_kind='gaussian', beta=2.0):
    nm2ev = 1240.0
    spec_x = wavelength_grid_nm()
    spec_x = nm2ev / spec_x
    spec_x = spec_x.unsqueeze(0).expand(len(preds), -1)
    if spectrum_type == 'Naive':
        preds = preds.numpy()
        mins = preds.min(axis=1, keepdims=True)
        maxs = preds.max(axis=1, keepdims=True)
        spec_y = torch.from_numpy((preds-mins) / (maxs-mins))
    elif spectrum_type == 'GMM':
        spec_y = spectrum_gmm(spec_x, preds[:,0:1], preds[:,1:2], 
                preds[:,2:3], preds[:,3:4], preds[:,4:5], 
                preds[:,5:6], preds[:,6:7], preds[:,7:8])
    elif spectrum_type == 'FC':
        n_S = (preds.shape[1]-2)//2
        spec_y = spectrum_fc(spec_x, preds[:,0:n_S], preds[:,n_S+2:2*n_S+2], 
                preds[:,n_S:n_S+1], preds[:,n_S+1:n_S+2],
                kernel_kind=kernel_kind, beta=beta)
    else:
        raise Exception("Undefined spectrum type")
    return spec_y
