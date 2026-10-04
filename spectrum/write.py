import torch
import numpy as np
import pandas as pd
from spectrum.reconstruct import reconstruct_spectrum

# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2025 Hasup Lee. All rights reserved.

def save_spectrum(preds, ids, wrt_file, spectrum_type='FC', kernel_kind='gaussian', beta=2.0):
    spec_y = reconstruct_spectrum(preds, spectrum_type, kernel_kind=kernel_kind, beta=beta)

    x = np.arange(400, 800, 0.5)
    strings = [f"{v:.1f}".rstrip("0").rstrip(".") for v in x]
    line_x = " ".join(strings)

    data = []
    for name, spec in zip(ids, spec_y):
        line_y = " ".join(f"{val:.6f}" for val in spec.tolist())
        data.append((name, line_x, line_y))

    df = pd.DataFrame(data, columns=["molecule_id", "Wavelength(nm)", "Intensity"])
    df.to_csv(wrt_file, index=False)

