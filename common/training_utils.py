"""Architecture-independent helpers shared by train_PaiNN.py and
train_Equiformer.py (Plan.md Step 2, G2).

Moved verbatim out of both scripts, where they were duplicated identically.
Geoformer's Lightning-based path (geoformer/module.py, geoformer/data.py)
is not touched here (Plan.md non-goal: no forced merge with the Geoformer
path).
"""

import os

import numpy as np
import pandas as pd


def load_split_from_npz(path):
    if not os.path.exists(path):
        raise Exception(f"npz file {path} is not exist")
    data = np.load(path)
    idx_train = data['idx_train']
    idx_val = data['idx_val']
    idx_test = data['idx_test']
    return idx_train.tolist(), idx_val.tolist(), idx_test.tolist()


def warmup_exponential_decay(step: int, hparams):
    alpha = min(1.0, float(step) / float(hparams.lr_warmup_steps))
    lr_scale = hparams.lr_warmup_factor * (1.0 - alpha) + alpha
    lr_exp = hparams.decay_rate ** (step / hparams.decay_step)
    return lr_scale * lr_exp


def save_pred(preds, ids, wrt_file, col_names):
    preds_df = pd.DataFrame(preds, columns=col_names)
    preds_df.insert(0, "molecule_id", ids)
    preds_df.to_csv(wrt_file, index=False)


def build_spectrum_targets(spectrum_type: str, n_mode: int = 3):
    """Return (targets, standardize) for a given --spectrum-type/--n-mode,
    matching the branch that used to live in each script's `if __name__ ==
    "__main__":` block (train_PaiNN.py / train_Equiformer.py)."""
    if spectrum_type == 'Naive':
        return [f'y{i}' for i in range(800)], False
    elif spectrum_type == 'GMM':
        return ['A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3'], True
    elif spectrum_type == 'FC':
        if n_mode == 3:
            targets = ['S1', 'S2', 'S3', 'C', 'E0', 'h1', 'h2', 'h3']
        else:
            targets = []
            for i in range(1, n_mode + 1):
                targets.append(f'S{i}({n_mode})')
            targets += [f'C({n_mode})', f'E0({n_mode})']
            for i in range(1, n_mode + 1):
                targets.append(f'h{i}({n_mode})')
        return targets, True
    else:
        raise Exception("Spectrum type Error")
