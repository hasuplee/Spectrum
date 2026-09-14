"""Training-loop-independent prediction skeleton (Plan.md Step 9, G4).

predict() runs a model forward + unnormalization for any of the three
backbones without needing an optimizer, scheduler, or training loop --
unlike engine.py's train_one_step()/evaluate(), which are still coupled to
the training loop, or geoformer/module.py's LNNP, which is coupled to
PyTorch Lightning.

This is a skeleton, not a finished inference service (PRD non-goal):
reconstructing an actual spectrum curve from the returned FC/GMM parameter
vector (via spectrum.physics.spectrum_fc/spectrum_gmm) is left to the
caller.
"""

import torch

from common.adapters import geoformer_adapter
from common.adapters._shared import engine_style_forward


def predict(model, batch, norm_factor, base_model: str) -> torch.Tensor:
    """norm_factor: [task_mean, task_std]. base_model: one of "PaiNN",
    "Equiformer", "Geoformer". PaiNN/Equiformer batches are torch_geometric
    Batch objects; Geoformer batches are the dict produced by
    geoformer.model.collating_geoformer.GeoformerDataCollator."""
    task_mean, task_std = norm_factor
    with torch.no_grad():
        if base_model == "Geoformer":
            pred = geoformer_adapter.forward(model, batch)
        elif base_model in ("PaiNN", "Equiformer"):
            pred = engine_style_forward(model, batch)
            pred = pred.view(pred.shape[0], -1)
        else:
            raise KeyError(f"Unknown base_model: {base_model!r}. Available: PaiNN, Equiformer, Geoformer")
        return pred * task_std + task_mean
