"""Training-loop-independent prediction skeleton (Plan.md Step 9, G4).

predict() runs a model forward + unnormalization for any of the three
backbones without needing an optimizer, scheduler, or training loop --
unlike engine.py's train_one_step()/evaluate(), which are still coupled to
the training loop, or geoformer/module.py's LNNP, which is coupled to
PyTorch Lightning.

predict() returns the (unnormalized) FC/GMM parameter vector. load_checkpoint()
and predict_curves() (Plan.md Step 2B) restore a trained checkpoint and turn
that vector into a spectrum curve via spectrum.reconstruct (imported, not
copied). Only PaiNN checkpoints can be loaded so far (Steps 2C/2D add the
other two backbones).
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from common.adapters import geoformer_adapter, painn_adapter
from common.adapters._shared import engine_style_forward
from common.data import load_dataset_splits
from spectrum.reconstruct import reconstruct_spectrum


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


_CHECKPOINT_BUILDERS = {"PaiNN": painn_adapter.build}


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    base_model: str
    args: object
    norm_factor: list


def load_checkpoint(path, base_model: str, norm_factor=None) -> LoadedCheckpoint:
    """Rebuilds a model from a checkpoint saved by train_PaiNN.py (currently PaiNN only).

    The checkpoint stores a pickled argparse.Namespace, so it is loaded with
    weights_only=False. That can execute arbitrary code: only load checkpoints
    this repository trained itself.

    norm_factor: [task_mean, task_std]. If omitted, it is [0, 1] when the
    training args had standardize=False, otherwise recomputed from the training
    split (the checkpoint does not store it).
    """
    if base_model not in _CHECKPOINT_BUILDERS:
        raise ValueError(
            f"Unsupported base_model: {base_model!r}. Available: {', '.join(_CHECKPOINT_BUILDERS)}"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    model = _CHECKPOINT_BUILDERS[base_model](args)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if norm_factor is None:
        norm_factor = _restore_norm_factor(args)
    return LoadedCheckpoint(model=model, base_model=base_model, args=args, norm_factor=norm_factor)


def _restore_norm_factor(args) -> list:
    if not args.standardize:
        num_targets = len(args.targets)
        return [torch.zeros(num_targets), torch.ones(num_targets)]
    _, _, _, task_mean, task_std = load_dataset_splits(args)
    return [torch.tensor(task_mean, dtype=torch.float32), torch.tensor(task_std, dtype=torch.float32)]


def predict_curves(loaded: LoadedCheckpoint, batch) -> torch.Tensor:
    """Returns the spectrum curves (batch_size, 800) for a batch."""
    params = predict(loaded.model, batch, loaded.norm_factor, loaded.base_model)
    return reconstruct_spectrum(
        params, loaded.args.spectrum_type, kernel_kind=loaded.args.lineshape, beta=loaded.args.beta
    )
