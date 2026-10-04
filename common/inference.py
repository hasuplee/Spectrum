"""Training-loop-independent prediction skeleton (Plan.md Step 9, G4).

predict() runs a model forward + unnormalization for any of the three
backbones without needing an optimizer, scheduler, or training loop --
unlike engine.py's train_one_step()/evaluate(), which are still coupled to
the training loop, or geoformer/module.py's LNNP, which is coupled to
PyTorch Lightning.

predict() returns the (unnormalized) FC/GMM parameter vector. load_checkpoint()
and predict_curves() (Plan.md Step 2B) restore a trained checkpoint and turn
that vector into a spectrum curve via spectrum.reconstruct (imported, not
copied). Checkpoints of all three backbones can be loaded.
"""

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import torch

from common.adapters import equiformer_adapter, geoformer_adapter, painn_adapter
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


def _args_style_loader(build):
    """PaiNN/Equiformer checkpoints: {'model': state_dict, 'args': Namespace, ...}."""
    def load(checkpoint):
        args = checkpoint["args"]
        model = build(args)
        model.load_state_dict(checkpoint["model"])
        return model, args
    return load


def _load_geoformer(checkpoint):
    """Lightning checkpoint: 'hyper_parameters' is a plain dict that calls the
    spectrum type 'spec_loss_type', and every state_dict key carries the
    LNNP 'model.' prefix."""
    args = Namespace(**checkpoint["hyper_parameters"])
    args.spectrum_type = args.spec_loss_type
    model = geoformer_adapter.build(args)
    model.load_state_dict({key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()})
    return model, args


_CHECKPOINT_LOADERS = {
    "PaiNN": _args_style_loader(painn_adapter.build),
    "Equiformer": _args_style_loader(equiformer_adapter.build),
    "Geoformer": _load_geoformer,
}


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    base_model: str
    args: object
    norm_factor: list


def load_checkpoint(path, base_model: str, norm_factor=None) -> LoadedCheckpoint:
    """Rebuilds a model from a checkpoint saved by train_PaiNN.py / train_Equiformer.py /
    train_Geoformer.py.

    The checkpoint stores a pickled argparse.Namespace, so it is loaded with
    weights_only=False. That can execute arbitrary code: only load checkpoints
    this repository trained itself.

    norm_factor: [task_mean, task_std]. If omitted: Geoformer is always [0, 1]
    (the model already unnormalizes its output internally); otherwise the
    task_mean/task_std stored in the checkpoint's args are used when present
    (Equiformer); otherwise it is [0, 1] when the training args had
    standardize=False, or recomputed from the training split (PaiNN checkpoints
    do not store it).
    """
    if base_model not in _CHECKPOINT_LOADERS:
        raise ValueError(
            f"Unsupported base_model: {base_model!r}. Available: {', '.join(_CHECKPOINT_LOADERS)}"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, args = _CHECKPOINT_LOADERS[base_model](checkpoint)
    model.eval()

    if norm_factor is None:
        norm_factor = _restore_norm_factor(base_model, args)
    return LoadedCheckpoint(model=model, base_model=base_model, args=args, norm_factor=norm_factor)


def _restore_norm_factor(base_model: str, args) -> list:
    if base_model == "Geoformer":
        num_targets = len(args.dataset_arg)
        return [torch.zeros(num_targets), torch.ones(num_targets)]
    stored_mean = getattr(args, "task_mean", None)
    stored_std = getattr(args, "task_std", None)
    if stored_mean is not None and stored_std is not None:
        return [torch.as_tensor(stored_mean, dtype=torch.float32).detach().cpu(),
                torch.as_tensor(stored_std, dtype=torch.float32).detach().cpu()]
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
