"""Geoformer model adapter (Plan.md Step 3, G3).

Geoformer keeps its Lightning-based training entry point
(train_Geoformer.py / geoformer/module.py's LNNP) unchanged (Plan.md
non-goal: no forced merge). This adapter only exposes the same
build(args)/forward(model, batch) interface as the other two backbones, for
callers (e.g. a future inference API, Step 9) that want a uniform way to
run any of the three models without knowing their individual conventions.
"""

import torch
from torch import nn

from geoformer.model.modeling_geoformer import create_model as _create_model


def build(args) -> nn.Module:
    """args needs the fields geoformer.model.modeling_geoformer.create_model
    reads from GeoformerConfig: max_z, embedding_dim, ffn_embedding_dim,
    num_layers, num_heads, cutoff, num_rbf, trainable_rbf, norm_type,
    dropout, attention_dropout, activation_dropout, activation_function,
    decoder_type, aggr, dataset_root, dataset_arg, mean, std, prior_model,
    num_classes, pad_token_id."""
    return _create_model(args)


def forward(model: nn.Module, batch) -> torch.Tensor:
    pred = model(z=batch["z"], pos=batch["pos"])
    return pred.view(pred.shape[0], -1)
