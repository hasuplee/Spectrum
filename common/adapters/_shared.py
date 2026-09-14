"""Shared PaiNN/Equiformer forward calling convention (Plan.md Step 6, G3).

Both the PaiNN wrapper (common/adapters/painn_adapter.py) and the
Equiformer models (common/adapters/equiformer_adapter.py) ignore f_in and
edge_d_index/edge_d_attr in their own forward(), but engine.py's
train_one_step/evaluate always pass them so that both backbones can be
called through one uniform site. That call site is this function.
"""

import torch


def engine_style_forward(model, batch) -> torch.Tensor:
    return model(
        f_in=batch.x,
        pos=batch.pos,
        batch=batch.batch,
        node_atom=batch.z,
        edge_d_index=batch.edge_d_index,
        edge_d_attr=batch.edge_d_attr,
    )
