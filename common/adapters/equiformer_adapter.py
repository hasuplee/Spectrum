"""Equiformer model adapter (Plan.md Step 3, G3).

Wraps Equiformer/registry.py's model_entrypoint lookup behind the common
build(args)/forward(model, batch) interface.
"""

from torch import nn

from Equiformer import model_entrypoint

from common.adapters._shared import engine_style_forward as forward


def build(args) -> nn.Module:
    """args needs: model_name, input_irreps, radius, num_basis, out_channels,
    drop_path; task_mean/task_std/atomref are optional (default None)."""
    create_model = model_entrypoint(args.model_name)
    return create_model(
        irreps_in=args.input_irreps,
        radius=args.radius,
        num_basis=args.num_basis,
        out_channels=args.out_channels,
        task_mean=getattr(args, "task_mean", None),
        task_std=getattr(args, "task_std", None),
        atomref=getattr(args, "atomref", None),
        drop_path=args.drop_path,
    )
