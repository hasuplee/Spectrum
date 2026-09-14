"""Model registry (Plan.md Step 3, G3).

Maps a --base-model name to its adapter module, each exposing
build(args) -> nn.Module and forward(model, batch) -> Tensor.
"""

from . import painn_adapter, equiformer_adapter, geoformer_adapter

ADAPTERS = {
    "PaiNN": painn_adapter,
    "Equiformer": equiformer_adapter,
    "Geoformer": geoformer_adapter,
}


def get_adapter(name: str):
    if name not in ADAPTERS:
        raise KeyError(f"Unknown model adapter: {name!r}. Available: {list(ADAPTERS)}")
    return ADAPTERS[name]
