"""CPU-friendly, dataset-independent synthetic fixtures for characterization tests.

These builders deliberately avoid loading IrDB/PtDB from disk so that
characterization tests stay fast and reproducible on CPU-only environments
(see Plan.md Step 1). Molecule sizes and hyperparameters are intentionally
tiny; they are not meant to represent realistic chemistry.
"""

import torch
from torch_geometric.data import Data, Batch

from geoformer.model.collating_geoformer import GeoformerDataCollator

# Atomic numbers usable by every backbone in this repo:
#   - Equiformer's atom lookup table covers {6, 7, 8, 9, 16, 17, 35, 77, 14, 78}
#     (Equiformer/graph_attention_transformer.py: `atoms` dict)
#   - PaiNN's fairchem embedding covers num_elements=83 by default
CARBON = 6
OXYGEN = 8

NUM_X = 20  # tiny synthetic "wavelength" grid size for spec_x/spec_y
X_DIM = 16  # matches dataset/IrDB.py's node feature width (10 one-hot + 6 extra)


def n_mode_target_count(n_mode: int = 3) -> int:
    """Number of FC-progression target columns for a given n_mode (see train_PaiNN.py)."""
    return 2 * n_mode + 2


def _make_molecule(num_atoms: int, num_classes: int, seed_offset: int, name: str) -> Data:
    generator = torch.Generator().manual_seed(1000 + seed_offset)
    z = torch.tensor(
        [CARBON if i % 2 == 0 else OXYGEN for i in range(num_atoms)], dtype=torch.long
    )
    # kept tight enough that every pairwise distance stays well under the
    # smallest cutoff radius used anywhere in this repo (5.0 Angstrom)
    pos = torch.randn(num_atoms, 3, generator=generator) * 0.4
    x = torch.zeros(num_atoms, X_DIM)

    idx = torch.arange(num_atoms)
    edge_d_dst = idx.repeat_interleave(num_atoms)
    edge_d_src = idx.repeat(num_atoms)
    edge_d_index = torch.stack([edge_d_dst, edge_d_src], dim=0)
    edge_d_attr = (pos[edge_d_dst] - pos[edge_d_src]).norm(dim=1)

    y = torch.rand(1, num_classes, generator=generator) * 0.5 + 0.25
    spec_x = torch.linspace(1.7, 2.9, NUM_X).unsqueeze(0)
    spec_y = torch.rand(1, NUM_X, generator=generator)

    return Data(
        x=x,
        z=z,
        pos=pos,
        edge_d_index=edge_d_index,
        edge_d_attr=edge_d_attr,
        y=y,
        spec_x=spec_x,
        spec_y=spec_y,
        name=name,
    )


def make_tiny_pyg_batch(n_mode: int = 3) -> Batch:
    """Tiny 2-molecule torch_geometric Batch usable by engine.py's train_one_step/evaluate
    with the PaiNN and Equiformer model wrappers (both ignore f_in/edge_d_* in forward())."""
    num_classes = n_mode_target_count(n_mode)
    mols = [
        _make_molecule(num_atoms=2, num_classes=num_classes, seed_offset=1, name="tiny_mol_0"),
        _make_molecule(num_atoms=3, num_classes=num_classes, seed_offset=2, name="tiny_mol_1"),
    ]
    return Batch.from_data_list(mols)


def make_tiny_geoformer_batch(n_mode: int = 3) -> dict:
    """Tiny 2-molecule batch dict for GeoformerForEnergyRegression.forward(z=..., pos=...),
    built through the real GeoformerDataCollator so padding/masking matches production."""
    num_classes = n_mode_target_count(n_mode)
    pyg_mols = [
        _make_molecule(num_atoms=2, num_classes=num_classes, seed_offset=1, name="tiny_mol_0"),
        _make_molecule(num_atoms=3, num_classes=num_classes, seed_offset=2, name="tiny_mol_1"),
    ]
    features = [{"z": m.z, "pos": m.pos, "y": m.y, "spec_x": m.spec_x, "spec_y": m.spec_y, "name": m.name}
                for m in pyg_mols]
    collator = GeoformerDataCollator(max_nodes=None)
    return collator(features)
