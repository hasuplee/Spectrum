"""PaiNN model adapter (Plan.md Step 3, G3).

The PaiNN wrapper class and build_painn() were moved verbatim out of
train_PaiNN.py so the same construction/forward logic can be reached
through the common build(args)/forward(model, batch) interface instead of
train_PaiNN.py's own calling convention.
"""

from types import SimpleNamespace

import torch
from torch import nn
from torch_cluster import radius_graph
from fairchem.core.models.painn.painn import PaiNN as _PaiNN

from common.adapters._shared import engine_style_forward as forward


def _rbf(dist, K, cutoff):
    centers = torch.linspace(0, cutoff, K, device=dist.device)
    gamma = 1.0 / ((cutoff / max(K, 1)) ** 2 + 1e-9)
    return torch.exp(-gamma * (dist.unsqueeze(-1) - centers) ** 2)


class PaiNN(nn.Module):
    def __init__(
        self,
        out_channels=1,
        cutoff=5.0,
        hidden_channels=128,
        num_layers=6,
        num_rbf=64,
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.backbone = _PaiNN(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            out_channels=out_channels,
        )

    @torch.no_grad()
    def _edge_index(self, pos, batch):
        return radius_graph(pos, r=self.cutoff, batch=batch, loop=False, max_num_neighbors=512)

    def forward(
        self,
        f_in,
        pos,
        batch,
        node_atom,
        **kwargs,
    ):
        edge_index = self._edge_index(pos, batch)
        rij = pos[edge_index[0]] - pos[edge_index[1]]
        dist = rij.norm(dim=-1)
        edge_attr = _rbf(dist, self.num_rbf, self.cutoff)
        data = SimpleNamespace(
                pos=pos,
                z=node_atom.long(),
                atomic_numbers=node_atom.long(),
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                natoms=torch.bincount(batch),
                )
        B = int(batch.max())+1
        data.pbc = torch.zeros(B, 3, dtype=torch.bool, device=pos.device)
        data.cell = torch.zeros(B, 3, 3, dtype=pos.dtype, device=pos.device)
        pred = self.backbone(data)

        return pred


def build(args) -> nn.Module:
    """args needs: out_channels, radius, num_basis, embed_dim, num_layers."""
    return PaiNN(
        out_channels=args.out_channels,
        cutoff=args.radius,
        num_rbf=args.num_basis,
        hidden_channels=args.embed_dim,
        num_layers=args.num_layers,
    )
