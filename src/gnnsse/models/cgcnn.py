"""Crystal Graph Convolutional Neural Network (Xie & Grossman 2018)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


class CGCNNConv(nn.Module):
    """Single CGCNN convolution layer."""

    def __init__(self, atom_fea_len: int, nbr_fea_len: int) -> None:
        super().__init__()
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return updated atom features."""
        n_atoms = atom_fea.shape[0]
        atom_nbr_fea = atom_fea[nbr_idx[:, 1]]      # neighbour features
        atom_self_fea = atom_fea[nbr_idx[:, 0]]     # central atom features

        total_fea = torch.cat([atom_self_fea, atom_nbr_fea, nbr_fea], dim=1)
        total_gated = self.fc_full(total_fea)
        total_gated = self.bn1(total_gated)
        nbr_filter, nbr_core = total_gated.chunk(2, dim=1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        nbr_sumed = nbr_filter * nbr_core

        # Scatter-add messages onto central atoms
        nbr_summed = torch.zeros(n_atoms, nbr_sumed.shape[1], device=atom_fea.device)
        nbr_summed.scatter_add_(0, nbr_idx[:, 0].unsqueeze(1).expand_as(nbr_sumed), nbr_sumed)

        out = self.softplus(self.bn2(atom_fea + nbr_summed))
        return out


class CGCNN(nn.Module):
    """CGCNN for classification or regression.

    Parameters
    ----------
    atom_fea_len:
        Initial atom embedding dimension (= node feature size after linear projection).
    n_conv:
        Number of graph-conv layers.
    h_fea_len:
        Hidden dim of the MLP head.
    n_h:
        Number of hidden layers in MLP head.
    n_tasks:
        Number of output heads (1 for single-task, >1 for multi-task).
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        n_tasks: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # node feature input size = 103 (one-hot element)
        node_in = 103
        # edge feature input size derived from Gaussian expansion (0..8 Å, step 0.2)
        edge_in = 41  # len(np.arange(0, 8.2, 0.2))

        self.embedding = nn.Linear(node_in, atom_fea_len)
        self.convs = nn.ModuleList(
            [CGCNNConv(atom_fea_len, edge_in) for _ in range(n_conv)]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        hidden_layers = []
        for _ in range(n_h):
            hidden_layers += [nn.Linear(h_fea_len, h_fea_len), nn.Softplus()]
            if dropout > 0:
                hidden_layers.append(nn.Dropout(dropout))
        self.fcs = nn.Sequential(*hidden_layers)

        self.heads = nn.ModuleList([nn.Linear(h_fea_len, 1) for _ in range(n_tasks)])
        self.n_tasks = n_tasks

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Build (n_edges, 2) index tensor expected by CGCNNConv
        nbr_idx = edge_index.T  # shape (E, 2): [src, dst]

        atom_fea = self.embedding(x)
        for conv in self.convs:
            atom_fea = conv(atom_fea, edge_attr, nbr_idx)

        # Global pooling
        pooled = global_mean_pool(atom_fea, batch)
        crys_fea = self.conv_to_fc_softplus(self.conv_to_fc(pooled))
        crys_fea = self.fcs(crys_fea)

        if self.n_tasks == 1:
            return self.heads[0](crys_fea).squeeze(-1)
        return torch.cat([h(crys_fea) for h in self.heads], dim=-1)
