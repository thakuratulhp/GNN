"""Atomistic Line Graph Neural Network (ALIGNN, Choudhary & DeCost 2021).

Re implementation of the architecture described in Choudhary, K. & DeCost, B.
"Atomistic Line Graph Neural Network for improved materials property
predictions", npj Comput. Mater. 7, 185 (2021), following the reference
implementation at https://github.com/usnistgov/alignn.

The building block is the edge gated graph convolution of Bresson & Laurent,
applied alternately to the bond angle line graph and to the crystal graph. For a
graph carrying node states h and edge states e:

    e_ij_hat = A h_i + B h_j + C e_ij
    sigma_ij = sigmoid(e_ij_hat)
    h_i'     = h_i + SiLU(norm(U h_i + sum_j sigma_ij * V h_j / sum_j sigma_ij))
    e_ij'    = e_ij + SiLU(norm(e_ij_hat))

Both updates are residual and normalised, which is what keeps the activations
bounded across layers. Note that this is a gated update over separately
projected source and destination states; it is not the CGCNN update rule, which
gates a single joint projection of the concatenated vector [h_i || h_j || e_ij].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution; updates node and edge states together."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int) -> None:
        super().__init__()
        # Edge gate: separate source, destination and edge projections.
        self.src_gate = nn.Linear(node_dim, out_dim)
        self.dst_gate = nn.Linear(node_dim, out_dim)
        self.edge_gate = nn.Linear(edge_dim, out_dim)
        # Node update.
        self.src_update = nn.Linear(node_dim, out_dim)
        self.dst_update = nn.Linear(node_dim, out_dim)

        self.norm_nodes = nn.BatchNorm1d(out_dim)
        self.norm_edges = nn.BatchNorm1d(out_dim)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        n_nodes = x.shape[0]

        e_hat = self.src_gate(x)[src] + self.dst_gate(x)[dst] + self.edge_gate(edge_attr)
        sigma = torch.sigmoid(e_hat)

        # Gated mean over incoming neighbours, normalised by the gate mass.
        messages = sigma * self.src_update(x)[src]
        numerator = scatter(messages, dst, dim=0, dim_size=n_nodes, reduce="sum")
        denominator = scatter(sigma, dst, dim=0, dim_size=n_nodes, reduce="sum") + 1e-6

        x_out = x + self.act(self.norm_nodes(self.dst_update(x) + numerator / denominator))
        e_out = edge_attr + self.act(self.norm_edges(e_hat))
        return x_out, e_out


class ALIGNNLayer(nn.Module):
    """One ALIGNN layer: line graph update, then crystal graph update.

    The line graph treats bonds as nodes and atom triplets as edges, so the
    angular states refine the bond states; the refined bond states are then the
    edge states of the crystal graph update.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.angle_update = EdgeGatedGraphConv(hidden, hidden, hidden)
        self.bond_update = EdgeGatedGraphConv(hidden, hidden, hidden)

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        lg_edge_attr: torch.Tensor,
        lg_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lg_edge_index.numel() > 0:
            edge_attr, lg_edge_attr = self.angle_update(
                edge_attr, lg_edge_attr, lg_edge_index
            )
        x, edge_attr = self.bond_update(x, edge_attr, edge_index)
        return x, edge_attr, lg_edge_attr


class ALIGNN(nn.Module):
    """ALIGNN for single task or multi task property prediction.

    Parameters
    ----------
    node_input_size:
        Width of the input atom features.
    edge_input_size:
        Width of the input bond features (radial basis expansion).
    triplet_input_size:
        Width of the input angle features (angular basis expansion).
    hidden_size:
        Hidden width shared by atom, bond and angle states.
    n_layers:
        Number of ALIGNN (line graph plus crystal graph) layers.
    n_gcn_layers:
        Number of plain edge gated layers applied on the crystal graph afterwards.
    n_tasks:
        Number of output heads.
    dropout:
        Dropout applied in the readout head.
    """

    def __init__(
        self,
        node_input_size: int = 103,
        edge_input_size: int = 41,
        triplet_input_size: int = 40,
        hidden_size: int = 256,
        n_layers: int = 4,
        n_gcn_layers: int = 4,
        n_tasks: int = 1,
        dropout: float = 0.0,
        checkpoint: bool = False,
    ) -> None:
        super().__init__()
        # Activation checkpointing recomputes each layer's internals during the
        # backward pass rather than holding them. Peak memory drops to roughly
        # one layer's worth at the cost of a second forward pass. The arithmetic
        # is unchanged, which matters here: the alternative of shrinking the
        # batch would alter the batch normalisation statistics and make runs at
        # different label counts incomparable.
        self.checkpoint = checkpoint
        self.node_embedding = nn.Sequential(
            nn.Linear(node_input_size, hidden_size), nn.SiLU()
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_input_size, hidden_size), nn.SiLU()
        )
        self.angle_embedding = nn.Sequential(
            nn.Linear(triplet_input_size, hidden_size), nn.SiLU()
        )

        self.alignn_layers = nn.ModuleList(
            [ALIGNNLayer(hidden_size) for _ in range(n_layers)]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(hidden_size, hidden_size, hidden_size)
                for _ in range(n_gcn_layers)
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size // 2, 1) for _ in range(n_tasks)])
        self.n_tasks = n_tasks

    def forward(self, data: Data) -> torch.Tensor:
        x = self.node_embedding(data.x)
        edge_attr = self.edge_embedding(data.edge_attr)

        lg_edge_index = getattr(data, "lg_edge_index", None)
        lg_edge_attr = getattr(data, "lg_edge_attr", None)
        if lg_edge_attr is not None and lg_edge_attr.numel() > 0:
            lg_edge_attr = self.angle_embedding(lg_edge_attr)
        else:
            lg_edge_index = torch.zeros(2, 0, dtype=torch.long, device=x.device)
            lg_edge_attr = torch.zeros(0, x.shape[-1], device=x.device)

        use_ckpt = self.checkpoint and self.training and torch.is_grad_enabled()
        for layer in self.alignn_layers:
            if use_ckpt:
                x, edge_attr, lg_edge_attr = cp.checkpoint(
                    layer, x, edge_attr, data.edge_index, lg_edge_attr,
                    lg_edge_index, use_reentrant=False,
                )
            else:
                x, edge_attr, lg_edge_attr = layer(
                    x, edge_attr, data.edge_index, lg_edge_attr, lg_edge_index
                )
        for layer in self.gcn_layers:
            if use_ckpt:
                x, edge_attr = cp.checkpoint(
                    layer, x, edge_attr, data.edge_index, use_reentrant=False,
                )
            else:
                x, edge_attr = layer(x, edge_attr, data.edge_index)

        out = self.readout(global_mean_pool(x, data.batch))
        if self.n_tasks == 1:
            return self.heads[0](out).squeeze(-1)
        return torch.cat([h(out) for h in self.heads], dim=-1)
