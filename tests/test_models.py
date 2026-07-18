"""Tests for the graph models.

The stability test is the important one. Without normalisation and a gate
normalised aggregation, an edge-gated stack of this depth can grow its
activations geometrically and diverge while still returning finite-looking
numbers early in training. This pins that activations stay bounded through the
full stack.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.loader import DataLoader

from gnnsse.features.obelix_graph import structure_to_graph
from gnnsse.models import ALIGNN, CGCNN


# Both models hardcode an edge input width of 41, which is the number of
# radial basis functions produced at the reference 8 A cutoff with 0.2 A
# spacing. Building graphs at any other cutoff silently yields a width the
# models cannot accept, so the graphs here use the same construction as the
# paper.
CUTOFF = 8.0
MAX_NEIGHBORS = 12


def _batch(n: int = 4):
    rng = np.random.default_rng(0)
    graphs = []
    for _ in range(n):
        s = Structure(
            Lattice.cubic(3.0 + 0.4 * rng.random()),
            ["Li", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        g = structure_to_graph(s, cutoff=CUTOFF, max_neighbors=MAX_NEIGHBORS)
        g.y = torch.tensor([float(rng.normal(-5, 2))])
        graphs.append(g)
    return next(iter(DataLoader(graphs, batch_size=n)))


def test_graph_width_matches_what_the_models_expect():
    """Guards the coupling between the cutoff and the models' input width."""
    s = Structure(Lattice.cubic(3.0), ["Li"], [[0, 0, 0]])
    g = structure_to_graph(s, cutoff=CUTOFF, max_neighbors=MAX_NEIGHBORS)
    assert g.edge_attr.shape[1] == 41
    assert g.x.shape[1] == 103
    assert g.lg_edge_attr.shape[1] == 40


@pytest.mark.parametrize("model_fn", [
    lambda: CGCNN(atom_fea_len=64, n_conv=3, h_fea_len=128),
    lambda: ALIGNN(hidden_size=64, n_layers=2, n_gcn_layers=2),
])
def test_forward_shape_and_finiteness(model_fn):
    batch = _batch()
    out = model_fn()(batch)
    assert out.shape == (batch.num_graphs,)
    assert torch.isfinite(out).all()


def test_alignn_activations_stay_bounded_through_the_stack():
    """An unnormalised stack of this depth can reach ~1e5 activations; guard it."""
    batch = _batch(8)
    m = ALIGNN(hidden_size=64, n_layers=4, n_gcn_layers=4)
    m.train()

    x = m.node_embedding(batch.x)
    e = m.edge_embedding(batch.edge_attr)
    lg = m.angle_embedding(batch.lg_edge_attr)
    for layer in m.alignn_layers:
        x, e, lg = layer(x, e, batch.edge_index, lg, batch.lg_edge_index)
        assert x.abs().max() < 1e3, "node activations diverging"
        assert e.abs().max() < 1e3, "edge activations diverging"


def test_checkpointing_does_not_change_the_gradient():
    """Checkpointing trades compute for memory and must be exact.

    The alternative of shrinking the batch is not exact: both architectures use
    batch normalisation, so micro batching changes the batch statistics.
    """
    batch = _batch(4)
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        m = ALIGNN(hidden_size=64, n_layers=2, n_gcn_layers=2, checkpoint=ckpt)
        m.train()
        torch.nn.functional.mse_loss(m(batch), batch.y.view(-1)).backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters()
                                if p.grad is not None]))
    rel = (grads[0] - grads[1]).abs().max() / grads[0].abs().max()
    assert rel < 1e-4, f"checkpointing altered the gradient (rel {rel:.2e})"
