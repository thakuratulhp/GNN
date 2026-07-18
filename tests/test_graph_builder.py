"""Tests for the crystal graph builder.

These pin the two properties that are easy to get wrong on experimental
structures and that fail silently rather than loudly: bond angles under periodic
images, and partially occupied sites.
"""

from __future__ import annotations

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from gnnsse.features.obelix_graph import structure_to_graph


def test_angles_exist_when_every_neighbour_is_a_periodic_image():
    """Bond vectors must come from the neighbour's image, not the in-cell site.

    In simple cubic Li every neighbour of the single site is an image of that
    same site. Taking the difference of in-cell cartesian coordinates gives the
    zero vector and an undefined angle, which would leave the line graph empty
    and strip ALIGNN of the only information distinguishing it from a
    distance-only network.
    """
    structure = Structure(Lattice.cubic(3.0), ["Li"], [[0, 0, 0]])
    graph = structure_to_graph(structure, cutoff=5.0, max_neighbors=12)

    assert graph.edge_index.shape[1] > 0
    assert graph.lg_edge_index.shape[1] > 0, "line graph is empty: angles were dropped"

    # Recover the angles from the basis and check they are physical.
    centres = np.linspace(0.0, np.pi, 40)
    angles = np.degrees(centres[graph.lg_edge_attr.argmax(1).numpy()])
    assert angles.min() > 1.0, "a zero angle means a zero-length bond vector"
    # simple cubic geometry: 90 and 180 degree pairs must both appear
    assert np.any(np.abs(angles - 90) < 6)
    assert np.any(np.abs(angles - 180) < 6)


def test_partial_occupancy_is_occupancy_weighted():
    """Four in five OBELiX structures are disordered; the builder must read them.

    A one hot over site.specie raises on a disordered site, so node features are
    the occupancy weighted sum of the element vectors present.
    """
    structure = Structure(
        Lattice.cubic(4.0),
        [{"Li": 0.5, "Na": 0.5}, {"O": 1.0}],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    graph = structure_to_graph(structure, cutoff=5.0, max_neighbors=12)

    li, na = graph.x[0, 2], graph.x[0, 10]  # Z=3 and Z=11, zero indexed
    assert li == 0.5 and na == 0.5
    assert torch.isclose(graph.x[0].sum(), torch.tensor(1.0))


def test_radial_basis_spans_the_requested_cutoff():
    """The basis must follow the cutoff rather than a hardcoded range."""
    structure = Structure(Lattice.cubic(3.0), ["Li"], [[0, 0, 0]])
    assert structure_to_graph(structure, cutoff=5.0).edge_attr.shape[1] == 26
    assert structure_to_graph(structure, cutoff=8.0).edge_attr.shape[1] == 41
