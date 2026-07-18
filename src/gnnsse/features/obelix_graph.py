"""Crystal graph construction for experimentally reported structures.

Two properties of experimental structures drive the design:

1. Bond vectors are image aware. Neighbours returned by pymatgen may live in an
   adjacent periodic image, so the displacement is taken from the neighbour's
   own cartesian coordinate rather than from the coordinate of the site it maps
   onto inside the unit cell. Using the in cell coordinate would yield a zero
   length vector whenever an atom neighbours its own image, which is the common
   case in small unit cells, and the enclosed bond angle would then be
   undefined.

2. Sites may be partially occupied. Roughly four in five OBELiX structures carry
   fractional site occupancies, so a node feature is the occupancy weighted sum
   of its element vectors rather than a single one hot.
"""

from __future__ import annotations

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data

N_ELEMENTS = 103

# Radial basis over the neighbour cutoff.
RBF_STEP = 0.2
# Angular basis over [0, pi].
N_ANGLE_BINS = 40
_ANGLE_CENTERS = np.linspace(0.0, np.pi, N_ANGLE_BINS)
_ANGLE_WIDTH = np.pi / N_ANGLE_BINS


def _rbf_centers(cutoff: float) -> np.ndarray:
    """Gaussian centres spanning the actual cutoff, not a hardcoded range."""
    return np.arange(0.0, cutoff + RBF_STEP, RBF_STEP)


def _rbf(d: float, centers: np.ndarray, width: float = 0.5) -> np.ndarray:
    return np.exp(-((d - centers) ** 2) / (2.0 * width**2))


def _angle_basis(theta: float) -> np.ndarray:
    return np.exp(-((theta - _ANGLE_CENTERS) ** 2) / (2.0 * _ANGLE_WIDTH**2))


def _site_features(site) -> np.ndarray:
    """Occupancy weighted element vector for one (possibly disordered) site."""
    feat = np.zeros(N_ELEMENTS, dtype=np.float32)
    for element, occupancy in site.species.items():
        z = getattr(element, "Z", None)
        if z is not None and 1 <= z <= N_ELEMENTS:
            feat[z - 1] += float(occupancy)
    return feat


def structure_to_graph(
    structure: Structure,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    build_line_graph: bool = True,
) -> Data:
    """Convert a pymatgen ``Structure`` into a PyG ``Data`` crystal graph.

    Parameters
    ----------
    structure:
        Input structure; may be disordered.
    cutoff:
        Neighbour search radius in Angstrom. The default of 8.0 with
        ``max_neighbors=12`` follows the ALIGNN reference construction.
    max_neighbors:
        Retain at most this many nearest neighbours per site.
    build_line_graph:
        Also emit ``lg_edge_index`` / ``lg_edge_attr`` for angular models.
    """
    centers = _rbf_centers(cutoff)
    n_sites = len(structure)

    x = torch.tensor(
        np.vstack([_site_features(s) for s in structure]), dtype=torch.float32
    )

    src: list[int] = []
    dst: list[int] = []
    edge_attr: list[np.ndarray] = []
    # Image aware displacement for each edge, needed for correct bond angles.
    bond_vecs: list[np.ndarray] = []

    site_coords = np.array([s.coords for s in structure])
    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)

    for i, neighbors in enumerate(all_neighbors):
        nearest = sorted(neighbors, key=lambda n: n[1])[:max_neighbors]
        for nbr in nearest:
            j = nbr[2]
            distance = float(nbr[1])
            if distance < 1e-8:
                continue
            src.append(i)
            dst.append(j)
            edge_attr.append(_rbf(distance, centers))
            # nbr.coords already carries the periodic image offset.
            bond_vecs.append(np.asarray(nbr.coords) - site_coords[i])

    if not src:
        raise ValueError("structure produced no edges within the cutoff")

    data = Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(np.array(edge_attr), dtype=torch.float32),
    )

    if build_line_graph:
        lg_edge_index, lg_edge_attr = _line_graph(np.array(src), np.array(bond_vecs))
        data.lg_edge_index = lg_edge_index
        data.lg_edge_attr = lg_edge_attr

    return data


def _line_graph(
    src: np.ndarray, bond_vecs: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Connect bonds that share a source atom, with the enclosed angle as feature."""
    from collections import defaultdict

    incident: dict[int, list[int]] = defaultdict(list)
    for bond_idx, atom in enumerate(src):
        incident[int(atom)].append(bond_idx)

    lg_src: list[int] = []
    lg_dst: list[int] = []
    lg_feats: list[np.ndarray] = []

    norms = np.linalg.norm(bond_vecs, axis=1)

    for bonds in incident.values():
        for a_pos, b1 in enumerate(bonds):
            for b2 in bonds[a_pos + 1 :]:
                n1, n2 = norms[b1], norms[b2]
                if n1 < 1e-8 or n2 < 1e-8:
                    continue
                cos_theta = np.clip(
                    float(np.dot(bond_vecs[b1], bond_vecs[b2]) / (n1 * n2)), -1.0, 1.0
                )
                feat = _angle_basis(float(np.arccos(cos_theta)))
                lg_src += [b1, b2]
                lg_dst += [b2, b1]
                lg_feats += [feat, feat]

    if not lg_src:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, N_ANGLE_BINS), dtype=torch.float32),
        )
    return (
        torch.tensor([lg_src, lg_dst], dtype=torch.long),
        torch.tensor(np.array(lg_feats), dtype=torch.float32),
    )
