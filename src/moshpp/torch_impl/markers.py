"""Local-frame marker parameterization (torch port of TransformedCoeffs / TransformedLms).

For each latent marker, we find its 3 nearest body vertices on the canonical
T-pose mesh and store the marker as 3 coefficients in an orthonormal frame
built from those vertices. As the body is posed/shaped, the marker location
is re-synthesized from the same local frame computed on the posed vertices.

Correspondences (the 3-NN per marker) are computed once on the T-pose and
frozen — matching the behavior the chumpy version converges to after a few
iterations, and giving LBFGS a stable graph to differentiate.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _knn3(reference: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Brute-force 3-NN. reference: (V, 3), query: (M, 3). Returns (M, 3)."""
    d = torch.cdist(query, reference)
    return d.topk(3, largest=False).indices


def _local_frame(
    v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an orthonormal frame from a triangle (v0, v1, v2)."""
    e1 = v1 - v0
    e2 = v2 - v0
    f1 = F.normalize(e1, dim=-1, eps=1e-12)
    f2 = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=1e-12)
    f3 = torch.cross(f1, f2, dim=-1)
    return f1, f2, f3


def compute_nn_idx(
    canonical_verts: torch.Tensor,
    marker_positions: torch.Tensor,
) -> torch.Tensor:
    """Non-differentiable: 3-NN indices on the canonical body."""
    return _knn3(canonical_verts, marker_positions)


def compute_coeffs(
    canonical_verts: torch.Tensor,  # (V, 3)
    marker_positions: torch.Tensor,  # (M, 3)
    nn_idx: torch.Tensor,  # (M, 3)
) -> torch.Tensor:
    """Local-frame coefficients of `marker_positions` in the triangle (v0, v1, v2)
    defined by the 3 nearest verts on `canonical_verts`. Differentiable in both
    `canonical_verts` (via betas) and `marker_positions` (via markers_latent)."""
    v0 = canonical_verts[nn_idx[:, 0]]
    v1 = canonical_verts[nn_idx[:, 1]]
    v2 = canonical_verts[nn_idx[:, 2]]
    f1, f2, f3 = _local_frame(v0, v1, v2)
    diff = marker_positions - v0
    return torch.stack(
        [(diff * f1).sum(-1), (diff * f2).sum(-1), (diff * f3).sum(-1)], dim=-1
    )


def build_local_frame(
    canonical_verts: torch.Tensor,
    marker_positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combined call for initialization: returns (nn_idx, coeffs)."""
    nn_idx = compute_nn_idx(canonical_verts, marker_positions)
    coeffs = compute_coeffs(canonical_verts, marker_positions, nn_idx)
    return nn_idx, coeffs


def synth_markers(
    verts: torch.Tensor,  # (B, V, 3)
    nn_idx: torch.Tensor,  # (M, 3)
    coeffs: torch.Tensor,  # (M, 3)
) -> torch.Tensor:
    """Synthesize marker positions from posed vertices using cached local frames.

    Returns: (B, M, 3)
    """
    v0 = verts[:, nn_idx[:, 0], :]
    v1 = verts[:, nn_idx[:, 1], :]
    v2 = verts[:, nn_idx[:, 2], :]
    f1, f2, f3 = _local_frame(v0, v1, v2)
    c = coeffs.unsqueeze(0)  # (1, M, 3)
    return v0 + c[..., 0:1] * f1 + c[..., 1:2] * f2 + c[..., 2:3] * f3
