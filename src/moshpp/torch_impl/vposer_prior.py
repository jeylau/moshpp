"""VPoser as a *prior* over a freely-optimized axis-angle body pose.

MoSh++ optimizes the raw 63-dim axis-angle pose and applies a GMM as a soft
penalty (see chmosh.py). We keep that structure but swap the GMM for VPoser:
`pose` stays a free variable and the prior is `||encode(pose).mean||^2`.

This matters. Parameterizing pose as `decode(z)` instead would make VPoser's
decoder a *hard* constraint rather than a soft one: any pose outside the
decoder's range becomes unreachable at any weight. Measured on this checkpoint,
fitting z to a plantarflexed ankle (-0.8 rad) saturates at a ~0.47 rad residual
no matter how large |z| grows or how small the prior weight is. A soft prior can
be outvoted by data; a decoder cannot. `decode_aa` is retained for warm-starting
and for callers that still want the latent parameterization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from pytorch3d.transforms import matrix_to_axis_angle


LATENT_DIM = 32


class FrozenVPoser(nn.Module):
    """Wraps a pretrained VPoser, frozen.

    `prior_term(pose)` is the intended entry point for fitting. `decode_aa` and
    `encode_mean` are exposed for warm-starting and inspection.
    """

    def __init__(
        self, expr_dir: Union[str, Path], device: Union[str, torch.device] = "cpu"
    ):
        super().__init__()
        vp, _ = load_model(
            str(expr_dir),
            model_code=VPoser,
            remove_words_in_model_weights="vp_model.",
            disable_grad=True,
        )
        self.vp = vp.to(device).eval()
        for p in self.vp.parameters():
            p.requires_grad = False
        self.device = torch.device(device)

    @property
    def latent_dim(self) -> int:
        return LATENT_DIM

    def decode_aa(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 32) -> body pose axis-angle (B, 63)."""
        out = self.vp.decode(z)
        pb = out["pose_body"]
        if pb.dim() == 4:
            # (B, 21, 3, 3) rotation matrices -> axis-angle
            aa = matrix_to_axis_angle(pb)
            return aa.reshape(z.shape[0], -1)
        return pb.reshape(z.shape[0], -1)

    def encode_mean(self, body_pose_aa: torch.Tensor) -> torch.Tensor:
        """Given an axis-angle pose (B, 63), return the latent mean (B, 32).

        Differentiable w.r.t. `body_pose_aa`, which is what lets us use it as a
        prior on a freely-optimized pose.
        """
        # VPoser V02 encodes from axis-angle directly
        q_z = self.vp.encode(body_pose_aa)
        return q_z.mean

    def prior_term(self, body_pose_aa: torch.Tensor) -> torch.Tensor:
        """Sum-of-squares VPoser prior on a free axis-angle pose (B, 63).

        Analogous to legacy `model.priors['pose'](model.pose[pose_body_ids])`:
        a soft cost that grows as the pose leaves the learned pose manifold.
        Note this does not bottom out at zero for the T-pose (|encode| ~ 3.2
        there) — the minimum sits at AMASS's relaxed mean pose, which is the
        intended behavior for a prior.
        """
        return (self.encode_mean(body_pose_aa) ** 2).sum()
