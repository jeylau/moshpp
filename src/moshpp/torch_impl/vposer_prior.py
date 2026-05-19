"""VPoser as a pose parameterization (not as a per-step prior).

We freeze VPoser, optimize a 32-d latent code z per frame, and decode it to
21-joint axis-angle body pose to feed SMPL-H. The "prior" term reduces to a
simple L2 on z.
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
    """Wraps a pretrained VPoser and exposes a single `decode_aa(z) -> (B, 63)`."""

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
        """Optional: given an axis-angle pose, return the latent mean. Useful
        for warm-starting z from a non-zero pose guess."""
        # VPoser V02 encodes from axis-angle directly
        q_z = self.vp.encode(body_pose_aa)
        return q_z.mean
