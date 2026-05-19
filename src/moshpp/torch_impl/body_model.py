"""Thin SMPL-H wrapper around the `smplx` package.

We expose only what MoSh needs: a forward pass producing posed vertices given
betas, body_pose (axis-angle, 21 joints = 63 dims), global_orient (3) and
transl (3). Hands are kept at their flat rest pose; we do not optimize them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import smplx
import torch
import torch.nn as nn


class SMPLHBodyModel(nn.Module):
    NUM_BODY_JOINTS = 21  # matches VPoser
    BODY_POSE_DIM = NUM_BODY_JOINTS * 3

    def __init__(
        self,
        model_path: Union[str, Path],
        gender: str = "male",
        num_betas: int = 10,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.gender = gender
        self.num_betas = num_betas

        # smplx.create wants the directory containing SMPLH_{GENDER}.pkl, OR
        # a direct path. We accept the direct .pkl path and split.
        model_path = Path(model_path)
        if model_path.is_file():
            model_folder = (
                model_path.parent.parent
            )  # smplh/SMPLH_MALE.pkl -> smplh/.. is the models dir
            # smplx looks for {model_folder}/smplh/SMPLH_{GENDER}.pkl
            # We require the user to point at the .pkl directly, then mimic the expected layout.
            model_folder = model_path.parent.parent
        else:
            model_folder = model_path

        self.smpl = smplx.create(
            model_path=str(model_folder),
            model_type="smplh",
            gender=gender,
            num_betas=num_betas,
            use_pca=False,
            flat_hand_mean=True,
            ext="pkl",
            batch_size=1,
        ).to(self.device)

        self.faces = torch.as_tensor(
            self.smpl.faces.astype(np.int64), device=self.device
        )
        self.v_template = self.smpl.v_template.detach().clone().to(self.device)
        self.num_verts = int(self.v_template.shape[0])

    def _zero_hand_pose(self, batch_size: int) -> torch.Tensor:
        # 15 joints per hand * 3 = 45 dims each, total 90
        return torch.zeros(batch_size, 90, device=self.device, dtype=torch.float32)

    def forward(
        self,
        betas: torch.Tensor,  # (B, num_betas)
        body_pose: torch.Tensor,  # (B, 63) axis-angle
        global_orient: torch.Tensor,  # (B, 3)
        transl: torch.Tensor,  # (B, 3)
    ) -> torch.Tensor:
        """Return posed vertices (B, V, 3)."""
        B = body_pose.shape[0]
        hand_pose = self._zero_hand_pose(B)
        # smplx SMPLH expects left_hand_pose and right_hand_pose separately when use_pca=False
        out = self.smpl(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=hand_pose[:, :45],
            right_hand_pose=hand_pose[:, 45:],
            transl=transl,
            return_verts=True,
        )
        return out.vertices

    @torch.no_grad()
    def template_vertices(self) -> torch.Tensor:
        """T-pose vertices with zero betas."""
        return self.v_template.clone()

    def canonical_verts(self, betas: torch.Tensor) -> torch.Tensor:
        """T-pose vertices with the given betas. Differentiable w.r.t. betas."""
        betas_b = betas.view(1, -1)
        zero3 = torch.zeros(1, 3, device=self.device)
        out = self.smpl(
            betas=betas_b,
            global_orient=zero3,
            body_pose=torch.zeros(1, self.BODY_POSE_DIM, device=self.device),
            left_hand_pose=torch.zeros(1, 45, device=self.device),
            right_hand_pose=torch.zeros(1, 45, device=self.device),
            transl=zero3,
            return_verts=True,
        )
        return out.vertices[0]

    @torch.no_grad()
    def root_joint(self, betas: torch.Tensor) -> torch.Tensor:
        """Root joint position (pelvis) under the given betas, zero pose, zero transl.
        Returns a (3,) tensor."""
        betas = betas.view(1, -1).to(self.device)
        zero3 = torch.zeros(1, 3, device=self.device)
        out = self.smpl(
            betas=betas,
            global_orient=zero3,
            body_pose=torch.zeros(1, self.BODY_POSE_DIM, device=self.device),
            left_hand_pose=torch.zeros(1, 45, device=self.device),
            right_hand_pose=torch.zeros(1, 45, device=self.device),
            transl=zero3,
            return_verts=False,
        )
        return out.joints[0, 0].detach()
