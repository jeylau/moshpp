"""Thin SMPL-H wrapper around the `smplx` package.

We expose only what MoSh needs: a forward pass producing posed vertices given
betas, body_pose (axis-angle, 21 joints = 63 dims), global_orient (3) and
transl (3). Hand pose is optional: pass `hand_pose` (B, 90) to articulate the
fingers, or leave it None to keep hands at their flat rest pose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import smplx
import torch
import torch.nn as nn
from smplx.lbs import (
    batch_rigid_transform,
    batch_rodrigues,
    blend_shapes,
    vertices2joints,
)


# MANO finger joint order within a 45-dim hand pose (15 joints x 3 axis-angle):
# index 0-2, middle 3-5, pinky 6-8, ring 9-11, thumb 12-14. Axis 1 of each
# metacarpophalangeal (the first joint of each finger) is the abduction axis.
_MCP_JOINT = {"index": 0, "middle": 3, "pinky": 6, "ring": 9}
# Sign that ADDUCTS (closes) each finger, measured per hand by perturbing each
# axis and keeping whichever direction reduces finger spread. The right hand
# mirrors the left, so getting this wrong splays a hand to ~103mm instead of
# closing it to ~38mm.
_ADDUCT_SIGN_L = {"index": +1.0, "middle": +1.0, "pinky": -1.0, "ring": -1.0}
_ADDUCT_SIGN_R = {k: -v for k, v in _ADDUCT_SIGN_L.items()}


def flat_fingers_together_hand(
    adduction: float = 0.40, device: Union[str, torch.device] = "cpu"
) -> torch.Tensor:
    """A (90,) hand pose: straight fingers, held together. Use with
    `flat_hand_mean=True`, whose zero pose is straight but splayed.

    Rotates each finger's MCP about its abduction axis to close the splay while
    leaving every flexion DoF at zero, so the hand stays flat. The default 0.40
    rad was solved against a swimmer whose hand measures 38.7mm finger spread /
    202.1mm wrist-to-fingertip reach; it yields 39.4mm / 198.9mm, where the two
    smplx presets manage only 66.8/203.9 (splayed) and 43.6/174.6 (curled).
    Re-solve `adduction` for a subject whose hand is held differently.
    """
    h = torch.zeros(90, device=device)
    for name, j in _MCP_JOINT.items():
        h[j * 3 + 1] = _ADDUCT_SIGN_L[name] * adduction  # left
        h[45 + j * 3 + 1] = _ADDUCT_SIGN_R[name] * adduction  # right
    return h


class VertexSubset:
    """Skinning tensors sliced down to a fixed set of vertices, for a fixed shape.

    Built by `SMPLHBodyModel.make_vertex_subset`. `betas` is baked in: the rest
    joints `J` come from the full shaped mesh (the joint regressor reads ~1655
    vertices, so it cannot be subset), but they depend only on `betas`, never on
    pose. That is exactly the stage II regime — shape frozen, pose free — so J is
    a constant and only the marker vertices need skinning each iteration.

    Rebuild the subset if `betas` changes; `forward_subset` cannot detect it.
    """

    def __init__(
        self,
        bm: "SMPLHBodyModel",
        vertex_ids: torch.Tensor,
        betas: Optional[torch.Tensor] = None,
    ):
        smpl = bm.smpl
        vids = torch.as_tensor(vertex_ids, dtype=torch.long, device=bm.device).reshape(
            -1
        )
        self.vertex_ids = vids
        num_betas = bm.num_betas
        if betas is None:
            betas = torch.zeros(num_betas, device=bm.device)
        betas = betas.detach().reshape(1, -1).to(bm.device)

        self.betas = betas.clone()
        with torch.no_grad():
            # Full shaped mesh: needed once, for the joint regressor.
            v_shaped_full = smpl.v_template + blend_shapes(betas, smpl.shapedirs)
            self.J = vertices2joints(smpl.J_regressor, v_shaped_full)  # (1, J, 3)
            self.v_shaped = v_shaped_full[:, vids]  # (1, S, 3)
            self.lbs_weights = smpl.lbs_weights[vids]  # (S, num_joints)
            # posedirs is (P, V*3) — flattened over vertices, so the column
            # indices for vertex v are 3v, 3v+1, 3v+2. Slicing the wrong axis
            # here silently produces a plausible-but-wrong mesh.
            cols = (vids.unsqueeze(1) * 3 + torch.arange(3, device=bm.device)).reshape(
                -1
            )
            self.posedirs = smpl.posedirs[:, cols]  # (P, S*3)


class SMPLHBodyModel(nn.Module):
    NUM_BODY_JOINTS = 21  # matches VPoser
    BODY_POSE_DIM = NUM_BODY_JOINTS * 3

    def __init__(
        self,
        model_path: Union[str, Path],
        gender: str = "male",
        num_betas: int = 10,
        device: Union[str, torch.device] = "cpu",
        flat_hand_mean: bool = True,
    ) -> None:
        """Neither hand preset is a straight hand with the fingers together.

        `flat_hand_mean=True` (the default here) gives a straight hand, but with
        the fingers splayed ~67mm apart. `False` uses the MANO mean, which bakes
        in ~3.29 rad of curl (up to 1.28 rad on one joint) and renders a visibly
        half-closed hand, though its fingers sit ~44mm apart. Legacy MoSh++ runs
        `use_hands_mean: true` (= `flat_hand_mean=False` here; the smplx flag is
        the negation of the legacy one).

        Only ~15 of a hand's 45 DoF are observable from a typical finger-marker
        set, so the unobserved 30 are decided by whatever the pose prior pulls
        toward — which is why the preset visibly dictates the rendered hand.
        Rather than pick the lesser evil, keep the straight preset and shift the
        prior's target with `hand_pose_mean` (see `flat_fingers_together_hand`).
        """
        super().__init__()
        self.device = torch.device(device)
        self.gender = gender
        self.num_betas = num_betas
        self.flat_hand_mean = flat_hand_mean

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
            flat_hand_mean=flat_hand_mean,
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

    def make_vertex_subset(
        self, vertex_ids: torch.Tensor, betas: Optional[torch.Tensor] = None
    ) -> "VertexSubset":
        """Precompute a skinning slice for `vertex_ids`, for use with `forward_subset`.

        A MoSh fit only ever reads ~150 of the mesh's 6890 vertices (the 3-NN of
        each marker), but `smplx` always skins all of them. Slicing the per-vertex
        tensors down to the ones we read is a large win in the inner loop.

        `betas` is baked into the result — rebuild if it changes.
        """
        return VertexSubset(self, vertex_ids, betas)

    def forward_subset(
        self,
        subset: "VertexSubset",
        body_pose: torch.Tensor,  # (B, 63)
        global_orient: torch.Tensor,  # (B, 3)
        transl: torch.Tensor,  # (B, 3)
        hand_pose: Optional[torch.Tensor] = None,  # (B, 90)
        betas: Optional[torch.Tensor] = None,  # (num_betas,) or (B, num_betas)
    ) -> torch.Tensor:
        """Skin only `subset`'s vertices. Returns (B, len(vertex_ids), 3).

        Equivalent to `forward(...)[:, vertex_ids]` for the betas the subset was
        built with, which are baked in and cannot vary — see `VertexSubset`. Pass
        `betas` to assert they still match what the subset was built for.
        """
        B = body_pose.shape[0]
        if betas is not None:
            want = betas.detach().reshape(-1, self.num_betas)[0]
            if not torch.allclose(want, subset.betas.reshape(-1), atol=1e-6):
                raise ValueError(
                    "betas do not match the ones this VertexSubset was built with. "
                    "The subset bakes in shape (its rest joints are precomputed); "
                    "rebuild it via make_vertex_subset(vertex_ids, betas)."
                )
        if hand_pose is None:
            hand_pose = self._zero_hand_pose(B)
        pose = torch.cat([global_orient, body_pose, hand_pose], dim=1)  # (B, 156)

        rot_mats = batch_rodrigues(pose.reshape(-1, 3)).view([B, -1, 3, 3])

        # Pose blend shapes, restricted to the subset's vertices.
        ident = torch.eye(3, dtype=rot_mats.dtype, device=rot_mats.device)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([B, -1])
        pose_offsets = torch.matmul(pose_feature, subset.posedirs).view(B, -1, 3)
        v_posed = pose_offsets + subset.v_shaped  # (B, S, 3)

        # Joints depend only on betas, which the subset baked in, so the rest
        # pose joints are constant and the transform chain is pose-only.
        _, A = batch_rigid_transform(
            rot_mats,
            subset.J.expand(B, -1, -1),
            self.smpl.parents,
            dtype=rot_mats.dtype,
        )

        num_joints = subset.J.shape[1]
        W = subset.lbs_weights.unsqueeze(0).expand([B, -1, -1])
        T = torch.matmul(W, A.view(B, num_joints, 16)).view(B, -1, 4, 4)

        homogen = torch.ones(
            [B, v_posed.shape[1], 1], dtype=v_posed.dtype, device=v_posed.device
        )
        v_homo = torch.matmul(
            T, torch.unsqueeze(torch.cat([v_posed, homogen], dim=2), dim=-1)
        )
        return v_homo[:, :, :3, 0] + transl.unsqueeze(1)

    def forward(
        self,
        betas: torch.Tensor,  # (B, num_betas)
        body_pose: torch.Tensor,  # (B, 63) axis-angle
        global_orient: torch.Tensor,  # (B, 3)
        transl: torch.Tensor,  # (B, 3)
        hand_pose: Optional[torch.Tensor] = None,  # (B, 90) axis-angle, or None
    ) -> torch.Tensor:
        """Return posed vertices (B, V, 3).

        `hand_pose` is (B, 90) = left (45) then right (45), axis-angle. When
        None the hands stay at the flat rest pose. Note that finger markers
        fitted against frozen hands can only be satisfied by rotating the
        wrist, which corrupts wrist orientation — either articulate the hands
        or drop the finger markers from the data term.
        """
        B = body_pose.shape[0]
        if hand_pose is None:
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

    def canonical_verts(
        self, betas: torch.Tensor, hand_pose: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """T-pose vertices with the given betas. Differentiable w.r.t. betas.

        `hand_pose` is (90,) or (1, 90). Pass the same static hand pose used for
        posing so that finger markers are placed on a hand in the configuration
        they will actually be synthesized from.
        """
        betas_b = betas.view(1, -1)
        zero3 = torch.zeros(1, 3, device=self.device)
        if hand_pose is None:
            hand_pose = self._zero_hand_pose(1)
        else:
            hand_pose = hand_pose.reshape(1, 90)
        out = self.smpl(
            betas=betas_b,
            global_orient=zero3,
            body_pose=torch.zeros(1, self.BODY_POSE_DIM, device=self.device),
            left_hand_pose=hand_pose[:, :45],
            right_hand_pose=hand_pose[:, 45:],
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
