"""Thin SMPL-H wrapper around the `smplx` package.

We expose only what MoSh needs: a forward pass producing posed vertices given
betas, body_pose (axis-angle, 21 joints = 63 dims), global_orient (3) and
transl (3). Hand pose is optional: pass `hand_pose` (B, 2*dof_per_hand) to
articulate the fingers, or leave it None to keep them at their rest pose.
Hands live in MANO's PCA space by default, as in legacy MoSh++.
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
        flat_hand_mean: bool = False,
        dof_per_hand: int = 24,
    ) -> None:
        """Hands are parameterized in MANO's PCA space, as in legacy MoSh++.

        `dof_per_hand` keeps the top-N MANO components per hand (legacy
        `dof_per_hand: 24`, moshpp_conf.yaml); pass 45 for the raw axis-angle
        space. The PCA basis is what makes hands identifiable from a handful of
        finger markers: 5 markers give 15 scalar observations, so raw 45-d leaves
        30 directions (67% of the space) determined by nothing but the prior,
        whereas at 12 components the markers pin every direction. It is a
        conditioning fix, not a plausibility guarantee — a large enough
        coefficient still leaves the anatomical region in any of these spaces.

        `flat_hand_mean=False` (default) matches legacy's `use_hands_mean: true`
        — note the smplx flag is the negation of the legacy one.

        It only moves the origin of the hand space; the PCA basis is identical
        either way, so it trades off against nothing but the rest pose. `False`
        centers on the MANO mean, which bakes in ~3.3 rad of curl: with a sparse
        finger-marker set most hand directions are unobserved, the fit cannot
        undo the curl, and hands render visibly half-closed. `True` centers on a
        straight hand instead — usually the better *look*, at no measured cost to
        marker accuracy — but it is a divergence from legacy, so it is opt-in.
        """
        super().__init__()
        self.device = torch.device(device)
        self.gender = gender
        self.num_betas = num_betas
        self.flat_hand_mean = flat_hand_mean
        self.dof_per_hand = dof_per_hand
        self.use_pca = dof_per_hand < 45
        # Dim of the per-hand pose vector this model accepts (both hands: x2).
        self.hand_pose_dim = 2 * dof_per_hand

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
            use_pca=self.use_pca,
            num_pca_comps=dof_per_hand,
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
        """Rest hand in whatever space this model uses: `2 * dof_per_hand`."""
        return torch.zeros(
            batch_size, self.hand_pose_dim, device=self.device, dtype=torch.float32
        )

    def hand_pose_to_aa(self, hand_pose: torch.Tensor) -> torch.Tensor:
        """(B, 2*dof_per_hand) in this model's hand space -> (B, 90) axis-angle.

        Mirrors what `smplx`'s own forward does internally, which our masked LBS
        path bypasses: PCA coefficients must be expanded through the components
        and offset by the hand mean before they can go through Rodrigues.
        """
        smpl = self.smpl
        left, right = (
            hand_pose[:, : self.dof_per_hand],
            hand_pose[:, self.dof_per_hand :],
        )
        if self.use_pca:
            left = torch.einsum("bi,ij->bj", left, smpl.left_hand_components)
            right = torch.einsum("bi,ij->bj", right, smpl.right_hand_components)
        aa = torch.cat([left, right], dim=1)
        # pose_mean holds the hand mean in its last 90 entries (zero when
        # flat_hand_mean=True). smplx adds it after the PCA expansion.
        return aa + smpl.pose_mean[-90:].unsqueeze(0)

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
        # smplx's forward expands PCA coeffs and adds pose_mean internally; this
        # path builds the axis-angle vector itself, so it must do the same.
        # pose_mean is zero over root+body, so only the hands need it.
        hand_aa = self.hand_pose_to_aa(hand_pose)  # (B, 90)
        pose = torch.cat([global_orient, body_pose, hand_aa], dim=1)  # (B, 156)

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
        hand_pose: Optional[torch.Tensor] = None,  # (B, 2*dof_per_hand), or None
    ) -> torch.Tensor:
        """Return posed vertices (B, V, 3).

        `hand_pose` is (B, 2*dof_per_hand) = left then right, in this model's
        hand space (MANO PCA coefficients unless dof_per_hand=45). When None the
        hands stay at their rest pose. Note that finger markers fitted against
        frozen hands can only be satisfied by rotating the wrist, which corrupts
        wrist orientation — either articulate the hands or drop the finger
        markers from the data term.
        """
        B = body_pose.shape[0]
        if hand_pose is None:
            hand_pose = self._zero_hand_pose(B)
        # smplx SMPLH expects left_hand_pose and right_hand_pose separately when use_pca=False
        d = self.dof_per_hand
        out = self.smpl(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=hand_pose[:, :d],
            right_hand_pose=hand_pose[:, d:],
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

        `hand_pose` is (2*dof_per_hand,) or (1, 2*dof_per_hand). Pass the same
        static hand pose used for posing so that finger markers are placed on a
        hand in the configuration they will actually be synthesized from.
        """
        betas_b = betas.view(1, -1)
        zero3 = torch.zeros(1, 3, device=self.device)
        if hand_pose is None:
            hand_pose = self._zero_hand_pose(1)
        else:
            hand_pose = hand_pose.reshape(1, self.hand_pose_dim)
        out = self.smpl(
            betas=betas_b,
            global_orient=zero3,
            body_pose=torch.zeros(1, self.BODY_POSE_DIM, device=self.device),
            left_hand_pose=hand_pose[:, : self.dof_per_hand],
            right_hand_pose=hand_pose[:, self.dof_per_hand :],
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
            left_hand_pose=torch.zeros(1, self.dof_per_hand, device=self.device),
            right_hand_pose=torch.zeros(1, self.dof_per_hand, device=self.device),
            transl=zero3,
            return_verts=False,
        )
        return out.joints[0, 0].detach()
