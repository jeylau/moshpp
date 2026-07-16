"""Stage I (shape + marker placement) and Stage II (per-frame pose) for the
torch port. Uses LBFGS with strong-Wolfe line search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from moshpp.torch_impl.body_model import SMPLHBodyModel
from moshpp.torch_impl.markers import (
    build_local_frame,
    compute_coeffs,
    compute_nn_idx,
    remap_nn_idx,
    synth_markers,
)
from moshpp.torch_impl.vposer_prior import FrozenVPoser


# ---------------------------------------------------------------------------
# Per-frame observations container
# ---------------------------------------------------------------------------


@dataclass
class FrameMarkers:
    """One frame of observed markers.

    obs_xyz: (k, 3) tensor of observed positions for the labels available this frame.
    label_idx: (k,) int tensor, indices into the global latent_labels list.
    """

    obs_xyz: torch.Tensor
    label_idx: torch.Tensor


def _frames_from_array(
    markers: np.ndarray,  # (T, M, 3)
    label_to_idx: Dict[str, int],
    labels: List[str],
    device: torch.device,
) -> List[FrameMarkers]:
    """Convert a dense (T, M, 3) numpy array into a list of FrameMarkers,
    dropping NaN / all-zero markers per frame. The returned indices are into
    the *global* latent_labels ordering (label_to_idx)."""
    T, M, _ = markers.shape
    out: List[FrameMarkers] = []
    for t in range(T):
        rows = []
        idxs = []
        for j in range(M):
            v = markers[t, j]
            if np.any(np.isnan(v)) or (v == 0).all():
                continue
            label = labels[j]
            if label not in label_to_idx:
                continue
            rows.append(v)
            idxs.append(label_to_idx[label])
        if not rows:
            out.append(
                FrameMarkers(
                    obs_xyz=torch.empty(0, 3, device=device),
                    label_idx=torch.empty(0, dtype=torch.long, device=device),
                )
            )
            continue
        out.append(
            FrameMarkers(
                obs_xyz=torch.as_tensor(
                    np.stack(rows), dtype=torch.float32, device=device
                ),
                label_idx=torch.as_tensor(idxs, dtype=torch.long, device=device),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Rigid alignment (SVD, single-frame). Used to seed root rotation/translation.
# ---------------------------------------------------------------------------


def rigid_align(
    src: torch.Tensor, tgt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Arun 1987 rigid alignment. src, tgt: (k, 3). Returns (R, t) such that
    R @ src + t ≈ tgt.

    Executed on CPU regardless of input device — `linalg.svd` and `det` either
    fall back or aren't implemented on MPS, and this is a one-shot init op so
    the device transfer cost is negligible.
    """
    device = src.device
    src_c = src.detach().cpu()
    tgt_c = tgt.detach().cpu()
    src_mean = src_c.mean(0, keepdim=True)
    tgt_mean = tgt_c.mean(0, keepdim=True)
    H = (src_c - src_mean).T @ (tgt_c - tgt_mean)
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    # det(R) > 0 check via det(V)·det(U); MPS lacks linalg.det but CPU is fine here.
    if torch.linalg.det(R) < 0:
        Vt = Vt.clone()
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = tgt_mean.T - R @ src_mean.T
    return R.to(device), t.squeeze(-1).to(device)


def _rotmat_to_aa(R: torch.Tensor) -> torch.Tensor:
    """3x3 -> axis-angle (3,)."""
    from pytorch3d.transforms import matrix_to_axis_angle

    return matrix_to_axis_angle(R.unsqueeze(0)).squeeze(0)


def _seed_root_from_rigid(
    R: torch.Tensor, tvec: torch.Tensor, root_joint: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a world-frame rigid (R, t) into SMPL (global_orient, transl).

    SMPL applies global_orient as a rotation about the root joint, then adds
    transl. To match a world transform `v -> R @ v + t` we need
        transl_smpl = t + R @ J_root - J_root
    so that R @ (v - J_root) + J_root + transl_smpl == R @ v + t.
    """
    aa = _rotmat_to_aa(R)
    transl = tvec + R @ root_joint - root_joint
    return aa, transl


# SMPL local frame: head along +Y. Rotating 180° around local Y flips the
# body's facing direction front/back without moving the pelvis (which sits on
# the local Y axis). Compose as `R_new = R @ _R_LOCAL_Y_180` to apply after R.
_R_LOCAL_Y_180 = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])


# ---------------------------------------------------------------------------
# Body pose masking
# ---------------------------------------------------------------------------

# SMPL joint numbering: 10 = L_Foot, 11 = R_Foot (the toe joints). `body_pose`
# covers joints 1..21, so joint j lives at body_pose[(j-1)*3 : (j-1)*3+3] —
# i.e. toes occupy [27:33]. Legacy drops these from the free variables unless
# `optimize_toes` is set (chmosh.py:389-390, 646-647, pose ids 30:36 on the
# root-inclusive vector).
_TOE_JOINTS = (10, 11)


def _body_pose_mask(optimize_toes: bool, device: torch.device) -> torch.Tensor:
    """(63,) multiplicative mask: 1.0 for free dims, 0.0 for frozen ones.

    Applied as `pose * mask`, which both pins the frozen dims at zero and
    zeroes their gradient, so LBFGS leaves them alone.
    """
    mask = torch.ones(SMPLHBodyModel.BODY_POSE_DIM, device=device)
    if not optimize_toes:
        for j in _TOE_JOINTS:
            mask[(j - 1) * 3 : (j - 1) * 3 + 3] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Stage configuration
# ---------------------------------------------------------------------------


@dataclass
class StageICfg:
    num_betas: int = 10
    optimize_betas: bool = True
    # If False, markers are pinned at vertex_i + normal_i * m2b_distance on the
    # betas-conditioned canonical body, with no free drift. Without the legacy
    # surface loss, free markers + free betas are jointly under-constrained and
    # the optimizer absorbs shape errors into marker drift instead of betas.
    optimize_markers_latent: bool = True
    m2b_distance: float = 0.0095
    # Anneal schedule matching legacy chmosh.py stagei_wt_annealing for SMPL-H.
    # `wt_data` scales inversely with anneal (data weight grows late); `wt_init`,
    # `wt_pose`, `wt_betas` scale proportionally (priors shrink late). `wt_surf`
    # is flat across steps.
    anneal: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125])
    # Default weights match legacy stage I (chmosh.py opt_weights.smplh):
    #   wt_data=75, wt_init=300, wt_pose=3, wt_betas=10, wt_surf=10000.
    # Tuned for marker-count of ~46; legacy scaled wt_data by (46/M), which we
    # don't replicate — callers with very different M may want to adjust.
    wt_data: float = 75.0
    wt_init: float = 300.0
    # VPoser prior on the freely-optimized body pose (analog of legacy wt_poseB,
    # which weighted a GMM over the same raw axis-angle pose).
    wt_pose: float = 3.0
    wt_betas: float = 10.0
    # Let the toe joints (SMPL 10/11) move. Legacy default is False: toes are
    # dropped from the free variables and toe/forefoot markers act on the ankle
    # instead. See _body_pose_mask.
    optimize_toes: bool = False
    # Estimate one static hand pose (90 = 45 left + 45 right, axis-angle offset
    # from the model's hand mean) shared across all reference frames.
    #
    # This is what makes finger markers usable. They are the only markers that
    # observe wrist flexion/extension — LIWR/LOWR sit almost exactly ON the
    # flexion axis, so they move only ~4.5 mm/rad about it (a 30° error moves
    # them ~2mm, under the noise floor) while finger markers move ~372 mm/rad,
    # ~83x more. But finger markers only carry that signal if the hand's shape
    # is right; against a mis-shaped hand they instead drag the wrist to a
    # wrong orientation. A per-frame hand is under-constrained from a handful
    # of finger markers, so we fit a single static one — appropriate whenever
    # the subject holds a roughly fixed hand shape.
    optimize_hand_pose: bool = True
    # L2 on the hand-pose offset, keeping it near the model's hand mean.
    # Legacy stagei_wt_poseH = 3.0 (moshpp_conf.yaml opt_weights.smplh).
    wt_poseH: float = 3.0
    # Anneal step from which the hand pose is free. Kept separate from
    # `markers_latent_free_from_step`: tying the two gives the hand only the
    # final step, and it then has to compete with markers that were freed on
    # the same step. The hand needs the body settled but not the markers, so it
    # gets its own (earlier) schedule.
    hand_pose_free_from_step: int = 1
    # Target the hand-pose prior pulls toward, as a (90,) offset from the body
    # model's own hand zero. Defaults to that zero. With `flat_hand_mean=True`
    # the zero is a straight but splayed hand; pass
    # `flat_fingers_together_hand()` to target a straight hand held together,
    # which is what the unobservable ~30 DoF will then render as.
    hand_pose_mean: Optional[torch.Tensor] = None
    # Per-marker multiplier on the init + surf terms, keyed by label. Values
    # below 1.0 let a marker relocate further from its seed vertex.
    #
    # SMPL's hand is not the subject's hand: for this dataset the fingertips sit
    # ~30mm closer to the wrist than the subject's, a *shape* error no hand
    # *pose* can absorb. Left unaddressed the optimizer instead rotates the
    # wrist to reach them, which is precisely the DoF we are trying to measure.
    # Loosening the finger markers' init lets stage I move them onto the right
    # spot on SMPL's hand once, so stage II's wrist is driven by geometry that
    # actually matches. Legacy has the same per-type knob (stagei_wt_init_finger).
    init_weights: Optional[Dict[str, float]] = None
    # Iso-surface constraint: pin each marker's normal-direction offset in its
    # local frame (coeffs[:, 1], see markers.py:_local_frame where f2 is the
    # triangle normal) to ±m2b_distance. Replaces the legacy scan2mesh `surf`
    # term. Un-annealed, matches legacy wt_surf=10000.
    wt_surf: float = 10000.0
    lbfgs_iters: int = 30
    lbfgs_lr: float = 1.0
    # Refresh nn_idx + init_markers against the current betas and marker
    # positions between anneal steps. Legacy chumpy recomputed 3-NN every
    # iteration; refreshing between LBFGS restarts keeps the in-step graph
    # stable while still tracking markers that slide tangentially.
    refresh_between_steps: bool = True
    # Hold markers_latent frozen for the first `markers_latent_free_from_step`
    # anneal steps, letting betas + z absorb the bulk of the data residual
    # before markers are allowed to refine. Otherwise the optimizer cheats by
    # sliding markers along the body instead of moving `z` away from the rest
    # pose (which carries a stronger prior under VPoser than under the legacy
    # GMM pose prior). Set to 0 to free markers from step 1.
    markers_latent_free_from_step: int = 2


@dataclass
class StageIICfg:
    wt_data: float = 1000.0
    # VPoser prior on the free body pose (legacy wt_poseB, GMM over raw pose).
    wt_pose: float = 0.05
    # Constant-velocity prior. Applied in *pose* space (rad^2), matching legacy
    # chmosh.py:626 which extrapolates opt_model.pose. A previous version of
    # this port penalized acceleration of the VPoser latent instead; a smooth
    # z-path is not a smooth pose-path, so the two are not interchangeable and
    # the old weights do not carry over.
    wt_velo: float = 1.0
    # Per-frame LBFGS (batch_size == 1)
    lbfgs_iters_first: int = 30
    lbfgs_iters: int = 8  # warm-started frames converge in just a few iters
    lbfgs_lr: float = 1.0
    # First-frame annealing schedule on the pose-prior weight, matching MoSh++
    # chmosh.py:637 (which anneals wt_pose × {10, 5, 1} on the initial frame to
    # ease the body out of the rest pose toward the observed markers).
    first_frame_wt_pose_anneal: tuple = (10.0, 5.0, 1.0)
    # See StageICfg.optimize_toes.
    optimize_toes: bool = False
    # Batched LBFGS (batch_size > 1). Set batch_size to the full sequence length
    # to optimize everything at once; smaller values chunk the sequence.
    batch_size: int = 1
    lbfgs_iters_batched: int = 80
    # Compose a 180° rotation around the SMPL local Y axis (= body vertical)
    # onto the rigid-alignment seed. Use when the SVD rigid init lands in the
    # back-facing basin (mesh visibly walks backwards). Pelvis is on local Y
    # so this flips facing direction without shifting body position.
    flip_root_seed_180: bool = False


# ---------------------------------------------------------------------------
# Stage I
# ---------------------------------------------------------------------------


def mosh_stagei(
    *,
    body_model: SMPLHBodyModel,
    vposer: FrozenVPoser,
    stagei_frames: List[FrameMarkers],
    latent_labels: List[str],
    marker_vids: torch.Tensor,  # (M,) int64, vertex id on canonical body for each latent marker
    cfg: StageICfg,
    marker_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Jointly estimate betas, latent marker placements, and per-reference-frame pose.

    `marker_vids[i]` is the seed body-vertex for `latent_labels[i]`.
    """
    device = body_model.device
    N = len(stagei_frames)
    M = len(latent_labels)
    logger.info(f"stage I: N={N} reference frames, M={M} markers")

    # Initial marker positions on the T-pose body, offset along vertex normals by m2b_distance.
    v_template = body_model.template_vertices()  # (V, 3)
    faces = body_model.faces  # (F, 3)
    # Vertex normals via face area-weighted accumulation (simple, no extra dep).
    vert_normals = _vertex_normals(v_template, faces)  # (V, 3)
    init_markers = (
        v_template[marker_vids] + vert_normals[marker_vids] * cfg.m2b_distance
    )  # (M, 3)

    # Build local frame from the *initial* marker positions on the T-pose.
    nn_idx, coeffs = build_local_frame(v_template, init_markers)
    # Iso-surface target for the surf term: each marker's coeffs[:, 1] should
    # stay at ±m2b_distance. Sign comes from the initial 3-NN triangle winding —
    # if the marker is outside the body, this sign points "outward" in the
    # local frame; refresh_between_steps re-derives it whenever nn_idx changes.
    target_normal_offset = torch.sign(coeffs[:, 1]) * cfg.m2b_distance

    # Free variables. markers_latent is only optimized when explicitly requested;
    # otherwise the markers are pinned at v[vid] + n[vid]*d on the betas-conditioned
    # canonical body inside the closure (gradient still flows to betas through that).
    if cfg.optimize_markers_latent:
        markers_latent = nn.Parameter(init_markers.clone())
    else:
        markers_latent = init_markers  # buffer, not a Parameter
    betas = nn.Parameter(torch.zeros(cfg.num_betas, device=device))
    # Free axis-angle body pose, initialized at the T-pose. Starting from zero
    # keeps this consistent with the rigid seed below, which aligns *T-pose*
    # markers to the observations. (The previous z-parameterization seeded from
    # T-pose markers but started the body at decode(z=0), which is not the
    # T-pose — a mismatch that biased the initial root estimate.)
    body_pose = nn.Parameter(
        torch.zeros(N, SMPLHBodyModel.BODY_POSE_DIM, device=device)
    )
    pose_mask = _body_pose_mask(cfg.optimize_toes, device)
    global_orient = nn.Parameter(torch.zeros(N, 3, device=device))
    transl = nn.Parameter(torch.zeros(N, 3, device=device))
    # One static hand pose shared by every reference frame, as an offset from
    # the body model's hand zero. Starts at `hand_pose_mean` (the prior's
    # target) rather than at the model zero.
    hand_mean = (
        torch.zeros(90, device=device)
        if cfg.hand_pose_mean is None
        else cfg.hand_pose_mean.to(device).reshape(90).detach()
    )
    hand_pose = nn.Parameter(hand_mean.clone())

    # (M, 1) per-marker multiplier on the init/surf terms.
    init_scale = torch.ones(M, device=device)
    if cfg.init_weights:
        for i, label in enumerate(latent_labels):
            if label in cfg.init_weights:
                init_scale[i] = float(cfg.init_weights[label])
        loosened = {k: v for k, v in cfg.init_weights.items() if k in latent_labels}
        if loosened:
            logger.info(f"stage I init-weight overrides: {loosened}")

    # Seed root rotation + translation per frame with a rigid alignment from
    # initial simulated markers to observed (corrected for the root joint pivot).
    with torch.no_grad():
        J_root = body_model.root_joint(betas)
        for f_idx, frame in enumerate(stagei_frames):
            if frame.obs_xyz.shape[0] < 3:
                continue
            sim = init_markers[frame.label_idx]
            R, t = rigid_align(sim, frame.obs_xyz)
            aa, transl_smpl = _seed_root_from_rigid(R, t, J_root)
            global_orient[f_idx].copy_(aa)
            transl[f_idx].copy_(transl_smpl)

    init_markers_snapshot = init_markers.clone()  # for tracking marker drift
    n_obs_total = sum(int(f.obs_xyz.shape[0]) for f in stagei_frames)

    for step, anneal in enumerate(cfg.anneal):
        wt_data = cfg.wt_data / max(anneal, 1e-6)
        wt_init = cfg.wt_init * anneal
        wt_pose = cfg.wt_pose * anneal
        wt_betas = cfg.wt_betas * anneal
        wt_poseH = cfg.wt_poseH * anneal

        # Stage I has two phases: shape+pose warmup (markers frozen), then
        # marker refinement. Markers stay frozen until step
        # `markers_latent_free_from_step` so betas+z absorb the bulk of the
        # data residual first.
        markers_free = (
            cfg.optimize_markers_latent and step >= cfg.markers_latent_free_from_step
        )
        if isinstance(markers_latent, nn.Parameter):
            markers_latent.requires_grad_(markers_free)

        params = [body_pose, global_orient, transl]
        if markers_free:
            params.append(markers_latent)
        if cfg.optimize_betas:
            params.append(betas)
        # Hold the hand at the mean until the body has settled: a hand freed
        # against a badly-posed arm just absorbs the arm's error.
        hand_free = cfg.optimize_hand_pose and step >= cfg.hand_pose_free_from_step
        hand_pose.requires_grad_(hand_free)
        if hand_free:
            params.append(hand_pose)

        phase = "refine" if markers_free else "warmup"
        logger.info(
            f"stagei step {step + 1}/{len(cfg.anneal)} anneal={anneal:.2f} "
            f"phase={phase} wt_data={wt_data:.1f} wt_init={wt_init:.2f} "
            f"wt_pose={wt_pose:.2f} hand={'free' if hand_free else 'frozen'}"
        )

        optimizer = torch.optim.LBFGS(
            params,
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_iters,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        def closure():
            optimizer.zero_grad()
            # Recompute coeffs every iteration so the marker layout (and betas, via
            # the canonical body verts) actually flow through to the data residual.
            # The canonical body carries the current hand pose too, so finger
            # markers are placed on the same hand shape they're synthesized from.
            can_verts = body_model.canonical_verts(betas, hand_pose)  # (V, 3)
            if cfg.optimize_markers_latent:
                live_coeffs = compute_coeffs(can_verts, markers_latent, nn_idx)
            else:
                # Marker positions glued to v[vid] + n[vid] * d on the *current*
                # canonical body. The vertex term carries gradient back to betas.
                can_normals = _vertex_normals(can_verts, faces)
                live_markers = (
                    can_verts[marker_vids] + can_normals[marker_vids] * cfg.m2b_distance
                )
                live_coeffs = compute_coeffs(can_verts, live_markers, nn_idx)
            pose_eff = body_pose * pose_mask  # (N, 63)
            betas_b = betas.unsqueeze(0).expand(N, -1)  # (N, num_betas)
            hand_b = hand_pose.unsqueeze(0).expand(N, -1)  # (N, 90)
            verts = body_model(
                betas_b, pose_eff, global_orient, transl, hand_b
            )  # (N, V, 3)
            sim_markers = synth_markers(verts, nn_idx, live_coeffs)  # (N, M, 3)

            data_term = _data_residual(sim_markers, stagei_frames, marker_weights)
            if cfg.optimize_markers_latent:
                # Per-marker init weight. Legacy keeps a separate wt_init per
                # marker type (chmosh.py:329-373, e.g. stagei_wt_init_finger);
                # collapsing it to one scalar pins every marker equally hard,
                # which is wrong when one region's shape mismatch is large.
                init_term = (
                    init_scale * ((markers_latent - init_markers) ** 2).sum(-1)
                ).sum()
                # Iso-surface constraint: keep coeffs[:, 1] (normal-direction
                # component in the local frame) at its initial ±m2b_distance.
                # Markers stay free to slide tangentially via coeffs[:, 0] and
                # coeffs[:, 2] but cannot drift toward/away from the body to
                # absorb data residuals that should be going into betas.
                surf_term = (
                    init_scale * (live_coeffs[:, 1] - target_normal_offset) ** 2
                ).sum()
            else:
                init_term = sim_markers.new_zeros(())
                surf_term = sim_markers.new_zeros(())
            pose_term = vposer.prior_term(pose_eff)
            betas_term = (
                (betas**2).sum()
                if cfg.optimize_betas
                else torch.tensor(0.0, device=device)
            )
            # Pull toward `hand_pose_mean`, not toward zero. ~30 of a hand's 45
            # DoF are unobservable from a typical finger-marker set, so whatever
            # this term targets is what those DoF render as.
            handH_term = ((hand_pose - hand_mean) ** 2).sum()

            loss = (
                wt_data * data_term
                + wt_init * init_term
                + cfg.wt_surf * surf_term
                + wt_pose * pose_term
                + wt_betas * betas_term
                + wt_poseH * handH_term
            )
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        # Diagnostics: data RMSE (independent of weights) + how far betas / markers moved.
        with torch.no_grad():
            can_verts = body_model.canonical_verts(betas, hand_pose)
            if cfg.optimize_markers_latent:
                live_coeffs = compute_coeffs(can_verts, markers_latent, nn_idx)
                live_markers_dbg = markers_latent
            else:
                can_normals = _vertex_normals(can_verts, faces)
                live_markers_dbg = (
                    can_verts[marker_vids] + can_normals[marker_vids] * cfg.m2b_distance
                )
                live_coeffs = compute_coeffs(can_verts, live_markers_dbg, nn_idx)
            pose_eff = body_pose * pose_mask
            betas_b = betas.unsqueeze(0).expand(N, -1)
            verts = body_model(
                betas_b,
                pose_eff,
                global_orient,
                transl,
                hand_pose.unsqueeze(0).expand(N, -1),
            )
            sim_markers = synth_markers(verts, nn_idx, live_coeffs)
            data_se = float(_data_residual(sim_markers, stagei_frames, marker_weights))
            rmse_mm = (data_se / max(n_obs_total, 1)) ** 0.5 * 1000.0
            betas_norm = float(betas.norm())
            mlat_drift_mm = (
                float((live_markers_dbg - init_markers_snapshot).norm(dim=-1).mean())
                * 1000.0
            )
            surf_dev_mm = (
                float((live_coeffs[:, 1] - target_normal_offset).abs().mean()) * 1000.0
            )
        drift_label = (
            "Δmarkers_latent" if cfg.optimize_markers_latent else "Δcan_markers"
        )
        logger.info(
            f"  loss={float(loss):.2f}  data_RMSE={rmse_mm:.1f}mm  "
            f"|betas|={betas_norm:.3f}  mean |{drift_label}|={mlat_drift_mm:.2f}mm  "
            f"surf_dev={surf_dev_mm:.2f}mm"
        )

        # Between-step refresh: track markers that slid tangentially, and rebase
        # the init term + surf target onto the current canonical body. Mirrors
        # legacy's per-iteration 3-NN recomputation but only between LBFGS
        # restarts so each in-step graph stays stable.
        if (
            cfg.refresh_between_steps
            and cfg.optimize_markers_latent
            and step < len(cfg.anneal) - 1
        ):
            with torch.no_grad():
                can_verts_refresh = body_model.canonical_verts(betas, hand_pose)
                # New 3-NN based on current marker positions on current canonical body
                nn_idx = compute_nn_idx(can_verts_refresh, markers_latent)
                refreshed_coeffs = compute_coeffs(
                    can_verts_refresh, markers_latent, nn_idx
                )
                # Preserve outward sign per marker in the new local frame
                target_normal_offset = (
                    torch.sign(refreshed_coeffs[:, 1]) * cfg.m2b_distance
                )
                # Rebase init target onto current betas-conditioned body so the
                # init term penalizes tangential drift only, not shape change.
                can_normals_refresh = _vertex_normals(can_verts_refresh, faces)
                init_markers = (
                    can_verts_refresh[marker_vids]
                    + can_normals_refresh[marker_vids] * cfg.m2b_distance
                )

    # Bake the final coeffs from the converged marker layout + final betas, so
    # stage II sees a marker placement that's consistent with the learned shape.
    with torch.no_grad():
        can_verts_final = body_model.canonical_verts(betas, hand_pose)
        if cfg.optimize_markers_latent:
            final_markers_latent = markers_latent
        else:
            n_final = _vertex_normals(can_verts_final, faces)
            final_markers_latent = (
                can_verts_final[marker_vids] + n_final[marker_vids] * cfg.m2b_distance
            )
        final_coeffs = compute_coeffs(can_verts_final, final_markers_latent, nn_idx)
        pose_eff = body_pose * pose_mask
        betas_b = betas.unsqueeze(0).expand(N, -1)
        verts = body_model(
            betas_b,
            pose_eff,
            global_orient,
            transl,
            hand_pose.unsqueeze(0).expand(N, -1),
        )
        sim_markers = synth_markers(verts, nn_idx, final_coeffs)
    _log_per_marker_rmse(
        _per_marker_rmse_table(
            sim_markers, stagei_frames, latent_labels, marker_weights
        ),
        "stage I per-marker RMSE (over ref frames, worst first):",
    )

    logger.info(
        f"stage I static hand pose: |offset from prior target| = "
        f"{float((hand_pose - hand_mean).norm()):.3f} rad"
    )

    return {
        "betas": betas.detach(),
        "markers_latent": final_markers_latent.detach(),
        "nn_idx": nn_idx,
        "coeffs": final_coeffs.detach(),
        "latent_labels": latent_labels,
        "hand_pose": hand_pose.detach(),
    }


# ---------------------------------------------------------------------------
# Stage II
# ---------------------------------------------------------------------------


def mosh_stageii(
    *,
    body_model: SMPLHBodyModel,
    vposer: FrozenVPoser,
    observed_frames: List[FrameMarkers],
    betas: torch.Tensor,  # (num_betas,)
    nn_idx: torch.Tensor,  # (M, 3) from stage I (or from template if v0)
    coeffs: torch.Tensor,  # (M, 3) from stage I (or from template if v0)
    cfg: StageIICfg,
    latent_labels: Optional[List[str]] = None,
    marker_weights: Optional[torch.Tensor] = None,
    hand_pose: Optional[torch.Tensor] = None,  # (90,) static, from stage I
) -> Dict[str, torch.Tensor]:
    """Pose estimation against observed markers (shape and marker placement frozen).

    `hand_pose` is the static per-subject hand from stage I, held fixed here.
    With the hand at a shape that matches the subject, finger markers become
    the primary observers of wrist flexion/extension — see StageICfg.

    Dispatches to per-frame LBFGS (``cfg.batch_size == 1``) or batched LBFGS
    over chunks (``cfg.batch_size > 1``).
    """
    if cfg.batch_size <= 1:
        out = _mosh_stageii_per_frame(
            body_model=body_model,
            vposer=vposer,
            observed_frames=observed_frames,
            betas=betas,
            nn_idx=nn_idx,
            coeffs=coeffs,
            cfg=cfg,
            marker_weights=marker_weights,
            hand_pose=hand_pose,
        )
    else:
        out = _mosh_stageii_batched(
            body_model=body_model,
            vposer=vposer,
            observed_frames=observed_frames,
            betas=betas,
            nn_idx=nn_idx,
            coeffs=coeffs,
            cfg=cfg,
            marker_weights=marker_weights,
            hand_pose=hand_pose,
        )

    if latent_labels is not None:
        _report_stageii_per_marker(
            body_model=body_model,
            observed_frames=observed_frames,
            stageii_out=out,
            nn_idx=nn_idx,
            coeffs=coeffs,
            betas=betas,
            latent_labels=latent_labels,
            marker_weights=marker_weights,
            hand_pose=hand_pose,
        )
    return out


def _report_stageii_per_marker(
    *,
    body_model,
    observed_frames,
    stageii_out,
    nn_idx,
    coeffs,
    betas,
    latent_labels,
    marker_weights=None,
    hand_pose=None,
) -> None:
    """Recompute simulated markers for every frame and log the per-marker RMSE table."""
    device = body_model.device
    T = stageii_out["body_pose"].shape[0]
    chunk = 64
    sim_chunks = []
    subset_vids, nn_idx_local = remap_nn_idx(nn_idx)
    subset = body_model.make_vertex_subset(subset_vids, betas)
    with torch.no_grad():
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            v = body_model.forward_subset(
                subset,
                stageii_out["body_pose"][s:e],
                stageii_out["global_orient"][s:e],
                stageii_out["transl"][s:e],
                None if hand_pose is None else hand_pose.unsqueeze(0).expand(e - s, -1),
            )
            sim_chunks.append(synth_markers(v, nn_idx_local, coeffs))
        sim_all = torch.cat(sim_chunks, dim=0)
    _log_per_marker_rmse(
        _per_marker_rmse_table(sim_all, observed_frames, latent_labels, marker_weights),
        "stage II per-marker RMSE (over all frames, worst first):",
    )


def _mosh_stageii_per_frame(
    *,
    body_model: SMPLHBodyModel,
    vposer: FrozenVPoser,
    observed_frames: List[FrameMarkers],
    betas: torch.Tensor,
    nn_idx: torch.Tensor,
    coeffs: torch.Tensor,
    cfg: StageIICfg,
    marker_weights: Optional[torch.Tensor] = None,
    hand_pose: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Original per-frame LBFGS. Kept verbatim for parity with prior runs."""
    device = body_model.device
    T = len(observed_frames)
    logger.info(f"stage II (per-frame): T={T} frames")

    out_global = torch.zeros(T, 3, device=device)
    out_body = torch.zeros(T, 63, device=device)
    out_transl = torch.zeros(T, 3, device=device)
    out_loss = torch.zeros(T, device=device)
    pose_mask = _body_pose_mask(cfg.optimize_toes, device)
    hand_b1 = None if hand_pose is None else hand_pose.reshape(1, 90)

    # See the batched path: skin only the vertices the markers read.
    subset_vids, nn_idx_local = remap_nn_idx(nn_idx)
    subset = body_model.make_vertex_subset(subset_vids, betas)
    logger.info(
        f"  masked LBS: skinning {len(subset_vids)} of {body_model.num_verts} vertices"
    )

    # Per-frame state
    pose_prev = None
    pose_prev2 = None
    global_prev = None
    transl_prev = None
    first = True

    # We rebuild the optimizer per frame to keep state isolated (LBFGS history
    # would otherwise be poisoned by the per-frame switch).
    M = nn_idx.shape[0]

    for t in range(T):
        frame = observed_frames[t]
        if frame.obs_xyz.shape[0] < 3:
            logger.warning(
                f"frame {t}: only {frame.obs_xyz.shape[0]} markers, skipping"
            )
            if not first:
                out_body[t] = out_body[t - 1]
                out_global[t] = out_global[t - 1]
                out_transl[t] = out_transl[t - 1]
            continue

        # Warm-start from previous frame
        pose_t = nn.Parameter(
            pose_prev.clone()
            if pose_prev is not None
            else torch.zeros(SMPLHBodyModel.BODY_POSE_DIM, device=device)
        )
        global_t = nn.Parameter(
            global_prev.clone()
            if global_prev is not None
            else torch.zeros(3, device=device)
        )
        transl_t = nn.Parameter(
            transl_prev.clone()
            if transl_prev is not None
            else torch.zeros(3, device=device)
        )

        if first:
            # Rigid-align the rest body to the observed markers
            with torch.no_grad():
                betas_b = betas.unsqueeze(0)
                # Use a zero-transl forward to recover where the body actually sits
                # under the current (R=I, t=0) so the rigid alignment is correct.
                verts = body_model.forward_subset(
                    subset,
                    (pose_t * pose_mask).unsqueeze(0),
                    torch.zeros(1, 3, device=device),
                    torch.zeros(1, 3, device=device),
                    hand_b1,
                )
                sim = synth_markers(verts, nn_idx_local, coeffs).squeeze(0)
                sim_sub = sim[frame.label_idx]
                R, tvec = rigid_align(sim_sub, frame.obs_xyz)
                if cfg.flip_root_seed_180:
                    R = R @ _R_LOCAL_Y_180.to(R)
                J_root = body_model.root_joint(betas)
                aa, transl_smpl = _seed_root_from_rigid(R, tvec, J_root)
                global_t.copy_(aa)
                transl_t.copy_(transl_smpl)

        # Velocity term needs the *previous* poses; freeze their values
        pose_prev_val = pose_prev.detach() if pose_prev is not None else None
        pose_prev2_val = pose_prev2.detach() if pose_prev2 is not None else None

        def build_closure(opt, wt_pose_mult: float):
            def closure():
                opt.zero_grad()
                pose_eff = (pose_t * pose_mask).unsqueeze(0)  # (1, 63)
                betas_b = betas.unsqueeze(0)
                verts = body_model.forward_subset(
                    subset,
                    pose_eff,
                    global_t.unsqueeze(0),
                    transl_t.unsqueeze(0),
                    hand_b1,
                    betas=betas,
                )
                sim = synth_markers(verts, nn_idx_local, coeffs).squeeze(0)  # (M, 3)
                sim_sub = sim[frame.label_idx]  # (k, 3)
                se = ((sim_sub - frame.obs_xyz) ** 2).sum(dim=-1)  # (k,)
                if marker_weights is not None:
                    se = se * marker_weights[frame.label_idx]
                data_term = se.sum()
                pose_term = vposer.prior_term(pose_eff)
                loss = (
                    cfg.wt_data * data_term + (cfg.wt_pose * wt_pose_mult) * pose_term
                )
                if pose_prev_val is not None and pose_prev2_val is not None:
                    # constant-velocity extrapolation in pose space (legacy chmosh.py:626)
                    pose_pred = 2 * pose_prev_val - pose_prev2_val
                    loss = loss + cfg.wt_velo * ((pose_t - pose_pred) ** 2).sum()
                loss.backward()
                return loss

            return closure

        if first:
            # 3-pass annealing on the pose-prior weight (mirrors chmosh.py:637).
            # Starting from the rest pose, a strong prior pulls toward plausible
            # body shapes first, then we relax it to let the data dominate.
            for wt_pose_mult in cfg.first_frame_wt_pose_anneal:
                optimizer = torch.optim.LBFGS(
                    [pose_t, global_t, transl_t],
                    lr=cfg.lbfgs_lr,
                    max_iter=cfg.lbfgs_iters_first,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=1e-7,
                    tolerance_change=1e-9,
                )
                loss = optimizer.step(build_closure(optimizer, wt_pose_mult))
        else:
            optimizer = torch.optim.LBFGS(
                [pose_t, global_t, transl_t],
                lr=cfg.lbfgs_lr,
                max_iter=cfg.lbfgs_iters,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
            )
            loss = optimizer.step(build_closure(optimizer, 1.0))

        out_global[t] = global_t.detach()
        out_transl[t] = transl_t.detach()
        out_loss[t] = float(loss)

        with torch.no_grad():
            pose_eff_f = (pose_t.detach() * pose_mask).unsqueeze(0)
            out_body[t] = pose_eff_f.squeeze(0)
            # Report per-marker RMSE in mm — the loss number alone is hard to read.
            verts_f = body_model.forward_subset(
                subset,
                pose_eff_f,
                global_t.detach().unsqueeze(0),
                transl_t.detach().unsqueeze(0),
                hand_b1,
            )
            sim_f = synth_markers(verts_f, nn_idx_local, coeffs).squeeze(0)
            sim_sub_f = sim_f[frame.label_idx]
            rmse_mm = float(((sim_sub_f - frame.obs_xyz) ** 2).mean().sqrt()) * 1000.0

        pose_prev2 = pose_prev
        pose_prev = pose_t.detach()
        global_prev = global_t.detach()
        transl_prev = transl_t.detach()
        first = False

        if t % 25 == 0:
            logger.info(f"  frame {t}/{T} loss={float(loss):.2f}  RMSE={rmse_mm:.1f}mm")

    return {
        "betas": betas.detach(),
        "body_pose": out_body,
        "global_orient": out_global,
        "transl": out_transl,
        "loss": out_loss,
    }


# ---------------------------------------------------------------------------
# Stage II — batched (joint LBFGS over chunks of frames)
# ---------------------------------------------------------------------------


def _mosh_stageii_batched(
    *,
    body_model: SMPLHBodyModel,
    vposer: FrozenVPoser,
    observed_frames: List[FrameMarkers],
    betas: torch.Tensor,
    nn_idx: torch.Tensor,
    coeffs: torch.Tensor,
    cfg: StageIICfg,
    marker_weights: Optional[torch.Tensor] = None,
    hand_pose: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Optimize ``cfg.batch_size`` frames jointly per LBFGS call.

    Frames within a chunk are coupled via the constant-velocity pose prior;
    boundaries between chunks are stitched by treating the previous chunk's
    last two poses as fixed and continuing the velocity prior across.
    """
    device = body_model.device
    T = len(observed_frames)
    B = max(1, cfg.batch_size)
    logger.info(f"stage II (batched): T={T} frames, batch_size={B}")

    out_global = torch.zeros(T, 3, device=device)
    out_transl = torch.zeros(T, 3, device=device)
    out_body = torch.zeros(T, 63, device=device)
    out_loss = torch.zeros(T, device=device)
    pose_mask = _body_pose_mask(cfg.optimize_toes, device)

    # Stage II holds shape and marker layout fixed, so only the ~150 vertices the
    # markers read ever matter. Skin just those instead of all 6890 (see
    # SMPLHBodyModel.make_vertex_subset). Valid only because betas is frozen here.
    subset_vids, nn_idx_local = remap_nn_idx(nn_idx)
    subset = body_model.make_vertex_subset(subset_vids, betas)
    logger.info(
        f"  masked LBS: skinning {len(subset_vids)} of {body_model.num_verts} vertices"
    )

    # Phase 1: rigid-align every frame independently from the rest body.
    # This gives each frame a sensible global_orient/transl before LBFGS.
    with torch.no_grad():
        J_root = body_model.root_joint(betas)
        body_pose0 = torch.zeros(1, SMPLHBodyModel.BODY_POSE_DIM, device=device)
        verts0 = body_model(
            betas.unsqueeze(0),
            body_pose0,
            torch.zeros(1, 3, device=device),
            torch.zeros(1, 3, device=device),
            None if hand_pose is None else hand_pose.reshape(1, 90),
        )
        sim0 = synth_markers(verts0, nn_idx, coeffs).squeeze(0)  # (M, 3)
        for t, frame in enumerate(observed_frames):
            if frame.obs_xyz.shape[0] < 3:
                # Carry over previous frame's seed if available
                if t > 0:
                    out_global[t] = out_global[t - 1]
                    out_transl[t] = out_transl[t - 1]
                continue
            sim_sub = sim0[frame.label_idx]
            R, tvec = rigid_align(sim_sub, frame.obs_xyz)
            if cfg.flip_root_seed_180:
                R = R @ _R_LOCAL_Y_180.to(R)
            aa, transl_smpl = _seed_root_from_rigid(R, tvec, J_root)
            out_global[t] = aa
            out_transl[t] = transl_smpl

    # Phase 2: chunked LBFGS.
    for chunk_start in range(0, T, B):
        chunk_end = min(chunk_start + B, T)
        chunk_T = chunk_end - chunk_start
        chunk_frames = observed_frames[chunk_start:chunk_end]

        # Warm-start each chunk from the previous chunk's last solved pose,
        # rather than from the T-pose — the first chunk has nothing to inherit.
        pose_init = out_body[chunk_start:chunk_end].clone()
        if chunk_start >= 1:
            pose_init[:] = out_body[chunk_start - 1]
        pose_chunk = nn.Parameter(pose_init)
        global_chunk = nn.Parameter(out_global[chunk_start:chunk_end].clone())
        transl_chunk = nn.Parameter(out_transl[chunk_start:chunk_end].clone())

        # Boundary pose values from the previous chunk (detached, no grad).
        pose_bm1 = out_body[chunk_start - 1].clone() if chunk_start >= 1 else None
        pose_bm2 = out_body[chunk_start - 2].clone() if chunk_start >= 2 else None

        betas_b = betas.unsqueeze(0).expand(chunk_T, -1)
        hand_bc = (
            None if hand_pose is None else hand_pose.reshape(1, 90).expand(chunk_T, -1)
        )

        optimizer = torch.optim.LBFGS(
            [pose_chunk, global_chunk, transl_chunk],
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_iters_batched,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        def closure():
            optimizer.zero_grad()
            pose_eff = pose_chunk * pose_mask  # (chunk_T, 63)
            verts = body_model.forward_subset(
                subset, pose_eff, global_chunk, transl_chunk, hand_bc, betas=betas
            )  # (chunk_T, S, 3)
            sim = synth_markers(verts, nn_idx_local, coeffs)  # (chunk_T, M, 3)

            data_term = sim.new_zeros(())
            for i, frame in enumerate(chunk_frames):
                if frame.obs_xyz.shape[0]:
                    sim_sub = sim[i, frame.label_idx]
                    se = ((sim_sub - frame.obs_xyz) ** 2).sum(dim=-1)
                    if marker_weights is not None:
                        se = se * marker_weights[frame.label_idx]
                    data_term = data_term + se.sum()

            pose_term = vposer.prior_term(pose_eff)

            # Constant-velocity prior in pose space within the chunk
            velo_term = sim.new_zeros(())
            if chunk_T >= 3:
                pred = 2 * pose_chunk[1:-1] - pose_chunk[:-2]
                velo_term = velo_term + ((pose_chunk[2:] - pred) ** 2).sum()

            # Stitch with previous chunk's last two frames
            if pose_bm1 is not None and pose_bm2 is not None:
                pred0 = 2 * pose_bm1 - pose_bm2
                velo_term = velo_term + ((pose_chunk[0] - pred0) ** 2).sum()
                if chunk_T >= 2:
                    pred1 = 2 * pose_chunk[0] - pose_bm1
                    velo_term = velo_term + ((pose_chunk[1] - pred1) ** 2).sum()

            loss = (
                cfg.wt_data * data_term
                + cfg.wt_pose * pose_term
                + cfg.wt_velo * velo_term
            )
            loss.backward()
            return loss

        final_loss = optimizer.step(closure)

        with torch.no_grad():
            out_global[chunk_start:chunk_end] = global_chunk.detach()
            out_transl[chunk_start:chunk_end] = transl_chunk.detach()
            body_pose = pose_chunk.detach() * pose_mask
            out_body[chunk_start:chunk_end] = body_pose

            # Per-frame loss + chunk RMSE for logging
            verts_final = body_model.forward_subset(
                subset,
                body_pose,
                global_chunk.detach(),
                transl_chunk.detach(),
                hand_bc,
            )
            sim_final = synth_markers(verts_final, nn_idx_local, coeffs)
            total_se = 0.0
            total_n = 0
            for i, frame in enumerate(chunk_frames):
                if frame.obs_xyz.shape[0]:
                    sim_sub = sim_final[i, frame.label_idx]
                    se = ((sim_sub - frame.obs_xyz) ** 2).sum()
                    out_loss[chunk_start + i] = float(se)
                    total_se += float(se)
                    total_n += frame.obs_xyz.shape[0]
            rmse_mm = (total_se / max(total_n, 1)) ** 0.5 * 1000.0

        logger.info(
            f"  chunk [{chunk_start:4d}:{chunk_end:4d}]  "
            f"loss={float(final_loss):.2f}  RMSE={rmse_mm:.1f}mm"
        )

    return {
        "betas": betas.detach(),
        "body_pose": out_body,
        "global_orient": out_global,
        "transl": out_transl,
        "loss": out_loss,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _per_marker_rmse_table(
    sim_per_frame: torch.Tensor,  # (N, M, 3) simulated markers per frame
    observed_frames: List[FrameMarkers],
    latent_labels: List[str],
    marker_weights: Optional[torch.Tensor] = None,
):
    """Returns [(label, rmse_mm, n_obs, weight)] sorted by rmse_mm descending.

    RMSE is always the raw geometric error, unweighted — `marker_weights` is
    carried through only so the caller can flag markers the fit ignored. A
    zero-weight marker with a large RMSE is expected, not a failure.
    """
    device = sim_per_frame.device
    M = sim_per_frame.shape[1]
    per_marker_se = torch.zeros(M, device=device)
    per_marker_n = torch.zeros(M, device=device)
    for t, frame in enumerate(observed_frames):
        if frame.obs_xyz.shape[0] == 0:
            continue
        sim_sub = sim_per_frame[t, frame.label_idx]
        se = ((sim_sub - frame.obs_xyz) ** 2).sum(dim=-1)  # (k,)
        per_marker_se.index_add_(0, frame.label_idx, se)
        per_marker_n.index_add_(0, frame.label_idx, torch.ones_like(se))
    rmse_mm = (per_marker_se / per_marker_n.clamp(min=1)).sqrt() * 1000.0
    rmse_list = rmse_mm.cpu().tolist()
    n_obs = per_marker_n.long().cpu().tolist()
    wts = marker_weights.cpu().tolist() if marker_weights is not None else [1.0] * M
    rows = [
        (latent_labels[i], rmse_list[i], n_obs[i], wts[i])
        for i in range(M)
        if n_obs[i] > 0
    ]
    rows.sort(key=lambda r: -r[1])
    return rows


def _log_per_marker_rmse(rows, header: str) -> None:
    if not rows:
        return
    width = max(len(r[0]) for r in rows)
    logger.info(f"{header}")
    for label, rmse, n, wt in rows:
        if wt == 0.0:
            note = "  [excluded, wt=0]"
        elif wt != 1.0:
            note = f"  [wt={wt:g}]"
        else:
            note = ""
        logger.info(f"  {label:<{width}}  {rmse:6.1f} mm  ({n} obs){note}")
    fitted = [r[1] for r in rows if r[3] != 0.0]
    if fitted:
        logger.info(
            f"  -> {len(fitted)} fitted markers: "
            f"mean {sum(fitted) / len(fitted):.1f} mm, max {max(fitted):.1f} mm"
        )


def _data_residual(
    sim_markers: torch.Tensor,
    frames: List[FrameMarkers],
    marker_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sum-of-squares over the (frame, marker) pairs that are actually observed.

    `marker_weights`: optional (M,) tensor multiplying each marker's squared
    residual. Set to 0.0 to drop a marker entirely; >1.0 to boost.
    """
    total = sim_markers.new_zeros(())
    for f_idx, frame in enumerate(frames):
        if frame.obs_xyz.shape[0] == 0:
            continue
        sim_sub = sim_markers[f_idx, frame.label_idx]
        se = ((sim_sub - frame.obs_xyz) ** 2).sum(dim=-1)  # (k,)
        if marker_weights is not None:
            se = se * marker_weights[frame.label_idx]
        total = total + se.sum()
    return total


def _vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals. verts: (V, 3), faces: (F, 3)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = torch.cross(
        v1 - v0, v2 - v0, dim=-1
    )  # not normalized -> area-weighted
    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], face_normals)
    vn.index_add_(0, faces[:, 1], face_normals)
    vn.index_add_(0, faces[:, 2], face_normals)
    return torch.nn.functional.normalize(vn, dim=-1, eps=1e-12)
