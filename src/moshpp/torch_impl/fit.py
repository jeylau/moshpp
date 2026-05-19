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
    synth_markers,
)
from moshpp.torch_impl.vposer_prior import FrozenVPoser, LATENT_DIM


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


# ---------------------------------------------------------------------------
# Stage configuration
# ---------------------------------------------------------------------------


@dataclass
class StageICfg:
    num_betas: int = 10
    optimize_betas: bool = True
    m2b_distance: float = 0.0095  # meters, used only for initial marker placement
    anneal: List[float] = field(default_factory=lambda: [10.0, 5.0, 2.0, 1.0, 0.5])
    wt_data: float = 1000.0
    wt_init: float = 50.0
    wt_z: float = 5.0
    wt_betas: float = 1.0
    lbfgs_iters: int = 30
    lbfgs_lr: float = 1.0


@dataclass
class StageIICfg:
    wt_data: float = 1000.0
    wt_z: float = 5.0
    wt_velo: float = 100.0
    # Per-frame LBFGS (batch_size == 1)
    lbfgs_iters_first: int = 30
    lbfgs_iters: int = 8  # warm-started frames converge in just a few iters
    lbfgs_lr: float = 1.0
    # First-frame annealing schedule on the z-prior weight, matching MoSh++
    # chmosh.py:637 (which anneals wt_pose × {10, 5, 1} on the initial frame to
    # ease the body out of the rest pose toward the observed markers).
    first_frame_wt_z_anneal: tuple = (10.0, 5.0, 1.0)
    # Batched LBFGS (batch_size > 1). Set batch_size to the full sequence length
    # to optimize everything at once; smaller values chunk the sequence.
    batch_size: int = 1
    lbfgs_iters_batched: int = 80


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

    # Free variables
    markers_latent = nn.Parameter(init_markers.clone())
    betas = nn.Parameter(torch.zeros(cfg.num_betas, device=device))
    z = nn.Parameter(torch.zeros(N, LATENT_DIM, device=device))
    global_orient = nn.Parameter(torch.zeros(N, 3, device=device))
    transl = nn.Parameter(torch.zeros(N, 3, device=device))

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

    params = [markers_latent, z, global_orient, transl]
    if cfg.optimize_betas:
        params.append(betas)

    init_markers_snapshot = init_markers.clone()  # for tracking marker drift
    n_obs_total = sum(int(f.obs_xyz.shape[0]) for f in stagei_frames)

    for step, anneal in enumerate(cfg.anneal):
        wt_data = cfg.wt_data / max(anneal, 1e-6)
        wt_init = cfg.wt_init * anneal
        wt_z = cfg.wt_z * anneal
        wt_betas = cfg.wt_betas * anneal
        logger.info(
            f"stagei step {step + 1}/{len(cfg.anneal)} anneal={anneal:.2f} "
            f"wt_data={wt_data:.1f} wt_init={wt_init:.2f} wt_z={wt_z:.2f}"
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
            # Recompute coeffs every iteration so markers_latent (and betas, via the
            # canonical body verts) actually flow through to the data residual.
            # This mirrors chumpy's TransformedCoeffs.on_changed in MoSh++.
            can_verts = body_model.canonical_verts(betas)  # (V, 3)
            live_coeffs = compute_coeffs(can_verts, markers_latent, nn_idx)  # (M, 3)
            body_pose = vposer.decode_aa(z)  # (N, 63)
            betas_b = betas.unsqueeze(0).expand(N, -1)  # (N, num_betas)
            verts = body_model(betas_b, body_pose, global_orient, transl)  # (N, V, 3)
            sim_markers = synth_markers(verts, nn_idx, live_coeffs)  # (N, M, 3)

            data_term = _data_residual(sim_markers, stagei_frames)
            init_term = ((markers_latent - init_markers) ** 2).sum()
            z_term = (z**2).sum()
            betas_term = (
                (betas**2).sum()
                if cfg.optimize_betas
                else torch.tensor(0.0, device=device)
            )

            loss = (
                wt_data * data_term
                + wt_init * init_term
                + wt_z * z_term
                + wt_betas * betas_term
            )
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        # Diagnostics: data RMSE (independent of weights) + how far betas / markers moved.
        with torch.no_grad():
            can_verts = body_model.canonical_verts(betas)
            live_coeffs = compute_coeffs(can_verts, markers_latent, nn_idx)
            body_pose = vposer.decode_aa(z)
            betas_b = betas.unsqueeze(0).expand(N, -1)
            verts = body_model(betas_b, body_pose, global_orient, transl)
            sim_markers = synth_markers(verts, nn_idx, live_coeffs)
            data_se = float(_data_residual(sim_markers, stagei_frames))
            rmse_mm = (data_se / max(n_obs_total, 1)) ** 0.5 * 1000.0
            betas_norm = float(betas.norm())
            mlat_drift_mm = (
                float((markers_latent - init_markers_snapshot).norm(dim=-1).mean())
                * 1000.0
            )
        logger.info(
            f"  loss={float(loss):.2f}  data_RMSE={rmse_mm:.1f}mm  "
            f"|betas|={betas_norm:.3f}  mean |Δmarkers_latent|={mlat_drift_mm:.2f}mm"
        )

    # Bake the final coeffs from the optimized markers_latent + final betas, so
    # stage II sees a marker placement that's consistent with the learned shape.
    with torch.no_grad():
        can_verts_final = body_model.canonical_verts(betas)
        final_coeffs = compute_coeffs(can_verts_final, markers_latent, nn_idx)
        body_pose = vposer.decode_aa(z)
        betas_b = betas.unsqueeze(0).expand(N, -1)
        verts = body_model(betas_b, body_pose, global_orient, transl)
        sim_markers = synth_markers(verts, nn_idx, final_coeffs)
    _log_per_marker_rmse(
        _per_marker_rmse_table(sim_markers, stagei_frames, latent_labels),
        "stage I per-marker RMSE (over ref frames, worst first):",
    )

    return {
        "betas": betas.detach(),
        "markers_latent": markers_latent.detach(),
        "nn_idx": nn_idx,
        "coeffs": final_coeffs.detach(),
        "latent_labels": latent_labels,
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
) -> Dict[str, torch.Tensor]:
    """Pose estimation against observed markers (shape and marker placement frozen).

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
        )
    return out


def _report_stageii_per_marker(
    *, body_model, observed_frames, stageii_out, nn_idx, coeffs, betas, latent_labels
) -> None:
    """Recompute simulated markers for every frame and log the per-marker RMSE table."""
    device = body_model.device
    T = stageii_out["body_pose"].shape[0]
    betas_b = betas.unsqueeze(0).expand(T, -1)
    chunk = 64
    sim_chunks = []
    with torch.no_grad():
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            v = body_model(
                betas_b[s:e],
                stageii_out["body_pose"][s:e],
                stageii_out["global_orient"][s:e],
                stageii_out["transl"][s:e],
            )
            sim_chunks.append(synth_markers(v, nn_idx, coeffs))
        sim_all = torch.cat(sim_chunks, dim=0)
    _log_per_marker_rmse(
        _per_marker_rmse_table(sim_all, observed_frames, latent_labels),
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
) -> Dict[str, torch.Tensor]:
    """Original per-frame LBFGS. Kept verbatim for parity with prior runs."""
    device = body_model.device
    T = len(observed_frames)
    logger.info(f"stage II (per-frame): T={T} frames")

    out_global = torch.zeros(T, 3, device=device)
    out_body = torch.zeros(T, 63, device=device)
    out_transl = torch.zeros(T, 3, device=device)
    out_z = torch.zeros(T, LATENT_DIM, device=device)
    out_loss = torch.zeros(T, device=device)

    # Per-frame state
    z_prev = None
    z_prev2 = None
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
                out_z[t] = out_z[t - 1]
                out_global[t] = out_global[t - 1]
                out_transl[t] = out_transl[t - 1]
            continue

        # Warm-start from previous frame
        z_t = nn.Parameter(
            z_prev.clone()
            if z_prev is not None
            else torch.zeros(LATENT_DIM, device=device)
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
                body_pose = vposer.decode_aa(z_t.unsqueeze(0))
                # Use a zero-transl forward to recover where the body actually sits
                # under the current (R=I, t=0) so the rigid alignment is correct.
                verts = body_model(
                    betas_b,
                    body_pose,
                    torch.zeros(1, 3, device=device),
                    torch.zeros(1, 3, device=device),
                )
                sim = synth_markers(verts, nn_idx, coeffs).squeeze(0)
                sim_sub = sim[frame.label_idx]
                R, tvec = rigid_align(sim_sub, frame.obs_xyz)
                J_root = body_model.root_joint(betas)
                aa, transl_smpl = _seed_root_from_rigid(R, tvec, J_root)
                global_t.copy_(aa)
                transl_t.copy_(transl_smpl)

        # Velocity term needs the *previous* z; freeze its value
        z_prev_val = z_prev.detach() if z_prev is not None else None
        z_prev2_val = z_prev2.detach() if z_prev2 is not None else None

        def build_closure(opt, wt_z_mult: float):
            def closure():
                opt.zero_grad()
                body_pose = vposer.decode_aa(z_t.unsqueeze(0))  # (1, 63)
                betas_b = betas.unsqueeze(0)
                verts = body_model(
                    betas_b, body_pose, global_t.unsqueeze(0), transl_t.unsqueeze(0)
                )
                sim = synth_markers(verts, nn_idx, coeffs).squeeze(0)  # (M, 3)
                sim_sub = sim[frame.label_idx]  # (k, 3)
                data_term = ((sim_sub - frame.obs_xyz) ** 2).sum()
                z_term = (z_t**2).sum()
                loss = cfg.wt_data * data_term + (cfg.wt_z * wt_z_mult) * z_term
                if z_prev_val is not None and z_prev2_val is not None:
                    # constant-velocity extrapolation in z-space
                    z_pred = 2 * z_prev_val - z_prev2_val
                    loss = loss + cfg.wt_velo * ((z_t - z_pred) ** 2).sum()
                loss.backward()
                return loss

            return closure

        if first:
            # 3-pass annealing on the z-prior weight (mirrors chmosh.py:637).
            # Starting from the rest pose, a strong prior pulls toward plausible
            # body shapes first, then we relax it to let the data dominate.
            for wt_z_mult in cfg.first_frame_wt_z_anneal:
                optimizer = torch.optim.LBFGS(
                    [z_t, global_t, transl_t],
                    lr=cfg.lbfgs_lr,
                    max_iter=cfg.lbfgs_iters_first,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=1e-7,
                    tolerance_change=1e-9,
                )
                loss = optimizer.step(build_closure(optimizer, wt_z_mult))
        else:
            optimizer = torch.optim.LBFGS(
                [z_t, global_t, transl_t],
                lr=cfg.lbfgs_lr,
                max_iter=cfg.lbfgs_iters,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
            )
            loss = optimizer.step(build_closure(optimizer, 1.0))

        out_z[t] = z_t.detach()
        out_global[t] = global_t.detach()
        out_transl[t] = transl_t.detach()
        out_loss[t] = float(loss)

        with torch.no_grad():
            out_body[t] = vposer.decode_aa(z_t.detach().unsqueeze(0)).squeeze(0)
            # Report per-marker RMSE in mm — the loss number alone is hard to read.
            body_pose = vposer.decode_aa(z_t.detach().unsqueeze(0))
            verts_f = body_model(
                betas.unsqueeze(0),
                body_pose,
                global_t.detach().unsqueeze(0),
                transl_t.detach().unsqueeze(0),
            )
            sim_f = synth_markers(verts_f, nn_idx, coeffs).squeeze(0)
            sim_sub_f = sim_f[frame.label_idx]
            rmse_mm = float(((sim_sub_f - frame.obs_xyz) ** 2).mean().sqrt()) * 1000.0

        z_prev2 = z_prev
        z_prev = z_t.detach()
        global_prev = global_t.detach()
        transl_prev = transl_t.detach()
        first = False

        if t % 25 == 0:
            logger.info(f"  frame {t}/{T} loss={float(loss):.2f}  RMSE={rmse_mm:.1f}mm")

    return {
        "betas": betas.detach(),
        "z": out_z,
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
) -> Dict[str, torch.Tensor]:
    """Optimize ``cfg.batch_size`` frames jointly per LBFGS call.

    Frames within a chunk are coupled via the constant-velocity z-space prior;
    boundaries between chunks are stitched by treating the previous chunk's
    last two z values as fixed and continuing the velocity prior across.
    """
    device = body_model.device
    T = len(observed_frames)
    B = max(1, cfg.batch_size)
    logger.info(f"stage II (batched): T={T} frames, batch_size={B}")

    out_z = torch.zeros(T, LATENT_DIM, device=device)
    out_global = torch.zeros(T, 3, device=device)
    out_transl = torch.zeros(T, 3, device=device)
    out_body = torch.zeros(T, 63, device=device)
    out_loss = torch.zeros(T, device=device)

    # Phase 1: rigid-align every frame independently from the rest body.
    # This gives each frame a sensible global_orient/transl before LBFGS.
    with torch.no_grad():
        J_root = body_model.root_joint(betas)
        body_pose0 = vposer.decode_aa(torch.zeros(1, LATENT_DIM, device=device))
        verts0 = body_model(
            betas.unsqueeze(0),
            body_pose0,
            torch.zeros(1, 3, device=device),
            torch.zeros(1, 3, device=device),
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
            aa, transl_smpl = _seed_root_from_rigid(R, tvec, J_root)
            out_global[t] = aa
            out_transl[t] = transl_smpl

    # Phase 2: chunked LBFGS.
    for chunk_start in range(0, T, B):
        chunk_end = min(chunk_start + B, T)
        chunk_T = chunk_end - chunk_start
        chunk_frames = observed_frames[chunk_start:chunk_end]

        z_chunk = nn.Parameter(out_z[chunk_start:chunk_end].clone())
        global_chunk = nn.Parameter(out_global[chunk_start:chunk_end].clone())
        transl_chunk = nn.Parameter(out_transl[chunk_start:chunk_end].clone())

        # Boundary z values from the previous chunk (detached, no grad).
        z_bm1 = out_z[chunk_start - 1].clone() if chunk_start >= 1 else None
        z_bm2 = out_z[chunk_start - 2].clone() if chunk_start >= 2 else None

        betas_b = betas.unsqueeze(0).expand(chunk_T, -1)

        optimizer = torch.optim.LBFGS(
            [z_chunk, global_chunk, transl_chunk],
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_iters_batched,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        def closure():
            optimizer.zero_grad()
            body_pose = vposer.decode_aa(z_chunk)  # (chunk_T, 63)
            verts = body_model(
                betas_b, body_pose, global_chunk, transl_chunk
            )  # (chunk_T, V, 3)
            sim = synth_markers(verts, nn_idx, coeffs)  # (chunk_T, M, 3)

            data_term = sim.new_zeros(())
            for i, frame in enumerate(chunk_frames):
                if frame.obs_xyz.shape[0]:
                    sim_sub = sim[i, frame.label_idx]
                    data_term = data_term + ((sim_sub - frame.obs_xyz) ** 2).sum()

            z_term = (z_chunk**2).sum()

            # Constant-velocity prior in z-space within the chunk
            velo_term = sim.new_zeros(())
            if chunk_T >= 3:
                pred = 2 * z_chunk[1:-1] - z_chunk[:-2]
                velo_term = velo_term + ((z_chunk[2:] - pred) ** 2).sum()

            # Stitch with previous chunk's last two frames
            if z_bm1 is not None and z_bm2 is not None:
                pred0 = 2 * z_bm1 - z_bm2
                velo_term = velo_term + ((z_chunk[0] - pred0) ** 2).sum()
                if chunk_T >= 2:
                    pred1 = 2 * z_chunk[0] - z_bm1
                    velo_term = velo_term + ((z_chunk[1] - pred1) ** 2).sum()

            loss = cfg.wt_data * data_term + cfg.wt_z * z_term + cfg.wt_velo * velo_term
            loss.backward()
            return loss

        final_loss = optimizer.step(closure)

        with torch.no_grad():
            out_z[chunk_start:chunk_end] = z_chunk.detach()
            out_global[chunk_start:chunk_end] = global_chunk.detach()
            out_transl[chunk_start:chunk_end] = transl_chunk.detach()
            body_pose = vposer.decode_aa(z_chunk.detach())
            out_body[chunk_start:chunk_end] = body_pose

            # Per-frame loss + chunk RMSE for logging
            verts_final = body_model(
                betas_b, body_pose, global_chunk.detach(), transl_chunk.detach()
            )
            sim_final = synth_markers(verts_final, nn_idx, coeffs)
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
        "z": out_z,
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
):
    """Returns [(label, rmse_mm, n_obs)] sorted by rmse_mm descending."""
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
    rows = [
        (latent_labels[i], rmse_list[i], n_obs[i]) for i in range(M) if n_obs[i] > 0
    ]
    rows.sort(key=lambda r: -r[1])
    return rows


def _log_per_marker_rmse(rows, header: str) -> None:
    if not rows:
        return
    width = max(len(r[0]) for r in rows)
    logger.info(f"{header}")
    for label, rmse, n in rows:
        logger.info(f"  {label:<{width}}  {rmse:6.1f} mm  ({n} obs)")


def _data_residual(
    sim_markers: torch.Tensor, frames: List[FrameMarkers]
) -> torch.Tensor:
    """Sum-of-squares over the (frame, marker) pairs that are actually observed."""
    total = sim_markers.new_zeros(())
    for f_idx, frame in enumerate(frames):
        if frame.obs_xyz.shape[0] == 0:
            continue
        sim_sub = sim_markers[f_idx, frame.label_idx]
        total = total + ((sim_sub - frame.obs_xyz) ** 2).sum()
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
