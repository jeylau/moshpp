"""High-level entry point: fit SMPL-H to a numpy array of 3D markers.

Inputs:
    markers     : (T, M, 3) float, meters, NaN where missing
    labels      : list of M strings (marker labels)
    marker_vids : dict {label: vertex_id_on_SMPLH_template}
    smplh_path  : path to SMPLH_MALE.pkl (or the models dir)
    vposer_dir  : path to the VPoser expr dir (contains V02_05.yaml + snapshots/)

Returns a dict with stage I and stage II results.

Usage:
    out = fit_smpl_to_markers(
        markers, labels, marker_vids,
        smplh_path="/Users/jessy/vscode/swim/models/smplh/SMPLH_MALE.pkl",
        vposer_dir="/Users/jessy/vscode/human_body_prior/support_data/dowloads/V02_05",
        gender="male",
        run_stage_i=True,
    )
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from loguru import logger

from moshpp.torch_impl.body_model import SMPLHBodyModel
from moshpp.torch_impl.fit import (
    StageICfg,
    StageIICfg,
    _frames_from_array,
    mosh_stagei,
    mosh_stageii,
)
from moshpp.torch_impl.markers import build_local_frame
from moshpp.torch_impl.vposer_prior import FrozenVPoser


def _pick_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_smpl_to_markers(
    markers: np.ndarray,
    labels: List[str],
    marker_vids: Dict[str, int],
    *,
    smplh_path: Union[str, Path],
    vposer_dir: Union[str, Path],
    gender: str = "male",
    num_betas: int = 10,
    stagei_frame_ids: Optional[List[int]] = None,
    stagei_num_frames: int = 12,
    run_stage_i: bool = True,
    betas: Optional[Union[np.ndarray, torch.Tensor]] = None,
    m2b_distance: float = 0.0095,
    device: Optional[str] = None,
    stagei_cfg: Optional[StageICfg] = None,
    stageii_cfg: Optional[StageIICfg] = None,
    stageii_batch_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Fit SMPL-H to a sequence of labeled 3D markers.

    If ``betas`` is provided, stage I is bypassed entirely and the marker
    layout is constructed on the betas-conditioned canonical body. Use this
    when you have a known body shape (e.g. AMASS fit) and only want to recover
    per-frame pose against new mocap data.
    """
    assert markers.ndim == 3 and markers.shape[2] == 3, "markers must be (T, M, 3)"
    assert markers.shape[1] == len(labels), (
        "labels length must match markers' second dim"
    )

    dev = _pick_device(device)
    logger.info(f"device: {dev}")

    # Unit sanity check. Indoor mocap in meters has max abs coord ~3.
    finite = markers[np.isfinite(markers)]
    abs_max = float(np.abs(finite).max()) if finite.size else 0.0
    logger.info(f"markers max abs coord: {abs_max:.3f}")
    if abs_max > 50:
        logger.warning(
            f"max abs coord {abs_max:.1f} looks like millimeters — auto-converting to meters"
        )
        markers = markers / 1000.0

    # Keep only labels that have a known vertex id.
    kept = [(i, label) for i, label in enumerate(labels) if label in marker_vids]
    if not kept:
        raise ValueError("No labels matched marker_vids.")
    latent_labels = [label for _, label in kept]
    marker_vids_t = torch.as_tensor(
        [marker_vids[label] for label in latent_labels], dtype=torch.long, device=dev
    )
    label_to_idx = {label: i for i, label in enumerate(latent_labels)}

    # Body model + VPoser
    body_model = SMPLHBodyModel(
        smplh_path, gender=gender, num_betas=num_betas, device=dev
    )
    vposer = FrozenVPoser(vposer_dir, device=dev)

    # ----- Stage I (or bypass) -----
    from moshpp.torch_impl.fit import _vertex_normals

    if betas is not None:
        # User-supplied betas: skip stage I, build marker layout on the
        # betas-conditioned canonical body.
        logger.info("skipping stage I — using user-supplied betas")
        betas_t = torch.as_tensor(betas, dtype=torch.float32, device=dev).flatten()
        if betas_t.shape[0] < num_betas:
            pad = torch.zeros(num_betas - betas_t.shape[0], device=dev)
            betas_t = torch.cat([betas_t, pad])
        elif betas_t.shape[0] > num_betas:
            logger.warning(
                f"truncating provided betas from {betas_t.shape[0]} to {num_betas}"
            )
            betas_t = betas_t[:num_betas]
        with torch.no_grad():
            can_verts = body_model.canonical_verts(betas_t)
        can_normals = _vertex_normals(can_verts, body_model.faces)
        init_markers = (
            can_verts[marker_vids_t] + can_normals[marker_vids_t] * m2b_distance
        )
        nn_idx, coeffs = build_local_frame(can_verts, init_markers)
        betas = betas_t.detach()
        markers_latent = init_markers
        stagei_out = {
            "betas": betas,
            "markers_latent": markers_latent,
            "nn_idx": nn_idx,
            "coeffs": coeffs,
            "latent_labels": latent_labels,
        }
    elif run_stage_i:
        if stagei_frame_ids is None:
            stagei_frame_ids = (
                np.linspace(0, markers.shape[0] - 1, num=stagei_num_frames)
                .astype(int)
                .tolist()
            )
        logger.info(f"stage I frames: {stagei_frame_ids}")
        stagei_frames = _frames_from_array(
            markers[stagei_frame_ids], label_to_idx, labels, dev
        )
        si_cfg = stagei_cfg or StageICfg(num_betas=num_betas, m2b_distance=m2b_distance)
        stagei_out = mosh_stagei(
            body_model=body_model,
            vposer=vposer,
            stagei_frames=stagei_frames,
            latent_labels=latent_labels,
            marker_vids=marker_vids_t,
            cfg=si_cfg,
        )
        betas = stagei_out["betas"]
        nn_idx = stagei_out["nn_idx"]
        coeffs = stagei_out["coeffs"]
        markers_latent = stagei_out["markers_latent"]
    else:
        # v0: zero betas, marker positions = template_vertex + normal * d
        logger.info("skipping stage I — using zero betas and template marker positions")
        v_template = body_model.template_vertices()

        vert_normals = _vertex_normals(v_template, body_model.faces)
        init_markers = (
            v_template[marker_vids_t] + vert_normals[marker_vids_t] * m2b_distance
        )
        nn_idx, coeffs = build_local_frame(v_template, init_markers)
        betas = torch.zeros(num_betas, device=dev)
        markers_latent = init_markers
        stagei_out = {
            "betas": betas,
            "markers_latent": markers_latent,
            "nn_idx": nn_idx,
            "coeffs": coeffs,
            "latent_labels": latent_labels,
        }

    # ----- Stage II -----
    observed = _frames_from_array(markers, label_to_idx, labels, dev)
    sii_cfg = stageii_cfg or StageIICfg()
    if stageii_batch_size is not None:
        sii_cfg = replace(sii_cfg, batch_size=stageii_batch_size)
    stageii_out = mosh_stageii(
        latent_labels=latent_labels,
        body_model=body_model,
        vposer=vposer,
        observed_frames=observed,
        betas=betas,
        nn_idx=nn_idx,
        coeffs=coeffs,
        cfg=sii_cfg,
    )

    return {
        "latent_labels": latent_labels,
        "marker_vids": marker_vids_t,
        "stagei": stagei_out,
        "stageii": stageii_out,
        "faces": body_model.faces,
        "gender": gender,
    }
