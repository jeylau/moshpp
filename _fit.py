# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moshpp.torch_impl import fit_smpl_to_markers
from moshpp.torch_impl.fit import StageICfg, StageIICfg

PROCESSED = Path("/Users/jessy/vscode/reac/fregly/processed")
SMPLH_DIR = Path("models/smplh")
VPOSER_DIR = Path("models/V02_05")
DEVICE = "cpu"  # or "cuda" / "mps"

GENDER_MAP = {"Male": "male", "Female": "female"}

stagei_cfg = StageICfg(markers_latent_free_from_step=3, wt_init=1000.0)
stageii_cfg = StageIICfg(
    wt_data=1000.0,
    wt_z=0.05,
    wt_velo=0.5,
    lbfgs_iters=50,
    batch_size=64,
    lbfgs_iters_batched=300,
)

manifest = pd.read_csv(PROCESSED / "manifest.csv")
marker_vids = json.loads((PROCESSED / "marker_vids.json").read_text())

# %%
for subject_dir, rows in manifest.groupby("subject_dir", sort=False):
    subj_path = PROCESSED / subject_dir
    labels = json.loads((subj_path / "labels.json").read_text())
    gender = GENDER_MAP[json.loads((subj_path / "subject.json").read_text())["gender"]]
    smplh_path = SMPLH_DIR / f"SMPLH_{gender.upper()}.pkl"

    # TODO: if fit_smpl_to_markers supports passing pre-fit Stage I outputs,
    # fit Stage I once per subject here (on a representative trial) and reuse
    # below with run_stage_i=False to avoid recomputing the marker layout.

    for row in rows.itertuples(index=False):
        out_path = PROCESSED / row.smpl_path
        if out_path.exists():
            continue

        markers = np.load(PROCESSED / row.markers_path)
        print(
            f"fitting {subject_dir}/{row.trial_stem} "
            f"({markers.shape[0]} frames, {markers.shape[1]} markers)"
        )

        out = fit_smpl_to_markers(
            markers=markers,
            labels=labels,
            marker_vids=marker_vids,
            smplh_path=smplh_path,
            vposer_dir=VPOSER_DIR,
            gender=gender,
            device=DEVICE,
            run_stage_i=True,
            stagei_cfg=stagei_cfg,
            stageii_cfg=stageii_cfg,
        )

        np.savez(
            out_path,
            body_pose=out["stageii"]["body_pose"].cpu().numpy(),
            global_orient=out["stageii"]["global_orient"].cpu().numpy(),
            transl=out["stageii"]["transl"].cpu().numpy(),
            betas=out["stageii"]["betas"].cpu().numpy(),
            gender=out["gender"],
            nn_idx=out["stagei"]["nn_idx"].cpu().numpy(),
            coeffs=out["stagei"]["coeffs"].cpu().numpy(),
            markers_latent=out["stagei"]["markers_latent"].cpu().numpy(),
            latent_labels=np.array(out["stagei"]["latent_labels"]),
        )
# %%
stem = "jw_tsgait_11"
temp = manifest[manifest.trial_stem == stem].itertuples(index=False)
row = next(temp)

subj_path = PROCESSED / row.subject_dir
labels = json.loads((subj_path / "labels.json").read_text())
gender = GENDER_MAP[json.loads((subj_path / "subject.json").read_text())["gender"]]
smplh_path = SMPLH_DIR / f"SMPLH_{gender.upper()}.pkl"

markers = np.load(PROCESSED / row.markers_path)

out = fit_smpl_to_markers(
    markers=markers,
    labels=labels,
    marker_vids=marker_vids,
    smplh_path=smplh_path,
    vposer_dir=VPOSER_DIR,
    gender=gender,
    device=DEVICE,
    run_stage_i=False,
    stagei_cfg=stagei_cfg,
    stageii_cfg=stageii_cfg,
)

np.savez(
    f"{stem}.npz",
    body_pose=out["stageii"]["body_pose"].cpu().numpy(),
    global_orient=out["stageii"]["global_orient"].cpu().numpy(),
    transl=out["stageii"]["transl"].cpu().numpy(),
    betas=out["stageii"]["betas"].cpu().numpy(),
    gender=out["gender"],
    nn_idx=out["stagei"]["nn_idx"].cpu().numpy(),
    coeffs=out["stagei"]["coeffs"].cpu().numpy(),
    markers_latent=out["stagei"]["markers_latent"].cpu().numpy(),
    latent_labels=np.array(out["stagei"]["latent_labels"]),
)

# %%
