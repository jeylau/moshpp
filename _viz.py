# %%
import numpy as np
import torch
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

patient = "1_JW"
stem = "jw_tsgait_11"
stem2 = "jw_tsgait_10"
feats = np.load(
    f"/Users/jessy/vscode/reac/fregly/processed/{patient}/features_fix/{stem}.npz"
)
feats2 = np.load(
    f"/Users/jessy/vscode/reac/fregly/processed/{patient}/features_fix/{stem2}.npz"
)
markers = np.load(
    f"/Users/jessy/vscode/reac/fregly/processed/{patient}/markers/{stem}.npy"
).astype(np.float32)
markers *= 1e-3
color_obs = np.broadcast_to(
    np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),
    markers.shape[:2] + (4,),
).copy()
pc_obs = PointClouds(
    points=markers,
    colors=color_obs,
    point_size=23.0,
    name="observed (input) markers",
)
# %%


def _make_smpl_seq(features, name: str) -> SMPLSequence:
    rot6d = torch.from_numpy(features["rot6d"].reshape((-1, 24, 6)))
    T = len(rot6d)
    mats = rotation_6d_to_matrix(rot6d)
    aa = matrix_to_axis_angle(mats)
    T = aa.shape[0]
    trans = np.nan_to_num(features["trans"].astype(np.float32))
    return SMPLSequence(
        smpl_layer=SMPLLayer(model_type="smpl", gender="neutral", num_betas=10),
        poses_root=aa[:, 0],
        poses_body=aa[:, 1:].reshape(T, -1),
        trans=trans,
        betas=np.tile(features["shape"][None].astype(np.float32), (T, 1)),
        name=name,
    )


C.update_conf({"smplx_models": "/Users/jessy/vscode/swim/models", "z_up": True})
v = Viewer()
v.scene.add(
    _make_smpl_seq(feats, "jw_tsgait_11"),
    _make_smpl_seq(feats2, "jw_tsgait_10"),
    pc_obs,
)
v.run()
# %%
