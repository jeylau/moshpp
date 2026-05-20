# MoSh++ pipeline: legacy chumpy implementation vs. minimal torch port

This document is a precise walkthrough of how the original MoSh++ fits an SMPL/SMPL-H/SMPL-X mesh to 3D marker data, mapped against the minimal `src/moshpp/torch_impl/` rewrite. It is meant as a reference while completing the port — every legacy mechanism is followed by the corresponding torch construct (or noted as deliberately dropped).

The legacy code lives in `src/moshpp/` (gated behind the `legacy` extra in `pyproject.toml`). The torch port lives in `src/moshpp/torch_impl/`.

---

## 0. The big picture

MoSh++ solves two coupled inverse problems against a stream of labeled 3D mocap markers:

| Stage | Free variables | Shared across frames? | What's frozen | Output |
|-------|----------------|----------------------|---------------|--------|
| **I — "shape"** | `betas`, per-marker latent positions on body surface, per-reference-frame pose + trans | `betas` and `markers_latent` shared; pose/trans per-frame | Marker layout (label → vertex id) | `betas`, `markers_latent`, marker-to-body coefficients |
| **II — "pose"** | per-frame `fullpose`, `trans` (+ DMPLs / expression if enabled) | nothing — independent per frame, only smoothness coupling | `betas`, `markers_latent` (locked to surface via local frame) | sequence of `fullpose[t]`, `trans[t]` |

The two stages are gated by I/O: stage I writes `*_stagei.pkl`, stage II writes `*_stageii.pkl`. They can be rerun independently as long as a `_stagei.pkl` exists.

The **central trick** is the *transformed-landmark* parameterization. Markers are not the raw 3D vertices of the body; each marker stores 3 coefficients in a local frame built from its 3 nearest body vertices on the canonical T-pose. As betas/pose change, the local frame rotates/translates with the body, and the marker position is re-synthesized. This lets the optimizer move markers *along the surface* without leaving it, and naturally accounts for marker-to-bone-distance offsets.

---

## 1. Configuration and entry point

### Legacy
- Config is OmegaConf YAML, base in `support_data/conf/moshpp_conf.yaml`, merged with kwargs/overrides via `MoSh.prepare_cfg()` (`src/moshpp/mosh_head.py:543`).
- Resolvers (e.g. `resolve_gender`, `resolve_mocap_ds_name`) live in `src/moshpp/tools/run_tools.py`.
- `MoSh(**cfg)` is the orchestrator; `run_moshpp_once(cfg)` calls `mp.mosh_stagei(...)` then `mp.mosh_stageii(...)` from `chmosh.py`.
- All paths are derived from `dirs.work_base_dir`, `mocap.fname`, `surface_model.type`, gender, subject — pickles are *cached on disk* (`stagei_fname`, `stageii_fname`) so a rerun short-circuits if those exist.

Key config sub-trees:
- `mocap.*`: c3d filename, units (mm by default), rotation, subject filtering, frame range, downsampling.
- `surface_model.*`: type (`smpl|smplh|smplx|mano|animal_*`), gender, num_betas (16), num_dmpls (8), num_expressions (80), `betas_expr_start_id` (300), `dof_per_hand` (24), `use_hands_mean`.
- `moshpp.*`: which body parts to optimize (`optimize_betas`, `_fingers`, `_face`, `_toes`, `_dynamics`), pose priors filenames, head-marker corr file, stage-I frame picker (`type ∈ {random, random_strict, manual}`, `num_frames=12`, `least_avail_markers`, `seed`).
- `opt_settings.weights`: per-model-type weight dicts (see `opt_weights.smplh` in the yaml). Hard-coded `num_train_markers = 46` is used to normalize `wt_data` against the actual count.

### Torch port
- No YAML. Two dataclasses, `StageICfg` and `StageIICfg` in `src/moshpp/torch_impl/fit.py`, hold every knob.
- One entry function `fit_smpl_to_markers(markers, labels, marker_vids, smplh_path, vposer_dir, …)` in `src/moshpp/torch_impl/api.py`. No on-disk caching, no OmegaConf resolvers.
- The mocap → numpy bridge happens *outside* the package: caller passes `markers: (T, M, 3)` (meters, NaN where missing) and a `marker_vids: dict[label, vid]`.
- Defaults: 10 betas (vs. 16), male, CPU/MPS/CUDA auto-pick.

**Deliberately dropped:** OmegaConf, multi-subject handling, `runtime.stagei_only`, on-disk caching, per-surface-model weight overrides, `verbosity` levels, visualization (`psbody` MeshViewer).

---

## 2. Marker layout and label management

The "marker layout" is the contract between the mocap labels (strings like `RKNE`, `LELB`) and the body model (SMPL-H vertex ids). It also stores marker types (`body`, `face`, `finger`, `finger_left`, …), per-type marker-to-bone distances, and per-marker colors.

### Legacy
- JSON files under `support_data/marker_layouts/`. Top-level shape is `{surface_model_type, markersets: [{type, distance_from_skin, indices: {label: vid}}, …]}`.
- `marker_layout_load(...)` (`src/moshpp/marker_layout/edit_tools.py:83`) builds an OrderedDict `marker_vids`, an OrderedDict `marker_type_mask` (boolean mask per type), `m2b_distance` (per-type float), `marker_colors`.
- A `general_labels_map` (`src/moshpp/marker_layout/labels_map.py`) renames legacy/dataset-specific labels to a canonical form (`R_KNEE → RKNE`, etc.).
- If no layout file exists yet, MoSh autogenerates one from the *observed* marker labels in the stage-I frames, via `marker_labels_to_marker_layout(...)` (`src/moshpp/marker_layout/create_marker_layout_for_mocaps.py`).
- After stage I, `MoSh.dump_stagei_marker_layout(...)` writes back the *optimized* layout (json + ply + c3d) where each marker's vid is the nearest body vertex to its converged latent position.

Distance-from-skin defaults: `body=0.0095 m`, `face=0.004 m`, `finger=0.005 m`. SMPL-X eyeballs are excluded from the 3-NN search via `support_data/smplx_eyeballs.npz` (loaded in `TransformedCoeffs`).

### Torch port
- No JSON. Caller passes `marker_vids: dict[label, int]` directly.
- Labels that have no vid are silently dropped (`api.py:91-98`). No per-type distance: one global `m2b_distance` (default `0.0095`).
- No labels-map rename, no marker-type masks, no autogeneration, no `dump_stagei_marker_layout`.

**Deliberately dropped:** marker types, per-type distance-from-skin, label remapping, head-marker correlation, eyeball exclusion (SMPL-H has no eyeballs, so moot for v1).

---

## 3. Mocap data ingestion

### Legacy — `tools/mocap_interface.py::MocapSession`
- Reads `.c3d` via `ezc3d`, also `.mat`, `.npz`, with optional subject filtering for multi-subject sessions.
- Applies `mocap_unit` (`mm`/`cm`/`m`) → meters and `mocap_rotate` (XYZ Euler in degrees).
- Exposes `markers: (T, N, 3)`, `labels: list[str]`, `frame_rate`, plus helpers `markers_asdict() -> dict[t, dict[label, xyz]]` and `marker_availability_mask(...)`.
- Used by frame_picker (stage I) and stage II.

### Legacy — `frame_picker.py`
- For stage I, three strategies select `num_frames` (default 12) reference frames:
  - `random_strict`: only frames where ≥ `least_avail_markers` (default 1.0 = all) of labels are non-NaN.
  - `random`: same idea but recursively lowers the threshold by 0.01 until enough frames are found.
  - `manual`: explicit list of `mocap_path_frameid` strings.
- Stops scanning after 100 candidates have been collected.

### Torch port
- Caller supplies a dense `markers: (T, M, 3)` numpy array. No c3d reader, no `MocapSession`, no rotation/unit handling beyond a sanity check (`api.py:82-89`: if abs-max > 50, assume mm and divide by 1000).
- Stage-I frame selection: linearly spaced indices across the sequence (`api.py:110-114`), or caller-provided `stagei_frame_ids`. No per-frame availability check; observations that are NaN or all-zero are dropped *inside* `_frames_from_array` (`fit.py:41-80`).

**Deliberately dropped:** `ezc3d`, `MocapSession`, multi-subject filtering, `mocap_rotate`, on-disk frame caching, the strict random samplers, the 100-candidate ceiling.

---

## 4. Body model loading

### Legacy — `models/bodymodel_loader.py::load_moshpp_models`
- Loads two flavors of the surface model via `SmplModelLBS` (chumpy wrapper around `load_surface_model`):
  - `can_model`: the **canonical** body in T-pose. Owns `betas`, which is the *shared* shape parameter across stage-I frames.
  - `opt_models`: a list of `num_beta_shared_models` (default 12) per-frame copies. Their `betas` *alias* `can_model.betas` (via `AliasedBetas` chumpy `Ch`) except when `optimize_face=True` for SMPL-X, where betas-vs-expression aliasing breaks chumpy.
- Attaches priors:
  - `priors['pose']` = GMM body prior (`create_gmm_body_prior` in `prior/gmm_prior_ch.py`) — 8-component GMM over 69-dim (SMPL) or 63-dim (SMPL-H/X) body axis-angle. **This is the original SMPLify-2016 prior**, loaded from a pickle.
  - `priors['betas']` = `AliasedBetas` (used as `||betas||²`-like term).
  - For animals: `smal_horse_prior`, `MaxMixtureDog` from `prior/horse_body_prior.py` / `prior/dog_body_prior.py`.
- Hands: 24 PCA components per hand by default. `use_hands_mean=True`.
- DMPLs: 8 dynamics components, loaded into `shapedirs` columns `[num_betas : num_betas+num_dmpls]` only in stage II if `optimize_dynamics=True` (SMPL/SMPL-H only).
- SMPL-X face: 80 expression components live in `betas[300:380]`. Stage I has to be split into two passes if you want both shape and expression, because chumpy can't share betas while indexing into the same array twice.

### Torch port — `body_model.py::SMPLHBodyModel`
- A thin `nn.Module` around `smplx.create(model_type="smplh", use_pca=False, flat_hand_mean=True, batch_size=1)`.
- Forward: `forward(betas, body_pose, global_orient, transl) -> (B, V, 3)` where `body_pose` is 63-dim (21 joints × 3). Hands stay at flat rest. No DMPLs, no expression.
- Helpers:
  - `template_vertices()`: T-pose with zero betas (cached `v_template`).
  - `canonical_verts(betas)`: T-pose under the given betas. **Differentiable**, used inside the stage-I closure.
  - `root_joint(betas)`: pelvis position in the rest pose. Used by the rigid-alignment seeder.
- No prior wiring inside the body model — the prior is handled by the VPoser parameterization (see §5).

**Deliberately dropped:** GMM body/hand priors, animal models, DMPLs, expressions, fingers as free variables, multi-frame `opt_models` list (the torch port batches frames inside one `smplx` forward call).

---

## 5. Pose parameterization and prior

This is the single largest architectural difference. The shape of the optimization variables changes here.

### Legacy
- Body pose is the raw axis-angle vector inside `model.pose` (chumpy `Ch`). Sizes depend on surface model:
  - SMPL: pose is 72 = root(3) + body(69).
  - SMPL-H: pose is 156 = root(3) + body(63) + L-hand(45) + R-hand(45).
  - SMPL-X: pose is 165 = root(3) + body(63) + jaw(3) + L-eye(3) + R-eye(3) + L-hand(45) + R-hand(45).
- Subsets are selected via index lists (`pose_root_ids`, `pose_body_ids`, `pose_finger_ids`, `pose_face_ids`); `optimize_toes=False` removes `pose[30:36]` from the free vars.
- **Pose prior**: GMM `priors['pose'](pose[pose_body_ids])` returns the Mahalanobis term against the closest component (`MaxMixtureComplete` in `prior/gmm_prior_ch.py`). Chumpy can't backprop through a torch network, which is why MoSh++ uses GMM rather than VPoser as the prior here.
- **Finger prior**: when `optimize_fingers=True`, raw L2 on the finger pose subset (no GMM).
- **Face prior**: L2 on jaw axis-angle and `betas[300:300+80]` (expression).

### Torch port — `vposer_prior.py::FrozenVPoser`
- The free pose variable is a 32-dim **VPoser latent code** `z`, not axis-angle.
- VPoser is loaded once via `human_body_prior.tools.model_loader.load_model(...)`, set to `eval()`, weights frozen.
- `decode_aa(z) -> (B, 63)` runs the VPoser decoder and converts the resulting rotation matrices (`(B, 21, 3, 3)`) to axis-angle via `pytorch3d.transforms.matrix_to_axis_angle`.
- `encode_mean(body_pose_aa)` (unused at runtime, kept for warm-start experiments).
- The prior term reduces to `(z**2).sum()` — a unit-Gaussian on the latent — replacing the GMM Mahalanobis entirely.

**Why VPoser-in-latent-space rather than VPoser-as-prior?** A VPoser-as-prior approach would still optimize `pose_body ∈ ℝ⁶³` and add `||VPoser.encode(pose_body).mean||²` to the loss. Putting VPoser inside the parameterization is strictly smaller (32 vs 63 free parameters), and the prior collapses to a closed-form L2.

**Why SMPL-H and not SMPL or SMPL-X?** VPoser was trained on 21 body joints. SMPL has 23 (head/jaw separated, no hand joints), which would need truncation. SMPL-X has 21 body joints + jaw/eyes/hands but only the 21 body joints are needed here. SMPL-H matches exactly.

**Deliberately dropped:** GMM body prior (and the SMPLify pickle), finger/face/eye optimization, DMPLs as a stage-II variable.

---

## 6. The transformed-landmark trick

This is the math that lets MoSh "slide" markers along the body surface. Both legacy and torch use the same construction; the torch version just freezes correspondences after init.

### Definitions
For a latent marker at position `m ∈ ℝ³` on the *canonical* body with vertices `V_can ∈ ℝ^(V×3)`:
1. Find the 3 nearest body vertices to `m` in `V_can`: indices `(i₀, i₁, i₂)`.
2. Define `e₁ = v_{i₁} − v_{i₀}`, `e₂ = v_{i₂} − v_{i₀}`.
3. Orthonormalize: `f₁ = e₁/‖e₁‖`, `f₂ = (e₁×e₂)/‖e₁×e₂‖`, `f₃ = f₁ × f₂`.
4. Project `m − v_{i₀}` into the frame: `coeffs = ((m−v_{i₀})·f₁, (m−v_{i₀})·f₂, (m−v_{i₀})·f₃)`.

On a *posed* body with vertices `V_pose`, rebuild the same frame from `(V_pose[i₀], V_pose[i₁], V_pose[i₂])` and synthesize:
`m_sim = V_pose[i₀] + c₁·f₁_pose + c₂·f₂_pose + c₃·f₃_pose`.

This makes `m_sim` covary with the body's pose/shape while staying at the same surface-local offset.

### Legacy — `transformed_lm.py`
- `TransformedCoeffs(Ch)` (`transformed_lm.py:45`): a chumpy node that computes `coeffs` from `(can_body, markers_latent)`. Its `on_changed` callback recomputes 3-NN via `sklearn.neighbors.NearestNeighbors(algorithm='kd_tree', n_neighbors=8)` **every time** the inputs change. It also handles degenerate triangles (collinear NN points) by advancing to the next neighbor (`NN_counter`).
- `TransformedLms(Ch)` (`transformed_lm.py:120`): synthesizes the marker positions from `(transformed_coeffs, can_body=opt_model_verts)`. Reads `coeffs.closest` (cached on `TransformedCoeffs`) and rebuilds `f₁, f₂, f₃` on the posed verts.
- For SMPL-X (`len(can_body) == 10475`), `TransformedCoeffs.no_eye_ball_vids` excludes eyeball vertices from the 3-NN search using `support_data/smplx_eyeballs.npz`.
- This setup gets recomputed every chumpy "tick", which dominates stage-I runtime.

### Torch port — `markers.py`
- `compute_nn_idx(canonical_verts, marker_positions)`: brute-force `torch.cdist + topk(3)`. Called **once** at the start of stage I and frozen.
- `compute_coeffs(canonical_verts, marker_positions, nn_idx)`: differentiable in *both* `canonical_verts` (via betas) and `marker_positions` (via `markers_latent`). Called inside the stage-I LBFGS closure so gradients flow to betas and to `markers_latent`.
- `synth_markers(verts, nn_idx, coeffs)`: stage-II forward. Vectorized: `(B, V, 3) -> (B, M, 3)`.

**Key behavioral choice (see memory: moshpp modernization decisions):**
> Stage I correspondences are computed **once** on the T-pose and frozen — not recomputed each iteration as chumpy did. LBFGS prefers a stable graph and the correspondences barely change in practice.

**Deliberately dropped:** collinear-triangle handling (`NN_counter` advance), eyeball exclusion, sklearn KD-tree.

---

## 7. Stage I — shape + marker placement

### Legacy — `chmosh.py::mosh_stagei`

1. **Optional warm start** (`chmosh.py:93-98`): if `betas_fname` is given, load betas from npz and write into `can_model.betas[:num_betas]`. If `v_template_fname` is given, betas are ignored and the v_template is locked in.
2. **Load marker layout** (`chmosh.py:120-124`) and auto-disable `optimize_face`/`optimize_fingers` if the corresponding marker type is missing from layout or observations.
3. **Load body models** via `load_moshpp_models(num_beta_shared_models=12)` so `can_model.betas` is aliased into all 12 `opt_models`.
4. **Init markers on the surface** via `prepare_mosh_markers_latent` (`chmosh.py:57-80`): for each labeled vid, set `markers_latent = v_template[vid] + vertex_normal[vid] * m2b_distance`. Build a `PtsToMesh` distance-to-surface object (using `scan2mesh`) that produces a signed surface distance for the *surf* loss.
5. **Build the transformed-landmark graph** (`chmosh.py:182-191`): one `TransformedCoeffs` from `(can_model, markers_latent)`, then one `TransformedLms` per `opt_model` (so each frame gets its own simulated markers). Also build `init_markers_latent` (the markers reprojected onto the T-pose) for the *init* loss.
6. **Match observed labels to latent labels** for each of the 12 ref frames (`chmosh.py:199-211`), keeping only non-NaN markers. Stack into `lm_diffs = obs - sim` and the per-frame `markers_obs`, `markers_sim`, `labels_obs`.
7. **Rigid alignment** via SVD (`rigid_transformations.py::perform_rigid_adjustment`) — for each frame, fit a global `(R, t)` from simulated to observed markers, write `R` as Rodrigues axis-angle into `pose[:3]` and `t` into `trans`. Optionally a dogleg minimize over `[pose[:3] + trans]` is run if `extra_initial_rigid_adjustment=True`.
8. **Optional head-marker correlation** (`chmosh.py:252-266`): if `head_marker_corr_fname` exists, the init loss for head markers is multiplied by a learned covariance matrix `corr ∈ ℝ^(k×k)` so head-marker errors are evaluated in a decorrelated space.
9. **Pose subset selection** (`chmosh.py:274-309`): per surface-model-type, decide which `pose_ids` are free. Toes (`pose[30:36]`) are excluded unless `optimize_toes=True`. Fingers and face are added only on the *last* 2 anneal steps (`detailed_step = tidx > len(annealing) - 3`).
10. **Annealing loop** over `stagei_wt_annealing = [1.0, 0.5, 0.25, 0.125]` (4 steps for `smplh`/`smplx`):
    - `wt_data` scales **inversely** with the anneal factor (data weight grows late).
    - `wt_poseB/H/F`, `wt_init_<type>`, `wt_betas`, `wt_expr` scale **proportionally** with the anneal factor (priors shrink late, letting the data dominate).
    - `wt_data` is further normalized: `* (num_train_markers / len(latent_labels))` with `num_train_markers = 46` hard-coded.
    - Loss terms assembled into `opt_objs`:
      - `data = (obs - sim) * wt_data` (concatenated across all 12 frames)
      - `poseB = GMM(pose[pose_body_ids]) * wt_poseB` per frame
      - `init_<type> = (markers_latent - init_markers_latent)[mask] * wt_init_<type>` — keeps markers near their template location, per marker type
      - `beta = can_model.priors['betas'] * wt_beta`
      - `surf = distance_to_surface_obj * wt_surf` — *signed* surface distance via scan2mesh, hard-pulls markers onto the body
      - on detailed steps: `poseH`, `poseF`, `expr`
    - Free vars: `trans + [markers_latent] + v_poses (+ v_face_exp + v_betas)`.
    - Optimizer: `ch.minimize(... method='dogleg', options={e_3: 1e-3, delta_0: 0.5, maxiter: 100})` — Powell's dogleg trust region.
11. **Final NN reassignment** (`chmosh.py:422-431`): re-assign each converged `markers_latent` to its single nearest body vertex on the *canonical* mesh, store as `markers_latent_vids` (label → vid). Also reassign all observed marker labels to their nearest vid on the *last posed* opt_model (`markers_latent_all_vids`).
12. **Save pickle** containing `betas`, `markers_latent`, `latent_labels`, `marker_meta`, `markers_latent_vids`, and a thick `stagei_debug_details` dict.

### Torch port — `fit.py::mosh_stagei`

1. **Init markers on the surface** (`fit.py:198-205`): `v_template + vertex_normal * m2b_distance`. Vertex normals via area-weighted accumulation (`_vertex_normals`, `fit.py:793-805`) — no pytorch3d dependency.
2. **Build local frame once** (`fit.py:208`): `build_local_frame(v_template, init_markers)` → `(nn_idx, coeffs)`. **Frozen for the entire stage I.**
3. **Define free variables** (`fit.py:211-215`):
   - `markers_latent: nn.Parameter` of shape `(M, 3)` (initialized to `init_markers`).
   - `betas: nn.Parameter` of shape `(num_betas,)` (initialized to zeros).
   - `z: nn.Parameter` of shape `(N, 32)` (initialized to zeros) — per-reference-frame VPoser latent.
   - `global_orient: nn.Parameter` of shape `(N, 3)`, `transl: nn.Parameter` of shape `(N, 3)`.
4. **Rigid alignment seed** (`fit.py:219-228`): SVD-based `rigid_align(sim, obs)` returns `(R, t)` mapping initial sim markers to observed markers. `_seed_root_from_rigid` corrects for the root joint pivot: `transl_smpl = t + R @ J_root - J_root` so that `R @ (v - J_root) + J_root + transl_smpl == R @ v + t`. Converts `R` to axis-angle via `pytorch3d.transforms.matrix_to_axis_angle`. SVD is run on CPU (MPS lacks `linalg.svd`/`det`).
5. **Annealing loop** over `cfg.anneal = [10.0, 5.0, 2.0, 1.0, 0.5]` (5 steps, larger range than legacy's 4):
   - `wt_data = wt_data / max(anneal, 1e-6)` — same inverse-anneal behavior as legacy.
   - `wt_init = wt_init * anneal`, `wt_z = wt_z * anneal`, `wt_betas = wt_betas * anneal` — same proportional behavior.
   - Closure (`fit.py:256-284`):
     - `can_verts = body_model.canonical_verts(betas)` — recomputes the T-pose with current betas (differentiable).
     - `live_coeffs = compute_coeffs(can_verts, markers_latent, nn_idx)` — **coeffs are recomputed every iteration** so that gradients flow into both betas and markers_latent. Only `nn_idx` is frozen (3-NN indices).
     - `body_pose = vposer.decode_aa(z)` — `(N, 63)`.
     - `verts = body_model(betas.expand(N,-1), body_pose, global_orient, transl)` — `(N, V, 3)`.
     - `sim_markers = synth_markers(verts, nn_idx, live_coeffs)` — `(N, M, 3)`.
     - Losses:
       - `data_term = Σ_f ||sim[f, label_idx_f] - obs_f||²` (`_data_residual`)
       - `init_term = ||markers_latent - init_markers||²`
       - `z_term = ||z||²` (VPoser prior)
       - `betas_term = ||betas||²`
     - `loss = wt_data * data + wt_init * init + wt_z * z + wt_betas * betas`
   - Optimizer: `torch.optim.LBFGS(line_search_fn='strong_wolfe', max_iter=30, tolerance_grad=1e-7, tolerance_change=1e-9)`, fresh per anneal step.
6. **Bake final coeffs** (`fit.py:310-312`): after convergence, recompute `final_coeffs = compute_coeffs(can_verts_final, markers_latent, nn_idx)` and pass *that* to stage II — so stage II sees a marker placement consistent with the learned shape.
7. **Diagnostics**: data RMSE in mm, `|betas|`, mean marker drift, per-marker RMSE table sorted worst-first.

### Stage I differences at a glance

| Aspect | Legacy | Torch port |
|---|---|---|
| Reference frames | 12 (picked by `frame_picker`) | 12 (linearly spaced) |
| Body-pose parameterization | raw axis-angle (63 dims) per frame | 32-d VPoser latent `z` per frame |
| Pose prior | 8-component GMM (SMPLify pickle) | `‖z‖²` |
| Body model copies | 12 chumpy `SmplModelLBS` sharing aliased `betas` | one `smplx.SMPLH`, batched forward over N |
| 3-NN correspondences | recomputed every iteration (sklearn KD-tree) | computed once on T-pose, frozen |
| Local-frame coeffs | static after each NN refresh | recomputed every closure call (differentiable in betas + markers_latent) |
| Surface loss (`surf`) | signed point-to-mesh via scan2mesh | **dropped** |
| Marker-type init weights | per-type (`body`, `face`, `finger`, …) | single global `wt_init` |
| Head-marker correlation | covariance-aware decorrelation | **dropped** |
| Toes / fingers / face / expression | gated by config; added late in anneal | **dropped** (body only) |
| Optimizer | chumpy dogleg, 100 iters per anneal step | LBFGS strong-Wolfe, 30 iters per anneal step |
| Anneal schedule | `[1, 0.5, 0.25, 0.125]` | `[10, 5, 2, 1, 0.5]` |
| Rigid seed | `rigid_landmark_transform` (SVD) + Rodrigues into `pose[:3]` | `rigid_align` + root-joint pivot correction → `global_orient` |
| Output identity | `betas`, `markers_latent`, `latent_labels`, `marker_meta`, `markers_latent_vids` | `betas`, `markers_latent`, `nn_idx`, `coeffs`, `latent_labels` |
| Caching | pickle on disk | in-memory only |

---

## 8. Stage II — per-frame pose fitting

### Legacy — `chmosh.py::mosh_stageii`

1. **Load mocap** via `MocapSession` again (no filtering on labels — keeps everything for visualization).
2. **Load body model** with `num_beta_shared_models=1`. Lock `can_model.betas` to the stage-I betas. Optionally enable DMPLs (`shapedirs[..., num_betas:num_betas+num_dmpls] = dmpl_pcs`, then `opt_model.dmpl = opt_model.betas[num_betas:num_betas+num_dmpls]`).
3. **Build the transformed-landmark graph** (`chmosh.py:502-503`): one `TransformedCoeffs(can_body=can_model.r, markers_latent=stagei_markers_latent)`, then one `TransformedLms` referencing `opt_model`. **`markers_latent` is now a frozen numpy array — not optimized in stage II.**
4. **Per-frame loop** over `range(start_fidx, end_fidx, ds_rate)`:
   - **Anneal `wt_pose`** based on missing markers: `anneal_factor = 1 + (n_missing / M) * stageii_wt_annealing`.
   - **Normalize `wt_data`** by present marker count: `wt_data = stageii_wt_data * (num_train_markers / k)`, `num_train_markers = 46`.
   - **Loss terms**:
     - `data = (sim - obs) * wt_data` — k present-this-frame markers only.
     - `poseB = GMM(pose[pose_body_ids]) * wt_pose` — body GMM.
     - `velo = (pose - extrapolated_pose) * wt_velo` — constant-velocity prior, `extrap = pose_prev + (pose_prev - pose_prev_prev)` (only after the first 2 frames).
     - `poseH` (fingers, L2), `poseF` (jaw, L2), `expr` (expression, L2) — gated by config.
     - `dmpl = opt_model.dmpl * wt_dmpl`, `extrap_dmpl = (dmpl - extrap_dmpl) * 6.0` — gated by `optimize_dynamics`.
   - **First active frame**: rigid alignment, then `for wt_pose_first in [10*wt_pose, 5*wt_pose, wt_pose]:` run dogleg minimize on `[trans, pose[pose_ids]]` — a 3-step anneal on the *pose* prior to ease the body out of the rest pose.
   - **Step 1** (subsequent frames): minimize on `[trans, pose[pose_root_ids + pose_body_ids]]` — warm start the root + body pose.
   - **Step 2**: minimize on `[trans, pose[pose_ids (+ finger + face)]]`, adding finger/face/expr/dmpl losses and free vars.
5. **Save** per-frame `fullpose`, `trans`, `markers_sim`, `markers_obs`, `labels_obs`, plus `dmpls`/`expression` if enabled, plus the original mocap `markers` and `labels`.

### Torch port — `fit.py::mosh_stageii`

Two implementations selected by `cfg.batch_size`:

#### Per-frame mode (`batch_size <= 1`) — `_mosh_stageii_per_frame`

For each frame `t`:
1. **Warm start** from frame `t-1`'s converged `(z, global, transl)`.
2. **First frame only**: rigid-align the rest body to the observed markers (`fit.py:473-492`), seed `global_orient` and `transl`. Then run a 3-pass anneal on `wt_z_mult` (default `(10.0, 5.0, 1.0)`) — directly mirroring legacy `chmosh.py:637`.
3. **Closure** (`fit.py:498-516`):
   - `body_pose = vposer.decode_aa(z_t.unsqueeze(0))` → `(1, 63)`.
   - `verts = body_model(betas.unsqueeze(0), body_pose, global_t, transl_t)`.
   - `sim = synth_markers(verts, nn_idx, coeffs)` — `coeffs` frozen from stage I.
   - `data_term = ||sim[label_idx] - obs||²`, `z_term = ||z||²`.
   - **Velocity prior** (after 2 frames): `loss += wt_velo * ||z_t - (2*z_prev - z_prev2)||²` — constant-velocity in *z-space* (vs. legacy's axis-angle space).
4. **LBFGS** with `max_iter=8` per subsequent frame (`max_iter=30` on first frame), `lr=1.0`, strong-Wolfe.

#### Batched mode (`batch_size > 1`) — `_mosh_stageii_batched`

1. **Phase 1**: rigid-align every frame independently from the rest body, fill `out_global[t]`, `out_transl[t]`.
2. **Phase 2**: chunked LBFGS over `cfg.batch_size` frames at once.
   - Free vars per chunk: `z_chunk: (B, 32)`, `global_chunk: (B, 3)`, `transl_chunk: (B, 3)`.
   - Closure runs one batched `body_model(...)` forward over `B` frames, accumulates per-frame data residuals, adds:
     - `z_term = ||z_chunk||²`
     - **Velocity within chunk**: `||z[2:] - (2*z[1:-1] - z[:-2])||²`
     - **Stitch across chunk boundaries**: use the previous chunk's last two z values (detached) to extrapolate into the current chunk's first one or two frames.
   - `max_iter = 80` per chunk.
3. Set `batch_size = T` to optimize the entire sequence jointly.

### Stage II differences at a glance

| Aspect | Legacy | Torch port |
|---|---|---|
| Free body-pose vars | `pose[pose_body_ids]` axis-angle | 32-d `z` (VPoser latent) |
| Free vars per frame | `trans`, `pose[pose_ids]` (+ `dmpl` + `v_face_exp`) | `z`, `global_orient`, `transl` |
| Body model state | single chumpy `opt_model` reused, in-place updates | stateless `smplx` forward call |
| Pose prior | GMM Mahalanobis | `‖z‖²` |
| Velocity prior | constant-velocity in axis-angle pose | constant-velocity in z (latent) |
| First-frame warmup | rigid + 3-step anneal on `wt_pose` | rigid + 3-step anneal on `wt_z` (mirrors legacy) |
| Optimizer | chumpy dogleg, 100 iters × 2 steps per frame | LBFGS strong-Wolfe, 8 iters per frame (30 first) — or batched 80 iters per chunk |
| Missing-marker handling | `wt_data` scaled by `num_train_markers / k`, `wt_pose` upweighted | dropped markers excluded from `data_term` directly; no weight adjustment |
| DMPLs / dynamics | optional via `optimize_dynamics` | dropped |
| Fingers / face / expression | optional, added in step 2 | dropped |
| Frame range | `range(start_fidx, end_fidx, ds_rate)` | full sequence (caller slices) |
| Visualization | `visualize_pose_estimate` (psbody MeshViewer) | dropped |
| Output schema | `fullpose: (T, full)`, `trans: (T, 3)` (+ dmpls, expression) | `body_pose: (T, 63)`, `global_orient: (T, 3)`, `transl: (T, 3)`, `z: (T, 32)`, `loss: (T,)` |

---

## 9. Surface loss (scan2mesh)

Legacy uses a *signed* point-to-mesh distance to keep markers exactly on the surface:
- `prepare_mosh_markers_latent` (`chmosh.py:57-80`) builds `PtsToMesh(sample_verts=markers_latent, reference_verts=can_model, reference_faces=can_mesh.f, normalize=False, signed=True)` from the `scan2mesh` subpackage.
- `surf` loss is `(PtsToMesh - desired_distances) * stagei_wt_surf` where `desired_distances = m2b_distance[per-marker]` (≈ 9.5 mm for body).
- This is what enforces "markers sit at the right offset from the body surface", letting `init` and `data` weights be relatively low.

The torch port **drops this term entirely** (see memory: moshpp modernization decisions). Rationale:
- `scan2mesh` is dead code (psbody-dependent, doesn't install).
- For clean mocap data, the `init` term (which keeps markers near their template placement) plus the `data` term is enough.
- If reintroduced later, the natural replacement is `pytorch3d.loss.point_mesh_face_distance` with sign recovered from the face normal.

---

## 10. Other dropped/deferred features

From the legacy code that the torch port deliberately ignores:

| Feature | Legacy location | Torch status |
|---|---|---|
| Multi-subject sessions | `mocap.multi_subject`, subject filtering in `MocapSession` | dropped |
| Animal models (horse, dog, rat) | `prior/horse_body_prior.py`, `prior/dog_body_prior.py`, `surface_model.type in ['animal_*', 'object']` | dropped |
| Rigid object fitting | `models/object_model.py::RigidObjectModel` | dropped |
| DMPLs (dynamics) | `chmosh.py:507-514`, `mosh_head.py:498-500` | dropped |
| SMPL-X face expressions | `chmosh.py:285-298`, `betas[300:380]` aliasing | dropped (SMPL-H has no face) |
| Finger PCA optimization | `optimize_fingers` paths everywhere | dropped (hands frozen at flat rest) |
| Head-marker covariance prior | `head_marker_corr_fname`, `chmosh.py:252-266` | dropped |
| Hand prior (`MaxMixturePriorHands`) | `prior/gmm_prior_ch.py:137-167` | dropped |
| Frame-picker strategies (`random_strict`, `random`, `manual`) | `frame_picker.py` | replaced by linearly-spaced indices |
| c3d I/O (`ezc3d`) | `tools/mocap_interface.py`, `marker_layout_to_c3d` | dropped (caller passes numpy) |
| Visualization (`psbody.mesh.MeshViewer`) | `tools/visualization.py` | dropped |
| Marker-layout autogeneration | `marker_layout/create_marker_layout_for_mocaps.py` | dropped (caller passes `marker_vids` dict) |
| `dump_stagei_marker_layout` (ply/c3d outputs) | `mosh_head.py:303` | dropped |
| `load_as_amass_npz` (re-pack to AMASS format) | `mosh_head.py:445` | dropped |
| `extra_initial_rigid_adjustment` | `chmosh.py:230-232` | dropped (one rigid seed is enough) |
| `wt_data` normalization by `num_train_markers=46` | `chmosh.py:327`, `chmosh.py:603` | dropped (caller tunes `wt_data`) |
| `mocap_rotate` (XYZ Euler in degrees) | `MocapSession`, `frame_picker` | dropped (caller pre-rotates) |
| `mm/cm/m` unit conversion | `MocapSession` | minimal heuristic (`api.py:82-89`) |

---

## 11. Loss-term comparison cheatsheet

| Term | Legacy stage I weight | Torch stage I default | Notes |
|---|---|---|---|
| `data` (markers) | `75.0 / anneal · (46 / M)` | `1000.0 / anneal` | torch doesn't normalize by M |
| `init` (markers_latent close to template) | `300.0 · anneal` per type | `50.0 · anneal` global | per-type weights collapsed |
| `surf` (point-to-mesh) | `10000.0` flat | — | dropped |
| `poseB` (body GMM) | `3.0 · anneal` | — | replaced by z prior |
| `z` prior (latent) | — | `5.0 · anneal` | new |
| `betas` (shape regularization) | `10.0 · anneal` | `1.0 · anneal` | both `‖βetas‖²` |
| `poseH`, `poseF`, `expr` | detailed-step only | — | dropped |

| Term | Legacy stage II weight | Torch stage II default |
|---|---|---|
| `data` | `400.0 · (46 / k)` | `1000.0` |
| `poseB` (body GMM) | `1.6 · anneal_missing` | — |
| `z` prior | — | `5.0` |
| `velo` (constant velocity) | `2.5` in pose space | `100.0` in z-space |
| `poseH`, `poseF`, `expr` | gated | — |
| `dmpl`, `extrap_dmpl` | gated | — |

---

## 12. Output schemas

### Legacy `stagei.pkl`
```python
{
    'betas': np.ndarray,                 # (num_betas,)
    'markers_latent': np.ndarray,        # (M, 3) — positions in canonical (T-pose) frame
    'latent_labels': list[str],          # length M
    'marker_meta': Markerlayout,         # full marker layout dict
    'markers_latent_vids': dict,         # label -> nearest body vid (post-opt)
    'v_template_fname': str (optional),
    'stagei_debug_details': {
        'opt_models_trans': list[np.ndarray],    # per ref-frame trans (12 × 3)
        'opt_models_pose': list[np.ndarray],     # per ref-frame pose (12 × full)
        'stagei_errs': dict[str, float],          # final sum-sq of each loss term
        'markers_latent_all_vids': dict,          # all observed labels → nearest vid on posed body
        'stagei_markers_sim': list,
        'stagei_markers_obs': list,
        'stagei_labels_obs': list,
        'stagei_fnames': list[str],               # picked frame ids
        'stagei_frames': list[dict],              # actual marker dicts
        'cfg': dict,                              # full resolved OmegaConf
        'stagei_elapsed_time': float,
        'v_template': np.ndarray (optional),
    },
}
```

### Legacy `stageii.pkl` (merged with stagei.pkl content)
```python
{
    # ... stagei keys ...
    'fullpose': np.ndarray,              # (T, full_pose_dim)
    'trans': np.ndarray,                 # (T, 3)
    'dmpls': np.ndarray (optional),      # (T, num_dmpls)
    'expression': np.ndarray (optional), # (T, num_expressions)
    'stageii_debug_details': {
        'stageii_errs': dict[str, np.ndarray],  # per-frame loss per term
        'markers_sim': list,
        'markers_obs': list,
        'labels_obs': list,
        'markers_orig': np.ndarray,             # all raw mocap markers
        'labels_orig': list[str],
        'mocap_fname': str,
        'mocap_frame_rate': float,
        'mocap_time_length': float,
        'stageii_elapsed_time': float,
        'cfg': dict,
    },
}
```

### Torch port return dict
```python
{
    'latent_labels': list[str],
    'marker_vids': torch.LongTensor,     # (M,)
    'stagei': {
        'betas': torch.Tensor,            # (num_betas,)
        'markers_latent': torch.Tensor,   # (M, 3) on the (post-opt) canonical body
        'nn_idx': torch.LongTensor,       # (M, 3)
        'coeffs': torch.Tensor,           # (M, 3) — local-frame coefficients
        'latent_labels': list[str],
    },
    'stageii': {
        'betas': torch.Tensor,            # (num_betas,)
        'z': torch.Tensor,                # (T, 32) — VPoser latent
        'body_pose': torch.Tensor,        # (T, 63) — axis-angle, 21 body joints
        'global_orient': torch.Tensor,    # (T, 3)
        'transl': torch.Tensor,           # (T, 3)
        'loss': torch.Tensor,             # (T,) — per-frame final SSE
    },
    'faces': torch.LongTensor,           # (F, 3)
    'gender': str,
}
```

---

## 13. End-to-end legacy call graph (one trial)

```
run_moshpp_once(cfg)
  └─ MoSh(cfg)                                        # mosh_head.py
       ├─ prepare_cfg(...)                            # OmegaConf merge
       └─ paths derived (stagei_fname, stageii_fname, marker_layout fname)
  └─ mp.mosh_stagei(chmosh.mosh_stagei)               # mosh_head.py:200
       ├─ prepare_stagei_frames()                     # frame_picker.load_marker_sessions_*
       │     └─ MocapSession(...) per c3d (ezc3d)
       ├─ marker_labels_to_marker_layout(...)         # autogen layout if missing
       └─ chmosh.mosh_stagei(...)                     # chmosh.py:83
             ├─ marker_layout_load(...)               # edit_tools.py:83
             ├─ load_moshpp_models(...)               # bodymodel_loader.py:81
             │     ├─ load_surface_model(...)         # smpl_fast_derivatives.py
             │     └─ create_gmm_body_prior(...)      # gmm_prior_ch.py:107
             ├─ prepare_mosh_markers_latent(...)      # chmosh.py:57
             │     └─ PtsToMesh(...)                  # scan2mesh/mesh_distance_main.py
             ├─ TransformedCoeffs(...) + TransformedLms(...)  # transformed_lm.py
             ├─ perform_rigid_adjustment(...)         # rigid_transformations.py:72
             ├─ for anneal in [1.0, 0.5, 0.25, 0.125]:
             │     ch.minimize(opt_objs, x0=free_vars, method='dogleg', ...)
             └─ pickle.dump(stagei_data, stagei_fname)
  └─ mp.mosh_stageii(chmosh.mosh_stageii)             # mosh_head.py:268
       └─ chmosh.mosh_stageii(...)                    # chmosh.py:458
             ├─ MocapSession(...)
             ├─ load_moshpp_models(num_beta_shared_models=1)
             ├─ TransformedCoeffs + TransformedLms (markers_latent locked)
             ├─ for frame t in range(start, end, ds_rate):
             │     if first: rigid + 3-step pose anneal
             │     else:
             │         ch.minimize step 1: [trans, pose_root+body]
             │         ch.minimize step 2: [trans, pose_all (+ dmpl, expr)]
             └─ pickle.dump(stageii_data, stageii_fname)
```

## 14. End-to-end torch call graph

```
fit_smpl_to_markers(markers, labels, marker_vids, smplh_path, vposer_dir, ...)   # api.py
  ├─ device pick (cuda > mps > cpu)
  ├─ mm→m heuristic
  ├─ label filtering against marker_vids
  ├─ SMPLHBodyModel(smplh_path, gender, num_betas, device)                       # body_model.py
  ├─ FrozenVPoser(vposer_dir, device)                                            # vposer_prior.py
  ├─ if run_stage_i:
  │     stagei_frame_ids = linspace(0, T-1, 12)
  │     stagei_frames = _frames_from_array(markers[ids], ...)                    # fit.py:41
  │     mosh_stagei(body_model, vposer, stagei_frames, ..., cfg=StageICfg(...))  # fit.py:180
  │       ├─ v_template + vertex_normal * m2b_distance   → init_markers
  │       ├─ build_local_frame(v_template, init_markers) → (nn_idx, coeffs)      # FROZEN
  │       ├─ params: markers_latent, betas, z, global_orient, transl
  │       ├─ per-frame rigid_align → seed global_orient + transl                 # fit.py:88
  │       └─ for anneal in [10, 5, 2, 1, 0.5]:
  │             LBFGS strong-Wolfe, 30 iters, closure recomputes coeffs each step
  │             (data + init + ||z||² + ||β||² losses)
  │       └─ bake final_coeffs = compute_coeffs(canonical_verts(betas), markers_latent, nn_idx)
  ├─ observed = _frames_from_array(markers, ...)                                 # fit.py:41
  └─ mosh_stageii(body_model, vposer, observed, betas, nn_idx, coeffs, cfg)      # fit.py:336
        ├─ if cfg.batch_size <= 1:  _mosh_stageii_per_frame(...)                 # fit.py:412
        │     for t in 0..T-1:
        │       warm-start z, global, transl from t-1
        │       if first: rigid_align + 3-pass anneal on wt_z * (10, 5, 1)
        │       else:     LBFGS 8 iters, closure = data + ||z||² + velo_in_z
        └─ else:                     _mosh_stageii_batched(...)                  # fit.py:588
              phase 1: per-frame rigid_align seeds global_orient + transl
              phase 2: chunked LBFGS 80 iters per chunk over (z, global, transl)
                       intra-chunk velocity prior in z-space + cross-chunk stitching
```

---

## 15. Practical mapping from a legacy `stageii.pkl` to torch outputs

For someone who has a legacy `stageii.pkl` and wants to convert to the torch schema:

| Legacy key | Torch key | Conversion |
|---|---|---|
| `stageii_pkl['betas'][:num_betas]` | `stageii['betas']` | identical (when num_betas matches) |
| `stageii_pkl['fullpose'][:, :3]` | `stageii['global_orient']` | identical |
| `stageii_pkl['fullpose'][:, 3:66]` | `stageii['body_pose']` | only SMPL-H/SMPL-X; for SMPL use `[:, 3:]` (truncate to first 21 joints) |
| `stageii_pkl['fullpose'][:, 66:]` | (frozen at zeros in torch) | hands; torch keeps them at flat rest |
| `stageii_pkl['trans']` | `stageii['transl']` | identical |
| `stageii_pkl['markers_latent']` | `stagei['markers_latent']` | identical |
| `stageii_pkl['latent_labels']` | `stagei['latent_labels']` | identical |
| `stageii_pkl['markers_latent_vids']` | (input `marker_vids` to torch) | reuse as `marker_vids` argument |

To re-encode a legacy `fullpose[:, 3:66]` into VPoser latent space for a torch run, use `FrozenVPoser.encode_mean(body_pose_aa)` — but in v1 this isn't called: stage II just zero-initializes `z` and lets LBFGS find it.

---

## 16. Open questions / things to verify before claiming parity

- **Anneal schedule**: the torch port uses 5 steps `[10, 5, 2, 1, 0.5]` vs. legacy's 4 steps `[1, 0.5, 0.25, 0.125]`. The directionality is the same (data ↑, priors ↓), but the magnitudes and rate are different. Worth sweeping once a walking trial is fitting end-to-end.
- **`wt_data` normalization by 46 markers**: legacy uses `* (num_train_markers / M)` everywhere. Torch hard-codes `wt_data`, so callers must tune for their marker count. Document this in `fit_smpl_to_markers` if needed.
- **3-NN frozen vs. relaxed**: legacy recomputes 3-NN every chumpy tick. The memory note says this barely changes in practice — but it's worth a sanity check on a real fit by comparing `nn_idx` before/after stage I.
- **Velocity prior in z-space vs. pose-space**: legacy is `||pose_t - extrap(pose_{t-1}, pose_{t-2})||` in raw axis-angle. Torch is the same form but in 32-d z. These have different metrics — z-space is smoother but `wt_velo=100` is doing a lot of work. Compare on a walking sequence with a quick foot strike.
- **First-frame rigid seed**: legacy applies `rigid_landmark_transform` directly to `pose[:3]` (Rodrigues) and `trans`. Torch applies it to `global_orient` and corrects `transl` for the root-joint pivot. Verify with a side-by-side rest-body alignment.
- **Hand pose**: legacy with `use_hands_mean=True` keeps hands at the mean MANO pose, not the flat rest. Torch uses `flat_hand_mean=True`, so default hands are flat. For SMPL-H bodies the legacy mean-hand pose may produce visibly bent fingers — confirm this isn't biasing rigid alignment.
