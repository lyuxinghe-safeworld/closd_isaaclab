# CLoSD-IsaacLab: Standalone Text-to-Motion with Isaac Lab

**Date**: 2026-03-15
**Status**: Draft

## 1. Goal

Port CLoSD's closed-loop text-to-motion pipeline to Isaac Lab by replacing CLoSD's PHC controller with ProtoMotions' pretrained motion tracker. The system takes a text prompt, generates motion via CLoSD's DiP diffusion model, and tracks it in Isaac Lab using ProtoMotions' SMPL humanoid, producing a video recording.

## 2. Background

### CLoSD Pipeline (Original)

CLoSD implements a closed-loop system:

1. **DiP (Diffusion Planner)**: An MDM-based transformer that generates motion in HumanML3D format (263-dim per frame at 20fps). Takes a 20-frame prefix and predicts 40 future frames. Uses DDPM with 10 diffusion steps and classifier-free guidance.

2. **HumanML3D format**: A delta-based representation where:
   - Dim 0: root angular velocity (Y-axis, delta)
   - Dims 1-2: root XZ linear velocity (local frame, delta)
   - Dim 3: root height (absolute)
   - Dims 4-66: RIC joint positions (21 joints x 3, root-relative, rotation-invariant)
   - Dims 67-192: 6D continuous rotations (21 joints x 6)
   - Dims 193-258: local joint velocities (22 joints x 3)
   - Dims 259-262: foot contacts (4 binary flags)

   Root position and rotation are recovered via cumulative summation of deltas. Joint positions are recovered by undoing the rotation-invariance transform using the accumulated root rotation.

3. **PHC controller**: An RL policy that tracks reference body positions. Uses only positions and velocities (obs_v7: position diffs + velocity diffs + reference positions, all heading-local). "Leaves IK for RL" — never consumes rotation data.

4. **Closed loop**: Sim state → `pose_to_hml()` (30fps Isaac positions → 20fps normalized HML with `recon_data` capturing root transform) → DiP → `hml_to_pose()` (HML → positions via `recover_from_ric()`, aligned to sim via two-step `recon_data` bridge) → controller → sim.

### ProtoMotions Motion Tracker

An MLP policy (PPO-trained on AMASS) that:
- **Input**: current robot proprioception + reference motion as full `RobotState` (body pos/rot/vel/ang_vel, dof_pos/dof_vel) at future timesteps
- **Output**: joint angle PD targets `[num_envs, 69]` (23 joints x 3 DOF)
- **Applied via**: Isaac Lab's built-in PD controller

The tracker expects `RobotState` from `MotionLib.get_motion_state()`, which provides all fields. Unlike CLoSD's position-only controller, it was trained with full rotation/DOF observations.

### Quaternion Convention Map

All quaternion handoffs must be explicit:

| System | Convention | Notes |
|--------|-----------|-------|
| HumanML3D / CLoSD internals | **wxyz** | `recover_root_rot_pos()` puts w at index 0; `qmul`/`qinv` in CLoSD assume wxyz |
| Isaac Gym (CLoSD sim) | **xyzw** | Isaac Gym Preview uses xyzw |
| ProtoMotions COMMON | **xyzw** | All algorithm-layer data; `RobotState` with `StateConversion.COMMON` |
| ProtoMotions SIM (Isaac Lab) | **wxyz** | Isaac Lab/PhysX uses wxyz; `DataConversionMapping.sim_w_last=False` auto-converts |

**Conversion points**:
- `hml_conversion.py`: internal math uses wxyz (matching CLoSD). `recon_data` stores wxyz quaternions.
- `robot_state_builder.py`: converts CLoSD wxyz → ProtoMotions xyzw before constructing `RobotState(state_conversion=COMMON)`.
- ProtoMotions' `DataConversionMapping` handles xyzw↔wxyz for sim I/O automatically.

### Key Porting Challenges

| Issue | Isaac Gym (CLoSD) | Isaac Lab (ProtoMotions) |
|-------|-------------------|--------------------------|
| Quaternion convention | wxyz (CLoSD internals), xyzw (Isaac Gym API) | wxyz (sim), xyzw (common) |
| Joint ordering | Depth-first (MJCF) | Breadth-first (USD), remapped via `DataConversionMapping` |
| Body names | Identical 24-body SMPL hierarchy | Same names, runtime reordering handled by ProtoMotions |
| PD control | `gym.set_dof_position_target_tensor()` | `ImplicitActuatorCfg` or `robot.set_joint_position_target()` |
| FPS | Diffusion 20fps, sim 30fps | Configurable via `sim.dt` + `decimation` |
| Height offset | `offset_height = 0.92` (Y-up), `offset = 0.0` in CLoSD rep_util.py | Z-up; configurable in `coord_transform.py` |

## 3. Architecture

### Project Structure

```
closd_isaaclab/
├── closd_isaaclab/
│   ├── __init__.py
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── motion_provider.py      # DiP wrapper, autoregressive closed-loop generation
│   │   ├── hml_conversion.py       # pose_to_hml / hml_to_pose (ported from CLoSD rep_util)
│   │   ├── rotation_solver.py      # Diffusion rotation extraction + analytical IK + FK verification
│   │   └── robot_state_builder.py  # HML output → ProtoMotions RobotState
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── closd_motion_lib.py     # MotionLib-compatible wrapper around diffusion provider
│   │   ├── closd_motion_manager.py # MimicMotionManager subclass for closed-loop
│   │   └── experiment.py           # ProtoMotions experiment config wiring
│   └── utils/
│       ├── __init__.py
│       ├── coord_transform.py      # Isaac↔SMPL coordinate transforms, joint reordering
│       └── fps_convert.py          # 20fps↔30fps bicubic interpolation
├── scripts/
│   ├── verify_diffusion.py         # Standalone diffusion → matplotlib MP4 (no sim)
│   ├── verify_rotations.py         # 6D rotation conversion correctness + FK consistency
│   ├── verify_tracking.py          # Offline .motion → ProtoMotions tracker → Isaac Lab video
│   └── run_closd_isaaclab.py       # Full closed-loop text-to-motion pipeline
├── tests/
│   ├── test_hml_conversion.py      # Round-trip pose_to_hml / hml_to_pose
│   ├── test_rotation_solver.py     # 6D conversion correctness against ground truth
│   ├── test_coord_transform.py     # Isaac↔SMPL transforms
│   └── test_robot_state_builder.py # RobotState assembly
├── pyproject.toml
└── README.md
```

### Component Design

#### 3.1 `diffusion/motion_provider.py` — DiP Wrapper

Wraps CLoSD's MDM for autoregressive closed-loop generation.

```
class DiffusionMotionProvider:
    __init__(model_path, mean_path, std_path, device, guidance=5.0,
             context_len=20, pred_len=40)

    generate_next_horizon(
        pose_buffer_isaac: [bs, context_len_30fps, 24, 3],  # Isaac Lab body positions
        recon_data_prev: {r_rot: [bs,4], r_pos: [bs,3]} | None,
        text_prompt: str | list[str]
    ) -> (
        positions_isaac: [bs, horizon_30fps, 24, 3],   # predicted positions in Isaac coords
        hml_raw: [bs, pred_len_20fps, 263],             # raw HML for rotation extraction
        recon_data: {r_rot, r_pos}                      # for next call
    )

    generate_standalone(
        text_prompt: str, num_seconds: float, prefix_mode="standing"
    ) -> (positions_smpl: [1, T, 22, 3], hml_raw: [1, T, 263])
        # Open-loop generation for verify_diffusion.py (no sim feedback)
```

Internally:
1. `pose_to_hml()`: 30fps Isaac → 20fps normalized HML + `recon_data`
2. MDM `p_sample_loop()`: 10 DDPM steps with classifier-free guidance
3. `hml_to_pose()`: HML → positions via `recover_from_ric()` + `recon_data` alignment + 20fps→30fps

#### 3.2 `diffusion/hml_conversion.py` — HumanML3D Conversion

Ported from CLoSD's `rep_util.py` and `motion_process_torch.py`. Two core functions:

**`pose_to_hml(positions_isaac, coord_transform)`**:
- Input: `[bs, T_30fps, 24, 3]` Isaac Lab body positions
- Append extrapolated dummy frame (for velocity computation)
- 30fps → 20fps bicubic downsampling
- Isaac → SMPL coordinate transform + joint reorder (mujoco→smpl, 24→22 joints)
- `extract_features_t2m()`: absolute positions → HML 263-dim deltas
  - Root angular velocity: quat diff between frames → arcsin
  - Root XZ velocity: position diff in local frame
  - Root height: absolute Y
  - RIC: subtract root XZ, rotate by root heading → egocentric
  - 6D rotations: IK to get per-joint quats → continuous 6D
  - Local velocities: joint position diffs in root frame
  - Foot contacts: velocity threshold on ankle/toe joints
- Save `recon_data = {r_rot, r_pos}` at frame `[-2]` (second-to-last input frame, which is the last frame with valid feature data since velocities consume one frame). `recon_data` quaternions are in **wxyz** convention.
- Normalize with mean/std
- Output: `[bs, T_20fps, 263]` + `recon_data`

**`hml_to_pose(hml_norm, recon_data, sim_at_hml_idx)`**:
- `sim_at_hml_idx`: the frame index within the generated HML sequence that corresponds to the sim's current state (typically `prefix_len - 1`). This is the alignment anchor.
- Unnormalize
- `recover_root_rot_pos()`: cumsum of angular deltas → absolute root rotation (wxyz); cumsum of rotated XZ velocity deltas → absolute root position; absolute root height from dim 3
- `recover_from_ric()`: rotate egocentric joint positions by accumulated root rotation, add root XZ → global 22-joint positions
- **Two-step alignment** (the critical stitching step):
  1. Extract HML's own root transform at `sim_at_hml_idx`: `hml_transform = {r_rot[sim_at_hml_idx], r_pos[sim_at_hml_idx]}`
  2. Zero out: subtract HML root XZ position, rotate by inverse of HML root rotation → motion centered at origin facing forward
  3. Apply sim's `recon_data`: rotate by sim root rotation, add sim root XZ position → motion placed at sim's actual world position
  This ensures seamless stitching between prediction and current sim state.
- 22→24 joints (add hands by extending wrist direction)
- SMPL → Isaac coordinate transform + joint reorder
- 20fps → 30fps bicubic upsampling
- Output: `[bs, T_30fps, 24, 3]`

#### 3.3 `diffusion/rotation_solver.py` — Rotation Derivation

Two modes controlled by a flag:

**`mode="diffusion"`**:
- Extract 21-joint 6D rotations from HML dims 67-192. **These are LOCAL (parent-relative) rotations**, not global — they are produced by `Skeleton.inverse_kinematics_np()` during HumanML3D data preprocessing, which computes parent-relative joint rotations.
- Convert 6D → local rotation matrices via Gram-Schmidt orthogonalization (the standard continuous 6D → SO(3) map)
- Prepend root rotation from `recover_root_rot_pos()` (wxyz quat → rotation matrix) as joint 0
- Extend to 24 joints (hands = identity rotation or wrist rotation copied)
- Reorder SMPL → mujoco joint order
- These are already local/parent-relative rotations — feed directly to ProtoMotions' `extract_qpos_from_transforms(kinematic_info, root_pos, joint_rot_mats)` → `dof_pos [bs, T, 69]`
- **Do NOT call `compute_joint_rot_mats_from_global_mats()`** — that function converts global→local, but our rotations are already local
- **FK consistency check**: run `compute_forward_kinematics_from_transforms(kinematic_info, root_pos, joint_rot_mats)` with the local rotations, compare resulting positions against `recover_from_ric()` positions. If mean error > 5cm, print `WARNING: Diffusion rotations show high FK inconsistency ({error}cm). Consider --rotation-mode analytical`.

**`mode="analytical"`**:
- Compute bone direction vectors from parent→child positions
- For each joint: align reference bone direction to actual direction via rotation
- Handle twist DOF via heuristic (e.g., minimize deviation from rest pose)
- Same downstream path: global → local rotations → `dof_pos`
- FK consistency check (should be tighter since rotations are derived from positions)

**`verify_consistency(positions, global_rot_quat, dof_pos) -> (mean_error, per_joint_error)`**:
- Runs FK from `dof_pos`, compares against `positions`
- Returns error in meters

#### 3.4 `diffusion/robot_state_builder.py` — Assemble RobotState

```
class RobotStateBuilder:
    __init__(kinematic_info, rotation_solver, sim_dt)

    build(positions_smpl: [bs, T, 24, 3],
          hml_raw: [bs, T, 263],
          dt: float
    ) -> cached motion data (positions, rotations, velocities, dof_pos, dof_vel)

    get_state_at_time(env_ids, times) -> RobotState
        # Interpolates cached data at requested times
        # Positions: linear interp
        # Rotations: SLERP
```

Velocity derivation:
- `rigid_body_vel`: `(pos[t+1] - pos[t-1]) / (2*dt)` central differencing
- `rigid_body_ang_vel`: from quaternion finite difference `2 * qmul(qinv(q[t]), q[t+1]) / dt`
- `dof_vel`: `(dof_pos[t+1] - dof_pos[t-1]) / (2*dt)`

Contact derivation:
- `rigid_body_contacts`: `[num_bodies]` per frame. HML dims 259-262 provide 4 foot contact flags (L_Ankle, L_Toe, R_Ankle, R_Toe). Map these to the corresponding body indices. All other bodies get contact=0.

**Performance note**: The IK in `pose_to_hml()` (via `Skeleton.inverse_kinematics_np()`) goes through NumPy/CPU. This is acceptable since it runs only once per planning horizon (~every 20 sim steps), not every frame.

#### 3.5 `integration/closd_motion_lib.py` — MotionLib Interface

```
class CLoSDMotionLib:
    """Duck-types MotionLib for MimicControl.get_context() and MimicMotionManager."""

    Properties (stubs for interface compatibility):
        num_motions() -> 1
        motion_lengths -> tensor([episode_length * dt])  # accessed by MimicMotionManager.get_done_tracks()
        motion_weights -> tensor([1.0])
        motion_file -> "closd_diffusion"  # for state_dict compatibility

    get_motion_state(motion_ids, motion_times) -> RobotState:
        # Delegates to robot_state_builder.get_state_at_time()
        # Returns interpolated state in COMMON format (xyzw quaternions)
        # RobotState includes rigid_body_contacts: foot contacts from HML dims 259-262
        #   mapped to body indices [L_Ankle, R_Ankle, L_Toe, R_Toe], all others zero.

    get_motion_length(motion_ids) -> tensor:
        # Returns motion_lengths[motion_ids] (called by MimicControl.get_context())
```

#### 3.6 `integration/closd_motion_manager.py` — Closed-Loop Manager

```
class CLoSDMotionManager(MimicMotionManager):
    __init__(..., motion_provider, robot_state_builder, pred_len_20fps=40)

    # planning_horizon_30fps is DERIVED, not a separate parameter:
    #   planning_horizon_30fps = int(pred_len_20fps * 30 / 20)  # = 60 for pred_len=40
    # This prevents silent inconsistency if pred_len changes.

    pose_buffer: [num_envs, context_len_30fps, 24, 3]  # sliding window of sim positions
    recon_data: {r_rot, r_pos}  # root state from last diffusion call (wxyz quaternions)
    frame_counter: int  # frames since last diffusion call

    post_physics_step():
        super().post_physics_step()  # advances self.motion_times += env_dt
        # 1. Get current sim body positions from simulator
        # 2. Append to pose_buffer (sliding window)
        # 3. Advance frame_counter
        # 4. If frame_counter % planning_horizon_30fps == 0:
        #      Call motion_provider.generate_next_horizon(pose_buffer, recon_data, prompt)
        #      Update robot_state_builder cache with new horizon
        #      Update recon_data for next call
        #      Reset motion_times to 0 for affected envs
        #      Set motion_lib.motion_lengths to new horizon duration
        #      (This prevents get_done_tracks() from signaling done mid-episode)

    sample_motions(env_ids):
        # On reset: clear pose_buffer, fill with current pose repeated
        # Reset recon_data to None (first diffusion call uses default)
        # Reset motion_times[env_ids] = 0
        # Set motion_lib.motion_lengths = [planning_horizon_30fps * env_dt]
```

**`motion_times` management**: Each time the diffusion generates a new horizon, `motion_times` is reset to 0 and `motion_lengths` is set to the new horizon's duration. This way `MimicMotionManager.get_done_tracks()` only signals done when the current horizon is exhausted (triggering a new diffusion call), not when the episode time exceeds some arbitrary value. The episode terminates based on `episode_length` from the experiment config, not from `get_done_tracks()`.

#### 3.7 `integration/experiment.py` — ProtoMotions Experiment Config

A ProtoMotions experiment file that wires everything together:
- Robot: SMPL humanoid (`smpl_humanoid.usda`)
- Simulator: Isaac Lab
- Agent: Load pretrained motion tracker from `data/pretrained_models/motion_tracker/smpl/last.ckpt`
- MotionManager: `CLoSDMotionManager` (custom)
- MotionLib: `CLoSDMotionLib` (custom, wraps diffusion)
- Control: `MimicControl` with appropriate `num_future_steps`
- Observations: Same as the pretrained tracker's config (loaded from `resolved_configs_inference.pt`)

### Coordinate Transform Constants

Ported from CLoSD `rep_util.py`:

```python
# Rotation matrices
to_isaac_mat = Rx(-pi/2)          # SMPL Y-up → Isaac Z-up
y180_rot = Ry(pi)                 # 180-degree Y rotation
smpl2sim_rot_mat = Rx(-pi)        # = Rx(-pi/2) @ Rx(-pi/2), a 180-deg X rotation

# Height offset (from CLoSD rep_util.py)
# offset_height = 0.92 (SMPL standing height in Y-up)
# offset = 0.0 (currently unused, FIXME in CLoSD)
# In Isaac Lab (Z-up), this maps to a Z offset. Configurable in coord_transform.py.

# Joint reordering (24 elements each)
smpl_2_mujoco = [0,1,4,7,10,2,5,8,11,3,6,9,12,15,13,16,18,20,22,14,17,19,21,23]
mujoco_2_smpl = [0,1,5,9,2,6,10,3,7,11,4,8,12,14,19,13,15,20,16,21,17,22,18,23]

# HumanML3D uses 22 joints (SMPL minus 2 hands)
# Mujoco/Isaac uses 24 joints (full SMPL)
# Hand positions derived by extending wrist direction by 0.08824
```

## 4. Verification Scripts

### 4.1 `scripts/verify_diffusion.py`

**Purpose**: Validate diffusion output independently of the simulator.

**What it does**:
1. Loads DiP model from CLoSD checkpoint
2. Generates motion from text prompt (open-loop, standing prefix)
3. Saves `motion.pt` (263-dim), `xyz.pt` (22-joint positions)
4. Renders matplotlib 3D skeleton MP4 (reuses CLoSD_t2m_standalone's `plot_3d_motion`)
5. Prints generation stats (frames, fps, guidance)

**CLI**: `python scripts/verify_diffusion.py --prompt "a person walks forward" --num-seconds 8 --output-dir outputs/verify_diffusion`

**Expected output**: MP4 showing a skeleton performing the described motion. No simulator needed.

### 4.2 `scripts/verify_rotations.py`

**Purpose**: Validate that the 6D rotation extraction and analytical IK produce correct rotations.

**What it does**:
1. **Ground truth test**: Loads a ProtoMotions `.motion` file (has pre-computed positions AND rotations). Extracts positions, runs both rotation solvers, compares against ground-truth rotations. Reports per-joint angular error.
2. **6D conversion unit test**: Takes known rotation matrices, converts to 6D, converts back, verifies round-trip identity.
3. **Diffusion output test**: Generates a diffusion motion, runs both solvers, reports FK consistency error for each. Compares the two solvers' outputs against each other.
4. Prints clear PASS/FAIL per test with error thresholds.

**CLI**: `python scripts/verify_rotations.py --motion-file /path/to/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion`

**Expected output**:
```
[PASS] 6D round-trip: max error 1.2e-7 rad
[PASS] Ground truth (diffusion mode): mean angular error 0.02 rad, FK pos error 0.8cm
[PASS] Ground truth (analytical mode): mean angular error 0.01 rad, FK pos error 0.3cm
[PASS] Diffusion output FK consistency (diffusion mode): 2.1cm
[PASS] Diffusion output FK consistency (analytical mode): 0.4cm
```

### 4.3 `scripts/verify_tracking.py`

**Purpose**: Validate ProtoMotions tracker + Isaac Lab work correctly with an offline motion.

**What it does**:
1. Loads ProtoMotions pretrained tracker checkpoint
2. Loads an offline `.motion` file via standard `MotionLib`
3. Runs `MimicMotionManager` + tracker policy in Isaac Lab
4. Records video (auto-record N frames or manual L-key toggle)

**CLI**: `python scripts/verify_tracking.py --checkpoint /path/to/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt --motion-file /path/to/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion --record-frames 300 --num-envs 1`

**Expected output**: MP4 of the SMPL humanoid tracking the sit-in-armchair motion in Isaac Lab.

### 4.4 `scripts/run_closd_isaaclab.py`

**Purpose**: Full closed-loop text-to-motion pipeline.

**What it does**:
1. Initializes DiP diffusion model
2. Initializes ProtoMotions tracker + Isaac Lab
3. Runs closed-loop: diffusion generates horizon → tracker follows → sim state feeds back
4. Records video

**CLI**: `python scripts/run_closd_isaaclab.py --prompt "a person walks forward then turns left" --rotation-mode diffusion --episode-length 300 --guidance 5.0 --record-frames 300`

**Flags**:
- `--prompt TEXT`: text condition
- `--rotation-mode {diffusion,analytical}`: how to derive rotations (default: diffusion)
- `--episode-length N`: number of sim frames (default: 300)
- `--guidance FLOAT`: classifier-free guidance scale (default: 5.0)
- `--record-frames N`: auto-stop recording after N frames
- `--num-envs N`: parallel environments (default: 1)
- `--headless`: run without viewer

## 5. Tests

### `tests/test_hml_conversion.py`
- Round-trip: take known positions → `pose_to_hml()` → `hml_to_pose()` → compare
- Delta accumulation: verify `recover_root_rot_pos()` cumsum matches known trajectory
- FPS conversion: verify 30→20→30 preserves motion within tolerance

### `tests/test_rotation_solver.py`
- 6D→matrix→6D round-trip identity
- Known rotations → positions via FK → rotation_solver → compare
- Both modes (diffusion, analytical) tested against ground truth
- FK consistency check for both modes

### `tests/test_coord_transform.py`
- Isaac→SMPL→Isaac round-trip identity
- Joint reorder smpl_2_mujoco / mujoco_2_smpl inverse check
- Specific known transforms verified numerically

### `tests/test_robot_state_builder.py`
- Velocity derivation: known linear motion → verify computed velocity
- Quaternion interpolation: verify SLERP at midpoint
- RobotState fields: verify all required fields present with correct shapes

## 6. Environment Setup & Reproduction

### Prerequisites

- GCP VM with GPU (tested on A100/V100)
- TurboVNC installed and running (display :1)
- Python 3.10+

### Step-by-step Setup

```bash
# 1. Clone/verify repos are present
ls ~/code/CLoSD ~/code/ProtoMotions ~/code/CLoSD_t2m_standalone

# 2. Download ProtoMotions pretrained checkpoint (if not done)
cd ~/code/ProtoMotions
sudo apt-get install -y git-lfs && git lfs install
git lfs pull --include="data/pretrained_models/motion_tracker/smpl/*"
# Or via curl:
curl -L -o data/pretrained_models/motion_tracker/smpl/last.ckpt \
  "https://github.com/NVlabs/ProtoMotions/raw/main/data/pretrained_models/motion_tracker/smpl/last.ckpt"

# 3. Download CLoSD dependencies (DiP checkpoint + HumanML3D data)
cd ~/code/CLoSD
python -c "from closd.utils.hf_handler import get_dependencies; get_dependencies()"

# 4. Set up the environment
# Option A: Use existing env_isaaclab
source ~/code/env_isaaclab/bin/activate
# Option B: Create new env (if needed)
# uv venv env_closd_isaaclab --python 3.10
# source env_closd_isaaclab/bin/activate

# 5. Install dependencies
cd ~/code/closd_isaaclab
pip install -e .
# Ensure CLoSD and ProtoMotions are importable
export PYTHONPATH="$HOME/code/CLoSD:$HOME/code/ProtoMotions:$HOME/code/CLoSD_t2m_standalone:$PYTHONPATH"

# 6. Set display for TurboVNC (GCP VM)
export DISPLAY=:1
```

### Reproduction Steps

```bash
# Step 1: Verify diffusion works (no simulator needed)
python scripts/verify_diffusion.py \
  --prompt "a person walks forward" \
  --num-seconds 8 \
  --output-dir outputs/verify_diffusion
# Expected: outputs/verify_diffusion/a_person_walks_forward.mp4

# Step 2: Verify rotation conversions are correct
python scripts/verify_rotations.py \
  --motion-file ~/code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion
# Expected: All tests PASS with error thresholds printed

# Step 3: Verify ProtoMotions tracker works in Isaac Lab
# (requires DISPLAY=:1 for rendering, or --headless)
python scripts/verify_tracking.py \
  --checkpoint ~/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
  --motion-file ~/code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion \
  --record-frames 300
# Expected: MP4 of humanoid tracking the sit motion

# Step 4: Run full CLoSD-IsaacLab pipeline
python scripts/run_closd_isaaclab.py \
  --prompt "a person walks forward then turns left" \
  --rotation-mode diffusion \
  --episode-length 300 \
  --record-frames 300
# Expected: MP4 of humanoid performing text-described motion in Isaac Lab

# Optional: Compare rotation modes
python scripts/run_closd_isaaclab.py \
  --prompt "a person walks forward" \
  --rotation-mode analytical \
  --episode-length 300 \
  --record-frames 300
```

### TurboVNC Notes (GCP VM)

```bash
# Start TurboVNC server if not running
/opt/TurboVNC/bin/vncserver :1

# For Isaac Lab rendering with VirtualGL
export DISPLAY=:1
# If vglrun is available:
vglrun -d :0 python scripts/run_closd_isaaclab.py ...
# Isaac Lab handles its own rendering; DISPLAY must be set

# For headless recording (no VNC needed):
python scripts/run_closd_isaaclab.py --headless --record-frames 300 ...
```

## 7. Dependencies

### Python packages (beyond existing envs)

```
# From CLoSD
torch, clip, transformers (for MDM text encoding)
scipy (for FPS interpolation)

# From ProtoMotions
isaaclab, isaacsim (for simulation)
lightning_fabric, easydict, omegaconf

# For visualization (verify_diffusion.py)
matplotlib, moviepy

# Shared
numpy
```

### External assets

| Asset | Source | Path |
|-------|--------|------|
| DiP checkpoint | HuggingFace `guytevet/CLoSD` (auto-downloaded) | `CLoSD/closd/diffusion_planner/save/DiP_no-target_.../model000200000.pt` |
| HML mean/std | HuggingFace `guytevet/CLoSD` or `CLoSD_t2m_standalone/standalone_t2m/assets/` | `t2m_mean.npy`, `t2m_std.npy` |
| Motion tracker | ProtoMotions Git LFS | `ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt` |
| SMPL humanoid USD | ProtoMotions repo | `ProtoMotions/protomotions/data/assets/usd/smpl_humanoid.usda` |
| SMPL humanoid MJCF | ProtoMotions repo | `ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml` |
| Example motion | ProtoMotions repo | `ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion` |

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Diffusion 6D rotations inconsistent with positions | Tracker gets conflicting reference → poor tracking | `--rotation-mode analytical` fallback; FK consistency warning at runtime |
| ProtoMotions tracker trained on AMASS motions, not diffusion output | Distribution mismatch → jittery tracking | Diffusion output is close to AMASS distribution (same training data). If poor, increase guidance or reduce planning horizon |
| FPS conversion artifacts (20↔30) | Jerky motion at boundaries | Bicubic interpolation (same as CLoSD), verified in tests |
| Isaac Lab joint ordering differs from Isaac Gym | Actions/observations sent to wrong joints | ProtoMotions' `DataConversionMapping` handles this automatically |
| Quaternion convention mismatch | Silent wrong orientations | All conversions explicitly documented with convention table; ProtoMotions auto-converts between wxyz (sim) and xyzw (common) |
| Diffusion produces NaN or degenerate output | Tracker falls, episode ruined | Check for NaN after each diffusion call; if detected, repeat last valid horizon and log warning. If FK error > 10cm, also fall back to last valid horizon |
