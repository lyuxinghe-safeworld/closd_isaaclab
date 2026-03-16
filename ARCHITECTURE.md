# CLoSD-IsaacLab: System Architecture

This document describes the internal architecture of the CLoSD-IsaacLab pipeline in detail, including component connections, tensor shapes, coordinate conventions, and the closed-loop replanning cycle.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Flowchart](#component-flowchart)
3. [Open-Loop Pipeline (Text to Physics State)](#open-loop-pipeline)
4. [Closed-Loop Replanning Cycle](#closed-loop-replanning-cycle)
5. [Tensor Reference](#tensor-reference)
6. [Module-by-Module Details](#module-by-module-details)
7. [Coordinate Transforms](#coordinate-transforms)
8. [Rotation Representations](#rotation-representations)
9. [Key Implementation Details](#key-implementation-details)

---

## System Overview

CLoSD-IsaacLab connects three external systems through a conversion pipeline:

```
 ┌──────────────┐     ┌────────────────────┐     ┌──────────────────┐
 │  CLoSD DiP   │────>│  closd_isaaclab    │────>│  ProtoMotions    │
 │  (diffusion) │     │  (this package)    │     │  SMPL Tracker    │
 └──────────────┘     └────────────────────┘     │  (Isaac Lab sim) │
                                                  └──────────────────┘
```

Given a text prompt like `"a person walks forward"`, the system:

1. **Generates** a 263-dim HumanML3D motion sequence via autoregressive diffusion (CLoSD DiP)
2. **Converts** the HumanML3D features into 24-joint positions in Isaac Lab's Z-up coordinate frame
3. **Solves** joint rotations and DOF positions for the MuJoCo SMPL skeleton
4. **Serves** the motion as a `RobotState` to ProtoMotions' physics-based tracker
5. **Replans** every 2 seconds by feeding simulator observations back into the diffusion model

---

## Component Flowchart

### Full System Architecture

```
                         "a person walks forward"
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  DiffusionMotionProvider  │
                    │  (motion_provider.py)     │
                    │                          │
                    │  • Builds standing prefix │
                    │  • Runs DiP diffusion     │
                    │  • Unnormalizes features   │
                    │  • Removes velocity bias   │
                    └────────┬─────────────────┘
                             │
               ┌─────────────┴─────────────┐
               │                           │
               ▼                           ▼
    hml_raw [bs, T, 263]      positions [bs, T, 22, 3]
    (unnormalized HumanML3D)  (SMPL Y-up, 20 fps)
               │                           │
               │                           ▼
               │              ┌────────────────────────┐
               │              │    HMLConversion        │
               │              │    (hml_conversion.py)  │
               │              │                        │
               │              │  • Align to sim frame   │
               │              │  • Add hands (22→24)    │
               │              │  • SMPL→Isaac coords    │
               │              │  • 20fps → 30fps        │
               │              └───────────┬────────────┘
               │                          │
               │                          ▼
               │           positions_isaac [bs, T_30, 24, 3]
               │           (Isaac Z-up, MuJoCo order, 30 fps)
               │                          │
               ▼                          ▼
    ┌──────────────────┐    ┌──────────────────────────┐
    │ RotationSolver   │    │   RobotStateBuilder       │
    │ (rotation_       │───>│   (robot_state_builder.py)│
    │  solver.py)      │    │                          │
    │                  │    │  • Finite-diff velocities │
    │ • Extract 6D     │    │  • Analytical IK → DOFs  │
    │   from HML       │    │  • Angular velocities     │
    │ • 6D→rot matrix  │    │  • Foot contacts          │
    │ • Root inversion │    └───────────┬──────────────┘
    └──────────────────┘                │
                                        ▼
                           ┌────────────────────────┐
                           │  Cached Physics State    │
                           │                        │
                           │  positions  [bs,T,24,3] │
                           │  velocities [bs,T,24,3] │
                           │  rotations  [bs,T,24,4] │
                           │  ang_vel    [bs,T,24,3] │
                           │  dof_pos    [bs,T,69]   │
                           │  dof_vel    [bs,T,69]   │
                           │  contacts   [bs,T,24]   │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │   CLoSDMotionLib        │
                           │   (closd_motion_lib.py) │
                           │                        │
                           │  Duck-types MotionLib   │
                           │  get_motion_state(t)    │
                           │  → RobotState           │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │  ProtoMotions Tracker   │
                           │  (MimicControl)         │
                           │                        │
                           │  PD control → torques   │
                           │  Isaac Lab physics step │
                           └───────────┬────────────┘
                                       │
                                       ▼
                              Simulated humanoid
                              moves in Isaac Lab
```

### Closed-Loop Feedback Path

```
     ProtoMotions Tracker
              │
              │ body positions [num_envs, 24, 3] (every physics step)
              ▼
   ┌──────────────────────────┐
   │  CLoSDMotionManager       │
   │  (closd_motion_manager.py)│
   │                          │
   │  pose_buffer              │◄── sliding window [bs, 30, 24, 3]
   │  [bs, context_len, 24, 3]│        (last ~1 sec at 30fps)
   │                          │
   │  Every 60 steps (~2s):   │
   │    call _replan()        │
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │  HMLConversion.pose_to_  │
   │  hml() [INVERSE path]    │
   │                          │
   │  Isaac→SMPL coords       │
   │  30fps→20fps             │
   │  extract HML features    │
   │  normalize               │
   └────────────┬─────────────┘
                │
                ▼
     hml_norm [bs, T_20, 263]
     (prefix for next diffusion)
                │
                ▼
   ┌──────────────────────────┐
   │  DiffusionMotionProvider  │
   │  .generate_next_horizon() │
   │                          │
   │  Diffusion conditioned    │
   │  on simulator prefix      │
   └────────────┬─────────────┘
                │
                ▼
       New horizon of motion
       (positions + HML features)
                │
                ▼
       RobotStateBuilder.build()
       → fresh cached state
                │
                ▼
       Reset motion_times to 0
       ───── loop continues ─────
```

---

## Open-Loop Pipeline

Step-by-step tensor transformations from text input to physics state:

```
Step 1: TEXT → DIFFUSION
━━━━━━━━━━━━━━━━━━━━━━━━

  Input:  text_prompt = "a person walks forward"
          num_seconds = 8

  DiffusionMotionProvider.generate_standalone()
    ├── Build standing prefix: [1, 20, 263] (normalized HML, all zeros = dataset mean)
    ├── Run DiP autoregressive sampling (20 fps, guidance=5.0)
    ├── Raw output: [1, T_20fps, 263] normalized HML
    ├── Unnormalize: raw = normalized * std + mean
    ├── Subtract prefix velocity bias from dims 1:3
    └── recover_from_ric() → joint positions

  Output: hml_raw    = [1, T_20fps, 263]   float32   unnormalized HumanML3D
          positions  = [1, T_20fps, 22, 3]  float32   SMPL joint XYZ (Y-up)


Step 2: SMPL POSITIONS → ISAAC POSITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Input: positions [1, T_20fps, 22, 3]  SMPL Y-up, 22 joints, 20 fps

  HMLConversion.hml_to_pose()
    ├── recover_root_rot_pos(hml_raw) → root rotation [1, T, 4] wxyz + root XZ [1, T, 3]
    ├── Alignment via recon_data:
    │     1. Zero HML root: subtract XZ, rotate by HML facing direction
    │     2. Apply sim root: inverse-rotate by sim facing, add sim XZ
    ├── Add hand joints:
    │     direction = normalize(wrist - elbow)
    │     hand_pos = wrist + direction * 0.08824m
    │     [1, T, 22, 3] → [1, T, 24, 3]
    ├── SMPL→Isaac coordinate transform:
    │     Apply rotation Rx(-π/2) @ Ry(-π) @ Rx(-π) @ Rz(π/2) in float64
    │     Reorder joints via smpl_2_mujoco index array
    └── FPS convert: bicubic interpolation 20fps → 30fps

  Output: positions_isaac = [1, T_30fps, 24, 3]  float32  Isaac Z-up, MuJoCo joint order


Step 3: POSITIONS → ROTATIONS + DOFs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Input: hml_raw [1, T_20fps, 263]

  RotationSolver.solve()
    ├── Extract 21-joint 6D rotations: hml_raw[..., 67:193] → [1, T, 21, 6]
    ├── cont6d_to_matrix(): Gram-Schmidt orthogonalization → [1, T, 21, 3, 3]
    ├── Recover root rotation:
    │     recover_root_rot_pos(hml_raw) → wxyz quat [1, T, 4]
    │     wxyz_quat_to_matrix() → [1, T, 3, 3]
    │     TRANSPOSE (invert: HML stores global→local, kinematic chain needs local→global)
    ├── Assemble 24-joint rotations:
    │     Joint 0:     root rotation matrix
    │     Joints 1-21: 6D-derived matrices
    │     Joints 22-23: identity (hands, no articulation)
    └── If kinematic_info: extract_qpos_from_transforms() → dof_pos [1, T, 69]

  Output: local_rot_mats_24 = [1, T_20fps, 24, 3, 3]  local rotation matrices
          dof_pos            = [1, T_20fps, 69]          MuJoCo joint angles


Step 4: BUILD PHYSICS STATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Input: positions_isaac [1, T_30fps, 24, 3]
         hml_raw         [1, T_20fps, 263]

  RobotStateBuilder.build()
    ├── Store positions: [1, T, 24, 3]
    ├── Velocities via finite differences:
    │     v[t] = (pos[t+1] - pos[t-1]) / (2 * dt)   (central diff, interior)
    │     v[0] = (pos[1] - pos[0]) / dt              (forward diff, boundary)
    │     → [1, T, 24, 3]
    ├── DOF positions: analytical IK on kinematic tree → [1, T, 69]
    ├── DOF velocities: finite differences on dof_pos → [1, T, 69]
    ├── Global rotations → xyzw quaternions: [1, T, 24, 4]
    ├── Angular velocities: [1, T, 24, 3]
    └── Foot contacts from HML dims 259-262:
          Resample to 30fps, map to body indices {3,4,7,8}
          → [1, T, 24] (sparse, mostly zeros)

  Output (cached):
    _positions:      [1, T_30fps, 24, 3]   joint positions (meters)
    _velocities:     [1, T_30fps, 24, 3]   linear velocities (m/s)
    _rotations:      [1, T_30fps, 24, 4]   xyzw quaternions
    _ang_velocities: [1, T_30fps, 24, 3]   angular velocities (rad/s)
    _dof_pos:        [1, T_30fps, 69]      joint angles (radians)
    _dof_vel:        [1, T_30fps, 69]      joint angular velocities (rad/s)
    _contacts:       [1, T_30fps, 24]      binary contact flags


Step 5: SERVE TO TRACKER
━━━━━━━━━━━━━━━━━━━━━━━━

  Input: motion_times [N] tensor of seconds (from ProtoMotions tracker query)

  CLoSDMotionLib.get_motion_state()
    ├── Convert time → frame: frame_idx = clamp(time / dt, 0, max_frame)
    ├── Query RobotStateBuilder.get_state_at_frames(frame_indices)
    └── Wrap in RobotState (ProtoMotions struct)

  Output: RobotState
    rigid_body_pos:      [N, 24, 3]
    rigid_body_rot:      [N, 24, 4]   (xyzw)
    rigid_body_vel:      [N, 24, 3]
    rigid_body_ang_vel:  [N, 24, 3]
    dof_pos:             [N, 69]
    dof_vel:             [N, 69]
    rigid_body_contacts: [N, 24]
```

---

## Closed-Loop Replanning Cycle

The closed-loop mode replans every **60 simulation steps** (~2 seconds at 30 fps).

### Timing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `context_len` | 20 frames @ 20fps (1.0s) | Diffusion prefix length |
| `pred_len` | 40 frames @ 20fps (2.0s) | Diffusion prediction horizon |
| `context_len_30fps` | 30 frames @ 30fps (1.0s) | Simulator pose buffer |
| `planning_horizon_30fps` | 60 frames @ 30fps (2.0s) | Steps between replans |
| `dt` | 1/30 s | Simulation timestep |

### Cycle Diagram

```
Time ──────────────────────────────────────────────────────────────►

Horizon 1                    Horizon 2                    Horizon 3
├────── 60 frames ──────────┤├────── 60 frames ──────────┤├───────

  ┌─ Diffusion generates     ┌─ pose_buffer (last 30     ┌─ ...
  │  motion [T=60 frames]    │  frames) → pose_to_hml()  │
  │                          │  → new diffusion prefix    │
  │  Tracker follows         │  → diffusion generates     │
  │  reference motion        │    new horizon             │
  │                          │                            │
  │  pose_buffer collects    │  Tracker follows new       │
  │  sim positions           │  reference motion          │
  │                          │                            │
  └─ frame_counter hits 60 ──┘  frame_counter hits 60 ───┘

  _replan() triggered        _replan() triggered
```

### Replan Data Flow

```
Simulator body positions [num_envs, 24, 3]
            │
            │  (collected for 60 steps into sliding window)
            ▼
pose_buffer [bs, 30, 24, 3]        (last 1 sec of sim, Isaac Z-up, 30fps)
            │
            │  HMLConversion.pose_to_hml()
            │    ├── Add dummy extrapolated frame (feature extraction loses last)
            │    ├── fps_convert(30→20)
            │    ├── isaac_to_smpl(): inverse coord transform, reorder, drop hands
            │    ├── extract_features_t2m(): SMPL positions → 263-dim HML
            │    └── normalize with mean/std
            ▼
hml_norm [bs, 20, 263]              (normalized HML, prefix for diffusion)
recon_data {r_rot: [bs,4], r_pos: [bs,3]}   (root alignment metadata)
            │
            │  DiffusionMotionProvider.generate_next_horizon()
            │    ├── Use last context_len frames as prefix
            │    ├── Run conditioned diffusion sampling
            │    ├── Unnormalize output
            │    ├── Recover positions
            │    └── hml_to_pose() with recon_data alignment
            ▼
positions_isaac [bs, 60, 24, 3]     (next 2 sec, Isaac Z-up, 30fps)
hml_raw [bs, 40, 263]               (unnormalized, for rotation extraction)
            │
            │  RobotStateBuilder.build()
            ▼
Cached state refreshed
            │
            │  Reset motion_times = 0
            ▼
Tracker reads from fresh horizon
```

---

## Tensor Reference

### HumanML3D Feature Vector (`[bs, T, 263]`)

```
Dimension Layout:
┌─────────┬──────┬──────────────────────────────────────────────────┐
│  Dims   │ Size │ Content                                          │
├─────────┼──────┼──────────────────────────────────────────────────┤
│  0      │  1   │ Root angular velocity (magnitude, rad/s)         │
│  1-2    │  2   │ Root XZ velocity (forward, lateral) (m/s)        │
│  3      │  1   │ Root height above ground (m)                     │
│  4-66   │  63  │ Root rotation + RIC features                     │
│  67-192 │ 126  │ 21-joint 6D rotations (21 joints × 6 dims)      │
│ 193-258 │  66  │ Joint positions/velocities in root frame         │
│ 259-262 │   4  │ Foot contacts: L_Ankle, L_Toe, R_Ankle, R_Toe   │
└─────────┴──────┴──────────────────────────────────────────────────┘

Extraction patterns:
  6D rotations:  hml_raw[..., 67:193].reshape(bs, T, 21, 6)
  Foot contacts: hml_raw[..., 259:263]   → binary {0, 1}
  Root velocity: hml_raw[..., 1:3]       → (vx, vz) in m/s
```

### Joint Position Tensors

| Stage | Shape | Joints | Coord Frame | FPS | Joint Order |
|-------|-------|--------|-------------|-----|-------------|
| Diffusion output | `[bs, T, 22, 3]` | 22 SMPL | Y-up | 20 | SMPL standard |
| After hand extension | `[bs, T, 24, 3]` | 22 + 2 hands | Y-up | 20 | SMPL + hands |
| Isaac Lab positions | `[bs, T, 24, 3]` | 24 | Z-up | 30 | MuJoCo reordered |

### Physics State Tensors (Cached in RobotStateBuilder)

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `_positions` | `[bs, T, 24, 3]` | meters | Joint positions in world frame |
| `_velocities` | `[bs, T, 24, 3]` | m/s | Linear velocities (finite diff) |
| `_rotations` | `[bs, T, 24, 4]` | unitless | Global rotations as **xyzw** quaternions |
| `_ang_velocities` | `[bs, T, 24, 3]` | rad/s | Angular velocities |
| `_dof_pos` | `[bs, T, 69]` | radians | 23 joints × 3 DOF each |
| `_dof_vel` | `[bs, T, 69]` | rad/s | Joint angular velocities |
| `_contacts` | `[bs, T, 24]` | binary | 1.0 at bodies {3,4,7,8} when in contact |

### RobotState (ProtoMotions Interface)

Served by `CLoSDMotionLib.get_motion_state()` — batch dimension collapsed:

| Field | Shape | Notes |
|-------|-------|-------|
| `rigid_body_pos` | `[N, 24, 3]` | N = number of queried time steps |
| `rigid_body_rot` | `[N, 24, 4]` | xyzw quaternion (w last) |
| `rigid_body_vel` | `[N, 24, 3]` | |
| `rigid_body_ang_vel` | `[N, 24, 3]` | |
| `dof_pos` | `[N, 69]` | |
| `dof_vel` | `[N, 69]` | |
| `rigid_body_contacts` | `[N, 24]` | |

---

## Module-by-Module Details

### `diffusion/motion_provider.py` — DiffusionMotionProvider

Wraps the CLoSD DiP diffusion model for text-conditioned motion generation.

**Key methods:**

| Method | Input | Output | When Used |
|--------|-------|--------|-----------|
| `generate_standalone()` | text prompt, num_seconds | `hml_raw [1,T,263]`, `positions [1,T,22,3]` | Open-loop (initial generation) |
| `generate_next_horizon()` | `pose_buffer [bs,T,24,3]`, recon_data, text | `positions_isaac [bs,T,24,3]`, `hml_raw`, recon_data | Closed-loop (replanning) |

**Hyperparameters:**
- `guidance = 5.0` (classifier-free guidance scale)
- `context_len = 20` frames at 20fps
- `pred_len = 40` frames at 20fps

---

### `diffusion/hml_conversion.py` — HMLConversion

Bidirectional conversion between HumanML3D (20fps, Y-up, 22 joints) and Isaac Lab (30fps, Z-up, 24 joints).

**Forward path** (`hml_to_pose`):

```
hml_norm [bs, T, 263]          ──unnormalize──►  hml_raw [bs, T, 263]
                                                       │
                                              recover_from_ric()
                                                       │
                                                       ▼
                                              positions [bs, T, 22, 3]  SMPL Y-up
                                                       │
                                              align via recon_data
                                                       │
                                              add hands (22→24)
                                                       │
                                              smpl_to_isaac()
                                                       │
                                              fps_convert(20→30)
                                                       │
                                                       ▼
                                              positions_isaac [bs, T_30, 24, 3]
```

**Inverse path** (`pose_to_hml`):

```
positions_isaac [bs, T_30, 24, 3]  ──add extrapolated frame──►  [bs, T_30+1, 24, 3]
                                                                        │
                                                               fps_convert(30→20)
                                                                        │
                                                               isaac_to_smpl() (24→22)
                                                                        │
                                                               extract_features_t2m()
                                                                        │
                                                               normalize
                                                                        │
                                                                        ▼
                                                               hml_norm [bs, T_20, 263]
                                                               recon_data {r_rot, r_pos}
```

**`recon_data`** is the alignment bridge between coordinate frames:
- `r_rot`: `[bs, 4]` — wxyz quaternion of root facing direction
- `r_pos`: `[bs, 3]` — root XZ position in world frame
- Passed from `pose_to_hml()` to `hml_to_pose()` so the next horizon aligns smoothly with the simulator's current state.

---

### `diffusion/rotation_solver.py` — RotationSolver

Extracts joint rotations from HumanML3D's embedded 6D representation.

**Pipeline:**

```
hml_raw[..., 67:193]  → reshape  → [bs, T, 21, 6]   (6D continuous rotation)
                                          │
                                   cont6d_to_matrix()
                                   (Gram-Schmidt ortho)
                                          │
                                          ▼
                                   [bs, T, 21, 3, 3]   (rotation matrices)

recover_root_rot_pos(hml_raw)  →  root quat [bs, T, 4] wxyz
                                          │
                                   wxyz_quat_to_matrix()
                                          │
                                   .transpose(-2,-1)   (INVERT: global→local to local→global)
                                          │
                                          ▼
                                   root_mat [bs, T, 3, 3]

Assemble:
  Joint  0:     root_mat                    [bs, T, 3, 3]
  Joints 1-21:  6D-derived matrices         [bs, T, 21, 3, 3]
  Joints 22-23: identity (hands)            [bs, T, 2, 3, 3]
  ────────────────────────────────────────────────────────────
  Result:       local_rot_mats_24           [bs, T, 24, 3, 3]
```

**6D → Matrix conversion** (`cont6d_to_matrix`):
```
Input:  [a1, a2, a3, b1, b2, b3]

  e1 = normalize([a1, a2, a3])
  e2 = normalize([b1, b2, b3] - dot([b1,b2,b3], e1) * e1)    # Gram-Schmidt
  e3 = cross(e1, e2)                                           # ensures det=+1

Output: [e1 | e2 | e3]   (3×3 rotation matrix)
```

---

### `diffusion/robot_state_builder.py` — RobotStateBuilder

Builds the complete physics state from positions and HML features.

**Velocity computation** (finite differences at dt = 1/30s):
```
Interior frames:  v[t] = (pos[t+1] - pos[t-1]) / (2 * dt)     (central)
First frame:      v[0] = (pos[1] - pos[0]) / dt                (forward)
Last frame:       v[-1] = (pos[-1] - pos[-2]) / dt             (backward)
```

**Foot contact mapping:**
```
HML dim 259 → MuJoCo body 3  (L_Ankle)
HML dim 260 → MuJoCo body 4  (L_Toe)
HML dim 261 → MuJoCo body 7  (R_Ankle)
HML dim 262 → MuJoCo body 8  (R_Toe)

All other bodies → 0 (no contact)
```

---

### `integration/closd_motion_lib.py` — CLoSDMotionLib

Duck-types ProtoMotions' `MotionLib` interface so the tracker can query diffusion-generated motion as if it were a pre-recorded `.motion` file.

**Interface mapping:**

| MotionLib method | CLoSDMotionLib behavior |
|-----------------|------------------------|
| `num_motions()` | Always returns 1 |
| `motion_lengths` | `[1]` tensor (current horizon duration) |
| `get_motion_state(ids, times)` | Converts times → frames, queries RobotStateBuilder |

**Fallback:** If RobotStateBuilder is empty (before first generation), returns zero-filled RobotState with identity quaternions.

---

### `integration/closd_motion_manager.py` — CLoSDMotionManager

Extends `MimicMotionManager` to add closed-loop replanning.

**State machine:**

```
┌─────────────┐  sample_motions()  ┌──────────────┐
│  RESET      │ ──────────────────►│  INITIAL     │
│  (env reset)│                    │  PLANNING    │
└─────────────┘                    │              │
                                   │ Fill buffer  │
                                   │ Clear recon  │
                                   │ _replan()    │
                                   └──────┬───────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │  TRACKING    │◄──────────┐
                                   │              │           │
                                   │ post_physics │           │
                                   │ _step()      │           │
                                   │              │           │
                                   │ Update buffer│           │
                                   │ Increment ctr│           │
                                   └──────┬───────┘           │
                                          │                   │
                                   counter % 60 == 0?         │
                                          │                   │
                                     YES  ▼              NO   │
                               ┌──────────────┐               │
                               │  REPLAN      │               │
                               │              │               │
                               │ _replan()    │───────────────┘
                               │ Reset times  │
                               └──────────────┘
```

---

### `utils/coord_transform.py` — CoordTransform

Converts between SMPL (Y-up, facing -Z) and Isaac Lab / MuJoCo (Z-up, facing +X).

**Transform matrix** (SMPL → Isaac):
```
R = Rx(-π/2) @ Ry(-π) @ Rx(-π) @ Rz(+π/2)

Where:
  Rz(+π/2)  : Align SMPL X-axis with MuJoCo Y-axis
  Rx(-π)    : Flip
  Ry(-π)    : Reverse facing direction
  Rx(-π/2)  : Rotate Y-up to Z-up
```

**Joint reordering** (SMPL index → MuJoCo index):
```
smpl_2_mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]

Example: SMPL joint 4 (L_Knee) → MuJoCo position 2
         SMPL joint 1 (L_Hip)  → MuJoCo position 1
```

**Hand extension** (22 → 24 joints):
```
hand_direction = normalize(wrist_pos - elbow_pos)
hand_pos = wrist_pos + hand_direction * 0.08824   (meters)

Left hand:  elbow=joint 18, wrist=joint 20 → new joint 22
Right hand: elbow=joint 19, wrist=joint 21 → new joint 23
```

---

### `utils/fps_convert.py` — fps_convert

Resamples motion tensors between frame rates using PyTorch interpolation.

```
fps_convert(data, src_fps=20, tgt_fps=30, mode="bicubic")

  Input:  [bs, T_src, ...]   (any trailing dimensions)
  Output: [bs, T_tgt, ...]

  T_tgt = int(T_src * tgt_fps / src_fps)

  Process:
    1. Flatten trailing dims → [bs, D, T_src]
    2. F.interpolate(mode="bicubic", align_corners=False)
    3. Reshape back → [bs, T_tgt, ...]
```

---

## Coordinate Transforms

### Coordinate Frame Comparison

```
     SMPL (HumanML3D)              Isaac Lab (MuJoCo)

          Y (up)                        Z (up)
          │                             │
          │                             │
          │                             │
          └──── X (right)               └──── X (forward)
         /                             /
        Z (backward)                  Y (left)
```

### Full Transform Chain (Forward)

```
SMPL Y-up positions [bs, T, 22, 3]
        │
        │  1. Extend hands: 22 → 24 joints
        ▼
SMPL Y-up positions [bs, T, 24, 3]
        │
        │  2. Apply rotation matrix R (computed in float64):
        │     pos_isaac = pos_smpl @ R.T
        ▼
Rotated positions [bs, T, 24, 3]
        │
        │  3. Reorder joints: SMPL order → MuJoCo order
        │     pos_reordered = pos_rotated[:, :, smpl_2_mujoco, :]
        ▼
Isaac Z-up positions [bs, T, 24, 3]    (MuJoCo joint order)
```

### Full Transform Chain (Inverse)

```
Isaac Z-up positions [bs, T, 24, 3]    (MuJoCo joint order)
        │
        │  1. Reorder joints: MuJoCo order → SMPL order
        │     pos_smpl_order = pos_isaac[:, :, mujoco_2_smpl, :]
        ▼
Rotated positions [bs, T, 24, 3]       (SMPL joint order)
        │
        │  2. Apply inverse rotation R⁻¹ = R.T (computed in float64)
        ▼
SMPL Y-up positions [bs, T, 24, 3]
        │
        │  3. Optionally drop hands: joints 22,23
        ▼
SMPL Y-up positions [bs, T, 22, 3]     (original SMPL order)
```

---

## Rotation Representations

Four rotation formats are used across the pipeline. Here is where each appears and how they convert:

```
┌─────────────────────┐
│  6D Continuous       │  Where: HumanML3D dims 67-192 (21 joints × 6)
│  [..., 6]            │
│                     │  Advantage: Smooth, no gimbal lock, no discontinuities
└────────┬────────────┘
         │ cont6d_to_matrix()  (Gram-Schmidt)
         ▼
┌─────────────────────┐
│  3×3 Rotation Matrix │  Where: RotationSolver output, intermediate computations
│  [..., 3, 3]         │
│                     │  Advantage: Composable via matrix multiply
└────────┬────────────┘
         │ matrix_to_quaternion()
         ▼
┌─────────────────────┐
│  xyzw Quaternion     │  Where: RobotState (ProtoMotions convention)
│  [..., 4]            │
│  (x, y, z, w)       │  Advantage: Compact, interpolation-friendly
└─────────────────────┘

┌─────────────────────┐
│  wxyz Quaternion     │  Where: HumanML3D root rotation, recon_data
│  [..., 4]            │
│  (w, x, y, z)       │  Note: CLoSD/HumanML3D convention (w FIRST)
└────────┬────────────┘
         │ wxyz_quat_to_matrix()
         ▼
      3×3 Matrix

IMPORTANT: xyzw ≠ wxyz. ProtoMotions uses xyzw. HumanML3D uses wxyz.
The conversion is: xyzw[..., [3,0,1,2]] → wxyz  (or vice versa).
```

---

## Key Implementation Details

### Prefix Velocity Bias Removal

The standing prefix (normalized HML = all zeros = dataset mean motion) carries an inherent forward velocity of ~0.17 m/s. Without removal, generated motion drifts forward even for stationary prompts.

```python
# In DiffusionMotionProvider.generate_standalone():
prefix_vel_bias = hml_raw[:, :context_len, 1:3].mean(dim=1, keepdim=True)
hml_raw[:, :, 1:3] -= prefix_vel_bias
```

### Root Rotation Inversion

HumanML3D stores root rotation as **global-to-local** (the rotation that brings global frame to the character's facing frame). Kinematic chains require **local-to-global**. The code transposes (inverts) the root rotation matrix:

```python
root_mat = wxyz_quat_to_matrix(r_rot_quat)    # global → local
root_mat = root_mat.transpose(-2, -1)          # local → global (invert orthogonal matrix)
```

### Height Floor Fix

Diffusion output may place feet below ground. A post-processing step ensures minimum clearance:

```python
min_z = pos_isaac[:, :, :, 2].min()
height_shift = max(0.015 - min_z.item(), 0.0)   # ensure ≥ 1.5cm above ground
pos_isaac[:, :, :, 2] += height_shift
```

### Extrapolated Dummy Frame

`extract_features_t2m()` computes velocities via forward differences and loses the last frame. Before calling it in the inverse path, a dummy frame is appended by linear extrapolation:

```python
# In HMLConversion.pose_to_hml():
last_vel = positions[:, -1:] - positions[:, -2:-1]
dummy = positions[:, -1:] + last_vel
positions_extended = torch.cat([positions, dummy], dim=1)
```

### NaN Guard in Replanning

Degenerate diffusion outputs can produce NaN positions. The replan step checks and skips:

```python
if torch.isnan(positions_isaac).any():
    log.warning("NaN in diffusion output — skipping replan")
    return  # keep previous horizon
```

---

## External Dependencies

| System | Repository | Role | Key Interface |
|--------|-----------|------|---------------|
| CLoSD | `~/code/CLoSD` | DiP diffusion model, HML utilities | `load_model_wo_clip()`, `recover_from_ric()` |
| CLoSD_t2m_standalone | `~/code/CLoSD_t2m_standalone` | HumanML3D feature extraction | `extract_features_t2m()`, `t2m_kinematic_chain` |
| ProtoMotions | `~/code/ProtoMotions` | SMPL tracker, `MotionLib`, `RobotState` | `MimicMotionManager`, `MimicControl`, `MotionLib` |
| Isaac Lab | (pip installed) | Physics simulation | GPU-accelerated rigid body simulation |

---

## Script Entry Points

| Script | Purpose | Requires Display | Key Output |
|--------|---------|-----------------|-----------|
| `verify_diffusion.py` | Test diffusion + conversion (no sim) | No | `motion.pt`, `skeleton.mp4` |
| `verify_rotations.py` | Test 6D rotation round-trips | No | PASS/FAIL checks |
| `verify_tracking.py` | Test tracker with offline `.motion` file | Yes | Isaac Lab visualization |
| `run_closd_isaaclab.py` | Full closed-loop pipeline | Yes | Live simulation |
| `diagnose_drift.py` | Debug horizontal drift | No | Position statistics |
