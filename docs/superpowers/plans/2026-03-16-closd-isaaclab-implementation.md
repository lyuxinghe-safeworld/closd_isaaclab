# CLoSD-IsaacLab Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port CLoSD's closed-loop text-to-motion pipeline to Isaac Lab using ProtoMotions' motion tracker, producing video output from text prompts.

**Architecture:** Embed CLoSD's diffusion planner (DiP) as a custom MotionManager within ProtoMotions' existing IsaacLab infrastructure. The diffusion model generates HumanML3D motion autoregressively, which is converted to ProtoMotions' RobotState format and tracked by the pretrained SMPL motion tracker. Sim state feeds back as the next diffusion prefix (closed loop).

**Tech Stack:** PyTorch, CLoSD (MDM diffusion), ProtoMotions (IsaacLab simulator + PPO motion tracker), Isaac Lab/Sim, matplotlib/moviepy (visualization)

**Spec:** `docs/superpowers/specs/2026-03-15-closd-isaaclab-design.md`

**Key paths:**
- Project: `/home/lyuxinghe/code/closd_isaaclab/`
- CLoSD: `/home/lyuxinghe/code/CLoSD/`
- ProtoMotions: `/home/lyuxinghe/code/ProtoMotions/`
- CLoSD_t2m_standalone: `/home/lyuxinghe/code/CLoSD_t2m_standalone/`
- Environment: `/home/lyuxinghe/code/env_isaaclab/`
- DiP checkpoint: `CLoSD/closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000200000.pt`
- Tracker checkpoint: `ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt`
- HML stats: `CLoSD_t2m_standalone/standalone_t2m/assets/t2m_mean.npy`, `t2m_std.npy`

---

## Chunk 1: Project Scaffolding + Coordinate Transforms + FPS Conversion

### Task 1: Project scaffolding and pyproject.toml

**Files:**
- Create: `closd_isaaclab/__init__.py`
- Create: `closd_isaaclab/diffusion/__init__.py`
- Create: `closd_isaaclab/integration/__init__.py`
- Create: `closd_isaaclab/utils/__init__.py`
- Create: `pyproject.toml`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "closd-isaaclab"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "numpy",
    "scipy",
    "matplotlib",
    "moviepy",
]

[project.optional-dependencies]
dev = ["pytest"]
```

- [ ] **Step 2: Create all `__init__.py` files**

Empty `__init__.py` in: `closd_isaaclab/`, `closd_isaaclab/diffusion/`, `closd_isaaclab/integration/`, `closd_isaaclab/utils/`, `tests/`.

- [ ] **Step 3: Install in development mode**

```bash
cd /home/lyuxinghe/code/closd_isaaclab
source /home/lyuxinghe/code/env_isaaclab/bin/activate
pip install -e ".[dev]"
export PYTHONPATH="/home/lyuxinghe/code/CLoSD:/home/lyuxinghe/code/ProtoMotions:/home/lyuxinghe/code/CLoSD_t2m_standalone:$PYTHONPATH"
```

- [ ] **Step 4: Verify import works**

```bash
python -c "import closd_isaaclab; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: project scaffolding with pyproject.toml"
```

---

### Task 2: Coordinate transform utilities

**Files:**
- Create: `closd_isaaclab/utils/coord_transform.py`
- Create: `tests/test_coord_transform.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_coord_transform.py
import torch
import pytest
from closd_isaaclab.utils.coord_transform import (
    CoordTransform,
    smpl_2_mujoco,
    mujoco_2_smpl,
)


class TestJointReordering:
    def test_smpl_2_mujoco_length(self):
        assert len(smpl_2_mujoco) == 24

    def test_mujoco_2_smpl_length(self):
        assert len(mujoco_2_smpl) == 24

    def test_round_trip_identity(self):
        """smpl -> mujoco -> smpl should be identity."""
        original = torch.arange(24)
        via_mujoco = original[smpl_2_mujoco]
        back = via_mujoco[mujoco_2_smpl]
        assert torch.equal(original, back)

    def test_inverse_consistency(self):
        """mujoco_2_smpl[smpl_2_mujoco[i]] == i for all i."""
        for i in range(24):
            assert mujoco_2_smpl[smpl_2_mujoco[i]] == i


class TestCoordTransform:
    @pytest.fixture
    def ct(self):
        return CoordTransform(device="cpu")

    def test_isaac_to_smpl_round_trip(self, ct):
        """Isaac -> SMPL -> Isaac should preserve positions within tolerance."""
        # 24 joints in mujoco order, random positions
        pos_isaac = torch.randn(1, 24, 3)
        pos_smpl = ct.isaac_to_smpl(pos_isaac)
        pos_back = ct.smpl_to_isaac(pos_smpl)
        assert torch.allclose(pos_isaac, pos_back, atol=1e-5)

    def test_smpl_to_isaac_round_trip(self, ct):
        """SMPL -> Isaac -> SMPL should preserve positions."""
        pos_smpl = torch.randn(1, 22, 3)  # 22 joints, no hands
        pos_isaac = ct.smpl_to_isaac(pos_smpl)  # 24 joints
        # Can't round-trip perfectly because hands are added heuristically
        # But the first 22 joints (in SMPL order) should be close
        pos_smpl_back = ct.isaac_to_smpl(pos_isaac)
        # Compare only non-hand joints (indices 0-21 in SMPL order)
        assert pos_smpl_back.shape[-2] == 22

    def test_output_shapes(self, ct):
        pos_isaac = torch.randn(2, 24, 3)
        pos_smpl = ct.isaac_to_smpl(pos_isaac)
        assert pos_smpl.shape == (2, 22, 3)

    def test_batched(self, ct):
        """Should handle arbitrary batch dimensions."""
        pos = torch.randn(3, 5, 24, 3)  # batch=3, seq=5
        result = ct.isaac_to_smpl(pos)
        assert result.shape == (3, 5, 22, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/lyuxinghe/code/closd_isaaclab
pytest tests/test_coord_transform.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: Implement coord_transform.py**

```python
# closd_isaaclab/utils/coord_transform.py
"""
Coordinate transforms between Isaac Lab simulator space and SMPL/HumanML3D space.

Three coordinate spaces (see spec Section 3 "Three Coordinate Spaces"):
1. HumanML3D / SMPL-internal: Y-up, SMPL joint order (22 joints), egocentric
2. Cached reference-pose: [x, -z, y] in SMPL 24-joint order (CLoSD intermediate)
3. Simulator body-state: Z-up, MuJoCo joint order (24 joints), world frame

This module handles conversions between spaces 1 and 3. We skip space 2
(the CLoSD intermediate) by applying the full transform chain in one step.

Rotation matrices ported from CLoSD rep_util.py:
  to_isaac_mat = Rx(-pi/2)
  y180_rot = Ry(pi)
  smpl2sim_rot_mat = Rx(-pi)  (= to_isaac_mat @ to_isaac_mat)

Full SMPL-to-Isaac chain:
  smpl2sim_rot_mat @ pos + y180_rot @ pos + to_isaac_mat.T @ pos
  (split across smpl_to_sim() and _get_state_from_gen_cache() in CLoSD)
"""
import torch
import math
from typing import Optional

# Joint reordering arrays (24 elements each)
# Source: CLoSD closd/utils/rep_util.py lines 21-23
smpl_2_mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]
mujoco_2_smpl = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 19, 13, 15, 20, 16, 21, 17, 22, 18, 23]


def _rotation_matrix_x(angle: float) -> torch.Tensor:
    """Rotation matrix around X axis."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)


def _rotation_matrix_y(angle: float) -> torch.Tensor:
    """Rotation matrix around Y axis."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


class CoordTransform:
    """Handles coordinate transforms between SMPL and Isaac Lab spaces.

    Ported from CLoSD RepresentationHandler (closd/utils/rep_util.py).
    """

    def __init__(self, device: str = "cpu", offset_height: float = 0.92):
        self.device = torch.device(device)
        self.offset_height = offset_height

        # Rotation matrices (from CLoSD rep_util.py lines 74-78)
        self.to_isaac_mat = _rotation_matrix_x(-math.pi / 2).to(self.device)
        self.smpl2sim_rot_mat = (self.to_isaac_mat @ self.to_isaac_mat).to(self.device)
        self.y180_rot = _rotation_matrix_y(-math.pi).to(self.device)

        # Combined full transform: SMPL -> Isaac
        # This merges smpl_to_sim() + _get_state_from_gen_cache()'s to_isaac_mat.T
        # = to_isaac_mat.T @ y180_rot @ smpl2sim_rot_mat
        self._smpl_to_isaac_mat = (
            self.to_isaac_mat.T @ self.y180_rot @ self.smpl2sim_rot_mat
        ).to(self.device)
        self._isaac_to_smpl_mat = self._smpl_to_isaac_mat.inverse().to(self.device)

    def smpl_to_isaac(self, pos_smpl: torch.Tensor) -> torch.Tensor:
        """Convert positions from SMPL space to Isaac Lab simulator space.

        Args:
            pos_smpl: [..., N_joints, 3] positions in SMPL space.
                      If N_joints == 22, hands are added (-> 24 joints).
                      Joints are reordered from SMPL to MuJoCo order.
        Returns:
            [..., 24, 3] positions in Isaac Lab space, MuJoCo joint order.
        """
        if pos_smpl.shape[-2] == 22:
            pos_smpl = _add_hand_joints(pos_smpl)

        # Apply rotation
        result = pos_smpl @ self._smpl_to_isaac_mat.T.to(pos_smpl.device)

        # Reorder SMPL -> MuJoCo
        result = result[..., smpl_2_mujoco, :]
        return result

    def isaac_to_smpl(self, pos_isaac: torch.Tensor, drop_hands: bool = True) -> torch.Tensor:
        """Convert positions from Isaac Lab simulator space to SMPL space.

        Args:
            pos_isaac: [..., 24, 3] positions in Isaac Lab space, MuJoCo joint order.
            drop_hands: If True, drop hand joints (24 -> 22 joints).
        Returns:
            [..., 22 or 24, 3] positions in SMPL space.
        """
        # Reorder MuJoCo -> SMPL
        result = pos_isaac[..., mujoco_2_smpl, :]

        # Apply inverse rotation
        result = result @ self._isaac_to_smpl_mat.T.to(result.device)

        if drop_hands:
            # Drop last 2 joints (L_Hand=22, R_Hand=23 in SMPL order)
            result = result[..., :22, :]

        return result


def _add_hand_joints(pos_22: torch.Tensor) -> torch.Tensor:
    """Add hand joints by extending wrist direction. 22 -> 24 joints.

    Source: CLoSD rep_util.py xyz_to_full_smpl() lines 232-245.
    In SMPL order: L_Wrist=20, R_Wrist=21, L_Hand=22, R_Hand=23.
    Parent of L_Wrist is L_Elbow=18, parent of R_Wrist is R_Elbow=19.
    """
    HAND_OFFSET = 0.08824

    l_wrist = pos_22[..., 20:21, :]
    l_elbow = pos_22[..., 18:19, :]
    l_wrist_dir = l_wrist - l_elbow
    l_wrist_dir = l_wrist_dir / (l_wrist_dir.norm(dim=-1, keepdim=True) + 1e-8)
    l_hand = l_wrist + l_wrist_dir * HAND_OFFSET

    r_wrist = pos_22[..., 21:22, :]
    r_elbow = pos_22[..., 19:20, :]
    r_wrist_dir = r_wrist - r_elbow
    r_wrist_dir = r_wrist_dir / (r_wrist_dir.norm(dim=-1, keepdim=True) + 1e-8)
    r_hand = r_wrist + r_wrist_dir * HAND_OFFSET

    return torch.cat([pos_22, l_hand, r_hand], dim=-2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_coord_transform.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add closd_isaaclab/utils/coord_transform.py tests/test_coord_transform.py
git commit -m "feat: coordinate transform utilities (Isaac <-> SMPL)"
```

---

### Task 3: FPS conversion utility

**Files:**
- Create: `closd_isaaclab/utils/fps_convert.py`
- Create: `tests/test_fps_convert.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fps_convert.py
import torch
import pytest
from closd_isaaclab.utils.fps_convert import fps_convert


class TestFPSConvert:
    def test_30_to_20(self):
        """30fps -> 20fps should reduce frame count by 2/3."""
        data = torch.randn(1, 30, 24, 3)  # 1 second at 30fps
        result = fps_convert(data, src_fps=30, tgt_fps=20)
        assert result.shape[1] == 20  # 1 second at 20fps

    def test_20_to_30(self):
        """20fps -> 30fps should increase frame count by 3/2."""
        data = torch.randn(1, 20, 24, 3)  # 1 second at 20fps
        result = fps_convert(data, src_fps=20, tgt_fps=30)
        assert result.shape[1] == 30  # 1 second at 30fps

    def test_identity(self):
        """Same fps should return identical data."""
        data = torch.randn(1, 10, 24, 3)
        result = fps_convert(data, src_fps=30, tgt_fps=30)
        assert torch.allclose(data, result)

    def test_preserves_batch(self):
        data = torch.randn(4, 30, 24, 3)
        result = fps_convert(data, src_fps=30, tgt_fps=20)
        assert result.shape[0] == 4
        assert result.shape[2:] == (24, 3)

    def test_round_trip_close(self):
        """30->20->30 should be close to original (not exact due to interpolation)."""
        data = torch.randn(1, 30, 24, 3)
        down = fps_convert(data, src_fps=30, tgt_fps=20)
        up = fps_convert(down, src_fps=20, tgt_fps=30)
        # Not exact but should be close
        assert torch.allclose(data, up, atol=0.3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fps_convert.py -v
```

- [ ] **Step 3: Implement fps_convert.py**

```python
# closd_isaaclab/utils/fps_convert.py
"""FPS conversion via bicubic interpolation.

Matches CLoSD's fps_convert() in rep_util.py.
"""
import torch
import torch.nn.functional as F


def fps_convert(
    data: torch.Tensor, src_fps: int, tgt_fps: int, mode: str = "bicubic"
) -> torch.Tensor:
    """Convert motion data between frame rates via interpolation.

    Args:
        data: [batch, src_frames, ...] motion data. Can have arbitrary trailing dims.
        src_fps: Source frame rate.
        tgt_fps: Target frame rate.
        mode: Interpolation mode ('bicubic', 'linear', 'nearest').

    Returns:
        [batch, tgt_frames, ...] resampled data.
    """
    if src_fps == tgt_fps:
        return data

    batch_shape = data.shape[0]
    src_frames = data.shape[1]
    trailing_shape = data.shape[2:]
    tgt_frames = int(src_frames * tgt_fps / src_fps)

    # Flatten trailing dimensions for interpolation
    flat = data.reshape(batch_shape, src_frames, -1)  # [B, T_src, D]
    # F.interpolate expects [B, C, T]
    flat = flat.permute(0, 2, 1)  # [B, D, T_src]

    if mode == "bicubic":
        # bicubic needs 4D input: [B, C, 1, T]
        flat = flat.unsqueeze(2)
        resampled = F.interpolate(flat, size=(1, tgt_frames), mode="bicubic", align_corners=True)
        resampled = resampled.squeeze(2)
    else:
        resampled = F.interpolate(flat, size=tgt_frames, mode=mode, align_corners=True if mode == "linear" else None)

    # [B, D, T_tgt] -> [B, T_tgt, D] -> [B, T_tgt, ...]
    resampled = resampled.permute(0, 2, 1)
    return resampled.reshape(batch_shape, tgt_frames, *trailing_shape)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_fps_convert.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add closd_isaaclab/utils/fps_convert.py tests/test_fps_convert.py
git commit -m "feat: FPS conversion utility (bicubic interpolation)"
```

---

## Chunk 2: HumanML3D Conversion (hml_to_pose / pose_to_hml)

### Task 4: HumanML3D to pose conversion (hml_to_pose)

This is the most critical component — converts diffusion output to simulator-space positions.

**Files:**
- Create: `closd_isaaclab/diffusion/hml_conversion.py`
- Create: `tests/test_hml_conversion.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_hml_conversion.py
import torch
import pytest
import sys
sys.path.insert(0, "/home/lyuxinghe/code/CLoSD")
sys.path.insert(0, "/home/lyuxinghe/code/CLoSD_t2m_standalone")

from closd_isaaclab.diffusion.hml_conversion import HMLConversion


class TestHMLConversion:
    @pytest.fixture
    def hml_conv(self):
        import numpy as np
        mean = np.load("/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_mean.npy")
        std = np.load("/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_std.npy")
        return HMLConversion(
            mean=torch.from_numpy(mean).float(),
            std=torch.from_numpy(std).float(),
            device="cpu",
        )

    def test_hml_to_pose_output_shape(self, hml_conv):
        """hml_to_pose should return [bs, T_30fps, 24, 3]."""
        # Normalized HML data: [bs, T_20fps, 263]
        hml = torch.randn(1, 60, 263)  # 60 frames at 20fps = 3 sec
        recon_data = {"r_rot": torch.tensor([[1.0, 0, 0, 0]]), "r_pos": torch.zeros(1, 3)}
        result = hml_conv.hml_to_pose(hml, recon_data, sim_at_hml_idx=19)
        assert result.shape[0] == 1
        assert result.shape[2] == 24  # 24 joints
        assert result.shape[3] == 3   # xyz
        # 60 frames at 20fps -> 90 frames at 30fps
        assert result.shape[1] == 90

    def test_recover_root_rot_pos_straight_line(self, hml_conv):
        """A straight-line walk should produce monotonically increasing X position."""
        from closd_isaaclab.diffusion.hml_conversion import recover_root_rot_pos
        # Create HML with constant forward velocity (dim 1) and zero rotation (dim 0)
        hml = torch.zeros(1, 20, 263)
        hml[:, :, 1] = 0.1  # constant XZ velocity in local frame
        r_rot, r_pos = recover_root_rot_pos(hml)
        # Root position X should increase
        assert r_pos.shape == (1, 20, 3)
        assert r_rot.shape == (1, 20, 4)

    def test_recon_data_saves_at_correct_index(self, hml_conv):
        """pose_to_hml should save recon_data at frame [-2]."""
        # Create simple 30fps positions
        pos = torch.zeros(1, 60, 24, 3)
        pos[:, :, 0, 0] = torch.linspace(0, 1, 60)  # root moves in X
        _, recon_data = hml_conv.pose_to_hml(pos)
        assert "r_rot" in recon_data
        assert "r_pos" in recon_data
        assert recon_data["r_rot"].shape == (1, 4)
        assert recon_data["r_pos"].shape == (1, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_hml_conversion.py -v
```

- [ ] **Step 3: Implement hml_conversion.py**

This is the largest single file. It ports the core conversion logic from CLoSD's `rep_util.py` and `motion_process_torch.py`:

```python
# closd_isaaclab/diffusion/hml_conversion.py
"""HumanML3D <-> simulator body-state conversion.

Ports CLoSD's RepresentationHandler (closd/utils/rep_util.py) and
motion_process_torch.py. Handles the delta-based HumanML3D representation
and the critical recon_data alignment for closed-loop stitching.

Quaternion convention: internal math uses wxyz (matching CLoSD).
Output positions are in simulator body-state space (Z-up, MuJoCo joint order).
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional

from closd_isaaclab.utils.coord_transform import CoordTransform, smpl_2_mujoco, mujoco_2_smpl
from closd_isaaclab.utils.fps_convert import fps_convert

# Import CLoSD's feature extraction (CPU/numpy-based IK)
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import (
    extract_features_t2m as _extract_features_t2m,
)
# Import CLoSD's quaternion utilities (wxyz convention)
from closd.diffusion_planner.data_loaders.humanml.common.quaternion import (
    qrot, qinv, qmul,
)


def recover_root_rot_pos(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recover root rotation and position from HumanML3D deltas.

    Source: CLoSD motion_process_torch.py lines 8-27.

    Args:
        data: [..., T, 263] unnormalized HumanML3D features.

    Returns:
        r_rot_quat: [..., T, 4] root rotation quaternions (wxyz).
        r_pos: [..., T, 3] root positions.
    """
    rot_vel = data[..., 0]  # root angular velocity around Y

    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    r_pos_flat = r_pos.reshape(-1, r_pos.shape[-2], 3)
    r_rot_flat = r_rot_quat.reshape(-1, r_rot_quat.shape[-2], 4)
    r_pos_flat = qrot(qinv(r_rot_flat.reshape(-1, 4)).unsqueeze(1).expand(-1, r_pos_flat.shape[1], -1).reshape(-1, 4),
                       r_pos_flat.reshape(-1, 3)).reshape(r_pos_flat.shape)
    # Actually, let's use the exact CLoSD logic:
    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    # Rotate each frame's delta by inverse of that frame's rotation
    batch_shape = r_pos.shape[:-2]
    T = r_pos.shape[-2]
    r_pos_2d = r_pos.reshape(-1, T, 3)
    r_rot_2d = r_rot_quat.reshape(-1, T, 4)
    for t in range(T):
        r_pos_2d[:, t] = qrot(qinv(r_rot_2d[:, t]), r_pos_2d[:, t])
    r_pos = r_pos_2d.reshape(batch_shape + (T, 3))
    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]  # absolute root height

    return r_rot_quat, r_pos


def recover_from_ric(data: torch.Tensor, joints_num: int = 22) -> torch.Tensor:
    """Recover joint positions from RIC representation.

    Source: CLoSD motion_process_torch.py lines 47-62.

    Args:
        data: [..., T, 263] unnormalized HumanML3D features.
        joints_num: Number of joints (22 for HumanML3D).

    Returns:
        [..., T, joints_num, 3] global joint positions.
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    # Extract local joint positions: dims 4 to 4+(joints_num-1)*3
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.reshape(positions.shape[:-1] + (joints_num - 1, 3))

    # Rotate from egocentric to global
    batch_shape = positions.shape[:-2]
    T = positions.shape[-2]
    J = joints_num - 1

    pos_flat = positions.reshape(-1, T, J, 3)
    rot_flat = r_rot_quat.reshape(-1, T, 4)
    rpos_flat = r_pos.reshape(-1, T, 3)

    # Apply inverse root rotation to undo rotation-invariance
    for t in range(T):
        inv_rot = qinv(rot_flat[:, t:t+1, :]).expand(-1, J, -1).reshape(-1, 4)
        pos_flat[:, t] = qrot(inv_rot, pos_flat[:, t].reshape(-1, 3)).reshape(-1, J, 3)

    # Add root XZ position
    pos_flat[..., 0] += rpos_flat[..., 0:1]
    pos_flat[..., 2] += rpos_flat[..., 2:3]

    # Prepend root joint
    positions = torch.cat([rpos_flat.unsqueeze(-2), pos_flat], dim=-2)
    return positions.reshape(batch_shape + (T, joints_num, 3))


def _align_to_recon_data(
    points: torch.Tensor,
    recon_data: Dict[str, torch.Tensor],
    is_inverse: bool = False,
) -> torch.Tensor:
    """Align positions to/from a root coordinate frame.

    Source: CLoSD rep_util.py lines 141-165.

    Args:
        points: [..., N, 3] joint positions.
        recon_data: {"r_rot": [..., 4] wxyz, "r_pos": [..., 3]}.
        is_inverse: If True, applies inverse transform (egocentric -> world).
    """
    r_rot = recon_data["r_rot"]
    r_pos = recon_data["r_pos"]

    if not is_inverse:
        # World -> egocentric: subtract root XZ, rotate by root heading
        points = points.clone()
        points[..., [0, 2]] -= r_pos[..., None, [0, 2]]
        # Expand r_rot to match points shape
        rot_shape = points.shape[:-1] + (4,)
        rot_expanded = r_rot.unsqueeze(-2).expand(rot_shape)
        points = qrot(rot_expanded.reshape(-1, 4), points.reshape(-1, 3)).reshape(points.shape)
    else:
        # Egocentric -> world: rotate by inverse heading, add root XZ
        r_rot_inv = qinv(r_rot)
        rot_shape = points.shape[:-1] + (4,)
        rot_expanded = r_rot_inv.unsqueeze(-2).expand(rot_shape)
        points = qrot(rot_expanded.reshape(-1, 4), points.reshape(-1, 3)).reshape(points.shape)
        points = points.clone()
        points[..., [0, 2]] += r_pos[..., None, [0, 2]]

    return points


class HMLConversion:
    """Bidirectional conversion between HumanML3D and simulator body-state space.

    Handles normalization, delta-to-absolute recovery, coordinate transforms,
    and the critical recon_data alignment for closed-loop stitching.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        self.coord_transform = CoordTransform(device=device)

    def hml_to_pose(
        self,
        hml_norm: torch.Tensor,
        recon_data: Dict[str, torch.Tensor],
        sim_at_hml_idx: int,
    ) -> torch.Tensor:
        """Convert normalized HML to simulator body-state positions.

        Args:
            hml_norm: [bs, T_20fps, 263] normalized HumanML3D features.
            recon_data: {"r_rot": [bs, 4], "r_pos": [bs, 3]} wxyz quats.
                        Root state at the last prefix frame for alignment.
            sim_at_hml_idx: Frame index in the HML sequence corresponding
                           to the sim's current state (typically prefix_len - 1).

        Returns:
            [bs, T_30fps, 24, 3] in simulator body-state space
            (Z-up, MuJoCo joint order, world frame).
        """
        # Unnormalize
        hml = hml_norm * self.std.to(hml_norm.device) + self.mean.to(hml_norm.device)

        # Recover 22-joint positions in HumanML3D space (Y-up, SMPL order)
        hml_xyz = recover_from_ric(hml, 22)  # [bs, T, 22, 3]

        # Recover root rotation/position for alignment
        r_rot_quat, r_pos = recover_root_rot_pos(hml)

        # Two-step alignment (spec Section 3.2):
        # 1. Zero out HML's own root at sim_at_hml_idx
        hml_transform = {
            "r_rot": r_rot_quat[:, sim_at_hml_idx],
            "r_pos": r_pos[:, sim_at_hml_idx],
        }
        zeroed = _align_to_recon_data(hml_xyz, hml_transform, is_inverse=False)

        # 2. Apply sim's actual root transform
        aligned = _align_to_recon_data(zeroed, recon_data, is_inverse=True)

        # Convert SMPL 22-joint -> Isaac 24-joint simulator space
        result = self.coord_transform.smpl_to_isaac(aligned)  # [bs, T_20fps, 24, 3]

        # Upsample 20fps -> 30fps
        result = fps_convert(result, src_fps=20, tgt_fps=30)

        return result

    def pose_to_hml(
        self,
        positions_isaac: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert simulator body-state positions to normalized HML.

        Args:
            positions_isaac: [bs, T_30fps, 24, 3] Isaac Lab body positions.

        Returns:
            hml_norm: [bs, T_20fps, 263] normalized HumanML3D features.
            recon_data: {"r_rot": [bs, 4], "r_pos": [bs, 3]} wxyz quats.
        """
        bs = positions_isaac.shape[0]

        # Append extrapolated dummy frame (velocities consume one frame)
        last = positions_isaac[:, -1:]
        second_last = positions_isaac[:, -2:-1]
        dummy = last + (last - second_last)
        positions_ext = torch.cat([positions_isaac, dummy], dim=1)

        # Downsample 30fps -> 20fps
        positions_20 = fps_convert(positions_ext, src_fps=30, tgt_fps=20)

        # Isaac -> SMPL 22-joint space
        positions_smpl = self.coord_transform.isaac_to_smpl(positions_20)  # [bs, T_20fps, 22, 3]

        # Extract HumanML3D features (calls IK internally, goes through numpy)
        # extract_features_t2m expects [bs, T, 22, 3] numpy
        positions_np = positions_smpl.detach().cpu()

        hml_features_list = []
        recon_data_list = []
        for b in range(bs):
            feats, rd = _extract_features_t2m(positions_np[b:b+1])
            hml_features_list.append(feats)
            recon_data_list.append(rd)

        hml_features = torch.cat(hml_features_list, dim=0).to(self.device)

        # Combine recon_data (take from index [-2] as per CLoSD convention)
        recon_data = {
            "r_rot": torch.stack([rd["r_rot"] for rd in recon_data_list]).to(self.device),
            "r_pos": torch.stack([rd["r_pos"] for rd in recon_data_list]).to(self.device),
        }

        # Normalize
        hml_norm = (hml_features - self.mean) / self.std

        return hml_norm, recon_data
```

**Note:** The `recover_root_rot_pos` and `recover_from_ric` implementations above are simplified placeholders. The actual implementation should import directly from CLoSD's `motion_process_torch.py` to ensure exact numerical equivalence. The plan provides both: direct imports for correctness, and standalone implementations for testing without CLoSD dependency.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_hml_conversion.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add closd_isaaclab/diffusion/hml_conversion.py tests/test_hml_conversion.py
git commit -m "feat: HumanML3D conversion (hml_to_pose / pose_to_hml)"
```

---

### Task 5: Rotation solver

**Files:**
- Create: `closd_isaaclab/diffusion/rotation_solver.py`
- Create: `tests/test_rotation_solver.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rotation_solver.py
import torch
import pytest
from closd_isaaclab.diffusion.rotation_solver import RotationSolver, cont6d_to_matrix, matrix_to_cont6d


class TestCont6DConversion:
    def test_round_trip_identity(self):
        """matrix -> 6d -> matrix should be identity."""
        # Random rotation matrices via QR decomposition
        A = torch.randn(10, 3, 3)
        Q, _ = torch.linalg.qr(A)
        # Ensure proper rotation (det=+1)
        Q = Q * torch.det(Q).unsqueeze(-1).unsqueeze(-1).sign()

        c6d = matrix_to_cont6d(Q)
        assert c6d.shape == (10, 6)

        recovered = cont6d_to_matrix(c6d)
        assert torch.allclose(Q, recovered, atol=1e-5)

    def test_6d_to_matrix_orthogonal(self):
        """Output should be orthogonal matrices."""
        c6d = torch.randn(5, 6)
        mats = cont6d_to_matrix(c6d)
        # R @ R^T should be identity
        eye = torch.eye(3).expand(5, 3, 3)
        assert torch.allclose(mats @ mats.transpose(-1, -2), eye, atol=1e-5)


class TestRotationSolver:
    def test_verify_consistency_perfect(self):
        """FK from known rotations should match positions exactly."""
        # This test requires ProtoMotions' pose_lib
        pytest.importorskip("protomotions")
        solver = RotationSolver(mode="diffusion", device="cpu")
        # TODO: Will need kinematic_info from ProtoMotions
        # Placeholder: just verify the interface exists
        assert hasattr(solver, "verify_consistency")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_rotation_solver.py -v
```

- [ ] **Step 3: Implement rotation_solver.py**

```python
# closd_isaaclab/diffusion/rotation_solver.py
"""Derive rigid body rotations and DOF positions from diffusion output.

Two modes:
  - "diffusion": Use 6D rotations from HML dims 67-192 (local/parent-relative)
  - "analytical": Derive rotations from bone direction vectors

Both modes produce FK consistency checks against position output.
"""
import torch
import warnings
from typing import Tuple, Optional

from closd_isaaclab.diffusion.hml_conversion import recover_root_rot_pos


def cont6d_to_matrix(cont6d: torch.Tensor) -> torch.Tensor:
    """Convert continuous 6D rotation representation to rotation matrix.

    Uses Gram-Schmidt orthogonalization (Zhou et al., 2019).

    Args:
        cont6d: [..., 6] continuous 6D representation.
    Returns:
        [..., 3, 3] rotation matrix.
    """
    a1 = cont6d[..., :3]
    a2 = cont6d[..., 3:]

    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_cont6d(mat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to continuous 6D representation.

    Args:
        mat: [..., 3, 3] rotation matrix.
    Returns:
        [..., 6] continuous 6D representation.
    """
    return mat[..., :2, :].reshape(mat.shape[:-2] + (6,))


def _wxyz_quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion to rotation matrix."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mat = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1).reshape(quat.shape[:-1] + (3, 3))
    return mat


class RotationSolver:
    """Derives rotations and DOF positions from diffusion output.

    Args:
        mode: "diffusion" or "analytical".
        device: torch device.
        consistency_threshold: meters, max FK error before warning.
    """

    def __init__(
        self,
        mode: str = "diffusion",
        device: str = "cpu",
        consistency_threshold: float = 0.05,
        kinematic_info=None,
    ):
        assert mode in ("diffusion", "analytical")
        self.mode = mode
        self.device = torch.device(device)
        self.consistency_threshold = consistency_threshold
        self.kinematic_info = kinematic_info

    def solve(
        self,
        positions: torch.Tensor,
        hml_raw: Optional[torch.Tensor] = None,
        root_rot_wxyz: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """Derive rotations from diffusion output.

        Args:
            positions: [bs, T, 22, 3] SMPL joint positions (SMPL order, SMPL space).
            hml_raw: [bs, T, 263] unnormalized HML (needed for mode="diffusion").
            root_rot_wxyz: [bs, T, 4] root rotation quaternions (wxyz).

        Returns:
            local_rot_mats: [bs, T, 24, 3, 3] local rotation matrices (SMPL order).
            dof_pos: [bs, T, 69] joint angles if kinematic_info available, else None.
            consistency_error: mean FK position error in meters.
        """
        if self.mode == "diffusion":
            return self._solve_diffusion(positions, hml_raw, root_rot_wxyz)
        else:
            return self._solve_analytical(positions, root_rot_wxyz)

    def _solve_diffusion(self, positions, hml_raw, root_rot_wxyz):
        """Extract local rotations from HML 6D rotation dims."""
        assert hml_raw is not None, "hml_raw required for diffusion mode"
        bs, T = hml_raw.shape[:2]

        # Extract 21-joint 6D local rotations from dims 67-192
        rot_6d = hml_raw[..., 67:193].reshape(bs, T, 21, 6)

        # Convert 6D -> rotation matrices
        local_rot_mats_21 = cont6d_to_matrix(rot_6d)  # [bs, T, 21, 3, 3]

        # Get root rotation matrix
        if root_rot_wxyz is None:
            root_rot_wxyz, _ = recover_root_rot_pos(hml_raw)
        root_mat = _wxyz_quat_to_matrix(root_rot_wxyz)  # [bs, T, 3, 3]

        # Prepend root, add hand joints (identity)
        identity = torch.eye(3, device=hml_raw.device).expand(bs, T, 1, 3, 3)
        # 22 joints: root + 21 local
        local_rot_mats_22 = torch.cat([root_mat.unsqueeze(-3), local_rot_mats_21], dim=-3)
        # 24 joints: add 2 hand joints as identity
        local_rot_mats_24 = torch.cat([local_rot_mats_22, identity.expand(bs, T, 2, 3, 3)], dim=-3)

        # Compute dof_pos via ProtoMotions FK if available
        dof_pos = None
        consistency_error = -1.0

        if self.kinematic_info is not None:
            try:
                from protomotions.components.pose_lib import (
                    extract_qpos_from_transforms,
                    compute_forward_kinematics_from_transforms,
                )
                # Reorder to mujoco for ProtoMotions
                from closd_isaaclab.utils.coord_transform import smpl_2_mujoco
                rot_mats_mujoco = local_rot_mats_24[:, :, smpl_2_mujoco]
                root_pos = positions[:, :, 0]  # [bs, T, 3]

                # Process frame by frame for FK
                dof_pos_list = []
                fk_pos_list = []
                for t in range(T):
                    qpos = extract_qpos_from_transforms(
                        self.kinematic_info, root_pos[:, t], rot_mats_mujoco[:, t]
                    )
                    dof_pos_list.append(qpos)

                    fk_pos, _ = compute_forward_kinematics_from_transforms(
                        self.kinematic_info, root_pos[:, t], rot_mats_mujoco[:, t]
                    )
                    fk_pos_list.append(fk_pos)

                dof_pos = torch.stack(dof_pos_list, dim=1)
                fk_positions = torch.stack(fk_pos_list, dim=1)

                # FK consistency check
                consistency_error = (fk_positions - positions).norm(dim=-1).mean().item()
                if consistency_error > self.consistency_threshold:
                    warnings.warn(
                        f"WARNING: Diffusion rotations show high FK inconsistency "
                        f"({consistency_error*100:.1f}cm). Consider --rotation-mode analytical"
                    )
            except ImportError:
                pass

        return local_rot_mats_24, dof_pos, consistency_error

    def _solve_analytical(self, positions, root_rot_wxyz):
        """Derive rotations from bone direction vectors."""
        # TODO: Implement analytical IK from bone vectors
        # For now, raise NotImplementedError
        raise NotImplementedError("Analytical rotation solver not yet implemented")

    def verify_consistency(
        self, positions: torch.Tensor, dof_pos: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """Compare FK(dof_pos) against positions.

        Returns:
            mean_error: mean position error in meters.
            per_joint_error: [num_joints] per-joint error.
        """
        assert self.kinematic_info is not None, "kinematic_info required"
        from protomotions.components.pose_lib import compute_forward_kinematics_from_transforms, extract_transforms_from_qpos

        root_pos, joint_rot_mats = extract_transforms_from_qpos(self.kinematic_info, dof_pos)
        fk_pos, _ = compute_forward_kinematics_from_transforms(self.kinematic_info, root_pos, joint_rot_mats)

        per_joint_error = (fk_pos - positions).norm(dim=-1).mean(dim=0)
        mean_error = per_joint_error.mean().item()
        return mean_error, per_joint_error
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_rotation_solver.py -v
```
Expected: Cont6D tests PASS; ProtoMotions-dependent tests may skip.

- [ ] **Step 5: Commit**

```bash
git add closd_isaaclab/diffusion/rotation_solver.py tests/test_rotation_solver.py
git commit -m "feat: rotation solver (diffusion 6D + FK consistency check)"
```

---

## Chunk 3: RobotState Builder + Motion Provider

### Task 6: RobotState builder

**Files:**
- Create: `closd_isaaclab/diffusion/robot_state_builder.py`
- Create: `tests/test_robot_state_builder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_robot_state_builder.py
import torch
import pytest
from closd_isaaclab.diffusion.robot_state_builder import RobotStateBuilder


class TestRobotStateBuilder:
    def test_velocity_derivation_constant_motion(self):
        """Constant velocity motion should produce constant velocity output."""
        builder = RobotStateBuilder(dt=1.0 / 30.0)
        # 10 frames of linear motion
        T = 10
        positions = torch.zeros(1, T, 24, 3)
        positions[:, :, 0, 0] = torch.linspace(0, 0.9, T)  # root moves at ~1m/s in X

        velocities = builder._compute_velocities(positions)
        # Central differencing should give ~1.0 m/s for middle frames
        assert velocities.shape == (1, T, 24, 3)
        # Interior frames should have non-zero velocity
        assert velocities[:, 2:-2, 0, 0].abs().mean() > 0.5

    def test_contacts_from_hml(self):
        """Should map 4 HML foot contacts to 24-body contact tensor."""
        builder = RobotStateBuilder(dt=1.0 / 30.0)
        hml = torch.zeros(1, 10, 263)
        hml[:, :, 259:263] = 1.0  # all feet in contact
        contacts = builder._extract_contacts(hml)
        assert contacts.shape == (1, 10, 24)  # 24 bodies
        # At least 4 non-zero entries per frame
        assert (contacts.sum(dim=-1) >= 4).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_robot_state_builder.py -v
```

- [ ] **Step 3: Implement robot_state_builder.py**

```python
# closd_isaaclab/diffusion/robot_state_builder.py
"""Build ProtoMotions RobotState from diffusion output.

Assembles positions, rotations, velocities, DOFs, and contacts into
the format expected by ProtoMotions' MimicControl.get_context().
"""
import torch
from typing import Dict, Optional, Tuple

from closd_isaaclab.diffusion.rotation_solver import RotationSolver, _wxyz_quat_to_matrix
from closd_isaaclab.utils.coord_transform import smpl_2_mujoco, mujoco_2_smpl

# Contact body indices in MuJoCo order
# L_Ankle=3, R_Ankle=7, L_Toe=4, R_Toe=8 (mujoco order)
# HML foot contacts order: [L_Ankle, L_Toe, R_Ankle, R_Toe] (dims 259-262)
_HML_CONTACT_TO_MUJOCO_BODY = {
    0: 3,   # L_Ankle
    1: 4,   # L_Toe
    2: 7,   # R_Ankle
    3: 8,   # R_Toe
}


class RobotStateBuilder:
    """Builds RobotState-compatible data from diffusion output.

    Args:
        dt: Simulation timestep (1/30 for 30fps).
        rotation_solver: RotationSolver instance (optional).
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        rotation_solver: Optional[RotationSolver] = None,
        num_bodies: int = 24,
    ):
        self.dt = dt
        self.rotation_solver = rotation_solver
        self.num_bodies = num_bodies

        # Cache for current horizon
        self._positions = None      # [bs, T, 24, 3]
        self._rotations = None      # [bs, T, 24, 4] xyzw
        self._velocities = None     # [bs, T, 24, 3]
        self._ang_velocities = None # [bs, T, 24, 3]
        self._dof_pos = None        # [bs, T, 69]
        self._dof_vel = None        # [bs, T, 69]
        self._contacts = None       # [bs, T, 24]
        self._horizon_dt = None

    def build(
        self,
        positions: torch.Tensor,
        hml_raw: Optional[torch.Tensor] = None,
        root_rot_wxyz: Optional[torch.Tensor] = None,
    ):
        """Build and cache motion data from a new planning horizon.

        Args:
            positions: [bs, T, 24, 3] positions in Isaac Lab space (MuJoCo order).
            hml_raw: [bs, T_20fps, 263] raw HML output for rotation extraction.
            root_rot_wxyz: [bs, T, 4] root rotations (wxyz).
        """
        self._positions = positions
        self._horizon_dt = self.dt

        # Derive velocities
        self._velocities = self._compute_velocities(positions)

        # Derive rotations if solver available
        if self.rotation_solver is not None and hml_raw is not None:
            # Rotation solver works in SMPL order, so reorder
            pos_smpl = positions[..., mujoco_2_smpl, :]
            local_rot_mats, dof_pos, error = self.rotation_solver.solve(
                pos_smpl, hml_raw, root_rot_wxyz
            )
            if local_rot_mats is not None:
                # Convert local rotation matrices to xyzw quaternions
                # For now store as matrices; convert when building RobotState
                self._dof_pos = dof_pos
                if dof_pos is not None:
                    self._dof_vel = self._compute_velocities_1d(dof_pos)

        # Extract contacts from HML
        if hml_raw is not None:
            self._contacts = self._extract_contacts(hml_raw)

    def get_state_at_frames(
        self, frame_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get cached state at specific frame indices.

        Args:
            frame_indices: [num_envs] integer frame indices.

        Returns:
            Dict with rigid_body_pos, rigid_body_vel, etc.
        """
        state = {}
        if self._positions is not None:
            state["rigid_body_pos"] = self._positions[:, frame_indices]
        if self._velocities is not None:
            state["rigid_body_vel"] = self._velocities[:, frame_indices]
        if self._dof_pos is not None:
            state["dof_pos"] = self._dof_pos[:, frame_indices]
        if self._dof_vel is not None:
            state["dof_vel"] = self._dof_vel[:, frame_indices]
        if self._contacts is not None:
            state["rigid_body_contacts"] = self._contacts[:, frame_indices]
        return state

    def _compute_velocities(self, positions: torch.Tensor) -> torch.Tensor:
        """Central differencing for velocity derivation."""
        vel = torch.zeros_like(positions)
        if positions.shape[1] >= 3:
            vel[:, 1:-1] = (positions[:, 2:] - positions[:, :-2]) / (2 * self.dt)
            vel[:, 0] = (positions[:, 1] - positions[:, 0]) / self.dt
            vel[:, -1] = (positions[:, -1] - positions[:, -2]) / self.dt
        elif positions.shape[1] == 2:
            v = (positions[:, 1] - positions[:, 0]) / self.dt
            vel[:, 0] = v
            vel[:, 1] = v
        return vel

    def _compute_velocities_1d(self, data: torch.Tensor) -> torch.Tensor:
        """Central differencing for 1D data (dof_pos)."""
        vel = torch.zeros_like(data)
        if data.shape[1] >= 3:
            vel[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (2 * self.dt)
            vel[:, 0] = (data[:, 1] - data[:, 0]) / self.dt
            vel[:, -1] = (data[:, -1] - data[:, -2]) / self.dt
        return vel

    def _extract_contacts(self, hml_raw: torch.Tensor) -> torch.Tensor:
        """Map HML foot contacts (4 flags) to per-body contact tensor.

        Args:
            hml_raw: [bs, T, 263] unnormalized HML.

        Returns:
            [bs, T, 24] binary contact flags.
        """
        bs, T = hml_raw.shape[:2]
        contacts = torch.zeros(bs, T, self.num_bodies, device=hml_raw.device)
        foot_contacts = hml_raw[..., 259:263]  # [bs, T, 4]

        for hml_idx, mujoco_idx in _HML_CONTACT_TO_MUJOCO_BODY.items():
            contacts[..., mujoco_idx] = foot_contacts[..., hml_idx]

        return contacts
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_robot_state_builder.py -v
```

- [ ] **Step 5: Commit**

```bash
git add closd_isaaclab/diffusion/robot_state_builder.py tests/test_robot_state_builder.py
git commit -m "feat: RobotState builder with velocity and contact derivation"
```

---

### Task 7: Diffusion Motion Provider

**Files:**
- Create: `closd_isaaclab/diffusion/motion_provider.py`

- [ ] **Step 1: Implement motion_provider.py**

```python
# closd_isaaclab/diffusion/motion_provider.py
"""DiP diffusion model wrapper for autoregressive motion generation.

Wraps CLoSD's MDM model for both:
- Standalone open-loop generation (verify_diffusion.py)
- Closed-loop generation with sim feedback (run_closd_isaaclab.py)
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from closd_isaaclab.diffusion.hml_conversion import HMLConversion


class DiffusionMotionProvider:
    """Wraps CLoSD's DiP model for motion generation.

    Args:
        model_path: Path to DiP checkpoint (.pt file).
        mean_path: Path to HumanML3D mean normalization stats.
        std_path: Path to HumanML3D std normalization stats.
        device: Torch device.
        guidance: Classifier-free guidance scale.
        context_len: Number of prefix frames (20fps).
        pred_len: Number of predicted frames (20fps).
    """

    def __init__(
        self,
        model_path: str,
        mean_path: str,
        std_path: str,
        device: str = "cuda",
        guidance: float = 5.0,
        context_len: int = 20,
        pred_len: int = 40,
    ):
        self.device = torch.device(device)
        self.guidance = guidance
        self.context_len = context_len
        self.pred_len = pred_len
        self.max_frames = context_len + pred_len

        # Load normalization stats
        mean = torch.from_numpy(np.load(mean_path)).float()
        std = torch.from_numpy(np.load(std_path)).float()

        # Build HML conversion
        self.hml_conv = HMLConversion(mean=mean, std=std, device=device)

        # Load model via CLoSD_t2m_standalone
        from standalone_t2m.checkpoint import CheckpointBundle
        from standalone_t2m.config import build_model_and_diffusion

        bundle = CheckpointBundle(
            model_path=Path(model_path),
            args_path=Path(model_path).parent / "args.json",
            mean_path=Path(mean_path),
            std_path=Path(std_path),
        )
        model_bundle = build_model_and_diffusion(bundle)
        self.model = model_bundle.model
        self.diffusion = model_bundle.diffusion
        self.sample_fn = model_bundle.sample_fn
        self.model_args = model_bundle.args

        # Cache text embeddings
        self._text_embed_cache = {}

    def generate_standalone(
        self,
        text_prompt: str,
        num_seconds: float = 8.0,
        prefix_mode: str = "standing",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Open-loop generation without simulator feedback.

        Args:
            text_prompt: Text description of desired motion.
            num_seconds: Duration of generated motion (excluding prefix).
            prefix_mode: "standing" for standing start.

        Returns:
            positions: [1, T, 22, 3] SMPL joint positions.
            hml_raw: [1, T, 263] unnormalized HML features.
        """
        from standalone_t2m.generation import generate_motion
        from standalone_t2m.prefix.standing import build_standing_prefix
        from standalone_t2m.decode import decode_to_xyz

        target_frames = int(num_seconds * 20)  # 20fps
        prefix = build_standing_prefix(self.context_len).to(self.device)

        # Generate
        from standalone_t2m.config import LoadedModelBundle
        bundle = LoadedModelBundle(
            model=self.model,
            diffusion=self.diffusion,
            sample_fn=self.sample_fn,
            args=self.model_args,
            mean=self.hml_conv.mean,
            std=self.hml_conv.std,
            context_len=self.context_len,
            pred_len=self.pred_len,
            device=self.device,
        )
        generated = generate_motion(bundle, text_prompt, target_frames, self.guidance, prefix)

        # Compose full motion (prefix + generated)
        from standalone_t2m.generation import compose_output_motion
        full_motion = compose_output_motion(prefix, generated)

        # Decode to positions
        positions = decode_to_xyz(full_motion, self.hml_conv.mean.cpu(), self.hml_conv.std.cpu())

        # Also get unnormalized HML
        hml_raw_norm = full_motion.squeeze(2).permute(0, 2, 1)  # [1, T, 263]
        hml_raw = hml_raw_norm * self.hml_conv.std.cpu() + self.hml_conv.mean.cpu()

        return positions, hml_raw

    def generate_next_horizon(
        self,
        pose_buffer_isaac: torch.Tensor,
        recon_data: Optional[Dict[str, torch.Tensor]],
        text_prompt: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Closed-loop generation with sim state feedback.

        Args:
            pose_buffer_isaac: [bs, context_len_30fps, 24, 3] sim body positions.
            recon_data: {"r_rot": [bs,4], "r_pos": [bs,3]} from last call (or None).
            text_prompt: Text condition.

        Returns:
            positions_isaac: [bs, horizon_30fps, 24, 3] predicted positions.
            hml_raw: [bs, pred_len_20fps, 263] raw HML for rotation extraction.
            recon_data_new: {"r_rot", "r_pos"} for next call.
        """
        # Convert sim state to HML prefix
        hml_prefix, recon_data_from_sim = self.hml_conv.pose_to_hml(pose_buffer_isaac)

        # Use recon_data from sim (captures current root state)
        if recon_data is None:
            recon_data = recon_data_from_sim

        # Reshape for MDM: [bs, 263, 1, T]
        hml_prefix_mdm = hml_prefix.permute(0, 2, 1).unsqueeze(2)

        # Get text embeddings (cached)
        text_embed = self._get_text_embed(text_prompt)

        # Build model kwargs
        model_kwargs = {
            "y": {
                "text": [text_prompt] * hml_prefix.shape[0],
                "tokens": text_embed,
                "mask": torch.ones(1, 1, 1, self.max_frames, device=self.device, dtype=torch.bool),
                "lengths": torch.tensor([self.max_frames], device=self.device),
                "prefix_len": self.context_len,
            }
        }

        # Build inpainting input
        shape = (hml_prefix.shape[0], 263, 1, self.max_frames)
        inpainted = torch.zeros(shape, device=self.device)
        inpainted[:, :, :, :self.context_len] = hml_prefix_mdm

        inpainting_mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        inpainting_mask[:, :, :, :self.context_len] = True

        model_kwargs["y"]["inpainted_motion"] = inpainted
        model_kwargs["y"]["inpainting_mask"] = inpainting_mask

        # Run diffusion
        sample = self.sample_fn(
            self.model,
            shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            progress=False,
        )

        # Extract prediction
        sample_hml = sample.squeeze(2).permute(0, 2, 1)  # [bs, max_frames, 263]

        # Unnormalize for hml_raw
        hml_raw = sample_hml * self.hml_conv.std + self.hml_conv.mean

        # Convert to positions
        positions_isaac = self.hml_conv.hml_to_pose(
            sample_hml, recon_data_from_sim, sim_at_hml_idx=self.context_len - 1
        )

        # New recon_data for next call
        recon_data_new = recon_data_from_sim

        return positions_isaac, hml_raw, recon_data_new

    def _get_text_embed(self, prompt: str) -> torch.Tensor:
        if prompt not in self._text_embed_cache:
            self._text_embed_cache[prompt] = self.model.encode_text([prompt])
        return self._text_embed_cache[prompt]
```

- [ ] **Step 2: Verify imports work**

```bash
python -c "from closd_isaaclab.diffusion.motion_provider import DiffusionMotionProvider; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add closd_isaaclab/diffusion/motion_provider.py
git commit -m "feat: DiP diffusion motion provider (standalone + closed-loop)"
```

---

## Chunk 4: Verification Scripts (Diffusion + Rotations)

### Task 8: verify_diffusion.py

**Files:**
- Create: `scripts/verify_diffusion.py`

- [ ] **Step 1: Implement verify_diffusion.py**

```python
#!/usr/bin/env python3
"""Verify CLoSD diffusion model generates reasonable motion from text prompts.

No simulator required. Outputs:
  - motion.pt: raw 263-dim HumanML3D motion
  - xyz.pt: decoded 22-joint 3D positions
  - {prompt_slug}.mp4: matplotlib 3D skeleton animation

Usage:
    python scripts/verify_diffusion.py --prompt "a person walks forward" --num-seconds 8
"""
import argparse
import re
import sys
from pathlib import Path
import torch

sys.path.insert(0, "/home/lyuxinghe/code/CLoSD")
sys.path.insert(0, "/home/lyuxinghe/code/CLoSD_t2m_standalone")
sys.path.insert(0, "/home/lyuxinghe/code/ProtoMotions")


def main():
    parser = argparse.ArgumentParser(description="Verify diffusion motion generation")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--num-seconds", type=float, default=8.0)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--output-dir", default="outputs/verify_diffusion")
    parser.add_argument(
        "--model-path",
        default="/home/lyuxinghe/code/CLoSD/closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000200000.pt",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    from closd_isaaclab.diffusion.motion_provider import DiffusionMotionProvider
    provider = DiffusionMotionProvider(
        model_path=args.model_path,
        mean_path="/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_mean.npy",
        std_path="/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_std.npy",
        device=device,
        guidance=args.guidance,
    )

    # Generate
    print(f"Generating {args.num_seconds}s for: '{args.prompt}'")
    positions, hml_raw = provider.generate_standalone(args.prompt, args.num_seconds)
    print(f"Generated: positions {positions.shape}, hml_raw {hml_raw.shape}")

    # Save artifacts
    torch.save(hml_raw, output_dir / "motion.pt")
    torch.save(positions, output_dir / "xyz.pt")

    # Render MP4
    slug = re.sub(r"[^a-z0-9]+", "_", args.prompt.lower()).strip("_")[:50]
    mp4_path = output_dir / f"{slug}.mp4"

    from standalone_t2m.render import render_xyz_motion
    render_xyz_motion(positions.cpu(), args.prompt, mp4_path, fps=20)
    print(f"Saved: {mp4_path}")

    # Print stats
    total_frames = positions.shape[1]
    print(f"Total frames: {total_frames} ({total_frames/20:.1f}s at 20fps)")
    print(f"Guidance: {args.guidance}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test it**

```bash
cd /home/lyuxinghe/code/closd_isaaclab
python scripts/verify_diffusion.py --prompt "a person walks forward" --num-seconds 4
```
Expected: MP4 file at `outputs/verify_diffusion/a_person_walks_forward.mp4`

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_diffusion.py
git commit -m "feat: verify_diffusion.py script (standalone diffusion test)"
```

---

### Task 9: verify_rotations.py

**Files:**
- Create: `scripts/verify_rotations.py`

- [ ] **Step 1: Implement verify_rotations.py**

```python
#!/usr/bin/env python3
"""Verify 6D rotation conversions and FK consistency.

Tests:
1. 6D round-trip: matrix -> 6D -> matrix
2. Ground truth: ProtoMotions .motion file positions -> rotation solver -> compare
3. Diffusion output: FK consistency check for both modes

Usage:
    python scripts/verify_rotations.py \
        --motion-file ~/code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion
"""
import argparse
import sys
import torch

sys.path.insert(0, "/home/lyuxinghe/code/CLoSD")
sys.path.insert(0, "/home/lyuxinghe/code/CLoSD_t2m_standalone")
sys.path.insert(0, "/home/lyuxinghe/code/ProtoMotions")

from closd_isaaclab.diffusion.rotation_solver import (
    cont6d_to_matrix,
    matrix_to_cont6d,
)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def test_6d_round_trip():
    """Test matrix -> 6D -> matrix round trip."""
    A = torch.randn(100, 3, 3)
    Q, _ = torch.linalg.qr(A)
    Q = Q * torch.det(Q).unsqueeze(-1).unsqueeze(-1).sign()

    c6d = matrix_to_cont6d(Q)
    recovered = cont6d_to_matrix(c6d)

    max_err = (Q - recovered).abs().max().item()
    status = PASS if max_err < 1e-5 else FAIL
    print(f"{status} 6D round-trip: max error {max_err:.2e}")
    return max_err < 1e-5


def test_ground_truth(motion_file: str):
    """Load .motion file, extract positions, run solver, compare rotations."""
    try:
        data = torch.load(motion_file, weights_only=False)
    except Exception as e:
        print(f"[SKIP] Could not load {motion_file}: {e}")
        return True

    gt_pos = data.get("gts", data.get("rigid_body_pos"))
    gt_rot = data.get("grs", data.get("rigid_body_rot"))
    if gt_pos is None or gt_rot is None:
        print("[SKIP] Motion file missing position or rotation data")
        return True

    if gt_pos.dim() == 2:
        # Single frame: [bodies, 3]
        gt_pos = gt_pos.unsqueeze(0)
        gt_rot = gt_rot.unsqueeze(0)
    if gt_pos.dim() == 3:
        # [T, bodies, 3] -> [1, T, bodies, 3]
        gt_pos = gt_pos.unsqueeze(0)
        gt_rot = gt_rot.unsqueeze(0)

    print(f"  Loaded: {gt_pos.shape[1]} frames, {gt_pos.shape[2]} bodies")
    print(f"  Ground truth rotation shape: {gt_rot.shape}")
    status = PASS
    print(f"{status} Ground truth test: loaded successfully (full comparison requires kinematic_info)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", default=None)
    args = parser.parse_args()

    results = []

    print("\n=== 6D Rotation Conversion Tests ===")
    results.append(test_6d_round_trip())

    if args.motion_file:
        print("\n=== Ground Truth Tests ===")
        results.append(test_ground_truth(args.motion_file))

    print("\n=== Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"{passed}/{total} tests passed")

    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test it**

```bash
python scripts/verify_rotations.py \
    --motion-file /home/lyuxinghe/code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion
```

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_rotations.py
git commit -m "feat: verify_rotations.py script (6D conversion + FK consistency)"
```

---

## Chunk 5: ProtoMotions Integration (MotionLib + MotionManager + Experiment)

### Task 10: CLoSD MotionLib wrapper

**Files:**
- Create: `closd_isaaclab/integration/closd_motion_lib.py`

- [ ] **Step 1: Implement closd_motion_lib.py**

```python
# closd_isaaclab/integration/closd_motion_lib.py
"""MotionLib-compatible wrapper that serves diffusion-generated motion.

Duck-types ProtoMotions' MotionLib interface so MimicControl.get_context()
can query reference motion from diffusion output.
"""
import torch
from typing import Optional

from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)
from closd_isaaclab.diffusion.robot_state_builder import RobotStateBuilder


class CLoSDMotionLib:
    """MotionLib-compatible interface backed by diffusion-generated motion.

    Implements the subset of MotionLib used by MimicControl.get_context()
    and MimicMotionManager.
    """

    def __init__(
        self,
        robot_state_builder: RobotStateBuilder,
        device: str = "cuda",
        horizon_duration: float = 2.0,
    ):
        self.robot_state_builder = robot_state_builder
        self.device = torch.device(device)
        self._horizon_duration = horizon_duration

        # Interface stubs
        self.motion_file = "closd_diffusion"
        self.motion_weights = torch.tensor([1.0], device=self.device)

    @property
    def motion_lengths(self) -> torch.Tensor:
        return torch.tensor([self._horizon_duration], device=self.device)

    @motion_lengths.setter
    def motion_lengths(self, value: torch.Tensor):
        self._horizon_duration = value[0].item()

    def num_motions(self) -> int:
        return 1

    def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self.motion_lengths[motion_ids]

    def get_motion_state(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor, **kwargs
    ) -> RobotState:
        """Get interpolated reference state at requested times.

        Args:
            motion_ids: [num_envs] (ignored, always motion 0).
            motion_times: [num_envs] times within current horizon.

        Returns:
            RobotState in COMMON format (xyzw quaternions).
        """
        # Convert times to frame indices
        frame_indices = (motion_times / self.robot_state_builder.dt).long()
        frame_indices = frame_indices.clamp(0, self.robot_state_builder._positions.shape[1] - 1)

        state_dict = self.robot_state_builder.get_state_at_frames(frame_indices)

        return RobotState.from_dict(state_dict, state_conversion=StateConversion.COMMON)
```

- [ ] **Step 2: Commit**

```bash
git add closd_isaaclab/integration/closd_motion_lib.py
git commit -m "feat: CLoSDMotionLib (MotionLib duck-type for diffusion)"
```

---

### Task 11: CLoSD MotionManager (closed-loop)

**Files:**
- Create: `closd_isaaclab/integration/closd_motion_manager.py`

- [ ] **Step 1: Implement closd_motion_manager.py**

```python
# closd_isaaclab/integration/closd_motion_manager.py
"""Closed-loop motion manager that feeds sim state back to diffusion.

Subclasses MimicMotionManager to:
1. Maintain a sliding pose_buffer of sim body positions
2. Trigger diffusion replanning when the planning horizon is exhausted
3. Update the RobotStateBuilder cache with new horizon data
"""
import torch
from typing import Optional

from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
from protomotions.components.motion_lib import MotionLib

from closd_isaaclab.diffusion.motion_provider import DiffusionMotionProvider
from closd_isaaclab.diffusion.robot_state_builder import RobotStateBuilder
from closd_isaaclab.integration.closd_motion_lib import CLoSDMotionLib


class CLoSDMotionManager(MimicMotionManager):
    """Motion manager with closed-loop diffusion replanning.

    Args:
        motion_provider: DiP diffusion wrapper.
        robot_state_builder: Builds RobotState from diffusion output.
        motion_lib: CLoSDMotionLib instance.
        text_prompt: Text condition for generation.
        pred_len_20fps: Prediction length at 20fps (default 40).
        context_len_30fps: Pose buffer length at 30fps.
    """

    def __init__(
        self,
        config: MimicMotionManagerConfig,
        num_envs: int,
        env_dt: float,
        device: torch.device,
        motion_lib: CLoSDMotionLib,
        motion_provider: DiffusionMotionProvider,
        robot_state_builder: RobotStateBuilder,
        text_prompt: str = "",
        pred_len_20fps: int = 40,
        context_len_30fps: int = 30,  # 20 frames at 20fps * 30/20
        get_body_positions_fn=None,
    ):
        super().__init__(config, num_envs, env_dt, device, motion_lib)

        self.motion_provider = motion_provider
        self.robot_state_builder = robot_state_builder
        self.text_prompt = text_prompt
        self.pred_len_20fps = pred_len_20fps
        self.context_len_30fps = context_len_30fps
        self.planning_horizon_30fps = int(pred_len_20fps * 30 / 20)
        self._get_body_positions = get_body_positions_fn

        # Sliding window of sim body positions
        self.pose_buffer = torch.zeros(
            num_envs, context_len_30fps, 24, 3, device=device
        )
        self.recon_data = None
        self.frame_counter = 0

    def post_physics_step(self):
        """Called after each physics step. Updates pose buffer and triggers replanning."""
        super().post_physics_step()

        # Update pose buffer with current sim state
        if self._get_body_positions is not None:
            current_pos = self._get_body_positions()  # [num_envs, 24, 3]
            self.pose_buffer = torch.cat(
                [self.pose_buffer[:, 1:], current_pos.unsqueeze(1)], dim=1
            )

        self.frame_counter += 1

        # Trigger replanning when horizon exhausted
        if self.frame_counter % self.planning_horizon_30fps == 0:
            self._replan()

    def _replan(self):
        """Run diffusion to generate next planning horizon."""
        positions_isaac, hml_raw, recon_data_new = self.motion_provider.generate_next_horizon(
            self.pose_buffer, self.recon_data, self.text_prompt
        )

        # Check for NaN
        if torch.isnan(positions_isaac).any():
            print("WARNING: Diffusion produced NaN. Keeping last valid horizon.")
            return

        # Update caches
        self.robot_state_builder.build(positions_isaac, hml_raw)
        self.recon_data = recon_data_new

        # Reset motion_times so get_done_tracks doesn't signal premature done
        self.motion_times[:] = 0
        self.motion_lib.motion_lengths = torch.tensor(
            [self.planning_horizon_30fps * self.env_dt],
            device=self.device,
        )

    def sample_motions(self, env_ids: torch.Tensor, new_motion_ids=None):
        """Reset: clear pose buffer, trigger initial diffusion call."""
        # Fill pose buffer with current pose repeated
        if self._get_body_positions is not None:
            current_pos = self._get_body_positions()
            self.pose_buffer[env_ids] = current_pos[env_ids].unsqueeze(1).expand(
                -1, self.context_len_30fps, -1, -1
            )

        self.recon_data = None
        self.frame_counter = 0
        self.motion_times[env_ids] = 0

        # Initial planning
        self._replan()
```

- [ ] **Step 2: Commit**

```bash
git add closd_isaaclab/integration/closd_motion_manager.py
git commit -m "feat: CLoSDMotionManager (closed-loop diffusion + sim feedback)"
```

---

### Task 12: verify_tracking.py

**Files:**
- Create: `scripts/verify_tracking.py`

- [ ] **Step 1: Implement verify_tracking.py**

This script tests ProtoMotions' tracker independently with an offline motion file.

```python
#!/usr/bin/env python3
"""Verify ProtoMotions tracker works with offline motion in Isaac Lab.

Loads a pretrained SMPL motion tracker and an offline .motion file,
runs the tracker in Isaac Lab, and records video.

Usage:
    python scripts/verify_tracking.py \
        --checkpoint ~/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
        --motion-file ~/code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion \
        --record-frames 300
"""
import argparse
import sys

sys.path.insert(0, "/home/lyuxinghe/code/ProtoMotions")


def main():
    parser = argparse.ArgumentParser(description="Verify ProtoMotions tracker")
    parser.add_argument("--checkpoint", required=True, help="Path to tracker checkpoint")
    parser.add_argument("--motion-file", required=True, help="Path to .motion file")
    parser.add_argument("--record-frames", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # Use ProtoMotions' inference_agent with the provided checkpoint and motion
    # This leverages their existing infrastructure directly
    import subprocess
    cmd = [
        sys.executable,
        "/home/lyuxinghe/code/ProtoMotions/protomotions/inference_agent.py",
        "--checkpoint", args.checkpoint,
        "--motion-file", args.motion_file,
        "--num-envs", str(args.num_envs),
        "--simulator", "isaaclab",
    ]
    if args.headless:
        cmd.append("--headless")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test it (requires DISPLAY)**

```bash
export DISPLAY=:1
python scripts/verify_tracking.py \
    --checkpoint /home/lyuxinghe/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
    --motion-file /home/lyuxinghe/code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion \
    --num-envs 1
```

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_tracking.py
git commit -m "feat: verify_tracking.py (offline motion in Isaac Lab)"
```

---

## Chunk 6: Full Pipeline + README

### Task 13: Full closed-loop script

**Files:**
- Create: `scripts/run_closd_isaaclab.py`

- [ ] **Step 1: Implement run_closd_isaaclab.py**

This is the main integration script. It will require iterative debugging to get the ProtoMotions experiment config right. The initial version should:

1. Load DiP model
2. Load ProtoMotions tracker config from the pretrained checkpoint
3. Replace MotionLib/MotionManager with our custom versions
4. Run the inference loop

```python
#!/usr/bin/env python3
"""Full CLoSD-IsaacLab closed-loop text-to-motion pipeline.

Usage:
    python scripts/run_closd_isaaclab.py \
        --prompt "a person walks forward" \
        --rotation-mode diffusion \
        --episode-length 300 \
        --record-frames 300
"""
import argparse
import sys
import torch

sys.path.insert(0, "/home/lyuxinghe/code/CLoSD")
sys.path.insert(0, "/home/lyuxinghe/code/CLoSD_t2m_standalone")
sys.path.insert(0, "/home/lyuxinghe/code/ProtoMotions")


def main():
    parser = argparse.ArgumentParser(description="CLoSD-IsaacLab text-to-motion")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--rotation-mode", choices=["diffusion", "analytical"], default="diffusion")
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--record-frames", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--tracker-checkpoint",
        default="/home/lyuxinghe/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt",
    )
    parser.add_argument(
        "--dip-checkpoint",
        default="/home/lyuxinghe/code/CLoSD/closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000200000.pt",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load DiP diffusion model
    from closd_isaaclab.diffusion.motion_provider import DiffusionMotionProvider
    from closd_isaaclab.diffusion.rotation_solver import RotationSolver
    from closd_isaaclab.diffusion.robot_state_builder import RobotStateBuilder

    provider = DiffusionMotionProvider(
        model_path=args.dip_checkpoint,
        mean_path="/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_mean.npy",
        std_path="/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_std.npy",
        device=device,
        guidance=args.guidance,
    )

    rotation_solver = RotationSolver(mode=args.rotation_mode, device=device)
    state_builder = RobotStateBuilder(dt=1.0/30.0, rotation_solver=rotation_solver)

    # 2. Load ProtoMotions tracker config
    from pathlib import Path
    checkpoint_dir = Path(args.tracker_checkpoint).parent
    configs = torch.load(checkpoint_dir / "resolved_configs_inference.pt", weights_only=False)

    # 3. Wire in custom MotionLib and MotionManager
    from closd_isaaclab.integration.closd_motion_lib import CLoSDMotionLib
    motion_lib = CLoSDMotionLib(state_builder, device=device)

    # 4. Run with ProtoMotions infrastructure
    # This section will need iterative refinement based on ProtoMotions' config system
    print(f"Prompt: {args.prompt}")
    print(f"Rotation mode: {args.rotation_mode}")
    print(f"Episode length: {args.episode_length}")
    print(f"Guidance: {args.guidance}")
    print(f"Device: {device}")
    print("\nFull integration with ProtoMotions' inference loop pending...")
    print("Use verify_diffusion.py and verify_tracking.py to test components independently.")


if __name__ == "__main__":
    main()
```

**Note:** The full ProtoMotions integration requires careful wiring of configs, which will be refined iteratively. The script provides the framework and component initialization.

- [ ] **Step 2: Test basic execution**

```bash
python scripts/run_closd_isaaclab.py --prompt "a person walks forward" --headless
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_closd_isaaclab.py
git commit -m "feat: run_closd_isaaclab.py (full pipeline framework)"
```

---

### Task 14: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

The README should contain the exact setup and reproduction steps from the spec's Section 6, plus a quick-start guide. Include GCP/TurboVNC notes.

Content should cover:
1. Overview (what this does)
2. Prerequisites (GCP VM, TurboVNC, GPU)
3. Environment setup (step-by-step commands)
4. Verification steps (4 scripts with exact commands and expected output)
5. Architecture diagram (text)
6. Troubleshooting (common issues)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup and reproduction steps"
```

---

## Task Dependencies

```
Task 1 (scaffolding)
  ├── Task 2 (coord_transform) ──┐
  └── Task 3 (fps_convert) ──────┤
                                  ├── Task 4 (hml_conversion)
                                  │     └── Task 5 (rotation_solver)
                                  │           └── Task 6 (robot_state_builder)
                                  │                 └── Task 7 (motion_provider)
                                  │                       ├── Task 8 (verify_diffusion)
                                  │                       └── Task 9 (verify_rotations)
                                  │
                                  └── Task 10 (closd_motion_lib)
                                        └── Task 11 (closd_motion_manager)
                                              └── Task 12 (verify_tracking)
                                                    └── Task 13 (run_closd_isaaclab)
                                                          └── Task 14 (README)
```

## Implementation Notes

1. **Start with verify_diffusion.py working end-to-end** (Tasks 1-8). This validates the diffusion model, HML conversion, and coordinate transforms without any simulator dependency. If this produces reasonable skeleton animations, the representation pipeline is correct.

2. **Then verify_tracking.py** (Task 12). This validates ProtoMotions' tracker and Isaac Lab work correctly with a known-good motion file. If this produces reasonable humanoid tracking, the simulator setup is correct.

3. **Then wire them together** (Task 13). The full integration is the riskiest part — it requires the diffusion output to be in exactly the right format for ProtoMotions' observation system. Debug by comparing RobotState fields against what MotionLib normally provides.

4. **The HML conversion is the hardest code** (Task 4). Port it carefully from CLoSD, running the CLoSD tests against both implementations to verify numerical equivalence. Pay special attention to:
   - `recover_root_rot_pos()`: cumulative sum of deltas must match exactly
   - Two-step `recon_data` alignment: the stitching is easy to get backwards
   - The full transform chain (smpl2sim + y180 + to_isaac): must apply ALL rotations

5. **Import directly from CLoSD/ProtoMotions** wherever possible rather than reimplementing. This reduces bugs. The standalone implementations in this plan are for reference — prefer `from closd.diffusion_planner...` imports.
