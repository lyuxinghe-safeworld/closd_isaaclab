"""Coordinate-space transforms between SMPL/HumanML3D and Isaac Lab body-state space.

Three spaces involved:
  1. HumanML3D / SMPL-internal: Y-up, 22 joints (SMPL order)
  2. Cached reference-pose space: [x, -z, y] intermediate (CLoSD internal)
  3. Simulator body-state space: Z-up, 24 joints (MuJoCo order)

Rotation matrices (ported from CLoSD rep_util.py):
  to_isaac_mat    = Rx(-pi/2)      — SMPL Y-up -> Isaac Z-up
  smpl2sim_rot_mat = Rx(-pi)       — to_isaac_mat @ to_isaac_mat
  y180_rot         = Ry(-pi)       — 180-degree Y rotation

Full SMPL->Isaac chain: to_isaac_mat.T @ y180_rot @ smpl2sim_rot_mat
"""

import math
import torch

# ---------------------------------------------------------------------------
# Joint reordering indices (from CLoSD rep_util.py lines 21-23)
# ---------------------------------------------------------------------------

# Converts 24-joint SMPL-extended ordering -> MuJoCo ordering
smpl_2_mujoco: list[int] = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]

# Converts 24-joint MuJoCo ordering -> SMPL-extended ordering
mujoco_2_smpl: list[int] = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 19, 13, 15, 20, 16, 21, 17, 22, 18, 23]

# SMPL joint indices used for hand extension
_L_WRIST = 20
_R_WRIST = 21
_L_ELBOW = 18
_R_ELBOW = 19
_HAND_OFFSET = 0.08824  # metres


def _rx(angle: float) -> torch.Tensor:
    """3x3 rotation matrix around the X axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([
        [1.0,  0.0,  0.0],
        [0.0,   c,   -s],
        [0.0,   s,    c],
    ], dtype=torch.float64)


def _ry(angle: float) -> torch.Tensor:
    """3x3 rotation matrix around the Y axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([
        [ c,   0.0,   s],
        [0.0,  1.0,  0.0],
        [-s,   0.0,   c],
    ], dtype=torch.float64)


class CoordTransform:
    """Bidirectional coordinate transform between SMPL space and Isaac Lab space.

    Usage
    -----
    ct = CoordTransform()
    isaac_pos = ct.smpl_to_isaac(smpl_pos)   # [..., 22 or 24, 3] -> [..., 24, 3]
    smpl_pos  = ct.isaac_to_smpl(isaac_pos)  # [..., 24, 3]       -> [..., 22, 3]

    The rotation matrices are stored in float64 for numerical accuracy.  All
    position tensors are cast to the input dtype on output so the caller's
    dtype is preserved.
    """

    def __init__(self) -> None:
        # Build component rotation matrices in float64 for numerical accuracy
        to_isaac_mat = _rx(-math.pi / 2)          # Rx(-pi/2)
        smpl2sim_rot_mat = _rx(-math.pi)           # Rx(-pi)  == to_isaac_mat @ to_isaac_mat
        y180_rot = _ry(-math.pi)                   # Ry(-pi)

        # Full SMPL -> Isaac combined rotation
        # CLoSD applies: pos @ smpl2sim.T, then pos @ y180.T, then pos @ to_isaac.T
        # Combined: pos @ (to_isaac @ y180 @ smpl2sim).T
        # So rot_mat is what we LEFT-multiply with pos (i.e. pos @ rot_mat)
        self.rot_mat: torch.Tensor = (to_isaac_mat @ y180_rot @ smpl2sim_rot_mat).T
        self.rot_mat_inv: torch.Tensor = self.rot_mat.T  # orthogonal -> inverse == transpose

        # Pre-compute index tensors
        self._smpl_2_mujoco = torch.tensor(smpl_2_mujoco, dtype=torch.long)
        self._mujoco_2_smpl = torch.tensor(mujoco_2_smpl, dtype=torch.long)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def smpl_to_isaac(self, pos_smpl: torch.Tensor) -> torch.Tensor:
        """Convert SMPL joint positions to Isaac Lab body-state space.

        Parameters
        ----------
        pos_smpl:
            [..., N, 3] tensor where N is 22 (SMPL) or 24 (SMPL + hands).

        Returns
        -------
        Tensor of shape [..., 24, 3] in Isaac Lab / MuJoCo joint order.
        """
        in_dtype = pos_smpl.dtype
        # Upcast to float64 for the rotation; we'll cast back at the end
        pos = pos_smpl.to(torch.float64)

        # 1. If only 22 joints, add hand joints first
        n_joints = pos.shape[-2]
        if n_joints == 22:
            pos = self._add_hand_joints(pos)
        elif n_joints != 24:
            raise ValueError(f"Expected 22 or 24 joints, got {n_joints}")

        # 2. Apply rotation:  pos_isaac = pos @ rot_mat.T  (broadcasts over batch dims)
        #    Equivalent to applying rot_mat to each column vector.
        rot = self.rot_mat.to(pos.device)
        pos_rot = pos @ rot.T  # [..., 24, 3]

        # 3. Reorder joints: SMPL-extended ordering -> MuJoCo ordering
        idx = self._smpl_2_mujoco.to(pos.device)
        return pos_rot[..., idx, :].to(in_dtype)

    def isaac_to_smpl(
        self,
        pos_isaac: torch.Tensor,
        drop_hands: bool = True,
    ) -> torch.Tensor:
        """Convert Isaac Lab body-state positions to SMPL space.

        Parameters
        ----------
        pos_isaac:
            [..., 24, 3] tensor in Isaac Lab / MuJoCo joint order.
        drop_hands:
            If True (default), return [..., 22, 3] by dropping the two hand
            joints (indices 22, 23) that were synthesised during smpl_to_isaac.

        Returns
        -------
        Tensor of shape [..., 22, 3] or [..., 24, 3] in SMPL joint order.
        """
        in_dtype = pos_isaac.dtype
        # Upcast to float64 for the rotation; we'll cast back at the end
        pos = pos_isaac.to(torch.float64)

        if pos.shape[-2] != 24:
            raise ValueError(f"Expected 24 joints (Isaac order), got {pos.shape[-2]}")

        # 1. Reorder joints: MuJoCo ordering -> SMPL-extended ordering
        idx = self._mujoco_2_smpl.to(pos.device)
        pos_smpl_order = pos[..., idx, :]

        # 2. Apply inverse rotation: pos_smpl = pos_isaac @ rot_mat  (rot_mat.T inverse)
        rot_inv = self.rot_mat_inv.to(pos.device)
        pos_rot = pos_smpl_order @ rot_inv.T  # [..., 24, 3]

        # 3. Optionally drop hand joints (indices 22, 23 in SMPL ordering)
        if drop_hands:
            return pos_rot[..., :22, :].to(in_dtype)
        return pos_rot.to(in_dtype)

    def _add_hand_joints(self, pos_22: torch.Tensor) -> torch.Tensor:
        """Extend 22-joint SMPL positions to 24 joints by adding hand joints.

        Hand joint positions are estimated by extending the forearm direction
        (elbow->wrist) from the wrist by ``_HAND_OFFSET`` metres.

        Parameters
        ----------
        pos_22:
            [..., 22, 3] tensor in SMPL joint order.

        Returns
        -------
        Tensor of shape [..., 24, 3].
        """
        l_wrist = pos_22[..., _L_WRIST, :]   # [..., 3]
        r_wrist = pos_22[..., _R_WRIST, :]
        l_elbow = pos_22[..., _L_ELBOW, :]
        r_elbow = pos_22[..., _R_ELBOW, :]

        l_dir = l_wrist - l_elbow
        r_dir = r_wrist - r_elbow

        l_norm = l_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        r_norm = r_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        l_hand = l_wrist + _HAND_OFFSET * l_dir / l_norm  # [..., 3]
        r_hand = r_wrist + _HAND_OFFSET * r_dir / r_norm

        # Append as joints 22 and 23
        l_hand = l_hand.unsqueeze(-2)  # [..., 1, 3]
        r_hand = r_hand.unsqueeze(-2)

        return torch.cat([pos_22, l_hand, r_hand], dim=-2)  # [..., 24, 3]
