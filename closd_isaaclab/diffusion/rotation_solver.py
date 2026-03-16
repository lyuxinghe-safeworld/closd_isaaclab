"""Rotation solver: extract and convert joint rotations from HumanML3D 263-dim features.

The diffusion model outputs 263-dim HumanML3D features. Dimensions 67-192 contain
21-joint 6D continuous rotations (LOCAL/parent-relative), produced by IK during
data preprocessing.

This module provides:
- cont6d_to_matrix: 6D continuous rotation -> 3x3 rotation matrix (Gram-Schmidt)
- matrix_to_cont6d: 3x3 rotation matrix -> 6D continuous rotation (first two rows)
- wxyz_quat_to_matrix: wxyz quaternion -> 3x3 rotation matrix
- RotationSolver: extracts per-joint local rotation matrices from hml_raw,
  optionally using kinematic_info from ProtoMotions to compute qpos and FK consistency.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import (
    recover_root_rot_pos,
)


# ---------------------------------------------------------------------------
# Low-level rotation conversions
# ---------------------------------------------------------------------------


def cont6d_to_matrix(cont6d: Tensor) -> Tensor:
    """Convert 6D continuous rotation representation to 3x3 rotation matrix.

    Uses Gram-Schmidt orthogonalization.

    Parameters
    ----------
    cont6d : Tensor
        [..., 6] 6D rotation representation.

    Returns
    -------
    Tensor
        [..., 3, 3] rotation matrix.
    """
    a1 = cont6d[..., :3]  # [..., 3]
    a2 = cont6d[..., 3:]  # [..., 3]

    # Gram-Schmidt
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    dot = (b1 * a2).sum(dim=-1, keepdim=True)  # [..., 1]
    b2_unnorm = a2 - dot * b1
    b2 = b2_unnorm / b2_unnorm.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    b3 = torch.linalg.cross(b1, b2)  # [..., 3]

    # Stack along second-to-last dim: [..., 3, 3]
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_cont6d(mat: Tensor) -> Tensor:
    """Convert 3x3 rotation matrix to 6D continuous rotation representation.

    Takes the first two rows and flattens them.

    Parameters
    ----------
    mat : Tensor
        [..., 3, 3] rotation matrix.

    Returns
    -------
    Tensor
        [..., 6] 6D representation (first two rows of mat concatenated).
    """
    # mat[..., 0, :] is first row, mat[..., 1, :] is second row
    row0 = mat[..., 0, :]  # [..., 3]
    row1 = mat[..., 1, :]  # [..., 3]
    return torch.cat([row0, row1], dim=-1)  # [..., 6]


def wxyz_quat_to_matrix(quat: Tensor) -> Tensor:
    """Convert wxyz quaternion to 3x3 rotation matrix.

    Parameters
    ----------
    quat : Tensor
        [..., 4] quaternion in wxyz convention (w first).

    Returns
    -------
    Tensor
        [..., 3, 3] rotation matrix.
    """
    # Normalize
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]

    # Build rotation matrix
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R00 = 1 - 2 * (yy + zz)
    R01 = 2 * (xy - wz)
    R02 = 2 * (xz + wy)
    R10 = 2 * (xy + wz)
    R11 = 1 - 2 * (xx + zz)
    R12 = 2 * (yz - wx)
    R20 = 2 * (xz - wy)
    R21 = 2 * (yz + wx)
    R22 = 1 - 2 * (xx + yy)

    # Stack into [..., 3, 3]
    row0 = torch.stack([R00, R01, R02], dim=-1)  # [..., 3]
    row1 = torch.stack([R10, R11, R12], dim=-1)
    row2 = torch.stack([R20, R21, R22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)  # [..., 3, 3]


# ---------------------------------------------------------------------------
# RotationSolver
# ---------------------------------------------------------------------------


class RotationSolver:
    """Extract local joint rotation matrices from HumanML3D features.

    Parameters
    ----------
    mode : str
        Operational mode. Currently only "diffusion" is supported.
        "analytical" raises NotImplementedError.
    device : str
        Torch device for computations.
    consistency_threshold : float
        FK consistency error threshold (metres) used for warnings.
    kinematic_info : optional
        ProtoMotions KinematicInfo object. When provided, enables qpos extraction
        and FK consistency checks.
    """

    def __init__(
        self,
        mode: str = "diffusion",
        device: str = "cpu",
        consistency_threshold: float = 0.05,
        kinematic_info=None,
    ):
        self.mode = mode
        self.device = device
        self.consistency_threshold = consistency_threshold
        self.kinematic_info = kinematic_info

    def solve(
        self,
        positions: Optional[Tensor],
        hml_raw: Optional[Tensor] = None,
        root_rot_wxyz: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[float]]:
        """Extract local rotation matrices and optionally compute qpos.

        Parameters
        ----------
        positions : Tensor or None
            [bs, T, N_joints, 3] joint positions (used for FK consistency check).
        hml_raw : Tensor, optional
            [bs, T, 263] raw (un-normalized) HumanML3D features. Required for
            mode="diffusion".
        root_rot_wxyz : Tensor, optional
            [bs, T, 4] root rotation quaternions in wxyz convention. If None,
            root rotation is recovered from hml_raw via recover_root_rot_pos.

        Returns
        -------
        local_rot_mats_24 : Tensor
            [bs, T, 24, 3, 3] local rotation matrices (joint 0 = root,
            joints 1-21 from 6D hml features, joints 22-23 = identity hands).
        dof_pos : Tensor or None
            [bs, T, nq] qpos vector if kinematic_info is provided, else None.
        consistency_error : float or None
            Mean FK position error (metres) if kinematic_info is provided, else None.
        """
        if self.mode == "analytical":
            raise NotImplementedError(
                "mode='analytical' is not yet implemented. Use mode='diffusion'."
            )

        if self.mode != "diffusion":
            raise ValueError(f"Unknown mode '{self.mode}'. Expected 'diffusion'.")

        # -----------------------------------------------------------------------
        # 1. Extract 6D rotations for 21 body joints from hml_raw[..., 67:193]
        # -----------------------------------------------------------------------
        # hml_raw: [bs, T, 263]  dims 67:193 → 126 values = 21 * 6
        raw_6d = hml_raw[..., 67:193]  # [bs, T, 126]
        bs, T = raw_6d.shape[:2]
        raw_6d = raw_6d.reshape(bs, T, 21, 6)  # [bs, T, 21, 6]

        # Convert to rotation matrices: [bs, T, 21, 3, 3]
        body_rot_mats = cont6d_to_matrix(raw_6d)

        # -----------------------------------------------------------------------
        # 2. Get root rotation matrix
        # -----------------------------------------------------------------------
        if root_rot_wxyz is not None:
            # root_rot_wxyz: [bs, T, 4]
            root_mat = wxyz_quat_to_matrix(root_rot_wxyz)  # [bs, T, 3, 3]
        else:
            # Recover root rotation from hml_raw via CLoSD's recover_root_rot_pos
            # recover_root_rot_pos returns (r_rot_quat, r_pos)
            # r_rot_quat: [bs, T, 4] in wxyz convention (CLoSD uses wxyz)
            #
            # IMPORTANT: HumanML3D convention — r_rot_quat represents the
            # global→local (facing frame) rotation, i.e. it maps world-frame
            # directions INTO the body's facing frame.  For the kinematic
            # chain we need the INVERSE (local→global), which is the transpose
            # of the rotation matrix.
            r_rot_quat, _ = recover_root_rot_pos(hml_raw)  # [bs, T, 4] wxyz
            root_mat = wxyz_quat_to_matrix(r_rot_quat)  # [bs, T, 3, 3]
            root_mat = root_mat.transpose(-2, -1)  # invert: global→local → local→global

        # root_mat: [bs, T, 3, 3] -> unsqueeze joint dim
        root_mat = root_mat.unsqueeze(2)  # [bs, T, 1, 3, 3]

        # -----------------------------------------------------------------------
        # 3. Build 24-joint local rotation matrices
        #    Joint 0:    root rotation matrix
        #    Joints 1-21: 6D-derived local rotations
        #    Joints 22-23: identity (hands)
        # -----------------------------------------------------------------------
        eye = torch.eye(3, dtype=body_rot_mats.dtype, device=body_rot_mats.device)
        hand_mats = eye.reshape(1, 1, 1, 3, 3).expand(bs, T, 2, 3, 3)  # [bs, T, 2, 3, 3]

        # Concatenate: [bs, T, 1+21+2, 3, 3] = [bs, T, 24, 3, 3]
        local_rot_mats_24 = torch.cat([root_mat, body_rot_mats, hand_mats], dim=2)

        # -----------------------------------------------------------------------
        # 4. Optionally extract qpos and compute FK consistency
        # -----------------------------------------------------------------------
        dof_pos = None
        consistency_error = None

        if self.kinematic_info is not None:
            from protomotions.components.pose_lib import (
                extract_qpos_from_transforms,
                compute_forward_kinematics_from_transforms,
            )

            # Flatten time dimension: treat each (batch, time) as an independent sample
            # local_rot_mats_24: [bs, T, 24, 3, 3] -> [bs*T, 24, 3, 3]
            rot_flat = local_rot_mats_24.reshape(bs * T, 24, 3, 3)

            # Root positions: [bs, T, 3] -> [bs*T, 3]
            if positions is not None:
                root_pos_flat = positions[:, :, 0, :].reshape(bs * T, 3)
            else:
                root_pos_flat = torch.zeros(bs * T, 3, device=rot_flat.device)

            # Extract qpos (SMPL uses 3-DOF ball joints → exp_map decomposition)
            # Returns [bs*T, Nq] where Nq = 3 (root pos) + 4 (root quat) + 69 (joints)
            qpos_flat = extract_qpos_from_transforms(
                self.kinematic_info, root_pos_flat, rot_flat,
                multi_dof_decomposition_method="exp_map",
            )  # [bs*T, 76]
            # Strip root pos (3) + root quat (4) = first 7 elements → keep joint DOFs only
            dof_pos_flat = qpos_flat[:, 7:]  # [bs*T, 69]
            dof_pos = dof_pos_flat.reshape(bs, T, -1)

            # FK consistency check
            if positions is not None:
                consistency_error, _ = self.verify_consistency(positions, dof_pos)

        return local_rot_mats_24, dof_pos, consistency_error

    def verify_consistency(
        self,
        positions: Tensor,
        dof_pos: Tensor,
    ) -> Tuple[float, Tensor]:
        """Verify FK consistency between joint positions and qpos.

        Runs forward kinematics from dof_pos and compares resulting world positions
        against the reference positions.

        Parameters
        ----------
        positions : Tensor
            [bs, T, N_joints, 3] reference world joint positions.
        dof_pos : Tensor
            [bs, T, nq] MuJoCo qpos vector.

        Returns
        -------
        mean_error : float
            Mean per-joint position error across all joints, frames, and batches (metres).
        per_joint_error : Tensor
            [N_joints] mean error per joint.
        """
        if self.kinematic_info is None:
            raise ValueError(
                "kinematic_info must be provided to verify FK consistency."
            )

        from protomotions.components.pose_lib import (
            extract_transforms_from_qpos,
            compute_forward_kinematics_from_transforms,
        )

        bs, T = dof_pos.shape[:2]
        dof_flat = dof_pos.reshape(bs * T, -1)

        # Recover transforms from qpos
        root_pos_fk, joint_rot_mats_fk = extract_transforms_from_qpos(
            self.kinematic_info, dof_flat
        )  # root_pos_fk: [bs*T, 3], joint_rot_mats_fk: [bs*T, Nb, 3, 3]

        # Compute FK world positions
        world_pos_fk, _ = compute_forward_kinematics_from_transforms(
            self.kinematic_info, root_pos_fk, joint_rot_mats_fk
        )  # [bs*T, Nb, 3]

        # Reshape FK positions back to [bs, T, Nb, 3]
        Nb = world_pos_fk.shape[1]
        world_pos_fk = world_pos_fk.reshape(bs, T, Nb, 3)

        # Compare against reference positions (may have different joint counts)
        N_ref = positions.shape[2]
        N_compare = min(Nb, N_ref)

        error = (world_pos_fk[:, :, :N_compare, :] - positions[:, :, :N_compare, :]).norm(dim=-1)
        per_joint_error = error.mean(dim=(0, 1))  # [N_compare]
        mean_error = error.mean().item()

        return mean_error, per_joint_error
