"""HumanML3D <-> Isaac Lab position conversion.

Wraps CLoSD's core math (recover_root_rot_pos, recover_from_ric,
extract_features_t2m) with our CoordTransform and fps_convert utilities.
"""

from typing import Dict, Tuple

import torch

# CLoSD imports — used directly for numerical equivalence
from closd.diffusion_planner.data_loaders.humanml.common.quaternion import qinv, qrot
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import (
    extract_features_t2m,
    recover_from_ric,
    recover_root_rot_pos,
)

from closd_isaaclab.utils.coord_transform import CoordTransform
from closd_isaaclab.utils.fps_convert import fps_convert


class HMLConversion:
    """Bidirectional conversion between normalized HumanML3D features and
    Isaac Lab body-state joint positions.

    Parameters
    ----------
    mean : torch.Tensor
        HumanML3D feature mean, shape [263].
    std : torch.Tensor
        HumanML3D feature std, shape [263].
    device : str
        Device for internal tensors.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, device: str = "cpu"):
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.device = device
        self.coord_transform = CoordTransform()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def hml_to_pose(
        self,
        hml_norm: torch.Tensor,
        recon_data: Dict[str, torch.Tensor],
        sim_at_hml_idx: int,
    ) -> torch.Tensor:
        """Convert normalized HML features to Isaac Lab joint positions.

        Parameters
        ----------
        hml_norm : torch.Tensor
            [bs, T_20fps, 263] normalized HumanML3D features.
        recon_data : dict
            ``{'r_rot': [bs, 4], 'r_pos': [bs, 3]}`` from the simulator's
            current state (wxyz quaternion convention).
        sim_at_hml_idx : int
            Frame index in the HML sequence that corresponds to the
            simulator's current state, used for alignment.

        Returns
        -------
        torch.Tensor
            [bs, T_30fps, 24, 3] Isaac Lab body-state positions.
        """
        # 1. Unnormalize
        hml = (hml_norm * self.std.to(hml_norm.device) + self.mean.to(hml_norm.device)).float()

        # 2. Recover 22-joint SMPL positions from RIC representation
        hml_xyz = recover_from_ric(hml, 22)  # [bs, T, 22, 3]

        # 3. Recover root rotation and position for alignment
        r_rot_quat, r_pos = recover_root_rot_pos(hml)
        hml_transform_at_sim = {
            "r_rot": r_rot_quat[:, sim_at_hml_idx],
            "r_pos": r_pos[:, sim_at_hml_idx],
        }

        # 4. Two-step alignment via recon_data
        #    Step a: zero out — subtract HML root XZ, rotate by HML root rotation
        zeroed = self._align_to_recon_data(hml_xyz, hml_transform_at_sim, is_inverse=False)
        #    Step b: apply sim transform — rotate by inverse of sim root rotation, add sim root XZ
        aligned = self._align_to_recon_data(zeroed, recon_data, is_inverse=True)

        # 5. Add hand joints (22 → 24) and apply SMPL → Isaac coordinate transform
        aligned_24 = self.coord_transform.smpl_to_isaac(aligned)  # [bs, T, 24, 3]

        # 6. FPS convert 20 → 30
        pose_30fps = fps_convert(aligned_24, src_fps=20, tgt_fps=30)
        return pose_30fps

    def pose_to_hml(
        self,
        positions_isaac: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert Isaac Lab positions to normalized HML features.

        Parameters
        ----------
        positions_isaac : torch.Tensor
            [bs, T_30fps, 24, 3] Isaac Lab body-state positions.

        Returns
        -------
        hml_norm : torch.Tensor
            [bs, T_20fps, 263] normalized HumanML3D features.
        recon_data : dict
            ``{'r_rot': [bs, 4], 'r_pos': [bs, 3]}`` saved at frame [-2]
            for stitching.
        """
        # 1. Append extrapolated dummy frame (last frame is lost during feature extraction)
        next_frame = positions_isaac[:, [-1]] + (
            positions_isaac[:, [-1]] - positions_isaac[:, [-2]]
        )
        positions_ext = torch.cat([positions_isaac, next_frame], dim=1)

        # 2. FPS convert 30 → 20
        positions_20fps = fps_convert(positions_ext, src_fps=30, tgt_fps=20)

        # 3. Isaac → SMPL coordinate transform (24 → 22 joints, drops hands)
        positions_smpl = self.coord_transform.isaac_to_smpl(positions_20fps)  # [bs, T, 22, 3]

        # 4. Extract HML features via CLoSD's extract_features_t2m
        #    Returns [bs, T-1, 263] and recon_data dict
        hml_features, recon_data = extract_features_t2m(positions_smpl)

        # 5. Normalize
        hml_norm = (hml_features - self.mean.to(hml_features.device)) / self.std.to(
            hml_features.device
        )

        return hml_norm, recon_data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_to_recon_data(
        points: torch.Tensor,
        recon_data: Dict[str, torch.Tensor],
        is_inverse: bool = False,
    ) -> torch.Tensor:
        """Apply or undo a root transform (rotation + XZ translation).

        When ``is_inverse=False``: zero out by subtracting XZ then rotating.
        When ``is_inverse=True``: restore by inverse-rotating then adding XZ.

        Ported from CLoSD ``RepresentationHandler.align_to_recon_data``.
        """
        points = points.clone()
        r_rot = recon_data["r_rot"]
        r_pos = recon_data["r_pos"]

        if is_inverse:
            r_rot = qinv(r_rot)

        # Expand r_rot and r_pos to match points dimensions
        for _ in range(points.dim() - 2):
            r_rot = r_rot.unsqueeze(1)
            r_pos = r_pos.unsqueeze(1)

        new_rot_shape = r_rot.shape[:1] + points.shape[1:-1] + r_rot.shape[-1:]

        if is_inverse:
            # Inverse: rotate then translate
            points = qrot(r_rot.expand(new_rot_shape), points)
            points[..., [0, 2]] += r_pos[..., [0, 2]]
        else:
            # Forward: translate then rotate
            points[..., [0, 2]] -= r_pos[..., [0, 2]]
            points = qrot(r_rot.expand(new_rot_shape), points)

        return points
