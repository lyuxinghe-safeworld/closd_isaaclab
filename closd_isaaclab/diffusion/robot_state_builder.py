"""RobotStateBuilder: construct RobotState-compatible data from diffusion output.

ProtoMotions' MimicControl expects reference motion as a RobotState with:
  - rigid_body_pos:      [bs, T, num_bodies, 3]  world positions
  - rigid_body_vel:      [bs, T, num_bodies, 3]  linear velocities
  - dof_pos:             [bs, T, nq]              joint positions (or None)
  - dof_vel:             [bs, T, nq]              joint velocities (or None)
  - rigid_body_contacts: [bs, T, num_bodies]      binary contact flags

Contact body mapping (HumanML3D dims 259-262 -> MuJoCo body indices):
  HML index 0 (dim 259) -> L_Ankle  (MuJoCo body index 3)
  HML index 1 (dim 260) -> L_Toe    (MuJoCo body index 4)
  HML index 2 (dim 261) -> R_Ankle  (MuJoCo body index 7)
  HML index 3 (dim 262) -> R_Toe    (MuJoCo body index 8)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor


# HML foot contact index -> MuJoCo body index mapping
_HML_TO_MUJOCO_CONTACT: List[int] = [3, 4, 7, 8]


class RobotStateBuilder:
    """Build and cache RobotState-compatible motion data from diffusion output.

    Parameters
    ----------
    dt : float
        Time step between frames (seconds). Default: 1/30.
    rotation_solver : RotationSolver, optional
        If provided, used to extract dof_pos from hml_raw rotations.
    num_bodies : int
        Number of rigid bodies (default 24 for SMPL-based humanoid).
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        rotation_solver=None,
        num_bodies: int = 24,
    ) -> None:
        self.dt = dt
        self.rotation_solver = rotation_solver
        self.num_bodies = num_bodies

        # Cached fields (set by build())
        self._positions: Optional[Tensor] = None
        self._velocities: Optional[Tensor] = None
        self._dof_pos: Optional[Tensor] = None
        self._dof_vel: Optional[Tensor] = None
        self._contacts: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        positions: Tensor,
        hml_raw: Optional[Tensor] = None,
        root_rot_wxyz: Optional[Tensor] = None,
    ) -> None:
        """Build and cache motion data from a new planning horizon.

        Parameters
        ----------
        positions : Tensor
            [bs, T, 24, 3] joint positions in Isaac Lab space (MuJoCo joint order).
        hml_raw : Tensor, optional
            [bs, T_20fps, 263] raw HumanML3D features. Used for rotation extraction
            and foot contact parsing. T_20fps may differ from T if fps differ.
        root_rot_wxyz : Tensor, optional
            [bs, T, 4] root rotation quaternions (wxyz). Passed to rotation_solver
            if provided.
        """
        # 1. Store positions
        self._positions = positions  # [bs, T, num_bodies, 3]

        # 2. Compute body velocities via finite differences
        self._velocities = self._compute_velocities(positions)  # [bs, T, num_bodies, 3]

        # 3. Extract rotations and dof_pos using ProtoMotions' FK pipeline
        #    This is the same pipeline used by convert_amass_to_proto.py to create
        #    .motion files. fk_from_transforms_with_velocities() is the single source
        #    of truth that ensures global rotations, positions, velocities, and dof_pos
        #    are all physically consistent.
        self._dof_pos = None
        self._dof_vel = None
        self._rotations = None      # [bs, T, num_bodies, 4] xyzw quaternions
        self._ang_velocities = None  # [bs, T, num_bodies, 3]
        if self.rotation_solver is not None and hml_raw is not None:
            from closd_isaaclab.utils.fps_convert import fps_convert

            # Solver works at 20fps (HML rate)
            local_rot_mats, _, _ = self.rotation_solver.solve(
                None, hml_raw=hml_raw, root_rot_wxyz=root_rot_wxyz
            )

            T_30 = positions.shape[1]
            bs = positions.shape[0]

            if local_rot_mats is not None and self.rotation_solver.kinematic_info is not None:
                from protomotions.components.pose_lib import (
                    fk_from_transforms_with_velocities,
                    extract_qpos_from_transforms,
                    compute_joint_rot_mats_from_global_mats,
                    compute_angular_velocity,
                )
                from protomotions.utils.rotations import matrix_to_quaternion

                bs_r, T_20 = local_rot_mats.shape[:2]
                ki = self.rotation_solver.kinematic_info

                # Get root positions at 20fps from our 30fps positions
                pos_20 = fps_convert(positions, src_fps=30, tgt_fps=20)
                if pos_20.shape[1] > T_20:
                    pos_20 = pos_20[:, :T_20]
                elif pos_20.shape[1] < T_20:
                    pad_p = pos_20[:, -1:].expand(-1, T_20 - pos_20.shape[1], -1, -1)
                    pos_20 = torch.cat([pos_20, pad_p], dim=1)

                # Process each batch element through ProtoMotions' FK pipeline.
                # Use FK-derived positions for consistency: the tracker needs
                # positions, rotations, and dof_pos that are all physically
                # consistent. FK ensures this by deriving everything from
                # the same local rotation matrices.
                all_pos_30 = []
                all_rot_30 = []
                all_vel_30 = []
                all_dof_pos_30 = []
                all_dof_vel_30 = []
                all_ang_vel_30 = []

                for b in range(bs):
                    root_pos_b = pos_20[b, :, 0, :]  # [T_20, 3]
                    local_rots_b = local_rot_mats[b]  # [T_20, 24, 3, 3]

                    # ProtoMotions' authoritative FK pipeline
                    motion_state = fk_from_transforms_with_velocities(
                        ki, root_pos_b, local_rots_b, fps=20, compute_velocities=True
                    )

                    # FK-derived positions (consistent with rotations)
                    fk_pos_20 = motion_state.rigid_body_pos  # [T_20, 24, 3]
                    global_rot_20 = motion_state.rigid_body_rot  # [T_20, 24, 4] xyzw
                    vel_20 = motion_state.rigid_body_vel  # [T_20, 24, 3]
                    ang_vel_20 = motion_state.rigid_body_ang_vel  # [T_20, 24, 3]

                    # DOF positions from local rotations
                    qpos = extract_qpos_from_transforms(
                        ki, root_pos_b, local_rots_b,
                        multi_dof_decomposition_method="exp_map",
                    )
                    dof_pos_20 = qpos[:, 7:]  # [T_20, 69]

                    # DOF velocities
                    dof_vel_20 = compute_angular_velocity(
                        local_rots_b.unsqueeze(0), fps=20
                    )
                    dof_vel_20 = dof_vel_20[0, :, 1:, :].reshape(T_20, -1)

                    # Upsample everything from 20fps to 30fps
                    def _upsample_and_pad(t, T_target):
                        t30 = fps_convert(t.unsqueeze(0), 20, 30)[0]
                        if t30.shape[0] > T_target:
                            return t30[:T_target]
                        elif t30.shape[0] < T_target:
                            pad = t30[-1:].expand(T_target - t30.shape[0], *t30.shape[1:])
                            return torch.cat([t30, pad], dim=0)
                        return t30

                    all_pos_30.append(_upsample_and_pad(fk_pos_20, T_30))
                    rot_30 = _upsample_and_pad(global_rot_20, T_30)
                    rot_30 = rot_30 / (rot_30.norm(dim=-1, keepdim=True) + 1e-8)
                    all_rot_30.append(rot_30)
                    all_vel_30.append(_upsample_and_pad(vel_20, T_30))
                    all_dof_pos_30.append(_upsample_and_pad(dof_pos_20, T_30))
                    all_dof_vel_30.append(_upsample_and_pad(dof_vel_20, T_30))
                    all_ang_vel_30.append(_upsample_and_pad(ang_vel_20, T_30))

                # Override positions with FK-derived (consistent) positions
                self._positions = torch.stack(all_pos_30).to(positions.device)
                self._velocities = torch.stack(all_vel_30).to(positions.device)
                self._rotations = torch.stack(all_rot_30).to(positions.device)
                self._dof_pos = torch.stack(all_dof_pos_30).to(positions.device)
                self._dof_vel = torch.stack(all_dof_vel_30).to(positions.device)
                self._ang_velocities = torch.stack(all_ang_vel_30).to(positions.device)

        # 4. Extract foot contacts from hml_raw
        self._contacts = None
        if hml_raw is not None:
            self._contacts = self._extract_contacts(hml_raw)  # [bs, T, num_bodies]

    def get_state_at_frames(self, frame_indices: List[int]) -> Dict[str, Optional[Tensor]]:
        """Retrieve cached state at specific frame indices.

        Parameters
        ----------
        frame_indices : list of int
            Indices into the time dimension of the cached data.

        Returns
        -------
        dict with keys:
            rigid_body_pos      : [bs, len(frame_indices), num_bodies, 3]
            rigid_body_vel      : [bs, len(frame_indices), num_bodies, 3]
            dof_pos             : [bs, len(frame_indices), nq] or None
            dof_vel             : [bs, len(frame_indices), nq] or None
            rigid_body_contacts : [bs, len(frame_indices), num_bodies] or None
        """
        idx = frame_indices

        pos = self._positions[:, idx, :, :] if self._positions is not None else None
        vel = self._velocities[:, idx, :, :] if self._velocities is not None else None
        dof_pos = self._dof_pos[:, idx, :] if self._dof_pos is not None else None
        dof_vel = self._dof_vel[:, idx, :] if self._dof_vel is not None else None

        if self._contacts is not None:
            contacts = self._contacts[:, idx, :]
        else:
            # Return zero contacts if none were extracted
            if pos is not None:
                bs = pos.shape[0]
                T = len(idx)
                contacts = torch.zeros(bs, T, self.num_bodies, dtype=pos.dtype, device=pos.device)
            else:
                contacts = None

        rot = self._rotations[:, idx, :, :] if self._rotations is not None else None
        ang_vel = self._ang_velocities[:, idx, :, :] if self._ang_velocities is not None else None

        return {
            "rigid_body_pos": pos,
            "rigid_body_rot": rot,
            "rigid_body_vel": vel,
            "rigid_body_ang_vel": ang_vel,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "rigid_body_contacts": contacts,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_velocities(self, positions: Tensor) -> Tensor:
        """Compute per-body velocities via finite differencing.

        Uses central differencing for interior frames:
            vel[t] = (pos[t+1] - pos[t-1]) / (2 * dt)

        Uses forward difference at t=0:
            vel[0] = (pos[1] - pos[0]) / dt

        Uses backward difference at t=T-1:
            vel[-1] = (pos[-1] - pos[-2]) / dt

        Parameters
        ----------
        positions : Tensor
            [bs, T, num_bodies, 3]

        Returns
        -------
        Tensor
            [bs, T, num_bodies, 3] velocities.
        """
        bs, T, nb, d = positions.shape
        vel = torch.zeros_like(positions)

        if T == 1:
            return vel  # single frame -> zero velocity

        # Forward difference at t=0
        vel[:, 0, :, :] = (positions[:, 1, :, :] - positions[:, 0, :, :]) / self.dt

        # Backward difference at t=T-1
        vel[:, -1, :, :] = (positions[:, -1, :, :] - positions[:, -2, :, :]) / self.dt

        # Central differencing for interior frames
        if T > 2:
            vel[:, 1:-1, :, :] = (
                positions[:, 2:, :, :] - positions[:, :-2, :, :]
            ) / (2.0 * self.dt)

        return vel

    def _compute_velocities_1d(self, data: Tensor) -> Tensor:
        """Compute velocities for 1D data (e.g. dof_pos) via finite differencing.

        Same boundary treatment as _compute_velocities.

        Parameters
        ----------
        data : Tensor
            [bs, T, D] — e.g. dof_pos

        Returns
        -------
        Tensor
            [bs, T, D] velocities.
        """
        bs, T, D = data.shape
        vel = torch.zeros_like(data)

        if T == 1:
            return vel

        vel[:, 0, :] = (data[:, 1, :] - data[:, 0, :]) / self.dt
        vel[:, -1, :] = (data[:, -1, :] - data[:, -2, :]) / self.dt

        if T > 2:
            vel[:, 1:-1, :] = (data[:, 2:, :] - data[:, :-2, :]) / (2.0 * self.dt)

        return vel

    def _extract_contacts(self, hml_raw: Tensor) -> Tensor:
        """Map HML foot contacts (dims 259-262) to a [bs, T, num_bodies] contact tensor.

        HML dims 259-262 are binary foot contact flags for:
            dim 259 -> L_Ankle -> MuJoCo body 3
            dim 260 -> L_Toe   -> MuJoCo body 4
            dim 261 -> R_Ankle -> MuJoCo body 7
            dim 262 -> R_Toe   -> MuJoCo body 8

        Parameters
        ----------
        hml_raw : Tensor
            [bs, T_20fps, 263] raw HumanML3D features.

        Returns
        -------
        Tensor
            [bs, T, num_bodies] float contact tensor, where T matches the
            time dimension of the cached positions.  If T != T_20fps, the
            contact signal is resampled (nearest-neighbour).
        """
        bs, T_20fps, _ = hml_raw.shape
        # Extract the 4 foot contact flags: [bs, T_20fps, 4]
        foot_contacts = hml_raw[..., 259:263]  # [bs, T_20fps, 4]

        # Target T from cached positions
        T_target = self._positions.shape[1] if self._positions is not None else T_20fps

        # Build full contact tensor [bs, T_20fps, num_bodies]
        contacts_full = torch.zeros(
            bs, T_20fps, self.num_bodies,
            dtype=hml_raw.dtype,
            device=hml_raw.device,
        )
        for hml_idx, body_idx in enumerate(_HML_TO_MUJOCO_CONTACT):
            contacts_full[:, :, body_idx] = foot_contacts[:, :, hml_idx]

        # Resample to T_target if needed (nearest neighbour along time axis)
        if T_20fps != T_target:
            # contacts_full: [bs, T_20fps, num_bodies]
            # We need to resample the time dimension
            contacts_full = contacts_full.permute(0, 2, 1)  # [bs, num_bodies, T_20fps]
            contacts_full = torch.nn.functional.interpolate(
                contacts_full.float(),
                size=T_target,
                mode="nearest",
            )  # [bs, num_bodies, T_target]
            contacts_full = contacts_full.permute(0, 2, 1)  # [bs, T_target, num_bodies]

        return contacts_full
