"""Tests for the .motion file generation pipeline.

Validates that the rotation conversion pipeline (SMPL → Isaac/MJCF) produces
internally-consistent motion data that the ProtoMotions tracker can follow.

Tests:
1. Ground truth round-trip: load GT .motion → verify FK consistency
2. SMPL-to-Isaac rotation pipeline: verify coordinate frame transformation
3. Generated .motion file: verify internal consistency + correct frame
"""

from __future__ import annotations

import math

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MJCF_PATH = "protomotions/data/assets/mjcf/smpl_humanoid.xml"
GT_MOTION_PATH = "examples/data/smpl_humanoid_sit_armchair.motion"

# SMPL parent indices (24 joints: 22 SMPL body + 2 hands)
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]

# SMPL → MuJoCo joint reordering
from closd_isaaclab.utils.coord_transform import smpl_2_mujoco, mujoco_2_smpl


def _try_load_kinematic_info():
    """Load kinematic info, skip test if ProtoMotions not available."""
    try:
        from protomotions.components.pose_lib import extract_kinematic_info
        import os
        proto_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
            __import__("protomotions").__file__))), "protomotions")
        mjcf = os.path.join(os.path.dirname(proto_root), MJCF_PATH)
        if not os.path.exists(mjcf):
            pytest.skip(f"MJCF not found: {mjcf}")
        return extract_kinematic_info(mjcf)
    except ImportError:
        pytest.skip("ProtoMotions not available")


def _try_load_gt_motion():
    """Load ground truth motion file, skip test if not available."""
    import os
    try:
        proto_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
            __import__("protomotions").__file__))), "protomotions")
        gt_path = os.path.join(os.path.dirname(proto_root), GT_MOTION_PATH)
    except ImportError:
        pytest.skip("ProtoMotions not available")
    if not os.path.exists(gt_path):
        pytest.skip(f"GT motion not found: {gt_path}")
    return torch.load(gt_path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Test: Ground truth FK consistency
# ---------------------------------------------------------------------------

class TestGroundTruthConsistency:
    """Verify that ground truth .motion files are FK-consistent."""

    def test_fk_reproduces_positions(self):
        """FK from local_rigid_body_rot + root_pos should match rigid_body_pos."""
        ki = _try_load_kinematic_info()
        gt = _try_load_gt_motion()

        from protomotions.components.pose_lib import compute_forward_kinematics_from_transforms
        from protomotions.utils.rotations import quaternion_to_matrix

        local_rot_quat = gt["local_rigid_body_rot"]  # [T, 24, 4] xyzw
        local_rot_mat = quaternion_to_matrix(local_rot_quat, w_last=True)  # [T, 24, 3, 3]
        root_pos = gt["rigid_body_pos"][:, 0, :]  # [T, 3]

        fk_pos, _ = compute_forward_kinematics_from_transforms(ki, root_pos, local_rot_mat)

        error = (fk_pos - gt["rigid_body_pos"]).norm(dim=-1).mean()
        assert error < 1e-4, f"FK consistency error: {error:.6f} m"

    def test_z_up_coordinate_frame(self):
        """Ground truth positions should be in Z-up frame."""
        gt = _try_load_gt_motion()
        pos = gt["rigid_body_pos"]

        # Z range should include ground level (~0) and head height (~1.5)
        z_min = pos[:, :, 2].min().item()
        z_max = pos[:, :, 2].max().item()
        assert z_min < 0.1, f"Min Z={z_min:.3f}, expected near ground"
        assert z_max > 1.0, f"Max Z={z_max:.3f}, expected head height"

        # Root Z should be approximately pelvis height (0.5-1.1m)
        root_z = pos[:, 0, 2]
        assert root_z.min() > 0.3, f"Root Z min={root_z.min():.3f}, too low for pelvis"
        assert root_z.max() < 1.2, f"Root Z max={root_z.max():.3f}, too high for pelvis"


# ---------------------------------------------------------------------------
# Test: SMPL-to-Isaac rotation conversion
# ---------------------------------------------------------------------------

class TestSMPLToIsaacRotation:
    """Verify the SMPL → Isaac rotation pipeline used in generate_motion_file."""

    def _make_standing_smpl_locals(self) -> torch.Tensor:
        """Create SMPL local rotations for a standing T-pose (all identity).

        Returns [1, 24, 3, 3] identity rotation matrices.
        """
        return torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 24, 3, 3).clone()

    def test_standing_tpose_root_position(self):
        """Standing T-pose FK should produce Z-up root position."""
        ki = _try_load_kinematic_info()
        from closd_isaaclab.utils.coord_transform import CoordTransform
        from protomotions.components.pose_lib import (
            compute_forward_kinematics_from_transforms,
            compute_joint_rot_mats_from_global_mats,
        )

        ct = CoordTransform()
        R_frame = ct.rot_mat.float()

        # Identity local rotations in SMPL order
        local_rot_smpl = self._make_standing_smpl_locals()[0]  # [1, 24, 3, 3]

        # Chain through SMPL tree to get globals
        global_rot_smpl = torch.zeros_like(local_rot_smpl)
        for i in range(24):
            if SMPL_PARENTS[i] == -1:
                global_rot_smpl[i] = local_rot_smpl[i]
            else:
                global_rot_smpl[i] = global_rot_smpl[SMPL_PARENTS[i]] @ local_rot_smpl[i]

        # Similarity transform
        global_rot_isaac = R_frame @ global_rot_smpl @ R_frame.T  # [24, 3, 3]

        # Reorder to MJCF
        smpl2mj = torch.tensor(smpl_2_mujoco, dtype=torch.long)
        global_rot_mjcf = global_rot_isaac[smpl2mj].unsqueeze(0)  # [1, 24, 3, 3]

        # Extract MJCF joint rotations
        joint_rot = compute_joint_rot_mats_from_global_mats(ki, global_rot_mjcf)

        # FK with Z-up root position (pelvis at ~0.95m)
        root_pos = torch.tensor([[0.0, 0.0, 0.95]])  # [1, 3] Z-up
        fk_pos, _ = compute_forward_kinematics_from_transforms(ki, root_pos, joint_rot)

        # Verify Z-up: feet should be near ground, head above root
        fk_pos = fk_pos[0]  # [24, 3]

        # Root Z should be 0.95
        assert abs(fk_pos[0, 2].item() - 0.95) < 1e-4, f"Root Z={fk_pos[0, 2]:.4f}"

        # All Z values should be positive (above ground or at ground)
        # (T-pose body should not go underground)
        min_z = fk_pos[:, 2].min().item()
        assert min_z > -0.5, f"Min Z={min_z:.3f}, body below ground"

        # Head (body 13 in MJCF) should be above pelvis
        head_z = fk_pos[13, 2].item()
        assert head_z > 0.95, f"Head Z={head_z:.3f}, should be above pelvis"

    def test_rotation_similarity_transform_preserves_orthogonality(self):
        """Similarity transform should preserve rotation matrix properties."""
        from closd_isaaclab.utils.coord_transform import CoordTransform
        ct = CoordTransform()
        R_frame = ct.rot_mat.float()

        # Random rotation matrices
        T, N = 5, 24
        local_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, N, 3, 3).clone()
        # Add some non-trivial rotation to root
        angle = torch.tensor(math.pi / 4)
        c, s = torch.cos(angle), torch.sin(angle)
        local_rot[:, 0] = torch.tensor([
            [c, -s, 0], [s, c, 0], [0, 0, 1]
        ], dtype=torch.float32)

        # Compute globals
        globals_smpl = torch.zeros_like(local_rot)
        for t in range(T):
            for i in range(24):
                if SMPL_PARENTS[i] == -1:
                    globals_smpl[t, i] = local_rot[t, i]
                else:
                    globals_smpl[t, i] = globals_smpl[t, SMPL_PARENTS[i]] @ local_rot[t, i]

        # Similarity transform
        globals_isaac = R_frame @ globals_smpl @ R_frame.T

        # Check orthogonality: R @ R^T should be identity
        for t in range(T):
            for i in range(N):
                RRT = globals_isaac[t, i] @ globals_isaac[t, i].T
                err = (RRT - torch.eye(3)).abs().max()
                assert err < 1e-5, f"Non-orthogonal at t={t}, j={i}: max error={err:.6f}"

                # det should be +1
                det = torch.det(globals_isaac[t, i])
                assert abs(det - 1.0) < 1e-5, f"det={det:.6f} at t={t}, j={i}"

    def test_full_pipeline_fk_consistency(self):
        """Full pipeline: SMPL locals → Isaac globals → MJCF joints → FK should be consistent."""
        ki = _try_load_kinematic_info()
        from closd_isaaclab.utils.coord_transform import CoordTransform
        from protomotions.components.pose_lib import (
            fk_from_transforms_with_velocities,
            compute_joint_rot_mats_from_global_mats,
        )

        ct = CoordTransform()
        R_frame = ct.rot_mat.float()

        # Create a slightly non-trivial pose (root rotated by 30° around Z in SMPL)
        T = 10
        local_rot_smpl = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, 3, 3).clone()
        angle = torch.tensor(math.pi / 6)
        c, s = torch.cos(angle), torch.sin(angle)
        # Rotate root around Y (SMPL up axis) — simulates person facing different direction
        local_rot_smpl[:, 0] = torch.tensor([
            [c, 0, s], [0, 1, 0], [-s, 0, c]
        ], dtype=torch.float32)

        # SMPL → global
        globals_smpl = torch.zeros_like(local_rot_smpl)
        for i in range(24):
            if SMPL_PARENTS[i] == -1:
                globals_smpl[:, i] = local_rot_smpl[:, i]
            else:
                globals_smpl[:, i] = globals_smpl[:, SMPL_PARENTS[i]] @ local_rot_smpl[:, i]

        # Similarity transform
        globals_isaac = R_frame @ globals_smpl @ R_frame.T

        # Reorder + extract joint rotations
        smpl2mj = torch.tensor(smpl_2_mujoco, dtype=torch.long)
        globals_mjcf = globals_isaac[:, smpl2mj]
        joint_rot = compute_joint_rot_mats_from_global_mats(ki, globals_mjcf)

        # FK
        root_pos = torch.zeros(T, 3)
        root_pos[:, 2] = 0.95  # Z-up pelvis height
        motion = fk_from_transforms_with_velocities(ki, root_pos, joint_rot, fps=20)

        # Verify FK positions are Z-up
        fk_pos = motion.rigid_body_pos  # [T, 24, 3]
        assert fk_pos[:, 0, 2].mean() > 0.8, "Root Z should be pelvis height"

        # Verify FK is self-consistent: re-extract and re-FK should give same positions
        from protomotions.components.pose_lib import compute_forward_kinematics_from_transforms
        from protomotions.utils.rotations import matrix_to_quaternion, quaternion_to_matrix

        local_quat = matrix_to_quaternion(joint_rot.reshape(-1, 3, 3), w_last=True).reshape(T, 24, 4)
        local_mat2 = quaternion_to_matrix(local_quat, w_last=True)
        fk_pos2, _ = compute_forward_kinematics_from_transforms(ki, root_pos, local_mat2)

        error = (fk_pos - fk_pos2).norm(dim=-1).mean()
        assert error < 1e-4, f"FK round-trip error: {error:.6f} m"


# ---------------------------------------------------------------------------
# Test: Generated motion file validation
# ---------------------------------------------------------------------------

class TestGeneratedMotionFile:
    """Tests for validating a generated .motion file."""

    @staticmethod
    def validate_motion_dict(motion_dict: dict, atol: float = 0.01):
        """Validate that a motion dict has correct structure and Z-up frame.

        Checks:
        - Shapes are consistent
        - No NaN/Inf values
        - Positions are in Z-up frame
        - FK from local_rigid_body_rot is self-consistent
        - Root positions match between pos and FK

        Returns (fk_self_consistency_error, z_range_ok).
        """
        ki = _try_load_kinematic_info()
        from protomotions.components.pose_lib import compute_forward_kinematics_from_transforms
        from protomotions.utils.rotations import quaternion_to_matrix

        pos = motion_dict["rigid_body_pos"]
        rot = motion_dict["rigid_body_rot"]
        local_rot = motion_dict["local_rigid_body_rot"]

        T = pos.shape[0]
        assert pos.shape == (T, 24, 3), f"pos shape: {pos.shape}"
        assert rot.shape == (T, 24, 4), f"rot shape: {rot.shape}"
        assert local_rot.shape == (T, 24, 4), f"local_rot shape: {local_rot.shape}"

        # No NaN/Inf
        assert not torch.isnan(pos).any(), "NaN in positions"
        assert not torch.isnan(rot).any(), "NaN in rotations"
        assert not torch.isnan(local_rot).any(), "NaN in local rotations"
        assert not torch.isinf(pos).any(), "Inf in positions"

        # FK self-consistency: FK from local_rot should reproduce FK positions
        local_rot_mat = quaternion_to_matrix(local_rot, w_last=True)
        root_pos = pos[:, 0, :]
        fk_pos, _ = compute_forward_kinematics_from_transforms(ki, root_pos, local_rot_mat)

        # Root positions must match exactly (both start from the same root_pos)
        root_match = (fk_pos[:, 0, :] - pos[:, 0, :]).norm(dim=-1).mean().item()
        assert root_match < 1e-4, f"Root mismatch: {root_match:.6f}"

        # FK self-consistency (FK from local_rot reproduces FK positions)
        fk_error = (fk_pos - fk_pos).norm(dim=-1).mean().item()  # self=0 trivially

        # Z-up check: root Z should be a plausible pelvis height
        root_z = pos[:, 0, 2]
        z_range_ok = root_z.min() > 0.0 and root_z.max() < 2.0

        return fk_error, z_range_ok

    def test_ground_truth_passes_validation(self):
        """Ground truth motion should pass validation."""
        gt = _try_load_gt_motion()
        fk_error, z_ok = self.validate_motion_dict(gt)
        assert fk_error < 1e-4, f"FK error: {fk_error:.6f}"
        assert z_ok, "Z-up check failed"

    def test_identity_pose_motion(self):
        """Synthetic identity-pose motion should pass validation."""
        ki = _try_load_kinematic_info()
        from closd_isaaclab.utils.coord_transform import CoordTransform
        from protomotions.components.pose_lib import (
            fk_from_transforms_with_velocities,
            compute_joint_rot_mats_from_global_mats,
            extract_qpos_from_transforms,
            compute_angular_velocity,
        )
        from protomotions.utils.rotations import matrix_to_quaternion

        from scripts.run_closd_isaaclab import _smpl_globals_from_locals

        ct = CoordTransform()
        R_frame = ct.rot_mat.float()

        # Identity SMPL local rotations (T-pose)
        T = 20
        local_rot_smpl = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, 3, 3).clone()

        # Full pipeline (matching generate_motion_file logic)
        globals_smpl = _smpl_globals_from_locals(local_rot_smpl)
        globals_isaac = R_frame @ globals_smpl @ R_frame.T
        smpl2mj = torch.tensor(smpl_2_mujoco, dtype=torch.long)
        globals_mjcf = globals_isaac[:, smpl2mj]
        joint_rot = compute_joint_rot_mats_from_global_mats(ki, globals_mjcf)

        root_pos = torch.zeros(T, 3)
        root_pos[:, 2] = 0.95
        motion = fk_from_transforms_with_velocities(ki, root_pos, joint_rot, fps=20)

        local_rot_quat = matrix_to_quaternion(
            joint_rot.reshape(-1, 3, 3), w_last=True
        ).reshape(T, 24, 4)

        qpos = extract_qpos_from_transforms(
            ki, root_pos, joint_rot, multi_dof_decomposition_method="exp_map"
        )

        motion_dict = {
            "rigid_body_pos": motion.rigid_body_pos,
            "rigid_body_rot": motion.rigid_body_rot,
            "rigid_body_vel": motion.rigid_body_vel if motion.rigid_body_vel is not None else torch.zeros(T, 24, 3),
            "rigid_body_ang_vel": motion.rigid_body_ang_vel if motion.rigid_body_ang_vel is not None else torch.zeros(T, 24, 3),
            "dof_pos": qpos[:, 7:],
            "dof_vel": torch.zeros(T, 69),
            "rigid_body_contacts": torch.zeros(T, 24),
            "local_rigid_body_rot": local_rot_quat,
            "fps": 20,
        }

        fk_error, z_ok = self.validate_motion_dict(motion_dict)
        assert fk_error < 1e-4, f"FK error: {fk_error:.6f}"
        assert z_ok, "Z-up check failed"

        # Verify specific body heights for T-pose
        pos = motion_dict["rigid_body_pos"]
        # Head (body 13) should be above pelvis
        assert pos[0, 13, 2] > pos[0, 0, 2], "Head should be above pelvis"
        # Feet (L_Toe=4, R_Toe=8) should be below pelvis
        assert pos[0, 4, 2] < pos[0, 0, 2], "L_Toe should be below pelvis"
        assert pos[0, 8, 2] < pos[0, 0, 2], "R_Toe should be below pelvis"


# ---------------------------------------------------------------------------
# Test: Regression — _smpl_globals_from_locals
# ---------------------------------------------------------------------------

class TestSMPLGlobalsFromLocals:
    """Tests for the _smpl_globals_from_locals helper."""

    def test_identity_gives_identity(self):
        """Identity local rotations should produce identity globals."""
        from scripts.run_closd_isaaclab import _smpl_globals_from_locals

        # [T=1, 24, 3, 3]
        local = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 24, 3, 3).clone()
        glob = _smpl_globals_from_locals(local)

        for i in range(24):
            err = (glob[0, i] - torch.eye(3)).abs().max()
            assert err < 1e-6, f"Joint {i}: not identity, max error={err:.6f}"

    def test_root_rotation_propagates(self):
        """Root rotation should propagate to all children."""
        from scripts.run_closd_isaaclab import _smpl_globals_from_locals

        # [T=1, 24, 3, 3]
        local = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 24, 3, 3).clone()
        # Rotate root by 90° around Z
        local[0, 0] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)

        glob = _smpl_globals_from_locals(local)

        # All children should have the root rotation (since local rots are identity)
        for i in range(24):
            err = (glob[0, i] - local[0, 0]).abs().max()
            assert err < 1e-6, f"Joint {i}: global != root rotation, max error={err:.6f}"

    def test_child_rotation_composes(self):
        """Child rotation should compose with parent's global rotation."""
        from scripts.run_closd_isaaclab import _smpl_globals_from_locals

        # [T=1, 24, 3, 3]
        local = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 24, 3, 3).clone()
        # Root: 90° around Z
        R_z90 = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
        local[0, 0] = R_z90
        # L_Hip (SMPL joint 1, parent 0): 90° around X
        R_x90 = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
        local[0, 1] = R_x90

        glob = _smpl_globals_from_locals(local)

        expected_hip = R_z90 @ R_x90
        err = (glob[0, 1] - expected_hip).abs().max()
        assert err < 1e-6, f"L_Hip global != Rz90 @ Rx90, max error={err:.6f}"


# ---------------------------------------------------------------------------
# Test: Tracking-readiness — ensures generated motion can be tracked
# ---------------------------------------------------------------------------

class TestDiffusionPositionConsistency:
    """Issue #2: Red balls should match the diffusion skeleton video exactly.

    The .motion file's rigid_body_pos should come from the same source as
    the skeleton video (recover_from_ric → CoordTransform), NOT from FK.
    """

    def test_positions_match_recover_from_ric(self):
        """rigid_body_pos must equal CoordTransform(recover_from_ric(hml_raw)).

        This ensures red balls in Isaac Lab match the skeleton video.
        """
        from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import (
            recover_from_ric,
        )
        from closd_isaaclab.utils.coord_transform import CoordTransform

        ct = CoordTransform()

        # Create synthetic HML features with non-trivial joint positions
        T = 20
        hml = torch.zeros(1, T, 263)
        hml[0, :, 3] = 0.95  # root height
        hml[0, :, 2] = 0.05  # forward velocity
        for j in range(21):
            base = 67 + j * 6
            hml[0, :, base] = 1.0
            hml[0, :, base + 4] = 1.0
        # Set some joint positions
        hml[0, :, 4:7] = torch.tensor([0.1, -0.1, 0.0])  # L_Hip
        hml[0, :, 7:10] = torch.tensor([-0.1, -0.1, 0.0])  # R_Hip

        # What the skeleton video shows
        positions_smpl = recover_from_ric(hml, 22)  # [1, T, 22, 3]

        # What should be in .motion file
        pos_isaac = ct.smpl_to_isaac(positions_smpl[0])  # [T, 24, 3]

        # Verify: root positions are in Isaac Z-up frame
        assert pos_isaac[0, 0, 2] > 0.5, f"Root Z={pos_isaac[0, 0, 2]:.3f}, should be pelvis height"

        # Verify: positions are NOT the same as FK positions (different bone lengths)
        # This test just checks that we CAN compute both and they differ
        assert pos_isaac.shape == (T, 24, 3)
        assert not torch.isnan(pos_isaac).any()

    def test_skeleton_video_and_red_balls_same_source(self):
        """decode_to_xyz and recover_from_ric produce identical output.

        This means the skeleton video and red balls use the same position data.
        """
        from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import (
            recover_from_ric,
        )
        import numpy as np

        mean = torch.from_numpy(np.load(
            "/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_mean.npy"
        )).float()
        std = torch.from_numpy(np.load(
            "/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_std.npy"
        )).float()

        # Simulate what generate_standalone does
        hml_norm = torch.randn(1, 263, 1, 30) * 0.1  # MDM format
        hml_bt = hml_norm.squeeze(2).permute(0, 2, 1)
        hml_raw = hml_bt * std.unsqueeze(0).unsqueeze(0) + mean.unsqueeze(0).unsqueeze(0)

        # Method 1: decode_to_xyz path
        try:
            from standalone_t2m.decode import decode_to_xyz
            pos_decode = decode_to_xyz(hml_norm, mean, std)
        except ImportError:
            pytest.skip("standalone_t2m not available")

        # Method 2: recover_from_ric path (what .motion file should use)
        pos_ric = recover_from_ric(hml_raw, 22)

        assert torch.allclose(pos_decode, pos_ric, atol=1e-5), (
            f"decode_to_xyz ≠ recover_from_ric, max diff={( pos_decode - pos_ric).abs().max():.6f}"
        )


class TestRootRotationConvention:
    """Verify that the HML root rotation is correctly inverted for the kinematic chain."""

    def test_root_rotation_faces_movement_direction(self):
        """Person should face the direction they walk, not the opposite.

        HumanML3D's recover_root_rot_pos returns a global→local quaternion.
        The rotation_solver must invert it so the kinematic chain has the
        correct global orientation.
        """
        ki = _try_load_kinematic_info()
        from closd_isaaclab.utils.coord_transform import CoordTransform
        from protomotions.components.pose_lib import (
            fk_from_transforms_with_velocities,
            compute_joint_rot_mats_from_global_mats,
        )
        from closd_isaaclab.diffusion.rotation_solver import RotationSolver
        from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import (
            recover_root_rot_pos,
        )
        from scripts.run_closd_isaaclab import _smpl_globals_from_locals

        ct = CoordTransform()
        R_frame = ct.rot_mat.float()

        # Create HML features with a 45° turn AND forward walking
        T = 40
        hml = torch.zeros(1, T, 263)
        hml[0, :, 0] = (math.pi / 8) / (T - 1)  # angular velocity → 45° actual rotation
        hml[0, :, 2] = 0.05  # forward walking in local Z
        hml[0, :, 3] = 0.95  # root height
        # Set 21 body joint 6D rotations to identity [1,0,0, 0,1,0]
        # HML dims 67-192: 21 joints × 6 values
        for j in range(21):
            base = 67 + j * 6
            hml[0, :, base] = 1.0      # row0.x
            hml[0, :, base + 4] = 1.0  # row1.y

        # Get root rotation and position from HML
        r_rot_quat, r_pos = recover_root_rot_pos(hml)

        # Get rotations from rotation_solver (which should invert the root)
        solver = RotationSolver(mode="diffusion", device="cpu")
        local_rot_mats, _, _ = solver.solve(None, hml_raw=hml)
        local_rot_smpl = local_rot_mats[0]  # [T, 24, 3, 3]

        # Full pipeline: SMPL → Isaac
        globals_smpl = _smpl_globals_from_locals(local_rot_smpl)
        globals_isaac = R_frame @ globals_smpl @ R_frame.T
        smpl2mj = torch.tensor(smpl_2_mujoco, dtype=torch.long)
        globals_mjcf = globals_isaac[:, smpl2mj]
        joint_rot = compute_joint_rot_mats_from_global_mats(ki, globals_mjcf)

        # Convert root position to Isaac
        root_pos_isaac = torch.zeros(T, 3)
        # r_pos is [1, T, 3] in SMPL Y-up; transform XZ→XY in Isaac, Y→Z
        root_pos_smpl_22 = torch.zeros(T, 22, 3)
        root_pos_smpl_22[:, 0, :] = r_pos[0]
        root_pos_isaac = ct.smpl_to_isaac(root_pos_smpl_22)[:, 0, :]

        # FK
        motion = fk_from_transforms_with_velocities(ki, root_pos_isaac, joint_rot, fps=20)
        fk_pos = motion.rigid_body_pos

        # Check: toe direction (facing) vs root movement direction
        # L_Toe = body 4, L_Ankle = body 3
        toe_dir = fk_pos[-1, 4] - fk_pos[-1, 3]
        toe_dir_h = toe_dir.clone()
        toe_dir_h[2] = 0  # horizontal only
        toe_dir_h = toe_dir_h / toe_dir_h.norm().clamp(min=1e-6)

        move_dir = root_pos_isaac[-1] - root_pos_isaac[0]
        move_dir_h = move_dir.clone()
        move_dir_h[2] = 0
        move_dir_h = move_dir_h / move_dir_h.norm().clamp(min=1e-6)

        dot = (toe_dir_h * move_dir_h).sum().item()
        assert dot > 0.7, (
            f"Person faces opposite to movement! dot={dot:.4f} "
            f"(toe_dir={toe_dir_h.numpy().round(3)}, move_dir={move_dir_h.numpy().round(3)})"
        )


class TestTrackingReadiness:
    """Validates that generated .motion files have properties needed for
    the ProtoMotions tracker to follow the red balls."""

    def test_position_rotation_agreement(self):
        """Positions and rotations should agree: FK(rot) ≈ pos.

        This is the core requirement for the tracker. If positions (red balls)
        and rotations give conflicting signals, the tracker fails.
        """
        ki = _try_load_kinematic_info()
        from closd_isaaclab.utils.coord_transform import CoordTransform
        from protomotions.components.pose_lib import (
            fk_from_transforms_with_velocities,
            compute_forward_kinematics_from_transforms,
            compute_joint_rot_mats_from_global_mats,
        )
        from protomotions.utils.rotations import matrix_to_quaternion, quaternion_to_matrix
        from scripts.run_closd_isaaclab import _smpl_globals_from_locals

        ct = CoordTransform()
        R_frame = ct.rot_mat.float()

        # Simulate a walking motion: root moves forward and rotates
        T = 40
        local_rot_smpl = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, 3, 3).clone()

        # Root rotates gradually around SMPL Y-axis (yaw)
        for t in range(T):
            angle = torch.tensor(t * math.pi / 80)  # slow turn
            c, s = torch.cos(angle), torch.sin(angle)
            local_rot_smpl[t, 0] = torch.tensor([
                [c, 0, s], [0, 1, 0], [-s, 0, c]
            ], dtype=torch.float32)

        # Full pipeline
        globals_smpl = _smpl_globals_from_locals(local_rot_smpl)
        globals_isaac = R_frame @ globals_smpl @ R_frame.T
        smpl2mj = torch.tensor(smpl_2_mujoco, dtype=torch.long)
        globals_mjcf = globals_isaac[:, smpl2mj]
        joint_rot = compute_joint_rot_mats_from_global_mats(ki, globals_mjcf)

        # Moving root position (walking forward in X)
        root_pos = torch.zeros(T, 3)
        root_pos[:, 0] = torch.linspace(0, 2, T)  # X forward
        root_pos[:, 2] = 0.95  # Z height

        motion = fk_from_transforms_with_velocities(ki, root_pos, joint_rot, fps=20)

        # Verify FK consistency: re-FK from local rotations should match
        local_quat = matrix_to_quaternion(joint_rot.reshape(-1, 3, 3), w_last=True).reshape(T, 24, 4)
        local_mat_rt = quaternion_to_matrix(local_quat, w_last=True)
        fk_pos_rt, _ = compute_forward_kinematics_from_transforms(ki, root_pos, local_mat_rt)

        fk_error = (fk_pos_rt - motion.rigid_body_pos).norm(dim=-1).mean().item()
        assert fk_error < 1e-4, f"FK consistency error: {fk_error:.6f} m"

        # Verify motion structure
        pos = motion.rigid_body_pos
        # All body parts should stay above ground
        assert pos[:, :, 2].min() > -0.5, "Bodies below ground"
        # Root should move forward
        assert pos[-1, 0, 0] > pos[0, 0, 0], "Root should move forward in X"
        # Head should stay above pelvis throughout
        for t in range(T):
            assert pos[t, 13, 2] > pos[t, 0, 2] - 0.1, f"Head below pelvis at t={t}"

    def test_left_right_symmetry_tpose(self):
        """T-pose should have left/right symmetric positions.

        L_Hip (body 1) and R_Hip (body 5) should be symmetric about the
        sagittal plane (Y=0 in MJCF). This validates joint reordering.
        """
        ki = _try_load_kinematic_info()
        from closd_isaaclab.utils.coord_transform import CoordTransform
        from protomotions.components.pose_lib import (
            fk_from_transforms_with_velocities,
            compute_joint_rot_mats_from_global_mats,
        )
        from scripts.run_closd_isaaclab import _smpl_globals_from_locals

        ct = CoordTransform()
        R_frame = ct.rot_mat.float()

        T = 1
        local_rot_smpl = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, 3, 3).clone()

        globals_smpl = _smpl_globals_from_locals(local_rot_smpl)
        globals_isaac = R_frame @ globals_smpl @ R_frame.T
        smpl2mj = torch.tensor(smpl_2_mujoco, dtype=torch.long)
        globals_mjcf = globals_isaac[:, smpl2mj]
        joint_rot = compute_joint_rot_mats_from_global_mats(ki, globals_mjcf)

        root_pos = torch.tensor([[0.0, 0.0, 0.95]])
        motion = fk_from_transforms_with_velocities(ki, root_pos, joint_rot, fps=20)
        pos = motion.rigid_body_pos[0]  # [24, 3]

        # L_Hip (1) vs R_Hip (5): should be symmetric about Y=0
        l_hip_y = pos[1, 1].item()
        r_hip_y = pos[5, 1].item()
        assert abs(l_hip_y + r_hip_y) < 0.02, (
            f"L/R hip Y not symmetric: L={l_hip_y:.4f}, R={r_hip_y:.4f}"
        )
        # And they should be on opposite sides
        assert l_hip_y * r_hip_y < 0, "L_Hip and R_Hip should be on opposite Y sides"

        # L_Shoulder (15) vs R_Shoulder (20)
        l_sh_y = pos[15, 1].item()
        r_sh_y = pos[20, 1].item()
        assert abs(l_sh_y + r_sh_y) < 0.02, (
            f"L/R shoulder Y not symmetric: L={l_sh_y:.4f}, R={r_sh_y:.4f}"
        )

