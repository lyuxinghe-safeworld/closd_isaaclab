"""Integration tests for the full pipeline data flow.

Tests the exact code path that run_closd_isaaclab.py exercises:
  diffusion output -> coord transform -> robot_state_builder -> closd_motion_lib
"""
import sys
import torch
import pytest

sys.path.insert(0, "/home/lyuxinghe/code/CLoSD")
sys.path.insert(0, "/home/lyuxinghe/code/ProtoMotions")
sys.path.insert(0, "/home/lyuxinghe/code/CLoSD_t2m_standalone")


class TestBuilderWith20fps30fpsMismatch:
    """The rotation solver works at 20fps (HML) while positions are at 30fps."""

    def test_build_with_mismatched_fps(self):
        """build() should handle 30fps positions + 20fps hml_raw without error."""
        from closd_isaaclab.diffusion.robot_state_builder import RobotStateBuilder
        from closd_isaaclab.diffusion.rotation_solver import RotationSolver

        solver = RotationSolver(mode="diffusion", device="cpu")
        builder = RobotStateBuilder(dt=1.0 / 30.0, rotation_solver=solver)

        # 30fps positions (270 frames) + 20fps HML (180 frames)
        positions_30 = torch.randn(1, 270, 24, 3)
        hml_20 = torch.randn(1, 180, 263)

        # This was crashing with shape mismatch
        builder.build(positions_30, hml_raw=hml_20)

        assert builder._positions.shape == (1, 270, 24, 3)
        assert builder._velocities.shape == (1, 270, 24, 3)

    def test_build_with_kinematic_info(self):
        """build() should produce dof_pos when kinematic_info is provided."""
        from closd_isaaclab.diffusion.robot_state_builder import RobotStateBuilder
        from closd_isaaclab.diffusion.rotation_solver import RotationSolver

        try:
            from protomotions.components.pose_lib import extract_kinematic_info
        except ImportError:
            pytest.skip("protomotions not available")

        import os
        mjcf = os.path.expanduser("~/code/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml")
        if not os.path.exists(mjcf):
            pytest.skip("MJCF not found")

        ki = extract_kinematic_info(mjcf)
        solver = RotationSolver(mode="diffusion", device="cpu", kinematic_info=ki)
        builder = RobotStateBuilder(dt=1.0 / 30.0, rotation_solver=solver)

        positions_30 = torch.randn(1, 270, 24, 3)
        hml_20 = torch.randn(1, 180, 263)

        builder.build(positions_30, hml_raw=hml_20)

        assert builder._dof_pos is not None, "dof_pos should be computed with kinematic_info"
        assert builder._dof_pos.shape[0] == 1
        assert builder._dof_pos.shape[1] == 270  # upsampled to 30fps
        assert builder._dof_pos.shape[2] == 69    # 23 joints * 3 DOF

        assert builder._rotations is not None
        assert builder._rotations.shape == (1, 270, 24, 4)  # xyzw quats

    def test_dof_vel_computed(self):
        """dof_vel should be computed from dof_pos via finite differencing."""
        from closd_isaaclab.diffusion.robot_state_builder import RobotStateBuilder
        from closd_isaaclab.diffusion.rotation_solver import RotationSolver

        try:
            from protomotions.components.pose_lib import extract_kinematic_info
        except ImportError:
            pytest.skip("protomotions not available")

        import os
        mjcf = os.path.expanduser("~/code/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml")
        if not os.path.exists(mjcf):
            pytest.skip("MJCF not found")

        ki = extract_kinematic_info(mjcf)
        solver = RotationSolver(mode="diffusion", device="cpu", kinematic_info=ki)
        builder = RobotStateBuilder(dt=1.0 / 30.0, rotation_solver=solver)

        positions_30 = torch.randn(1, 270, 24, 3)
        hml_20 = torch.randn(1, 180, 263)

        builder.build(positions_30, hml_raw=hml_20)

        assert builder._dof_vel is not None
        assert builder._dof_vel.shape == builder._dof_pos.shape


class TestMotionLibReturnsAllFields:
    """CLoSDMotionLib.get_motion_state must return all fields MimicControl needs."""

    def test_all_fields_present(self):
        """RobotState must have pos, rot, vel, ang_vel, dof_pos, dof_vel."""
        from closd_isaaclab.diffusion.robot_state_builder import RobotStateBuilder
        from closd_isaaclab.diffusion.rotation_solver import RotationSolver
        from closd_isaaclab.integration.closd_motion_lib import CLoSDMotionLib

        try:
            from protomotions.components.pose_lib import extract_kinematic_info
        except ImportError:
            pytest.skip("protomotions not available")

        import os
        mjcf = os.path.expanduser("~/code/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml")
        if not os.path.exists(mjcf):
            pytest.skip("MJCF not found")

        ki = extract_kinematic_info(mjcf)
        solver = RotationSolver(mode="diffusion", device="cpu", kinematic_info=ki)
        builder = RobotStateBuilder(dt=1.0 / 30.0, rotation_solver=solver)

        positions_30 = torch.randn(1, 90, 24, 3)
        hml_20 = torch.randn(1, 60, 263)
        builder.build(positions_30, hml_raw=hml_20)

        lib = CLoSDMotionLib(builder, device="cpu", horizon_duration=3.0)

        motion_ids = torch.zeros(4, dtype=torch.long)
        motion_times = torch.tensor([0.0, 0.5, 1.0, 1.5])

        state = lib.get_motion_state(motion_ids, motion_times)

        assert state.rigid_body_pos is not None, "rigid_body_pos is None"
        assert state.rigid_body_rot is not None, "rigid_body_rot is None"
        assert state.rigid_body_vel is not None, "rigid_body_vel is None"
        assert state.rigid_body_ang_vel is not None, "rigid_body_ang_vel is None"
        assert state.dof_pos is not None, "dof_pos is None"
        assert state.dof_vel is not None, "dof_vel is None"

        # Shape checks: [N, num_bodies, dim] or [N, num_dofs]
        N = 4
        assert state.rigid_body_pos.shape == (N, 24, 3)
        assert state.rigid_body_rot.shape == (N, 24, 4)
        assert state.dof_pos.shape == (N, 69)

    def test_zero_state_all_fields(self):
        """Zero state (before build) must also have all fields."""
        from closd_isaaclab.diffusion.robot_state_builder import RobotStateBuilder
        from closd_isaaclab.integration.closd_motion_lib import CLoSDMotionLib

        builder = RobotStateBuilder(dt=1.0 / 30.0)
        lib = CLoSDMotionLib(builder, device="cpu")

        motion_ids = torch.zeros(2, dtype=torch.long)
        motion_times = torch.tensor([0.0, 0.5])

        state = lib.get_motion_state(motion_ids, motion_times)

        assert state.rigid_body_pos is not None
        assert state.rigid_body_rot is not None
        assert state.rigid_body_ang_vel is not None
        assert state.dof_pos is not None
        assert state.dof_vel is not None
