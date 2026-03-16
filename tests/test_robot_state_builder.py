"""Tests for robot_state_builder.py.

Tests cover:
1. Constant velocity: linear motion at ~1m/s -> velocity output should be ~1.0 for interior frames
2. Contacts mapping: Set all HML foot contacts to 1.0 -> at least 4 nonzero entries per frame
3. Output shapes: build() then get_state_at_frames() returns correct shapes
4. Zero velocity for static pose: all-same positions -> all-zero velocities
"""

import sys

import pytest
import torch

for p in [
    "/home/lyuxinghe/code/CLoSD",
    "/home/lyuxinghe/code/ProtoMotions",
    "/home/lyuxinghe/code/CLoSD_t2m_standalone",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from closd_isaaclab.diffusion.robot_state_builder import RobotStateBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_positions(bs, T, num_bodies=24):
    """Create random stationary positions [bs, T, num_bodies, 3]."""
    return torch.zeros(bs, T, num_bodies, 3)


def _make_hml_raw(bs, T_20fps, foot_contact_value=0.0):
    """Create synthetic hml_raw [bs, T_20fps, 263] with specified foot contact values."""
    hml = torch.zeros(bs, T_20fps, 263)
    # Set foot contact flags (dims 259-262)
    hml[..., 259:263] = foot_contact_value
    return hml


# ---------------------------------------------------------------------------
# Test: Constant velocity
# ---------------------------------------------------------------------------

class TestConstantVelocity:
    """Linear motion at ~1m/s should yield velocity ~1.0 for interior frames."""

    def test_velocity_magnitude_interior_frames(self):
        """Interior frames should have velocity magnitude ~1.0 m/s."""
        dt = 1.0 / 30.0
        builder = RobotStateBuilder(dt=dt)
        bs, T, num_bodies = 1, 10, 24

        # Build positions with constant velocity of 1 m/s along x-axis
        # pos[t] = t * dt * 1.0 in x
        positions = torch.zeros(bs, T, num_bodies, 3)
        for t in range(T):
            positions[:, t, :, 0] = t * dt * 1.0  # 1 m/s along x

        builder.build(positions)
        state = builder.get_state_at_frames(list(range(T)))
        vel = state["rigid_body_vel"]  # [bs, T, num_bodies, 3]

        # Interior frames (indices 1 to T-2) should have velocity ~1.0 along x
        interior_vel_x = vel[:, 1:-1, :, 0]  # [bs, T-2, num_bodies]
        assert torch.allclose(interior_vel_x, torch.ones_like(interior_vel_x), atol=1e-4), \
            f"Expected velocity ~1.0 for interior frames, got max diff: {(interior_vel_x - 1.0).abs().max():.6f}"

    def test_velocity_y_z_zero_for_x_only_motion(self):
        """Y and Z velocity components should be zero for purely x-direction motion."""
        dt = 1.0 / 30.0
        builder = RobotStateBuilder(dt=dt)
        bs, T, num_bodies = 1, 10, 24

        positions = torch.zeros(bs, T, num_bodies, 3)
        for t in range(T):
            positions[:, t, :, 0] = t * dt * 1.0

        builder.build(positions)
        state = builder.get_state_at_frames(list(range(T)))
        vel = state["rigid_body_vel"]

        # Y and Z components should be zero
        assert torch.allclose(vel[..., 1], torch.zeros_like(vel[..., 1]), atol=1e-6), \
            "Y velocity should be zero for x-only motion"
        assert torch.allclose(vel[..., 2], torch.zeros_like(vel[..., 2]), atol=1e-6), \
            "Z velocity should be zero for x-only motion"


# ---------------------------------------------------------------------------
# Test: Zero velocity for static pose
# ---------------------------------------------------------------------------

class TestZeroVelocityStatic:
    """All-same positions should yield all-zero velocities."""

    def test_zero_velocity(self):
        builder = RobotStateBuilder(dt=1.0 / 30.0)
        bs, T, num_bodies = 2, 8, 24

        # All frames have the same position
        positions = torch.ones(bs, T, num_bodies, 3) * 0.5

        builder.build(positions)
        state = builder.get_state_at_frames(list(range(T)))
        vel = state["rigid_body_vel"]

        assert torch.allclose(vel, torch.zeros_like(vel), atol=1e-6), \
            f"Expected all-zero velocities for static pose, max: {vel.abs().max():.8f}"

    def test_zero_dof_vel_static(self):
        """When no rotation_solver is provided, dof_vel should still be zero for static pose."""
        builder = RobotStateBuilder(dt=1.0 / 30.0)
        bs, T, num_bodies = 1, 6, 24

        positions = torch.zeros(bs, T, num_bodies, 3)
        builder.build(positions)
        state = builder.get_state_at_frames(list(range(T)))

        # dof_vel should be None or all zeros (no rotation solver)
        # With no rotation solver, dof_pos is None -> dof_vel is None
        assert state["dof_vel"] is None, "dof_vel should be None when rotation_solver is not provided"


# ---------------------------------------------------------------------------
# Test: Contacts mapping
# ---------------------------------------------------------------------------

class TestContactsMapping:
    """HML foot contacts (dims 259-262) should map to 24-body contact tensor."""

    def test_all_foot_contacts_set(self):
        """Setting all 4 foot contact flags to 1.0 should produce >=4 nonzero entries per frame."""
        builder = RobotStateBuilder(dt=1.0 / 30.0)
        bs, T_20fps = 1, 10
        T = T_20fps  # same for simplicity

        positions = _make_positions(bs, T)
        hml_raw = _make_hml_raw(bs, T_20fps, foot_contact_value=1.0)

        builder.build(positions, hml_raw=hml_raw)
        state = builder.get_state_at_frames(list(range(T)))
        contacts = state["rigid_body_contacts"]  # [bs, T, 24]

        # Each frame should have at least 4 nonzero contact entries
        for t in range(T):
            nonzero = (contacts[:, t, :] > 0).sum(dim=-1)
            assert (nonzero >= 4).all(), \
                f"Frame {t}: expected >=4 nonzero contacts, got {nonzero}"

    def test_no_contacts_when_zero(self):
        """With all foot contact flags at 0, contact tensor should be all zeros."""
        builder = RobotStateBuilder(dt=1.0 / 30.0)
        bs, T_20fps = 1, 10
        T = T_20fps

        positions = _make_positions(bs, T)
        hml_raw = _make_hml_raw(bs, T_20fps, foot_contact_value=0.0)

        builder.build(positions, hml_raw=hml_raw)
        state = builder.get_state_at_frames(list(range(T)))
        contacts = state["rigid_body_contacts"]

        assert torch.allclose(contacts, torch.zeros_like(contacts), atol=1e-6), \
            "Expected all-zero contacts when foot flags are 0"

    def test_contact_body_indices(self):
        """Contacts should appear at correct MuJoCo body indices (3, 4, 7, 8)."""
        builder = RobotStateBuilder(dt=1.0 / 30.0)
        bs, T_20fps = 1, 5
        T = T_20fps

        positions = _make_positions(bs, T)
        hml_raw = _make_hml_raw(bs, T_20fps, foot_contact_value=1.0)

        builder.build(positions, hml_raw=hml_raw)
        state = builder.get_state_at_frames(list(range(T)))
        contacts = state["rigid_body_contacts"]  # [bs, T, 24]

        expected_indices = [3, 4, 7, 8]
        for idx in expected_indices:
            assert (contacts[:, :, idx] > 0).all(), \
                f"Expected contact at body index {idx}, but got zero"


# ---------------------------------------------------------------------------
# Test: Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    """build() then get_state_at_frames() should return correct shapes."""

    def test_rigid_body_pos_shape(self):
        bs, T, num_bodies = 2, 12, 24
        builder = RobotStateBuilder(dt=1.0 / 30.0, num_bodies=num_bodies)
        positions = _make_positions(bs, T, num_bodies)
        builder.build(positions)
        state = builder.get_state_at_frames(list(range(T)))
        assert state["rigid_body_pos"].shape == (bs, T, num_bodies, 3), \
            f"Expected ({bs}, {T}, {num_bodies}, 3), got {state['rigid_body_pos'].shape}"

    def test_rigid_body_vel_shape(self):
        bs, T, num_bodies = 2, 12, 24
        builder = RobotStateBuilder(dt=1.0 / 30.0, num_bodies=num_bodies)
        positions = _make_positions(bs, T, num_bodies)
        builder.build(positions)
        state = builder.get_state_at_frames(list(range(T)))
        assert state["rigid_body_vel"].shape == (bs, T, num_bodies, 3), \
            f"Expected ({bs}, {T}, {num_bodies}, 3), got {state['rigid_body_vel'].shape}"

    def test_contacts_shape(self):
        bs, T, num_bodies = 2, 12, 24
        builder = RobotStateBuilder(dt=1.0 / 30.0, num_bodies=num_bodies)
        positions = _make_positions(bs, T, num_bodies)
        hml_raw = _make_hml_raw(bs, T)
        builder.build(positions, hml_raw=hml_raw)
        state = builder.get_state_at_frames(list(range(T)))
        assert state["rigid_body_contacts"].shape == (bs, T, num_bodies), \
            f"Expected ({bs}, {T}, {num_bodies}), got {state['rigid_body_contacts'].shape}"

    def test_dof_pos_shape_without_solver(self):
        """Without rotation_solver, dof_pos and dof_vel should be None."""
        bs, T = 2, 12
        builder = RobotStateBuilder(dt=1.0 / 30.0)
        positions = _make_positions(bs, T)
        builder.build(positions)
        state = builder.get_state_at_frames(list(range(T)))
        assert state["dof_pos"] is None, "dof_pos should be None without rotation_solver"
        assert state["dof_vel"] is None, "dof_vel should be None without rotation_solver"

    def test_subset_frame_indices(self):
        """get_state_at_frames with a subset should return the right number of frames."""
        bs, T, num_bodies = 1, 20, 24
        builder = RobotStateBuilder(dt=1.0 / 30.0, num_bodies=num_bodies)
        positions = _make_positions(bs, T, num_bodies)
        builder.build(positions)
        indices = [0, 5, 10, 15]
        state = builder.get_state_at_frames(indices)
        assert state["rigid_body_pos"].shape == (bs, len(indices), num_bodies, 3), \
            f"Expected ({bs}, {len(indices)}, {num_bodies}, 3), got {state['rigid_body_pos'].shape}"
        assert state["rigid_body_vel"].shape == (bs, len(indices), num_bodies, 3), \
            f"Velocity shape mismatch"

    def test_state_keys_present(self):
        """State dict should have all required keys."""
        builder = RobotStateBuilder(dt=1.0 / 30.0)
        bs, T = 1, 5
        positions = _make_positions(bs, T)
        builder.build(positions)
        state = builder.get_state_at_frames(list(range(T)))
        expected_keys = {"rigid_body_pos", "rigid_body_vel", "dof_pos", "dof_vel", "rigid_body_contacts"}
        assert set(state.keys()) == expected_keys, \
            f"Missing keys: {expected_keys - set(state.keys())}"

    def test_build_multiple_times(self):
        """build() should reset and update cache correctly on subsequent calls."""
        builder = RobotStateBuilder(dt=1.0 / 30.0)
        bs, T, num_bodies = 1, 10, 24

        positions1 = _make_positions(bs, T, num_bodies)
        builder.build(positions1)
        state1 = builder.get_state_at_frames(list(range(T)))

        positions2 = torch.ones(bs, T, num_bodies, 3)
        builder.build(positions2)
        state2 = builder.get_state_at_frames(list(range(T)))

        assert not torch.allclose(state1["rigid_body_pos"], state2["rigid_body_pos"]), \
            "Second build should overwrite cached positions"


# ---------------------------------------------------------------------------
# Test: _compute_velocities internals
# ---------------------------------------------------------------------------

class TestComputeVelocities:
    """Test the central differencing velocity computation."""

    def test_forward_diff_at_start(self):
        """At t=0, velocity should use forward difference: (pos[1] - pos[0]) / dt."""
        dt = 1.0 / 30.0
        builder = RobotStateBuilder(dt=dt)
        bs, T, num_bodies = 1, 5, 24

        positions = torch.zeros(bs, T, num_bodies, 3)
        for t in range(T):
            positions[:, t, :, 0] = t * dt * 2.0  # 2 m/s along x

        builder.build(positions)
        state = builder.get_state_at_frames([0])
        vel_t0 = state["rigid_body_vel"][:, 0, :, 0]
        expected = torch.ones_like(vel_t0) * 2.0
        assert torch.allclose(vel_t0, expected, atol=1e-4), \
            f"Forward diff at t=0 failed: expected 2.0, got {vel_t0.mean():.6f}"

    def test_backward_diff_at_end(self):
        """At t=T-1, velocity should use backward difference: (pos[-1] - pos[-2]) / dt."""
        dt = 1.0 / 30.0
        builder = RobotStateBuilder(dt=dt)
        bs, T, num_bodies = 1, 5, 24

        positions = torch.zeros(bs, T, num_bodies, 3)
        for t in range(T):
            positions[:, t, :, 0] = t * dt * 2.0

        builder.build(positions)
        state = builder.get_state_at_frames([T - 1])
        vel_last = state["rigid_body_vel"][:, 0, :, 0]  # index 0 in the returned slice
        expected = torch.ones_like(vel_last) * 2.0
        assert torch.allclose(vel_last, expected, atol=1e-4), \
            f"Backward diff at t=T-1 failed: expected 2.0, got {vel_last.mean():.6f}"

    def test_central_diff_at_interior(self):
        """Interior frames should use central differencing."""
        dt = 1.0 / 30.0
        builder = RobotStateBuilder(dt=dt)
        bs, T, num_bodies = 1, 7, 24

        # Quadratic position: x(t) = t^2 * dt^2 -> v(t) = 2t*dt
        positions = torch.zeros(bs, T, num_bodies, 3)
        for t in range(T):
            positions[:, t, :, 0] = (t * dt) ** 2

        builder.build(positions)
        # At t=3 (interior), central diff: (pos[4] - pos[2]) / (2*dt)
        state = builder.get_state_at_frames([3])
        vel_t3 = state["rigid_body_vel"][:, 0, :, 0]
        # central diff: ((3*dt)^2 - (1*dt)^2) cannot use this since indices are 4 and 2
        # Actually: (pos[4] - pos[2]) / (2*dt) = ((4*dt)^2 - (2*dt)^2) / (2*dt)
        #         = dt^2 * (16 - 4) / (2*dt) = dt * 12 / 2 = 6*dt
        expected_vel = 6 * dt
        assert torch.allclose(vel_t3, torch.ones_like(vel_t3) * expected_vel, atol=1e-5), \
            f"Central diff at t=3 failed: expected {expected_vel:.6f}, got {vel_t3.mean():.6f}"
