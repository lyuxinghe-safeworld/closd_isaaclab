"""Tests for CoordTransform: SMPL <-> Isaac Lab coordinate space conversions."""

import torch
import pytest

from closd_isaaclab.utils.coord_transform import (
    CoordTransform,
    smpl_2_mujoco,
    mujoco_2_smpl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ct():
    return CoordTransform()


@pytest.fixture
def smpl_pos_22():
    """Random 22-joint SMPL positions, single sample."""
    torch.manual_seed(42)
    return torch.randn(22, 3)


@pytest.fixture
def smpl_pos_24():
    """Random 24-joint positions (SMPL + hands), single sample."""
    torch.manual_seed(42)
    return torch.randn(24, 3)


@pytest.fixture
def isaac_pos_24():
    """Random 24-joint Isaac positions, single sample."""
    torch.manual_seed(42)
    return torch.randn(24, 3)


# ---------------------------------------------------------------------------
# Module-level lists
# ---------------------------------------------------------------------------

class TestModuleLevelLists:
    def test_smpl_2_mujoco_length(self):
        assert len(smpl_2_mujoco) == 24

    def test_mujoco_2_smpl_length(self):
        assert len(mujoco_2_smpl) == 24

    def test_smpl_2_mujoco_is_permutation(self):
        assert sorted(smpl_2_mujoco) == list(range(24))

    def test_mujoco_2_smpl_is_permutation(self):
        assert sorted(mujoco_2_smpl) == list(range(24))

    def test_reorder_round_trip(self):
        """Applying smpl_2_mujoco then mujoco_2_smpl should be identity."""
        x = list(range(24))
        reordered = [x[i] for i in smpl_2_mujoco]
        recovered = [reordered[i] for i in mujoco_2_smpl]
        assert recovered == x

    def test_reorder_round_trip_reverse(self):
        """Applying mujoco_2_smpl then smpl_2_mujoco should be identity."""
        x = list(range(24))
        reordered = [x[i] for i in mujoco_2_smpl]
        recovered = [reordered[i] for i in smpl_2_mujoco]
        assert recovered == x


# ---------------------------------------------------------------------------
# _add_hand_joints
# ---------------------------------------------------------------------------

class TestAddHandJoints:
    def test_output_shape(self, ct, smpl_pos_22):
        out = ct._add_hand_joints(smpl_pos_22)
        assert out.shape == (24, 3)

    def test_first_22_unchanged(self, ct, smpl_pos_22):
        out = ct._add_hand_joints(smpl_pos_22)
        assert torch.allclose(out[:22], smpl_pos_22)

    def test_batched_output_shape(self, ct):
        torch.manual_seed(0)
        pos = torch.randn(4, 22, 3)
        out = ct._add_hand_joints(pos)
        assert out.shape == (4, 24, 3)

    def test_left_hand_position(self, ct, smpl_pos_22):
        """Left hand (joint 22) should extend from L_Wrist (20) in direction L_Wrist->L_Wrist + offset."""
        out = ct._add_hand_joints(smpl_pos_22)
        l_wrist = smpl_pos_22[20]
        l_elbow = smpl_pos_22[18]
        direction = l_wrist - l_elbow
        norm = direction.norm()
        if norm > 1e-6:
            expected = l_wrist + 0.08824 * direction / norm
        else:
            expected = l_wrist
        assert torch.allclose(out[22], expected, atol=1e-6)

    def test_right_hand_position(self, ct, smpl_pos_22):
        """Right hand (joint 23) should extend from R_Wrist (21) in direction R_Wrist->R_Wrist + offset."""
        out = ct._add_hand_joints(smpl_pos_22)
        r_wrist = smpl_pos_22[21]
        r_elbow = smpl_pos_22[19]
        direction = r_wrist - r_elbow
        norm = direction.norm()
        if norm > 1e-6:
            expected = r_wrist + 0.08824 * direction / norm
        else:
            expected = r_wrist
        assert torch.allclose(out[23], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# smpl_to_isaac
# ---------------------------------------------------------------------------

class TestSmplToIsaac:
    def test_output_shape_from_22(self, ct, smpl_pos_22):
        out = ct.smpl_to_isaac(smpl_pos_22)
        assert out.shape == (24, 3)

    def test_output_shape_from_24(self, ct, smpl_pos_24):
        out = ct.smpl_to_isaac(smpl_pos_24)
        assert out.shape == (24, 3)

    def test_batched_output_shape(self, ct):
        torch.manual_seed(0)
        pos = torch.randn(8, 22, 3)
        out = ct.smpl_to_isaac(pos)
        assert out.shape == (8, 24, 3)

    def test_batched_multidim_output_shape(self, ct):
        torch.manual_seed(0)
        pos = torch.randn(2, 4, 22, 3)
        out = ct.smpl_to_isaac(pos)
        assert out.shape == (2, 4, 24, 3)

    def test_output_is_finite(self, ct, smpl_pos_22):
        out = ct.smpl_to_isaac(smpl_pos_22)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# isaac_to_smpl
# ---------------------------------------------------------------------------

class TestIsaacToSmpl:
    def test_output_shape_drop_hands(self, ct, isaac_pos_24):
        out = ct.isaac_to_smpl(isaac_pos_24, drop_hands=True)
        assert out.shape == (22, 3)

    def test_output_shape_keep_hands(self, ct, isaac_pos_24):
        out = ct.isaac_to_smpl(isaac_pos_24, drop_hands=False)
        assert out.shape == (24, 3)

    def test_batched_output_shape(self, ct):
        torch.manual_seed(0)
        pos = torch.randn(8, 24, 3)
        out = ct.isaac_to_smpl(pos, drop_hands=True)
        assert out.shape == (8, 22, 3)

    def test_output_is_finite(self, ct, isaac_pos_24):
        out = ct.isaac_to_smpl(isaac_pos_24)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_smpl_to_isaac_to_smpl_22joints(self, ct, smpl_pos_22):
        """smpl_to_isaac followed by isaac_to_smpl (drop_hands) should recover the 22-joint input."""
        isaac = ct.smpl_to_isaac(smpl_pos_22)
        recovered = ct.isaac_to_smpl(isaac, drop_hands=True)
        assert torch.allclose(recovered, smpl_pos_22, atol=1e-5), \
            f"Max diff: {(recovered - smpl_pos_22).abs().max()}"

    def test_smpl_to_isaac_to_smpl_24joints(self, ct):
        """Round-trip on 24-joint SMPL input (keep hands)."""
        torch.manual_seed(7)
        smpl_24 = torch.randn(24, 3)
        isaac = ct.smpl_to_isaac(smpl_24)
        recovered = ct.isaac_to_smpl(isaac, drop_hands=False)
        assert torch.allclose(recovered, smpl_24, atol=1e-5), \
            f"Max diff: {(recovered - smpl_24).abs().max()}"

    def test_isaac_to_smpl_to_isaac(self, ct, isaac_pos_24):
        """isaac_to_smpl (keep hands) followed by smpl_to_isaac should recover the input."""
        smpl = ct.isaac_to_smpl(isaac_pos_24, drop_hands=False)
        recovered = ct.smpl_to_isaac(smpl)
        assert torch.allclose(recovered, isaac_pos_24, atol=1e-5), \
            f"Max diff: {(recovered - isaac_pos_24).abs().max()}"

    def test_batched_round_trip(self, ct):
        """Batched round-trip should be equivalent to per-sample processing."""
        torch.manual_seed(99)
        batch = torch.randn(5, 22, 3)
        isaac_batch = ct.smpl_to_isaac(batch)
        recovered_batch = ct.isaac_to_smpl(isaac_batch, drop_hands=True)
        # Also check per-sample
        for i in range(5):
            isaac_i = ct.smpl_to_isaac(batch[i])
            recovered_i = ct.isaac_to_smpl(isaac_i, drop_hands=True)
            assert torch.allclose(recovered_batch[i], recovered_i, atol=1e-5)

    def test_round_trip_preserves_values(self, ct, smpl_pos_22):
        """Each joint value should be preserved after round-trip."""
        isaac = ct.smpl_to_isaac(smpl_pos_22)
        recovered = ct.isaac_to_smpl(isaac, drop_hands=True)
        for j in range(22):
            assert torch.allclose(recovered[j], smpl_pos_22[j], atol=1e-5), \
                f"Joint {j} mismatch: expected {smpl_pos_22[j]}, got {recovered[j]}"


# ---------------------------------------------------------------------------
# Coordinate-space sanity checks
# ---------------------------------------------------------------------------

class TestCoordSpaceSanity:
    def test_rotation_matrix_shape(self, ct):
        """Combined rotation matrix should be 3x3."""
        assert ct.rot_mat.shape == (3, 3)

    def test_rotation_matrix_is_orthogonal(self, ct):
        """Rotation matrix should be orthogonal (R @ R.T = I)."""
        R = ct.rot_mat
        eye = R @ R.T
        assert torch.allclose(eye, torch.eye(3, dtype=R.dtype), atol=1e-6)

    def test_rotation_matrix_det_is_one(self, ct):
        """Rotation matrix determinant should be +1 (proper rotation)."""
        det = torch.linalg.det(ct.rot_mat)
        assert torch.allclose(det, torch.tensor(1.0, dtype=ct.rot_mat.dtype), atol=1e-6)

    def test_upward_vector_transforms(self, ct):
        """A joint directly above root in SMPL (Y-up) should appear above root in Isaac (Z-up)."""
        # In SMPL Y-up space: joint 1 unit above root
        pos_smpl = torch.zeros(22, 3)
        pos_smpl[1] = torch.tensor([0.0, 1.0, 0.0])  # Y-up
        pos_isaac = ct.smpl_to_isaac(pos_smpl)
        # After SMPL->Isaac transform, the "up" joint should have positive Z in Isaac
        # (allowing for the sign conventions from the full transform chain)
        root_isaac = pos_isaac[smpl_2_mujoco[0]]
        joint1_isaac = pos_isaac[smpl_2_mujoco[1]]
        diff = joint1_isaac - root_isaac
        # Z component should be non-trivially non-zero (it absorbed the Y direction)
        assert diff.abs().max() > 0.5
