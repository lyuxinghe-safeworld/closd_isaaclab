"""Tests for rotation_solver.py.

Tests cover:
1. cont6d_to_matrix / matrix_to_cont6d round-trip on random rotation matrices.
2. cont6d_to_matrix outputs are orthogonal (R @ R^T ≈ I) and have det ≈ +1.
3. wxyz_quat_to_matrix produces valid rotation matrices.
4. RotationSolver interface: constructor attributes and solve() for mode="diffusion".
"""

import sys
import math

import pytest
import torch

# Ensure CLoSD and related packages are importable
for p in [
    "/home/lyuxinghe/code/CLoSD",
    "/home/lyuxinghe/code/ProtoMotions",
    "/home/lyuxinghe/code/CLoSD_t2m_standalone",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from closd_isaaclab.diffusion.rotation_solver import (
    cont6d_to_matrix,
    matrix_to_cont6d,
    wxyz_quat_to_matrix,
    RotationSolver,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_rotation_matrices(shape, seed=42):
    """Return random proper rotation matrices via QR decomposition."""
    torch.manual_seed(seed)
    A = torch.randn(*shape, 3, 3)
    Q, R = torch.linalg.qr(A)
    # Fix sign so det == +1
    d = torch.linalg.det(Q).sign().unsqueeze(-1).unsqueeze(-1)
    Q = Q * d
    return Q.float()


# ---------------------------------------------------------------------------
# cont6d_to_matrix
# ---------------------------------------------------------------------------

class TestCont6dToMatrix:
    """Tests for cont6d_to_matrix."""

    def test_output_shape_1d(self):
        x = torch.randn(6)
        out = cont6d_to_matrix(x)
        assert out.shape == (3, 3), f"Expected (3, 3), got {out.shape}"

    def test_output_shape_batched(self):
        x = torch.randn(4, 6)
        out = cont6d_to_matrix(x)
        assert out.shape == (4, 3, 3), f"Expected (4, 3, 3), got {out.shape}"

    def test_output_shape_multidim(self):
        x = torch.randn(2, 5, 21, 6)
        out = cont6d_to_matrix(x)
        assert out.shape == (2, 5, 21, 3, 3), f"Expected (2, 5, 21, 3, 3), got {out.shape}"

    def test_output_is_orthogonal(self):
        """R @ R^T should be identity for all outputs."""
        torch.manual_seed(7)
        x = torch.randn(16, 6)
        R = cont6d_to_matrix(x)
        eye = R @ R.transpose(-1, -2)
        expected = torch.eye(3).expand_as(eye)
        assert torch.allclose(eye, expected, atol=1e-5), \
            f"Max deviation from I: {(eye - expected).abs().max():.6f}"

    def test_output_det_is_one(self):
        """det(R) should be +1 (proper rotation)."""
        torch.manual_seed(13)
        x = torch.randn(16, 6)
        R = cont6d_to_matrix(x)
        det = torch.linalg.det(R)
        ones = torch.ones_like(det)
        assert torch.allclose(det, ones, atol=1e-5), \
            f"det range: [{det.min():.6f}, {det.max():.6f}]"

    def test_identity_input(self):
        """A well-conditioned input aligned with x/y axes should give valid rotation."""
        x = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        R = cont6d_to_matrix(x)
        eye = R @ R.T
        assert torch.allclose(eye, torch.eye(3), atol=1e-5)


# ---------------------------------------------------------------------------
# matrix_to_cont6d
# ---------------------------------------------------------------------------

class TestMatrixToCont6d:
    """Tests for matrix_to_cont6d."""

    def test_output_shape_1d(self):
        R = _random_rotation_matrices(())
        out = matrix_to_cont6d(R)
        assert out.shape == (6,), f"Expected (6,), got {out.shape}"

    def test_output_shape_batched(self):
        R = _random_rotation_matrices((8,))
        out = matrix_to_cont6d(R)
        assert out.shape == (8, 6), f"Expected (8, 6), got {out.shape}"

    def test_output_shape_multidim(self):
        R = _random_rotation_matrices((2, 5, 21))
        out = matrix_to_cont6d(R)
        assert out.shape == (2, 5, 21, 6), f"Expected (2, 5, 21, 6), got {out.shape}"

    def test_first_three_match_first_row(self):
        """matrix_to_cont6d should return the first two rows of R."""
        R = _random_rotation_matrices((4,))
        c6d = matrix_to_cont6d(R)
        assert torch.allclose(c6d[..., :3], R[..., 0, :], atol=1e-6), \
            "First 3 elements should match first row of R"

    def test_last_three_match_second_row(self):
        R = _random_rotation_matrices((4,))
        c6d = matrix_to_cont6d(R)
        assert torch.allclose(c6d[..., 3:], R[..., 1, :], atol=1e-6), \
            "Last 3 elements should match second row of R"


# ---------------------------------------------------------------------------
# Round-trip: matrix -> cont6d -> matrix
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """matrix_to_cont6d then cont6d_to_matrix should recover the original matrix."""

    def test_round_trip_single(self):
        R = _random_rotation_matrices(())
        c6d = matrix_to_cont6d(R)
        R_recovered = cont6d_to_matrix(c6d)
        assert torch.allclose(R, R_recovered, atol=1e-5), \
            f"Max diff: {(R - R_recovered).abs().max():.6f}"

    def test_round_trip_batched(self):
        R = _random_rotation_matrices((32,))
        c6d = matrix_to_cont6d(R)
        R_recovered = cont6d_to_matrix(c6d)
        assert torch.allclose(R, R_recovered, atol=1e-5), \
            f"Max diff: {(R - R_recovered).abs().max():.6f}"

    def test_round_trip_multidim(self):
        R = _random_rotation_matrices((2, 5, 21))
        c6d = matrix_to_cont6d(R)
        R_recovered = cont6d_to_matrix(c6d)
        assert torch.allclose(R, R_recovered, atol=1e-5), \
            f"Max diff: {(R - R_recovered).abs().max():.6f}"


# ---------------------------------------------------------------------------
# wxyz_quat_to_matrix
# ---------------------------------------------------------------------------

class TestWxyzQuatToMatrix:
    """Tests for wxyz_quat_to_matrix."""

    def test_output_shape_single(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])  # identity wxyz
        R = wxyz_quat_to_matrix(q)
        assert R.shape == (3, 3), f"Expected (3, 3), got {R.shape}"

    def test_output_shape_batched(self):
        q = torch.randn(8, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = wxyz_quat_to_matrix(q)
        assert R.shape == (8, 3, 3), f"Expected (8, 3, 3), got {R.shape}"

    def test_identity_quaternion(self):
        """Identity quaternion (1, 0, 0, 0) in wxyz should give identity matrix."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        R = wxyz_quat_to_matrix(q)
        assert torch.allclose(R, torch.eye(3), atol=1e-6), \
            f"Identity quat should give I, got:\n{R}"

    def test_output_is_orthogonal(self):
        """Output should be a valid rotation matrix."""
        torch.manual_seed(3)
        q = torch.randn(16, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = wxyz_quat_to_matrix(q)
        eye = R @ R.transpose(-1, -2)
        expected = torch.eye(3).expand_as(eye)
        assert torch.allclose(eye, expected, atol=1e-5), \
            f"Max deviation: {(eye - expected).abs().max():.6f}"

    def test_output_det_is_one(self):
        torch.manual_seed(5)
        q = torch.randn(16, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = wxyz_quat_to_matrix(q)
        det = torch.linalg.det(R)
        assert torch.allclose(det, torch.ones_like(det), atol=1e-5)

    def test_known_90deg_rotation(self):
        """90° rotation around Z: wxyz = (cos45, 0, 0, sin45)."""
        angle = math.pi / 2
        q = torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        R = wxyz_quat_to_matrix(q)
        # Expected: x->y, y->-x, z->z
        expected = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        assert torch.allclose(R, expected, atol=1e-6), \
            f"Got:\n{R}\nExpected:\n{expected}"


# ---------------------------------------------------------------------------
# RotationSolver interface
# ---------------------------------------------------------------------------

class TestRotationSolverInterface:
    """Verify RotationSolver exists with correct attributes."""

    def test_instantiation_defaults(self):
        solver = RotationSolver()
        assert solver.mode == "diffusion"
        assert solver.device == "cpu"
        assert solver.consistency_threshold == 0.05
        assert solver.kinematic_info is None

    def test_instantiation_custom(self):
        solver = RotationSolver(
            mode="diffusion",
            device="cpu",
            consistency_threshold=0.1,
            kinematic_info=None,
        )
        assert solver.consistency_threshold == 0.1

    def test_has_solve_method(self):
        solver = RotationSolver()
        assert callable(getattr(solver, "solve", None))

    def test_has_verify_consistency_method(self):
        solver = RotationSolver()
        assert callable(getattr(solver, "verify_consistency", None))

    def test_analytical_mode_raises(self):
        solver = RotationSolver(mode="analytical")
        with pytest.raises(NotImplementedError):
            solver.solve(positions=None)


class TestRotationSolverSolve:
    """Test RotationSolver.solve() for mode='diffusion'."""

    def _make_hml_raw(self, bs, T, seed=0):
        """Create synthetic hml_raw [bs, T, 263] with valid 6D rotations in dims 67:193."""
        torch.manual_seed(seed)
        hml = torch.zeros(bs, T, 263)
        # Create 21 valid rotation matrices and encode as cont6d
        R = _random_rotation_matrices((bs, T, 21), seed=seed)
        c6d = matrix_to_cont6d(R)  # [bs, T, 21, 6]
        hml[..., 67:193] = c6d.reshape(bs, T, 126)
        return hml

    def test_solve_returns_tuple_of_three(self):
        solver = RotationSolver()
        bs, T = 2, 10
        hml_raw = self._make_hml_raw(bs, T)
        positions = torch.zeros(bs, T, 22, 3)
        # Provide root_rot_wxyz to bypass recover_root_rot_pos
        root_rot_wxyz = torch.zeros(bs, T, 4)
        root_rot_wxyz[..., 0] = 1.0  # identity wxyz
        result = solver.solve(positions, hml_raw=hml_raw, root_rot_wxyz=root_rot_wxyz)
        assert len(result) == 3, f"Expected 3 outputs, got {len(result)}"

    def test_local_rot_mats_shape(self):
        """local_rot_mats_24 should be [bs, T, 24, 3, 3]."""
        solver = RotationSolver()
        bs, T = 2, 10
        hml_raw = self._make_hml_raw(bs, T)
        positions = torch.zeros(bs, T, 22, 3)
        root_rot_wxyz = torch.zeros(bs, T, 4)
        root_rot_wxyz[..., 0] = 1.0
        local_rot_mats_24, dof_pos, consistency_error = solver.solve(
            positions, hml_raw=hml_raw, root_rot_wxyz=root_rot_wxyz
        )
        assert local_rot_mats_24.shape == (bs, T, 24, 3, 3), \
            f"Expected ({bs}, {T}, 24, 3, 3), got {local_rot_mats_24.shape}"

    def test_local_rot_mats_are_orthogonal(self):
        """Each output rotation matrix should satisfy R @ R^T ≈ I."""
        solver = RotationSolver()
        bs, T = 2, 5
        hml_raw = self._make_hml_raw(bs, T)
        positions = torch.zeros(bs, T, 22, 3)
        root_rot_wxyz = torch.zeros(bs, T, 4)
        root_rot_wxyz[..., 0] = 1.0
        local_rot_mats_24, _, _ = solver.solve(
            positions, hml_raw=hml_raw, root_rot_wxyz=root_rot_wxyz
        )
        eye = local_rot_mats_24 @ local_rot_mats_24.transpose(-1, -2)
        expected = torch.eye(3).expand_as(eye)
        assert torch.allclose(eye, expected, atol=1e-5), \
            f"Max deviation from I: {(eye - expected).abs().max():.6f}"

    def test_dof_pos_is_none_without_kinematic_info(self):
        """Without kinematic_info, dof_pos should be None."""
        solver = RotationSolver(kinematic_info=None)
        bs, T = 1, 8
        hml_raw = self._make_hml_raw(bs, T)
        positions = torch.zeros(bs, T, 22, 3)
        root_rot_wxyz = torch.zeros(bs, T, 4)
        root_rot_wxyz[..., 0] = 1.0
        _, dof_pos, _ = solver.solve(
            positions, hml_raw=hml_raw, root_rot_wxyz=root_rot_wxyz
        )
        assert dof_pos is None, f"Expected None dof_pos without kinematic_info, got {dof_pos}"

    def test_consistency_error_is_none_without_kinematic_info(self):
        """Without kinematic_info, consistency_error should be None."""
        solver = RotationSolver(kinematic_info=None)
        bs, T = 1, 8
        hml_raw = self._make_hml_raw(bs, T)
        positions = torch.zeros(bs, T, 22, 3)
        root_rot_wxyz = torch.zeros(bs, T, 4)
        root_rot_wxyz[..., 0] = 1.0
        _, _, consistency_error = solver.solve(
            positions, hml_raw=hml_raw, root_rot_wxyz=root_rot_wxyz
        )
        assert consistency_error is None, \
            f"Expected None consistency_error without kinematic_info, got {consistency_error}"

    def test_identity_root_preserved(self):
        """With identity root_rot_wxyz, joint 0 rotation should be identity."""
        solver = RotationSolver()
        bs, T = 1, 4
        hml_raw = self._make_hml_raw(bs, T)
        positions = torch.zeros(bs, T, 22, 3)
        root_rot_wxyz = torch.zeros(bs, T, 4)
        root_rot_wxyz[..., 0] = 1.0  # identity
        local_rot_mats_24, _, _ = solver.solve(
            positions, hml_raw=hml_raw, root_rot_wxyz=root_rot_wxyz
        )
        R0 = local_rot_mats_24[:, :, 0, :, :]  # [bs, T, 3, 3]
        expected = torch.eye(3).expand_as(R0)
        assert torch.allclose(R0, expected, atol=1e-5), \
            f"Root rotation (identity quat) max diff: {(R0 - expected).abs().max():.6f}"

    def test_hand_joints_are_identity(self):
        """Joints 22 and 23 (hands, appended as identity) should be identity matrices."""
        solver = RotationSolver()
        bs, T = 1, 4
        hml_raw = self._make_hml_raw(bs, T)
        positions = torch.zeros(bs, T, 22, 3)
        root_rot_wxyz = torch.zeros(bs, T, 4)
        root_rot_wxyz[..., 0] = 1.0
        local_rot_mats_24, _, _ = solver.solve(
            positions, hml_raw=hml_raw, root_rot_wxyz=root_rot_wxyz
        )
        R22 = local_rot_mats_24[:, :, 22, :, :]
        R23 = local_rot_mats_24[:, :, 23, :, :]
        eye = torch.eye(3).expand(bs, T, 3, 3)
        assert torch.allclose(R22, eye, atol=1e-6), \
            f"Joint 22 (L_hand) should be identity, max diff: {(R22 - eye).abs().max():.6f}"
        assert torch.allclose(R23, eye, atol=1e-6), \
            f"Joint 23 (R_hand) should be identity, max diff: {(R23 - eye).abs().max():.6f}"

    def test_no_nan_in_output(self):
        """Outputs should not contain NaN."""
        solver = RotationSolver()
        bs, T = 2, 10
        hml_raw = self._make_hml_raw(bs, T)
        positions = torch.zeros(bs, T, 22, 3)
        root_rot_wxyz = torch.zeros(bs, T, 4)
        root_rot_wxyz[..., 0] = 1.0
        local_rot_mats_24, _, _ = solver.solve(
            positions, hml_raw=hml_raw, root_rot_wxyz=root_rot_wxyz
        )
        assert not torch.isnan(local_rot_mats_24).any(), "Output contains NaN"
