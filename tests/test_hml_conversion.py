"""Tests for HMLConversion: hml_to_pose and pose_to_hml."""

import sys
import os
import numpy as np
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

from closd_isaaclab.diffusion.hml_conversion import HMLConversion


@pytest.fixture
def mean_std():
    """Load the real t2m mean and std."""
    base = "/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets"
    mean = torch.from_numpy(np.load(os.path.join(base, "t2m_mean.npy"))).float()
    std = torch.from_numpy(np.load(os.path.join(base, "t2m_std.npy"))).float()
    return mean, std


@pytest.fixture
def converter(mean_std):
    mean, std = mean_std
    return HMLConversion(mean, std, device="cpu")


class TestHmlToPoseShape:
    """Test 1: hml_to_pose output shape is [bs, T_30fps, 24, 3]."""

    def test_output_shape(self, converter, mean_std):
        bs, T_20 = 2, 10
        mean, std = mean_std
        # Create random normalized HML data
        hml_norm = torch.randn(bs, T_20, 263)

        # Create recon_data (identity rotation, origin position)
        recon_data = {
            "r_rot": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * bs),  # [bs, 4] wxyz identity
            "r_pos": torch.zeros(bs, 3),
        }
        sim_at_hml_idx = 0

        result = converter.hml_to_pose(hml_norm, recon_data, sim_at_hml_idx)

        # T_30fps = int(T_20 * 30 / 20) = 15
        expected_T30 = int(T_20 * 30 / 20)
        assert result.shape == (bs, expected_T30, 24, 3), f"Got shape {result.shape}"
        assert not torch.isnan(result).any(), "Output contains NaN"


class TestRecoverRootRotPos:
    """Test 2: recover_root_rot_pos on known data — constant forward velocity → monotonically increasing position."""

    def test_monotonic_forward(self, mean_std):
        from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import (
            recover_root_rot_pos,
        )

        bs, T = 1, 20
        # Create HML-like data with zero angular velocity and constant forward XZ velocity
        data = torch.zeros(bs, T, 263)
        # dim 0: angular velocity = 0
        # dims 1-2: XZ velocity = constant forward
        data[:, :, 1] = 0.05  # constant X velocity
        data[:, :, 2] = 0.0  # zero Z velocity
        # dim 3: root height
        data[:, :, 3] = 0.9  # constant height

        r_rot_quat, r_pos = recover_root_rot_pos(data)

        # With zero angular velocity, root rotation should stay identity-like
        # Position X should be monotonically increasing (cumsum of positive velocity)
        x_positions = r_pos[0, :, 0]
        for i in range(1, T):
            assert x_positions[i] >= x_positions[i - 1], (
                f"X position not monotonically increasing at frame {i}: "
                f"{x_positions[i].item()} < {x_positions[i-1].item()}"
            )

        # Y should equal the height
        assert torch.allclose(r_pos[0, :, 1], data[0, :, 3], atol=1e-5)


class TestReconData:
    """Test 3: recon_data is saved with r_rot and r_pos keys with correct shapes."""

    def test_recon_data_keys_and_shapes(self, converter):
        bs, T_30 = 2, 15
        # Create simple Isaac-space positions
        positions_isaac = torch.randn(bs, T_30, 24, 3)
        # Make positions somewhat realistic (spread around origin)
        positions_isaac[..., 2] += 0.9  # Z-up height

        _, recon_data = converter.pose_to_hml(positions_isaac)

        assert "r_rot" in recon_data, "recon_data missing 'r_rot'"
        assert "r_pos" in recon_data, "recon_data missing 'r_pos'"
        assert recon_data["r_rot"].shape == (bs, 4), (
            f"r_rot shape {recon_data['r_rot'].shape}, expected ({bs}, 4)"
        )
        assert recon_data["r_pos"].shape == (bs, 3), (
            f"r_pos shape {recon_data['r_pos'].shape}, expected ({bs}, 3)"
        )


class TestRoundTrip:
    """Test 4: Round-trip approximate preservation (pose → hml → pose)."""

    def test_round_trip(self, converter):
        bs = 1
        # We need enough frames for the pipeline. 30fps input → 20fps internal
        # Need at least ~6 frames at 20fps (so extract_features_t2m works), meaning ~9 at 30fps
        # But extract_features loses 1 frame, and we add a dummy, so use a generous count.
        T_30 = 30

        # Create plausible joint positions: a standing pose replicated
        # Use a simple T-pose-like configuration in Isaac space (Z-up)
        positions_isaac = torch.zeros(bs, T_30, 24, 3)
        # Spread joints vertically (Z-up)
        for j in range(24):
            positions_isaac[:, :, j, 2] = 0.04 * j  # height
            positions_isaac[:, :, j, 0] = 0.01 * (j % 5)  # slight X spread
            positions_isaac[:, :, j, 1] = 0.01 * (j % 3)  # slight Y spread

        # Forward: pose → hml
        hml_norm, recon_data = converter.pose_to_hml(positions_isaac)

        # The hml output should have T_20fps - 1 frames due to velocity computation
        # Reverse: hml → pose
        sim_at_hml_idx = 0
        reconstructed = converter.hml_to_pose(hml_norm, recon_data, sim_at_hml_idx)

        # The number of frames may differ slightly due to fps conversions
        # Compare the overlapping frames
        T_compare = min(positions_isaac.shape[1], reconstructed.shape[1])

        # Root joint (joint 0) trajectory should be roughly preserved
        orig_root = positions_isaac[:, :T_compare, 0, :]
        recon_root = reconstructed[:, :T_compare, 0, :]

        # Check that root height is approximately preserved (within reasonable tolerance)
        height_diff = (orig_root[..., 2] - recon_root[..., 2]).abs().mean()
        assert height_diff < 0.5, f"Root height difference too large: {height_diff.item()}"

        # Check overall shape matches
        assert reconstructed.shape[0] == bs
        assert reconstructed.shape[2] == 24
        assert reconstructed.shape[3] == 3
