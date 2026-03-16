"""Diagnostic test: analytical IK → FK roundtrip position accuracy.

Measures per-joint position error when positions go through:
  raw diffused positions → analytical_ik() → FK → reconstructed positions

Root cause investigation for red balls not matching diffused motion.
"""

from __future__ import annotations

import os

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ki():
    """Load KinematicInfo from MJCF, skip if unavailable."""
    try:
        from protomotions.components.pose_lib import extract_kinematic_info
        proto_root = os.path.dirname(os.path.dirname(os.path.abspath(
            __import__("protomotions").__file__)))
        mjcf = os.path.join(proto_root, "protomotions/data/assets/mjcf/smpl_humanoid.xml")
        if not os.path.exists(mjcf):
            pytest.skip(f"MJCF not found: {mjcf}")
        return extract_kinematic_info(mjcf)
    except ImportError:
        pytest.skip("ProtoMotions not available")


def _fk_positions(ki, root_pos, joint_rot_mats):
    """Run FK and return world positions [T, 24, 3]."""
    from protomotions.components.pose_lib import (
        compute_forward_kinematics_from_transforms,
    )
    pos, _ = compute_forward_kinematics_from_transforms(ki, root_pos, joint_rot_mats)
    return pos


def _ik_then_fk(ki, positions):
    """Run analytical IK → extract local rots → FK. Return FK positions."""
    from closd_isaaclab.diffusion.robot_state_builder import analytical_ik
    from protomotions.components.pose_lib import (
        compute_joint_rot_mats_from_global_mats,
        compute_forward_kinematics_from_transforms,
    )

    global_rots = analytical_ik(positions, ki)  # [T, 24, 3, 3]
    joint_rots = compute_joint_rot_mats_from_global_mats(ki, global_rots)  # [T, 24, 3, 3]
    root_pos = positions[:, 0, :]  # [T, 3]
    fk_pos, _ = compute_forward_kinematics_from_transforms(ki, root_pos, joint_rots)
    return fk_pos, global_rots


def _print_per_joint_error(ki, positions, fk_pos):
    """Print per-joint position error with bone length comparison."""
    T = positions.shape[0]
    per_joint_err = (fk_pos - positions).norm(dim=-1).mean(dim=0)  # [24]

    print("\n" + "=" * 80)
    print(f"{'Body':<15} {'MJCF idx':>8} {'Parent':>8} {'Pos err (mm)':>14} "
          f"{'MJCF bone (mm)':>15} {'Diffused bone (mm)':>19} {'Bone len err (mm)':>18}")
    print("-" * 80)

    for i in range(24):
        pi = ki.parent_indices[i]
        mjcf_bone_len = ki.local_pos[i].norm().item() * 1000  # mm

        if pi >= 0:
            diffused_bone = (positions[:, i] - positions[:, pi]).norm(dim=-1).mean().item() * 1000
            bone_len_err = diffused_bone - mjcf_bone_len
        else:
            diffused_bone = 0.0
            bone_len_err = 0.0

        err_mm = per_joint_err[i].item() * 1000
        parent_name = ki.body_names[pi] if pi >= 0 else "---"
        print(f"{ki.body_names[i]:<15} {i:>8} {parent_name:>8} {err_mm:>14.2f} "
              f"{mjcf_bone_len:>15.1f} {diffused_bone:>19.1f} {bone_len_err:>18.1f}")

    mean_err = per_joint_err.mean().item() * 1000
    max_err = per_joint_err.max().item() * 1000
    max_joint = ki.body_names[per_joint_err.argmax().item()]
    print("-" * 80)
    print(f"Mean: {mean_err:.2f} mm | Max: {max_err:.2f} mm ({max_joint})")
    print("=" * 80)

    return per_joint_err


# ---------------------------------------------------------------------------
# Test 1: FK-generated positions should roundtrip perfectly
# ---------------------------------------------------------------------------

class TestIKFKWithFKPositions:
    """IK→FK on positions that came from FK should have zero error.

    This isolates whether the IK algorithm itself is correct (independent of
    bone length mismatch).
    """

    def test_tpose_roundtrip(self):
        """T-pose FK positions → IK → FK should have zero error."""
        ki = _load_ki()

        # Generate T-pose FK positions (bone lengths match MJCF by definition)
        T = 5
        joint_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, 3, 3).clone()
        root_pos = torch.zeros(T, 3)
        root_pos[:, 2] = 0.95

        fk_pos = _fk_positions(ki, root_pos, joint_rot)

        # Roundtrip: IK → FK
        fk_pos_rt, _ = _ik_then_fk(ki, fk_pos)
        per_joint = (fk_pos_rt - fk_pos).norm(dim=-1).mean(dim=0)

        _print_per_joint_error(ki, fk_pos, fk_pos_rt)

        mean_err = per_joint.mean().item()
        assert mean_err < 0.001, f"T-pose roundtrip error: {mean_err*1000:.2f} mm (should be ~0)"

    def test_bent_pose_roundtrip(self):
        """Non-trivial FK pose → IK → FK should have near-zero error.

        Uses known joint rotations to create a crouching/bent pose, then
        roundtrips through IK→FK. Since bone lengths match MJCF, the only
        error source is the IK algorithm itself.
        """
        ki = _load_ki()
        import math

        T = 10
        joint_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, 3, 3).clone()
        root_pos = torch.zeros(T, 3)
        root_pos[:, 2] = 0.75  # lower = crouching

        # Bend knees: rotate L_Knee (body 2) and R_Knee (body 6) around X
        for t in range(T):
            angle = torch.tensor(math.pi / 4 * (t + 1) / T)  # up to 45°
            c, s = torch.cos(angle), torch.sin(angle)
            knee_rot = torch.tensor([
                [1, 0, 0], [0, c, -s], [0, s, c]
            ], dtype=torch.float32)
            joint_rot[t, 2] = knee_rot   # L_Knee
            joint_rot[t, 6] = knee_rot   # R_Knee

        # Bend hips: rotate L_Hip (body 1) and R_Hip (body 5) forward
        for t in range(T):
            angle = torch.tensor(-math.pi / 6 * (t + 1) / T)  # forward lean
            c, s = torch.cos(angle), torch.sin(angle)
            hip_rot = torch.tensor([
                [1, 0, 0], [0, c, -s], [0, s, c]
            ], dtype=torch.float32)
            joint_rot[t, 1] = hip_rot  # L_Hip
            joint_rot[t, 5] = hip_rot  # R_Hip

        # Generate FK positions
        fk_pos = _fk_positions(ki, root_pos, joint_rot)

        # Roundtrip
        fk_pos_rt, _ = _ik_then_fk(ki, fk_pos)

        print("\n--- Bent pose (knees 0-45°, hips forward) ---")
        per_joint = _print_per_joint_error(ki, fk_pos, fk_pos_rt)

        mean_err = per_joint.mean().item()
        max_err = per_joint.max().item()
        max_joint = ki.body_names[per_joint.argmax().item()]
        assert mean_err < 0.005, (
            f"Bent pose roundtrip mean error: {mean_err*1000:.2f} mm "
            f"(max: {max_err*1000:.2f} mm at {max_joint})"
        )


# ---------------------------------------------------------------------------
# Test 2: Diffused-like positions (potentially mismatched bone lengths)
# ---------------------------------------------------------------------------

class TestIKFKWithDiffusedPositions:
    """IK→FK on diffused positions that may have different bone lengths.

    This tests the actual use case: diffusion model produces positions with
    bone lengths that differ from the MJCF skeleton.
    """

    def _make_diffused_positions(self, ki, bone_scale=1.0, knee_scale=1.0):
        """Create positions by scaling MJCF bone lengths.

        bone_scale: global scale factor for all bones
        knee_scale: additional scale for knee-related bones (thigh + shin)
        """
        T = 10
        joint_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, 3, 3).clone()
        root_pos = torch.zeros(T, 3)
        root_pos[:, 2] = 0.95

        # Get FK positions with exact MJCF bones
        fk_pos = _fk_positions(ki, root_pos, joint_rot)

        # Now scale bone lengths to simulate diffused positions
        scaled_pos = fk_pos.clone()
        knee_bodies = {2, 3, 4, 6, 7, 8}  # L_Knee chain + R_Knee chain

        for i in range(24):
            pi = ki.parent_indices[i]
            if pi < 0:
                continue
            bone_vec = fk_pos[:, i] - fk_pos[:, pi]
            scale = bone_scale
            if i in knee_bodies:
                scale *= knee_scale
            # Replace with scaled bone
            scaled_pos[:, i] = scaled_pos[:, pi] + bone_vec * scale

        return scaled_pos

    def test_exact_bone_lengths(self):
        """With exact MJCF bone lengths, roundtrip should be perfect."""
        ki = _load_ki()
        positions = self._make_diffused_positions(ki, bone_scale=1.0, knee_scale=1.0)
        fk_pos_rt, _ = _ik_then_fk(ki, positions)

        print("\n--- Exact bone lengths (scale=1.0) ---")
        per_joint = _print_per_joint_error(ki, positions, fk_pos_rt)

        mean_err = per_joint.mean().item()
        assert mean_err < 0.001, f"Exact bones roundtrip error: {mean_err*1000:.2f} mm"

    def test_globally_scaled_bones(self):
        """5% globally scaled bones should show uniform error."""
        ki = _load_ki()
        positions = self._make_diffused_positions(ki, bone_scale=1.05, knee_scale=1.0)
        fk_pos_rt, _ = _ik_then_fk(ki, positions)

        print("\n--- Global 5% scale (all bones 5% longer) ---")
        per_joint = _print_per_joint_error(ki, positions, fk_pos_rt)

    def test_knee_scaled_bones(self):
        """10% knee-chain scaling should show large knee/ankle/toe error."""
        ki = _load_ki()
        positions = self._make_diffused_positions(ki, bone_scale=1.0, knee_scale=1.10)
        fk_pos_rt, _ = _ik_then_fk(ki, positions)

        print("\n--- Knee chain 10% longer (thigh + shin + ankle + toe) ---")
        per_joint = _print_per_joint_error(ki, positions, fk_pos_rt)

        # Knee error should be large
        l_knee_err = per_joint[2].item() * 1000
        r_knee_err = per_joint[6].item() * 1000
        print(f"\nL_Knee error: {l_knee_err:.2f} mm, R_Knee error: {r_knee_err:.2f} mm")

    def test_realistic_diffusion_noise(self):
        """Realistic per-joint position noise (±2cm) simulating diffusion."""
        ki = _load_ki()

        T = 20
        joint_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, 3, 3).clone()
        root_pos = torch.zeros(T, 3)
        root_pos[:, 2] = 0.95

        fk_pos = _fk_positions(ki, root_pos, joint_rot)

        # Add gaussian noise to simulate diffusion imprecision
        torch.manual_seed(42)
        noise = torch.randn_like(fk_pos) * 0.02  # ±2cm std
        noise[:, 0, :] = 0  # keep root exact
        noisy_pos = fk_pos + noise

        fk_pos_rt, _ = _ik_then_fk(ki, noisy_pos)

        print("\n--- Realistic diffusion noise (±2cm gaussian) ---")
        per_joint = _print_per_joint_error(ki, noisy_pos, fk_pos_rt)


# ---------------------------------------------------------------------------
# Test 3: Actual diffusion output (if available)
# ---------------------------------------------------------------------------

class TestIKFKWithActualDiffusion:
    """Test with actual diffusion-generated positions."""

    def test_with_generated_motion_file(self):
        """Load a previously generated .motion file and check roundtrip."""
        from glob import glob

        # Look for any generated motion files
        patterns = [
            "/home/lyuxinghe/code/closd_isaaclab/output/*/generated.motion",
            "/home/lyuxinghe/code/closd_isaaclab/outputs/*/generated.motion",
        ]
        motion_files = []
        for p in patterns:
            motion_files.extend(glob(p))

        if not motion_files:
            pytest.skip("No generated .motion files found")

        ki = _load_ki()
        motion_path = sorted(motion_files)[-1]  # most recent
        print(f"\nUsing motion file: {motion_path}")

        motion_dict = torch.load(motion_path, map_location="cpu", weights_only=False)
        positions = motion_dict["rigid_body_pos"]  # [T, 24, 3]

        # Skip stabilization prefix (first 60 frames are static)
        positions = positions[60:]
        if positions.shape[0] > 50:
            positions = positions[:50]

        fk_pos_rt, _ = _ik_then_fk(ki, positions)

        print(f"\n--- Actual diffusion output ({positions.shape[0]} frames) ---")
        per_joint = _print_per_joint_error(ki, positions, fk_pos_rt)

    def test_diffusion_bone_lengths_vs_mjcf(self):
        """Compare bone lengths in diffusion output vs MJCF."""
        from glob import glob

        patterns = [
            "/home/lyuxinghe/code/closd_isaaclab/output/*/generated.motion",
            "/home/lyuxinghe/code/closd_isaaclab/outputs/*/generated.motion",
        ]
        motion_files = []
        for p in patterns:
            motion_files.extend(glob(p))

        if not motion_files:
            pytest.skip("No generated .motion files found")

        ki = _load_ki()
        motion_path = sorted(motion_files)[-1]
        motion_dict = torch.load(motion_path, map_location="cpu", weights_only=False)
        positions = motion_dict["rigid_body_pos"][60:]  # skip stabilization

        print(f"\n{'Body':<15} {'MJCF len (mm)':>14} {'Diff mean (mm)':>15} "
              f"{'Diff std (mm)':>14} {'Ratio':>8}")
        print("-" * 70)

        for i in range(24):
            pi = ki.parent_indices[i]
            if pi < 0:
                continue
            mjcf_len = ki.local_pos[i].norm().item() * 1000
            diff_bones = (positions[:, i] - positions[:, pi]).norm(dim=-1)
            diff_mean = diff_bones.mean().item() * 1000
            diff_std = diff_bones.std().item() * 1000
            ratio = diff_mean / mjcf_len if mjcf_len > 0.1 else float('nan')
            print(f"{ki.body_names[i]:<15} {mjcf_len:>14.1f} {diff_mean:>15.1f} "
                  f"{diff_std:>14.1f} {ratio:>8.3f}")


# ---------------------------------------------------------------------------
# Test 4: Retarget + IK/FK roundtrip (should be perfect)
# ---------------------------------------------------------------------------

class TestRetargetIKFK:
    """After retargeting bone lengths to MJCF, IK→FK should be exact."""

    def test_retarget_preserves_directions(self):
        """Retargeted bones should point in the same direction as originals."""
        ki = _load_ki()
        from closd_isaaclab.diffusion.robot_state_builder import retarget_bone_lengths

        T = 10
        joint_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, 3, 3).clone()
        root_pos = torch.zeros(T, 3)
        root_pos[:, 2] = 0.95
        fk_pos = _fk_positions(ki, root_pos, joint_rot)

        # Scale bones unevenly to simulate diffusion
        scaled_pos = fk_pos.clone()
        scales = {2: 1.1, 3: 1.05, 6: 0.95, 7: 1.08, 9: 1.5, 13: 1.3}
        for i in range(24):
            pi = ki.parent_indices[i]
            if pi < 0:
                continue
            s = scales.get(i, 1.0)
            bone = fk_pos[:, i] - fk_pos[:, pi]
            scaled_pos[:, i] = scaled_pos[:, pi] + bone * s

        retargeted = retarget_bone_lengths(scaled_pos, ki)

        for i in range(24):
            pi = ki.parent_indices[i]
            if pi < 0:
                continue
            orig_dir = scaled_pos[:, i] - scaled_pos[:, pi]
            orig_dir = orig_dir / orig_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            ret_dir = retargeted[:, i] - retargeted[:, pi]
            ret_dir = ret_dir / ret_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            cos_sim = (orig_dir * ret_dir).sum(dim=-1).mean()
            assert cos_sim > 0.9999, (
                f"{ki.body_names[i]}: direction changed, cos_sim={cos_sim:.6f}"
            )

    def test_retarget_gives_mjcf_bone_lengths(self):
        """Retargeted positions should have exact MJCF bone lengths."""
        ki = _load_ki()
        from closd_isaaclab.diffusion.robot_state_builder import retarget_bone_lengths

        T = 10
        joint_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, 3, 3).clone()
        root_pos = torch.zeros(T, 3)
        root_pos[:, 2] = 0.95
        fk_pos = _fk_positions(ki, root_pos, joint_rot)

        # Scale all bones by 1.15 (simulating HumanML3D skeleton)
        scaled_pos = fk_pos.clone()
        for i in range(24):
            pi = ki.parent_indices[i]
            if pi < 0:
                continue
            bone = fk_pos[:, i] - fk_pos[:, pi]
            scaled_pos[:, i] = scaled_pos[:, pi] + bone * 1.15

        retargeted = retarget_bone_lengths(scaled_pos, ki)

        for i in range(24):
            pi = ki.parent_indices[i]
            if pi < 0:
                continue
            ret_bone_len = (retargeted[:, i] - retargeted[:, pi]).norm(dim=-1).mean().item()
            mjcf_len = ki.local_pos[i].norm().item()
            assert abs(ret_bone_len - mjcf_len) < 1e-5, (
                f"{ki.body_names[i]}: bone len {ret_bone_len*1000:.2f} mm != MJCF {mjcf_len*1000:.2f} mm"
            )

    def test_retarget_then_ik_fk_is_exact(self):
        """Retarget → IK → FK should give 0 error (bones match MJCF)."""
        ki = _load_ki()
        from closd_isaaclab.diffusion.robot_state_builder import retarget_bone_lengths

        # Create positions with mismatched bone lengths
        T = 10
        joint_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, 3, 3).clone()
        root_pos = torch.zeros(T, 3)
        root_pos[:, 2] = 0.95
        fk_pos = _fk_positions(ki, root_pos, joint_rot)

        # Heavily scale bones (mimicking worst-case HumanML3D mismatch)
        scaled_pos = fk_pos.clone()
        for i in range(24):
            pi = ki.parent_indices[i]
            if pi < 0:
                continue
            bone = fk_pos[:, i] - fk_pos[:, pi]
            scale = 1.0 + 0.3 * ((i * 7) % 11) / 10.0  # 1.0 to 1.3
            scaled_pos[:, i] = scaled_pos[:, pi] + bone * scale

        # Retarget
        retargeted = retarget_bone_lengths(scaled_pos, ki)

        # IK → FK
        fk_pos_rt, _ = _ik_then_fk(ki, retargeted)

        print("\n--- Retarget (up to 30% bone mismatch) → IK → FK ---")
        per_joint = _print_per_joint_error(ki, retargeted, fk_pos_rt)

        mean_err = per_joint.mean().item()
        assert mean_err < 0.001, f"Retarget→IK→FK error: {mean_err*1000:.2f} mm (should be ~0)"

    def test_retarget_then_ik_fk_with_gt_motion(self):
        """GT motion (already MJCF bones) → retarget is no-op → IK/FK exact."""
        ki = _load_ki()
        from closd_isaaclab.diffusion.robot_state_builder import retarget_bone_lengths

        gt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(
                __import__("protomotions").__file__))),
            "examples/data/smpl_humanoid_sit_armchair.motion",
        )
        if not os.path.exists(gt_path):
            pytest.skip("GT motion not found")

        gt = torch.load(gt_path, map_location="cpu", weights_only=False)
        positions = gt["rigid_body_pos"][:50]

        retargeted = retarget_bone_lengths(positions, ki)

        # Should be a no-op (GT already has MJCF bone lengths)
        retarget_diff = (retargeted - positions).norm(dim=-1).mean().item()
        print(f"\nGT retarget diff: {retarget_diff*1000:.4f} mm (should be ~0)")
        assert retarget_diff < 1e-5, f"Retarget changed GT positions: {retarget_diff*1000:.4f} mm"

        # IK → FK should be exact
        fk_pos_rt, _ = _ik_then_fk(ki, retargeted)
        per_joint = (fk_pos_rt - retargeted).norm(dim=-1).mean(dim=0)
        mean_err = per_joint.mean().item()
        print(f"GT retarget→IK→FK error: {mean_err*1000:.4f} mm")
        assert mean_err < 0.001, f"Error: {mean_err*1000:.2f} mm"
