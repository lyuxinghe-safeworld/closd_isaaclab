#!/usr/bin/env python3
"""Diagnose horizontal drift in diffusion-generated motion.

Inspects the raw HML features and recovered positions to determine
whether root velocity (HML dims 1-2) is causing spurious drift.

Usage:
    python scripts/diagnose_drift.py --prompt "a person is jumping"
    python scripts/diagnose_drift.py --prompt "a person walks forward"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "code/CLoSD"))
sys.path.insert(0, str(Path.home() / "code/CLoSD_t2m_standalone"))
sys.path.insert(0, str(Path.home() / "code/ProtoMotions"))

import argparse
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num-seconds", type=float, default=8.0)
    args = parser.parse_args()

    from closd_isaaclab.diffusion.motion_provider import DiffusionMotionProvider

    provider = DiffusionMotionProvider()
    positions, hml_raw = provider.generate_standalone(args.prompt, args.num_seconds)
    # positions: [1, T, 22, 3]  (SMPL Y-up from recover_from_ric)
    # hml_raw:   [1, T, 263]    (unnormalized HML features)

    pos = positions[0].cpu()  # [T, 22, 3]
    hml = hml_raw[0].cpu()    # [T, 263]
    T = pos.shape[0]

    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"Frames: {T}, Duration: {T/20:.1f}s")
    print(f"{'='*60}")

    # ---------------------------------------------------------------
    # 1. HML root features
    # ---------------------------------------------------------------
    # dim 0: root angular velocity (Y-axis)
    # dims 1-2: root XZ linear velocity (in facing frame)
    # dim 3: root height (Y, absolute)
    root_ang_vel = hml[:, 0]       # [T]
    root_xz_vel = hml[:, 1:3]     # [T, 2] — velocity in facing frame
    root_height = hml[:, 3]        # [T]

    print(f"\n--- HML Root Velocity Features (dims 1-2) ---")
    print(f"  root_xz_vel shape: {root_xz_vel.shape}")
    print(f"  X-vel (dim1): mean={root_xz_vel[:,0].mean():.6f}, std={root_xz_vel[:,0].std():.6f}, "
          f"min={root_xz_vel[:,0].min():.6f}, max={root_xz_vel[:,0].max():.6f}")
    print(f"  Z-vel (dim2): mean={root_xz_vel[:,1].mean():.6f}, std={root_xz_vel[:,1].std():.6f}, "
          f"min={root_xz_vel[:,1].min():.6f}, max={root_xz_vel[:,1].max():.6f}")
    print(f"  Ang-vel (dim0): mean={root_ang_vel.mean():.6f}, std={root_ang_vel.std():.6f}")
    print(f"  Height (dim3): mean={root_height.mean():.4f}, min={root_height.min():.4f}, max={root_height.max():.4f}")

    # ---------------------------------------------------------------
    # 2. Recovered root position (from cumsum of velocities)
    # ---------------------------------------------------------------
    root_pos = pos[:, 0, :]  # [T, 3] — SMPL Y-up (X=right, Y=up, Z=forward)
    print(f"\n--- Recovered Root Position (joint 0, SMPL Y-up) ---")
    print(f"  Start: X={root_pos[0,0]:.4f}, Y={root_pos[0,1]:.4f}, Z={root_pos[0,2]:.4f}")
    print(f"  End:   X={root_pos[-1,0]:.4f}, Y={root_pos[-1,1]:.4f}, Z={root_pos[-1,2]:.4f}")
    total_xz_displacement = torch.sqrt(
        (root_pos[-1, 0] - root_pos[0, 0])**2 + (root_pos[-1, 2] - root_pos[0, 2])**2
    )
    print(f"  Total XZ displacement: {total_xz_displacement:.4f} m")
    print(f"  Total Y displacement:  {(root_pos[-1,1] - root_pos[0,1]):.4f} m")

    # Per-frame root displacement
    root_delta = torch.diff(root_pos, dim=0)  # [T-1, 3]
    root_xz_speed = torch.sqrt(root_delta[:, 0]**2 + root_delta[:, 2]**2) * 20  # m/s
    print(f"  Mean XZ speed: {root_xz_speed.mean():.4f} m/s")
    print(f"  Max XZ speed:  {root_xz_speed.max():.4f} m/s")

    # ---------------------------------------------------------------
    # 3. Foot positions — are feet moving?
    # ---------------------------------------------------------------
    # SMPL joints: 7=left_ankle, 8=right_ankle, 10=left_toe, 11=right_toe
    foot_joints = {"left_ankle": 7, "right_ankle": 8, "left_toe": 10, "right_toe": 11}
    print(f"\n--- Foot Joint Displacements ---")
    for name, idx in foot_joints.items():
        fpos = pos[:, idx, :]
        total_disp = torch.sqrt(
            (fpos[-1, 0] - fpos[0, 0])**2 + (fpos[-1, 2] - fpos[0, 2])**2
        )
        delta = torch.diff(fpos, dim=0)
        speed = torch.sqrt(delta[:, 0]**2 + delta[:, 2]**2) * 20
        print(f"  {name}: total_XZ_disp={total_disp:.4f}m, mean_speed={speed.mean():.4f}m/s, max_speed={speed.max():.4f}m/s")

    # ---------------------------------------------------------------
    # 4. RIC joint positions (relative to root) — are they moving?
    # ---------------------------------------------------------------
    # HML dims 4:67 = 21 joints * 3 = RIC positions (relative to root, in facing frame)
    ric = hml[:, 4:67].reshape(T, 21, 3)
    ric_delta = torch.diff(ric, dim=0)
    ric_speed = ric_delta.norm(dim=-1) * 20  # [T-1, 21] m/s
    print(f"\n--- RIC (Root-Relative) Joint Motion ---")
    print(f"  Mean speed across all joints: {ric_speed.mean():.4f} m/s")
    print(f"  Max speed across all joints:  {ric_speed.max():.4f} m/s")
    # Feet in RIC (indices offset by 1 since root is excluded): 6=L_ankle, 7=R_ankle, 9=L_toe, 10=R_toe
    for name, ric_idx in [("left_ankle", 6), ("right_ankle", 7), ("left_toe", 9), ("right_toe", 10)]:
        speed = ric_speed[:, ric_idx]
        print(f"  {name} (RIC): mean_speed={speed.mean():.4f}m/s, max_speed={speed.max():.4f}m/s")

    # ---------------------------------------------------------------
    # 5. Foot contacts from HML
    # ---------------------------------------------------------------
    foot_contacts = hml[:, 259:263]  # [T, 4]: r_ankle, r_toe, l_ankle, l_toe
    print(f"\n--- Foot Contacts (HML dims 259-262) ---")
    for i, name in enumerate(["r_ankle", "r_toe", "l_ankle", "l_toe"]):
        pct = (foot_contacts[:, i] > 0.5).float().mean() * 100
        print(f"  {name}: {pct:.1f}% contact")

    # ---------------------------------------------------------------
    # 6. Frame-by-frame root position (sample every N frames)
    # ---------------------------------------------------------------
    print(f"\n--- Root Position Over Time (every 20 frames = 1s) ---")
    print(f"  {'Frame':>6} {'Time':>6} {'X':>8} {'Y':>8} {'Z':>8} {'XZ_disp':>10}")
    ref = root_pos[0]
    for i in range(0, T, 20):
        disp = torch.sqrt((root_pos[i, 0] - ref[0])**2 + (root_pos[i, 2] - ref[2])**2)
        print(f"  {i:6d} {i/20:5.1f}s {root_pos[i,0]:8.4f} {root_pos[i,1]:8.4f} {root_pos[i,2]:8.4f} {disp:10.4f}")
    # Print last frame too
    if (T-1) % 20 != 0:
        i = T-1
        disp = torch.sqrt((root_pos[i, 0] - ref[0])**2 + (root_pos[i, 2] - ref[2])**2)
        print(f"  {i:6d} {i/20:5.1f}s {root_pos[i,0]:8.4f} {root_pos[i,1]:8.4f} {root_pos[i,2]:8.4f} {disp:10.4f}")

    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    print("  If root XZ displacement is large but foot RIC speeds are low,")
    print("  the drift is from HML root velocity features (dims 1-2),")
    print("  NOT from actual limb movement.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
