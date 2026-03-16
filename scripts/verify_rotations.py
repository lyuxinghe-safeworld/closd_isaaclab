#!/usr/bin/env python3
"""Verify 6D rotation conversions and FK consistency.

Tests:
1. 6D round-trip: matrix -> 6D -> matrix (should be identity)
2. Ground truth: Load ProtoMotions .motion file, verify data loads correctly
3. cont6d_to_matrix output orthogonality check

Usage:
    python scripts/verify_rotations.py \
        --motion-file ~/code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path setup — must happen before any project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path.home() / "code/CLoSD"))
sys.path.insert(0, str(Path.home() / "code/CLoSD_t2m_standalone"))
sys.path.insert(0, str(Path.home() / "code/ProtoMotions"))

import torch

DEFAULT_MOTION_FILE = (
    Path.home() / "code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion"
)

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify 6D rotation conversions and FK consistency.",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=str(DEFAULT_MOTION_FILE),
        help="Path to .motion file for ground truth test (default: %(default)s)",
    )
    return parser.parse_args()


def random_rotation_matrices(n: int) -> torch.Tensor:
    """Generate n random proper rotation matrices (det=+1) via QR decomposition."""
    # Draw random matrices and QR-decompose
    rand = torch.randn(n, 3, 3)
    Q, R = torch.linalg.qr(rand)
    # Flip sign of rows so det(Q) = +1
    sign = torch.linalg.det(Q).sign().unsqueeze(-1).unsqueeze(-1)  # [n, 1, 1]
    Q = Q * sign
    return Q  # [n, 3, 3]


def test_roundtrip() -> bool:
    """Test 1: 6D round-trip — matrix -> 6D -> matrix."""
    from closd_isaaclab.diffusion.rotation_solver import cont6d_to_matrix, matrix_to_cont6d

    print("Test 1: 6D round-trip (matrix -> 6D -> matrix) ...")

    n = 100
    R_orig = random_rotation_matrices(n)               # [100, 3, 3]
    cont6d = matrix_to_cont6d(R_orig)                  # [100, 6]
    R_reconstructed = cont6d_to_matrix(cont6d)         # [100, 3, 3]

    max_err = (R_orig - R_reconstructed).abs().max().item()
    threshold = 1e-5
    passed = max_err < threshold
    status = PASS if passed else FAIL
    print(f"  Max reconstruction error: {max_err:.2e} (threshold: {threshold:.0e})")
    print(f"  Result: {status}")
    return passed


def test_orthogonality() -> bool:
    """Test 2: cont6d_to_matrix output orthogonality (R @ R^T ≈ I, det ≈ 1)."""
    from closd_isaaclab.diffusion.rotation_solver import cont6d_to_matrix

    print("Test 2: Orthogonality check (random 6D -> matrix) ...")

    n = 100
    # Random 6D vectors (not necessarily from a valid rotation matrix)
    cont6d = torch.randn(n, 6)
    R = cont6d_to_matrix(cont6d)  # [n, 3, 3]

    # R @ R^T should be identity
    eye = torch.eye(3).unsqueeze(0).expand(n, -1, -1)  # [n, 3, 3]
    RRt = torch.bmm(R, R.transpose(-1, -2))            # [n, 3, 3]
    ortho_err = (RRt - eye).abs().max().item()

    # det(R) should be +1
    dets = torch.linalg.det(R)  # [n]
    det_err = (dets - 1.0).abs().max().item()

    threshold_ortho = 1e-5
    threshold_det = 1e-5
    passed = ortho_err < threshold_ortho and det_err < threshold_det
    status = PASS if passed else FAIL
    print(f"  Max |R @ R^T - I| error: {ortho_err:.2e} (threshold: {threshold_ortho:.0e})")
    print(f"  Max |det(R) - 1| error:  {det_err:.2e} (threshold: {threshold_det:.0e})")
    print(f"  Result: {status}")
    return passed


def test_ground_truth_load(motion_file: str) -> bool:
    """Test 3: Load ProtoMotions .motion file and verify shapes."""
    print(f"Test 3: Ground truth load from '{motion_file}' ...")

    path = Path(motion_file).expanduser()
    if not path.exists():
        print(f"  Motion file not found: {path}")
        print(f"  Result: {FAIL}")
        return False

    try:
        data = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  Failed to load motion file: {exc}")
        print(f"  Result: {FAIL}")
        return False

    if not isinstance(data, dict):
        print(f"  Expected dict, got {type(data)}")
        print(f"  Result: {FAIL}")
        return False

    print(f"  Keys: {list(data.keys())}")

    # Check for position data
    pos_key = None
    for key in ("rigid_body_pos", "gts", "positions"):
        if key in data:
            pos_key = key
            break

    # Check for rotation data
    rot_key = None
    for key in ("rigid_body_rot", "local_rigid_body_rot", "grs", "rotations"):
        if key in data:
            rot_key = key
            break

    has_pos = pos_key is not None
    has_rot = rot_key is not None

    if has_pos:
        pos_tensor = data[pos_key]
        print(f"  Position data ({pos_key!r}): shape={tuple(pos_tensor.shape)}, dtype={pos_tensor.dtype}")
    else:
        print("  WARNING: No position data found (tried 'rigid_body_pos', 'gts', 'positions')")

    if has_rot:
        rot_tensor = data[rot_key]
        print(f"  Rotation data ({rot_key!r}): shape={tuple(rot_tensor.shape)}, dtype={rot_tensor.dtype}")
    else:
        print("  WARNING: No rotation data found (tried 'rigid_body_rot', 'local_rigid_body_rot', 'grs', 'rotations')")

    # Print fps if present
    if "fps" in data:
        print(f"  fps: {data['fps']}")

    passed = has_pos and has_rot
    status = PASS if passed else FAIL
    print(f"  Result: {status}")
    return passed


def main():
    args = parse_args()

    print("=" * 60)
    print("Rotation Conversion Verification")
    print("=" * 60)
    print()

    results = []

    results.append(test_roundtrip())
    print()

    results.append(test_orthogonality())
    print()

    results.append(test_ground_truth_load(args.motion_file))
    print()

    print("=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    overall = PASS if n_pass == n_total else FAIL
    print(f"Summary: {n_pass}/{n_total} tests passed — {overall}")
    print("=" * 60)

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
