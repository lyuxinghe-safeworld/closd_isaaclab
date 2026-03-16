#!/usr/bin/env python3
"""Verify ProtoMotions tracker works with offline motion in Isaac Lab.

This tests the ProtoMotions motion tracker independently of the diffusion pipeline.
It loads a pretrained SMPL tracker and an offline .motion file, runs in Isaac Lab.

Usage:
    python scripts/verify_tracking.py \
        --checkpoint ~/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
        --motion-file ~/code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion \
        --num-envs 1
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_CHECKPOINT = (
    Path.home() / "code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt"
)
DEFAULT_MOTION_FILE = (
    Path.home() / "code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion"
)
INFERENCE_AGENT = Path.home() / "code/ProtoMotions/protomotions/inference_agent.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify ProtoMotions motion tracker with an offline motion file in Isaac Lab."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to tracker checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=str(DEFAULT_MOTION_FILE),
        help="Path to .motion file (default: %(default)s)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments (default: %(default)s)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        default="isaaclab",
        choices=["isaacgym", "isaaclab", "newton"],
        help="Simulator backend (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cmd = [
        sys.executable,
        str(INFERENCE_AGENT),
        "--checkpoint", args.checkpoint,
        "--motion-file", args.motion_file,
        "--num-envs", str(args.num_envs),
        "--simulator", args.simulator,
    ]

    if args.headless:
        cmd.append("--headless")

    print("Running command:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
