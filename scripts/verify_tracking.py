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

    # ProtoMotions uses relative paths for USD assets — must run from its root
    protomotions_root = Path.home() / "code" / "ProtoMotions"

    import os
    env = os.environ.copy()

    # Isaac Sim needs libomniclient.so on LD_LIBRARY_PATH
    isaacsim_lib = Path.home() / "code" / "env_isaaclab" / "lib" / "python3.11" / "site-packages" / "isaacsim"
    omniclient_dir = isaacsim_lib / "kit" / "extscore" / "omni.client.lib" / "bin"
    if omniclient_dir.exists():
        env["LD_LIBRARY_PATH"] = str(omniclient_dir) + ":" + env.get("LD_LIBRARY_PATH", "")

    # NCCL fix for GCP VMs without InfiniBand
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_NET"] = "Socket"
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")

    print("Running command:")
    print(f"  cwd: {protomotions_root}")
    print(f"  cmd: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(protomotions_root), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
