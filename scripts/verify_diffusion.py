#!/usr/bin/env python3
"""Verify CLoSD diffusion model generates reasonable motion from text prompts.

No simulator required. Outputs:
  - motion.pt: raw 263-dim HumanML3D motion
  - xyz.pt: decoded 22-joint 3D positions
  - {prompt_slug}.mp4: matplotlib 3D skeleton animation

Usage:
    python scripts/verify_diffusion.py --prompt "a person walks forward" --num-seconds 8
"""

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path setup — must happen before any project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path.home() / "code/CLoSD"))
sys.path.insert(0, str(Path.home() / "code/CLoSD_t2m_standalone"))
sys.path.insert(0, str(Path.home() / "code/ProtoMotions"))

import torch

DEFAULT_MODEL_PATH = (
    Path.home()
    / "code/CLoSD/closd/diffusion_planner/save"
    / "DiP_no-target_10steps_context20_predict40/model000200000.pt"
)
DEFAULT_MEAN_PATH = (
    Path.home()
    / "code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_mean.npy"
)
DEFAULT_STD_PATH = (
    Path.home()
    / "code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_std.npy"
)


def slugify(text: str) -> str:
    """Convert a text prompt to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80]  # cap length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify CLoSD diffusion model generates reasonable motion from text prompts.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired motion.",
    )
    parser.add_argument(
        "--num-seconds",
        type=float,
        default=8.0,
        help="Generation duration in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=2.5,
        help="Classifier-free guidance scale (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/verify_diffusion",
        help="Directory for output files (default: %(default)s).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to DiP checkpoint .pt file (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Create DiffusionMotionProvider
    # ------------------------------------------------------------------
    from closd_isaaclab.diffusion.motion_provider import DiffusionMotionProvider

    print(f"Loading DiP model from: {args.model_path}")
    provider = DiffusionMotionProvider(
        model_path=args.model_path,
        mean_path=str(DEFAULT_MEAN_PATH),
        std_path=str(DEFAULT_STD_PATH),
        guidance=args.guidance,
    )

    # ------------------------------------------------------------------
    # 2. Generate motion
    # ------------------------------------------------------------------
    print(f"Generating motion for prompt: '{args.prompt}'")
    print(f"  Duration : {args.num_seconds}s")
    print(f"  Guidance : {args.guidance}")

    positions, hml_raw = provider.generate_standalone(args.prompt, args.num_seconds)
    # positions : [1, T, 22, 3]
    # hml_raw   : [1, T, 263]

    num_frames = positions.shape[1]
    fps = 20

    # ------------------------------------------------------------------
    # 3. Save motion.pt and xyz.pt
    # ------------------------------------------------------------------
    motion_path = output_dir / "motion.pt"
    xyz_path = output_dir / "xyz.pt"

    torch.save(hml_raw, motion_path)
    torch.save(positions, xyz_path)

    print(f"Saved motion.pt  -> {motion_path}")
    print(f"Saved xyz.pt     -> {xyz_path}")

    # ------------------------------------------------------------------
    # 4. Render MP4
    # ------------------------------------------------------------------
    slug = slugify(args.prompt)
    mp4_path = output_dir / f"{slug}.mp4"

    print(f"Rendering animation to: {mp4_path}")
    try:
        from standalone_t2m.render import render_xyz_motion
        render_xyz_motion(positions, args.prompt, mp4_path, fps=fps)
    except Exception as e:
        print(f"  Vendored renderer failed ({type(e).__name__}: {e})")
        print(f"  Using fallback renderer...")
        _render_skeleton_fallback(positions[0].cpu().numpy(), mp4_path, args.prompt, fps=fps)

    # ------------------------------------------------------------------
    # 5. Print summary stats
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Generation complete")
    print(f"  Prompt    : {args.prompt}")
    print(f"  Frames    : {num_frames}")
    print(f"  FPS       : {fps}")
    print(f"  Duration  : {num_frames / fps:.2f}s")
    print(f"  Guidance  : {args.guidance}")
    print(f"  Output dir: {output_dir.resolve()}")
    print("=" * 60)


def _render_skeleton_fallback(xyz: "np.ndarray", mp4_path: Path, title: str, fps: int = 20):
    """Fallback matplotlib skeleton renderer when vendored code has compatibility issues.

    Args:
        xyz: [T, 22, 3] joint positions (numpy).
        mp4_path: Output path.
        title: Plot title.
        fps: Frame rate.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    # HumanML3D kinematic chain (joint connectivity)
    kinematic_chain = [
        [0, 2, 5, 8, 11],      # right leg
        [0, 1, 4, 7, 10],      # left leg
        [0, 3, 6, 9, 12, 15],  # spine + head
        [9, 14, 17, 19, 21],   # right arm
        [9, 13, 16, 18, 20],   # left arm
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    T, J, _ = xyz.shape
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Compute axis limits from data
    mins = xyz.min(axis=(0, 1))
    maxs = xyz.max(axis=(0, 1))
    center = (mins + maxs) / 2
    span = (maxs - mins).max() * 0.6

    def update(frame):
        ax.clear()
        ax.set_title(f"{title}\nFrame {frame}/{T}", fontsize=10)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        pts = xyz[frame]
        for chain, color in zip(kinematic_chain, colors):
            chain_pts = pts[chain]
            ax.plot3D(chain_pts[:, 0], chain_pts[:, 1], chain_pts[:, 2], color=color, linewidth=2)
        ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], s=10, c="black", zorder=5)

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(str(mp4_path), writer=writer)
    plt.close(fig)
    print(f"  Saved: {mp4_path}")


if __name__ == "__main__":
    main()
