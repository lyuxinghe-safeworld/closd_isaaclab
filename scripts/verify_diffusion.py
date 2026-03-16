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
    from standalone_t2m.render import render_xyz_motion

    slug = slugify(args.prompt)
    mp4_path = output_dir / f"{slug}.mp4"

    print(f"Rendering animation to: {mp4_path}")
    render_xyz_motion(positions, args.prompt, mp4_path, fps=fps)

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


if __name__ == "__main__":
    main()
