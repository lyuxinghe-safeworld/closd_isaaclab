#!/usr/bin/env python3
"""Full CLoSD-IsaacLab closed-loop text-to-motion pipeline.

Generates motion from text via DiP diffusion, tracks it in Isaac Lab
using ProtoMotions' pretrained SMPL motion tracker, with closed-loop
sim state feedback to the diffusion model.

Usage:
    python scripts/run_closd_isaaclab.py \
        --prompt "a person walks forward" \
        --rotation-mode diffusion \
        --episode-length 300 \
        --clip-length 5

The script has two execution modes:

  1. Full pipeline (default): Initializes DiP diffusion, ProtoMotions tracker,
     CLoSDMotionLib/Manager, and runs closed-loop tracking in Isaac Lab.

  2. Diffusion-only fallback (--diffusion-only): Runs standalone diffusion
     generation, prints joint position statistics, and exits. Useful for
     verifying the diffusion model independently of ProtoMotions.

Setup:
    cd /home/lyuxinghe/code/closd_isaaclab
    source /home/lyuxinghe/code/env_isaaclab/bin/activate
    export PYTHONPATH="/home/lyuxinghe/code/CLoSD:/home/lyuxinghe/code/ProtoMotions:/home/lyuxinghe/code/CLoSD_t2m_standalone:$PYTHONPATH"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_DIP_CHECKPOINT = (
    "/home/lyuxinghe/code/CLoSD/closd/diffusion_planner/save/"
    "DiP_no-target_10steps_context20_predict40/model000200000.pt"
)
DEFAULT_TRACKER_CHECKPOINT = (
    "/home/lyuxinghe/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt"
)
DEFAULT_MEAN_PATH = (
    "/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_mean.npy"
)
DEFAULT_STD_PATH = (
    "/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_std.npy"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLoSD-IsaacLab: closed-loop text-to-motion in Isaac Lab",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Text prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default="a person walks forward",
        help="Natural-language motion description.",
    )

    # Diffusion settings
    parser.add_argument(
        "--rotation-mode",
        type=str,
        default="diffusion",
        choices=["diffusion"],
        help="How to extract joint rotations from diffusion output.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale for DiP.",
    )

    # Episode settings
    parser.add_argument(
        "--episode-length",
        type=int,
        default=300,
        help="Maximum episode length in sim steps (at 30 fps).",
    )
    parser.add_argument(
        "--clip-length",
        type=float,
        default=0,
        help="Max video clip length in seconds (0 = use full motion duration).",
    )
    parser.add_argument(
        "--video-output",
        type=str,
        default="",
        help="Path for output video (default: outputs/closd_pipeline/<prompt>.mp4).",
    )

    # Environment settings
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without viewer.",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        default="isaaclab",
        choices=["isaacgym", "isaaclab"],
        help="Simulator backend.",
    )

    # Checkpoints
    parser.add_argument(
        "--tracker-checkpoint",
        type=str,
        default=DEFAULT_TRACKER_CHECKPOINT,
        help="Path to ProtoMotions tracker checkpoint.",
    )
    parser.add_argument(
        "--dip-checkpoint",
        type=str,
        default=DEFAULT_DIP_CHECKPOINT,
        help="Path to DiP diffusion checkpoint.",
    )
    parser.add_argument(
        "--mean-path",
        type=str,
        default=DEFAULT_MEAN_PATH,
        help="Path to HumanML3D mean .npy.",
    )
    parser.add_argument(
        "--std-path",
        type=str,
        default=DEFAULT_STD_PATH,
        help="Path to HumanML3D std .npy.",
    )

    # Mode selection
    parser.add_argument(
        "--diffusion-only",
        action="store_true",
        default=False,
        help="Run diffusion standalone (skip ProtoMotions tracker).",
    )

    return parser


# Parse arguments before any heavy imports (isaaclab must be imported before torch)
parser = create_parser()
args, unknown_args = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Conditional simulator import (must happen before torch)
# ---------------------------------------------------------------------------
AppLauncher = None
if not args.diffusion_only:
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import torch and everything else
import torch  # noqa: E402


# ===================================================================
# Phase 1: Diffusion model initialization
# ===================================================================

def init_diffusion(args) -> "DiffusionMotionProvider":
    """Initialize the DiP diffusion motion provider."""
    from closd_isaaclab.diffusion.motion_provider import DiffusionMotionProvider

    log.info("Initializing DiP diffusion model ...")
    log.info("  checkpoint : %s", args.dip_checkpoint)
    log.info("  guidance   : %s", args.guidance)

    provider = DiffusionMotionProvider(
        model_path=args.dip_checkpoint,
        mean_path=args.mean_path,
        std_path=args.std_path,
        device="cuda",
        guidance=args.guidance,
    )
    log.info(
        "  context_len=%d  pred_len=%d",
        provider.context_len,
        provider.pred_len,
    )
    return provider


# ===================================================================
# Phase 2: Rotation solver and RobotStateBuilder
# ===================================================================

def init_rotation_solver(args):
    """Initialize the RotationSolver with kinematic_info for dof_pos extraction."""
    from closd_isaaclab.diffusion.rotation_solver import RotationSolver

    log.info("Initializing RotationSolver (mode=%s) ...", args.rotation_mode)

    # Load kinematic_info from ProtoMotions MJCF — needed to convert
    # local rotation matrices to dof_pos via extract_qpos_from_transforms
    kinematic_info = None
    try:
        from protomotions.components.pose_lib import extract_kinematic_info
        mjcf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__import__("protomotions").__file__))),
            "protomotions", "data", "assets", "mjcf", "smpl_humanoid.xml",
        )
        if os.path.exists(mjcf_path):
            kinematic_info = extract_kinematic_info(mjcf_path)
            log.info("  Loaded kinematic_info from %s", mjcf_path)
        else:
            log.warning("  MJCF not found at %s — dof_pos will not be computed", mjcf_path)
    except Exception as e:
        log.warning("  Failed to load kinematic_info: %s", e)

    solver = RotationSolver(
        mode=args.rotation_mode, device="cuda", kinematic_info=kinematic_info
    )
    return solver


def init_robot_state_builder(rotation_solver):
    """Initialize the RobotStateBuilder."""
    from closd_isaaclab.diffusion.robot_state_builder import RobotStateBuilder

    log.info("Initializing RobotStateBuilder ...")
    builder = RobotStateBuilder(
        dt=1.0 / 30.0,
        rotation_solver=rotation_solver,
        num_bodies=24,
    )
    return builder


# ===================================================================
# Phase 3: ProtoMotions tracker config loading
# ===================================================================

def load_tracker_configs(checkpoint_path: str):
    """Load resolved configs from the ProtoMotions tracker checkpoint.

    Returns
    -------
    dict
        Dictionary with keys: robot, simulator, terrain, scene_lib,
        motion_lib, env, agent.
    """
    ckpt_path = Path(checkpoint_path)
    resolved_path = ckpt_path.parent / "resolved_configs_inference.pt"

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Could not find resolved_configs_inference.pt at {resolved_path}. "
            f"Ensure the tracker checkpoint directory contains this file."
        )

    log.info("Loading tracker configs from %s ...", resolved_path)
    configs = torch.load(resolved_path, map_location="cpu", weights_only=False)

    log.info("  Loaded config keys: %s", list(configs.keys()))
    return configs


# ===================================================================
# Phase 4: CLoSDMotionLib and CLoSDMotionManager
# ===================================================================

def init_closd_motion_lib(robot_state_builder, device="cuda"):
    """Create the CLoSDMotionLib that duck-types MotionLib."""
    from closd_isaaclab.integration.closd_motion_lib import CLoSDMotionLib

    log.info("Initializing CLoSDMotionLib ...")
    motion_lib = CLoSDMotionLib(
        robot_state_builder=robot_state_builder,
        device=device,
        horizon_duration=2.0,
    )
    return motion_lib


def init_closd_motion_manager(
    motion_lib,
    motion_provider,
    robot_state_builder,
    text_prompt: str,
    num_envs: int,
    env_dt: float,
    pred_len_20fps: int,
    device="cuda",
):
    """Create the CLoSDMotionManager with closed-loop replanning."""
    from closd_isaaclab.integration.closd_motion_manager import CLoSDMotionManager
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    log.info("Initializing CLoSDMotionManager ...")
    log.info("  text_prompt     : %s", text_prompt)
    log.info("  num_envs        : %d", num_envs)
    log.info("  pred_len_20fps  : %d", pred_len_20fps)

    config = MimicMotionManagerConfig()
    manager = CLoSDMotionManager(
        config=config,
        num_envs=num_envs,
        env_dt=env_dt,
        device=device,
        motion_lib=motion_lib,
        motion_provider=motion_provider,
        robot_state_builder=robot_state_builder,
        text_prompt=text_prompt,
        pred_len_20fps=pred_len_20fps,
        context_len_30fps=30,  # ~1 second at 30 fps
        get_body_positions_fn=None,  # Wired later after env creation
    )
    return manager


# ===================================================================
# Phase 5: ProtoMotions environment and agent wiring
# ===================================================================

def build_protomotions_env_and_agent(
    tracker_configs: dict,
    closd_motion_lib,
    args,
):
    """Build the ProtoMotions environment and agent with CLoSD motion lib.

    This wires together:
    - Tracker configs (robot, simulator, terrain, scene_lib, env, agent)
    - CLoSDMotionLib as the motion source (replaces file-based MotionLib)
    - ProtoMotions env + agent for physics-based tracking

    Returns
    -------
    env, agent, simulator
    """
    from dataclasses import asdict

    from lightning.fabric import Fabric

    from protomotions.utils.component_builder import (
        build_scene_lib_from_config,
        build_simulator_from_config,
        build_terrain_from_config,
    )
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    robot_config = tracker_configs["robot"]
    simulator_config = tracker_configs["simulator"]
    terrain_config = tracker_configs.get("terrain")
    scene_lib_config = tracker_configs["scene_lib"]
    env_config = tracker_configs["env"]
    agent_config = tracker_configs["agent"]

    # --- Make robot asset path absolute (resolves relative USD path issue) ---
    asset_root = getattr(robot_config.asset, "asset_root", "")
    if asset_root and not os.path.isabs(asset_root):
        import protomotions
        proto_root = Path(protomotions.__file__).parent.parent
        abs_root = str(proto_root / asset_root)
        log.info("  asset_root: %s -> %s", asset_root, abs_root)
        robot_config.asset.asset_root = abs_root

    # --- Apply overrides ---
    simulator_config.num_envs = args.num_envs
    simulator_config.headless = args.headless

    # Switch simulator if needed
    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        log.info(
            "Switching simulator from '%s' to '%s'",
            current_simulator,
            args.simulator,
        )
        from protomotions.simulator.factory import update_simulator_config_for_test
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # Override episode length
    if hasattr(env_config, "max_episode_length"):
        env_config.max_episode_length = args.episode_length
        log.info("  max_episode_length -> %d", args.episode_length)

    # --- Fabric ---
    fabric_config = FabricConfig(
        devices=1,
        num_nodes=1,
        loggers=[],
        callbacks=[],
    )
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()
    device = fabric.device

    # --- IsaacLab AppLauncher ---
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {
            "headless": args.headless,
            "device": str(device),
        }
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    # --- Friction conversion ---
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # --- Build terrain, scene_lib, simulator ---
    terrain = build_terrain_from_config(
        terrain_config, simulator_config.num_envs, device
    )
    scene_lib = build_scene_lib_from_config(
        scene_lib_config, simulator_config.num_envs, device, terrain
    )

    # NOTE: We intentionally skip build_motion_lib_from_config because we use
    # CLoSDMotionLib instead. The closd_motion_lib is passed directly to the env.

    simulator = build_simulator_from_config(
        simulator_config, robot_config, terrain, scene_lib, device,
        **simulator_extra_params,
    )

    # --- Create env ---
    # TODO: The env expects a MotionLib; we pass closd_motion_lib which duck-types it.
    # If the env constructor does type-specific operations on motion_lib (e.g. calling
    # methods not on CLoSDMotionLib), this is where debugging starts.
    from protomotions.envs.base_env.env import BaseEnv
    EnvClass = get_class(env_config._target_)
    log.info("Creating env: %s", env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=closd_motion_lib,
        simulator=simulator,
    )

    # --- Create agent ---
    agent_kwargs = {"root_dir": Path(args.tracker_checkpoint).parent}
    AgentClass = get_class(agent_config._target_)
    log.info("Creating agent: %s", agent_config._target_)
    agent = AgentClass(
        config=agent_config, env=env, fabric=fabric, **agent_kwargs
    )
    agent.setup()
    agent.load(args.tracker_checkpoint, load_env=False)

    return env, agent, simulator, device


# ===================================================================
# Phase 6: Closed-loop execution
# ===================================================================

def _smpl_globals_from_locals(local_rot_mats: torch.Tensor) -> torch.Tensor:
    """Compute SMPL global rotation matrices from SMPL local (parent-relative) rotations.

    Chains through the SMPL kinematic tree (24 joints in SMPL order).

    Parameters
    ----------
    local_rot_mats : Tensor
        [T, 24, 3, 3] local rotation matrices in SMPL joint order.

    Returns
    -------
    Tensor
        [T, 24, 3, 3] global rotation matrices in SMPL joint order.
    """
    # SMPL parent indices (24 joints: 22 SMPL + 2 hands)
    SMPL_PARENTS = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
    ]
    T = local_rot_mats.shape[0]
    global_rot = torch.zeros_like(local_rot_mats)
    for i in range(24):
        if SMPL_PARENTS[i] == -1:
            global_rot[:, i] = local_rot_mats[:, i]
        else:
            global_rot[:, i] = global_rot[:, SMPL_PARENTS[i]] @ local_rot_mats[:, i]
    return global_rot


def generate_motion_file(motion_provider, prompt, output_dir, rotation_solver):
    """Generate diffusion motion and save as ProtoMotions .motion file.

    Follows the same rotation conversion pipeline as ProtoMotions'
    convert_amass_to_proto.py:
      1. Extract SMPL local rotations from HML features
      2. Compute SMPL global rotations (chain through SMPL tree)
      3. Apply coordinate frame rotation (SMPL Y-up → Isaac Z-up)
      4. Reorder globals to MJCF joint order
      5. Extract MJCF-compatible joint rotations
      6. Run FK for internally-consistent positions, rotations, velocities

    Returns path to the .motion file.
    """
    import re
    from closd_isaaclab.utils.coord_transform import CoordTransform, smpl_2_mujoco
    from protomotions.components.pose_lib import (
        fk_from_transforms_with_velocities,
        extract_qpos_from_transforms,
        compute_angular_velocity,
        compute_joint_rot_mats_from_global_mats,
    )
    from protomotions.utils.rotations import matrix_to_quaternion

    ki = rotation_solver.kinematic_info

    log.info("Generating diffusion motion for: '%s'", prompt)
    positions_smpl, hml_raw = motion_provider.generate_standalone(prompt, num_seconds=8.0)
    positions_smpl = positions_smpl.cpu()
    hml_raw = hml_raw.cpu()
    log.info("  Generated: %s positions, %s HML", list(positions_smpl.shape), list(hml_raw.shape))

    # Save skeleton video
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(positions_smpl, output_dir / "xyz.pt")
    try:
        slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:50]
        mp4_path = output_dir / f"{slug}_skeleton.mp4"
        try:
            from standalone_t2m.render import render_xyz_motion
            render_xyz_motion(positions_smpl, prompt, mp4_path, fps=20)
        except Exception:
            from scripts.verify_diffusion import _render_skeleton_fallback
            _render_skeleton_fallback(positions_smpl[0].numpy(), mp4_path, prompt, fps=20)
        log.info("  Saved skeleton video: %s", mp4_path)
    except Exception as e:
        log.warning("  Failed to render skeleton video: %s", e)

    T_20 = hml_raw.shape[1]

    # Convert diffusion positions to Isaac Z-up coords (MJCF body order)
    ct = CoordTransform()
    pos_isaac_20 = ct.smpl_to_isaac(positions_smpl[0])[:T_20]  # [T_20, 24, 3]
    root_pos_isaac = pos_isaac_20[:, 0, :]  # [T_20, 3] — Z-up

    # ---------------------------------------------------------------
    # Rotation pipeline: analytical IK from Isaac-space positions
    # ---------------------------------------------------------------
    # The diffusion model's 6D rotations (hml_raw[67:193]) are unreliable
    # for arm joints — they produce T-pose arms while positions show correct
    # arm-down poses. Instead, derive rotations from positions via analytical
    # IK, which guarantees position-rotation consistency.
    from closd_isaaclab.diffusion.robot_state_builder import analytical_ik

    ki = rotation_solver.kinematic_info
    global_rot_mjcf = analytical_ik(pos_isaac_20, ki)  # [T_20, 24, 3, 3]

    # Extract MJCF-compatible local joint rotations
    joint_rot_mjcf = compute_joint_rot_mats_from_global_mats(
        ki, global_rot_mjcf
    )  # [T_20, 24, 3, 3]

    # FK with Isaac root_pos and MJCF joint rotations
    # Produces internally-consistent positions, rotations, velocities
    motion = fk_from_transforms_with_velocities(
        ki, root_pos_isaac, joint_rot_mjcf, fps=20, compute_velocities=True
    )

    # Log FK vs diffusion position consistency
    fk_pos = motion.rigid_body_pos  # [T_20, 24, 3]
    fk_diff = (fk_pos - pos_isaac_20).norm(dim=-1).mean()
    log.info("  FK vs diffusion position diff: %.4f m (mean per-joint)", fk_diff.item())

    # ---------------------------------------------------------------
    # Extract remaining fields for .motion file
    # ---------------------------------------------------------------
    # Use DIFFUSION positions for rigid_body_pos so the red ball markers
    # match the diffusion skeleton video exactly (same source: recover_from_ric).
    # Use FK-derived rotations for rigid_body_rot (correct facing direction
    # from the rotation pipeline).
    # The tracker weights positions (0.5) more than rotations (0.3), so it
    # primarily follows the diffusion positions while maintaining correct
    # orientation from the FK rotations.

    # Fix height on diffusion positions: ensure feet above ground
    min_z = pos_isaac_20[:, :, 2].min()
    height_shift = max(0.015 - min_z.item(), 0.0)
    pos_isaac_20_fixed = pos_isaac_20.clone()
    pos_isaac_20_fixed[:, :, 2] += height_shift
    if height_shift > 0:
        log.info("  Height fix: shifted Z by %.4f m", height_shift)

    # Diffusion-based velocities (finite difference on diffusion positions)
    diff_vel = torch.zeros_like(pos_isaac_20_fixed)
    if T_20 >= 3:
        diff_vel[1:-1] = (pos_isaac_20_fixed[2:] - pos_isaac_20_fixed[:-2]) * (20 / 2)
        diff_vel[0] = (pos_isaac_20_fixed[1] - pos_isaac_20_fixed[0]) * 20
        diff_vel[-1] = (pos_isaac_20_fixed[-1] - pos_isaac_20_fixed[-2]) * 20

    # Local rotation quaternions (MJCF order) for MotionLib interpolation
    local_rot_quat = matrix_to_quaternion(
        joint_rot_mjcf.reshape(-1, 3, 3), w_last=True
    ).reshape(T_20, 24, 4)

    # DOF positions and velocities
    qpos = extract_qpos_from_transforms(
        ki, root_pos_isaac, joint_rot_mjcf, multi_dof_decomposition_method="exp_map"
    )
    dof_pos = qpos[:, 7:]  # [T_20, 69]
    dof_vel = compute_angular_velocity(joint_rot_mjcf[:, 1:, :, :], fps=20)
    dof_vel = dof_vel.reshape(T_20, -1)  # [T_20, 69]

    # Contacts from HML
    hml = hml_raw[0][:T_20]
    fc = hml[:, 259:263]
    contacts = torch.zeros(T_20, 24)
    contacts[:, 3] = (fc[:, 0] > 0.5).float()
    contacts[:, 4] = (fc[:, 1] > 0.5).float()
    contacts[:, 7] = (fc[:, 2] > 0.5).float()
    contacts[:, 8] = (fc[:, 3] > 0.5).float()

    # Prepend stabilization frames (3 seconds at 20fps = 60 frames)
    STAB = 60
    def _prepend_still(t, n):
        return torch.cat([t[0:1].expand(n, *t.shape[1:]), t])

    # rigid_body_pos: diffusion positions (matches skeleton video)
    # rigid_body_rot: FK-derived rotations (correct facing from rotation pipeline)
    # rigid_body_vel: diffusion-based velocities (consistent with positions)
    # rigid_body_ang_vel: FK-derived angular velocities (consistent with rotations)
    motion_dict = {
        "rigid_body_pos": _prepend_still(pos_isaac_20_fixed, STAB),
        "rigid_body_rot": _prepend_still(motion.rigid_body_rot, STAB),
        "rigid_body_vel": torch.cat([torch.zeros(STAB, 24, 3), diff_vel]),
        "rigid_body_ang_vel": torch.cat([torch.zeros(STAB, 24, 3), motion.rigid_body_ang_vel]),
        "dof_pos": _prepend_still(dof_pos, STAB),
        "dof_vel": torch.cat([torch.zeros(STAB, 69), dof_vel]),
        "rigid_body_contacts": _prepend_still(contacts, STAB),
        "local_rigid_body_rot": _prepend_still(local_rot_quat, STAB),
        "fps": 20,
    }

    motion_path = str((output_dir / "generated.motion").resolve())
    torch.save(motion_dict, motion_path)
    T_total = motion_dict["rigid_body_pos"].shape[0]
    log.info("  Saved .motion: %s (%d frames, %.1fs)", motion_path, T_total, T_total / 20)
    log.info("  Stabilization prefix: %d frames (%.1fs)", STAB, STAB / 20)

    return motion_path


def run_with_motion_file(motion_path, args):
    """Run ProtoMotions tracker with recording and auto-termination.

    Builds the env/agent inline (no subprocess), captures viewport frames
    as PNGs each step, and compiles to MP4 on exit.  Frames are written
    incrementally so the video is recoverable even after a force-quit.
    """
    import atexit
    import re
    from dataclasses import asdict

    from lightning.fabric import Fabric

    from protomotions.utils.component_builder import build_all_components
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    # ------------------------------------------------------------------
    # Determine duration / max steps
    # ------------------------------------------------------------------
    motion_data = torch.load(motion_path, map_location="cpu", weights_only=False)
    motion_fps = motion_data.get("fps", 20)
    motion_frames = motion_data["rigid_body_pos"].shape[0]
    motion_duration_s = motion_frames / motion_fps

    sim_fps = 30
    clip_s = args.clip_length if args.clip_length > 0 else motion_duration_s
    clip_s = min(clip_s, motion_duration_s)
    max_steps = int(clip_s * sim_fps)

    log.info("=" * 60)
    log.info("Running tracker with recording")
    log.info("  motion_file     : %s", motion_path)
    log.info("  motion_duration : %.1f s  (%d frames @ %d fps)",
             motion_duration_s, motion_frames, motion_fps)
    log.info("  clip_length     : %.1f s  (%d sim steps @ %d fps)",
             clip_s, max_steps, sim_fps)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    if args.video_output:
        video_path = Path(args.video_output)
    else:
        slug = re.sub(r"[^a-z0-9]+", "_", args.prompt.lower()).strip("_")[:50]
        video_path = Path("outputs/closd_pipeline") / f"{slug}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    frames_dir = video_path.with_suffix("") / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    log.info("  frames_dir      : %s", frames_dir)
    log.info("  video_output    : %s", video_path)

    # ------------------------------------------------------------------
    # Compile-video helper (called on normal exit AND via atexit)
    # ------------------------------------------------------------------
    _video_compiled = False

    def compile_video():
        nonlocal _video_compiled
        if _video_compiled:
            return
        _video_compiled = True
        pngs = sorted(frames_dir.glob("*.png"))
        if not pngs:
            log.warning("No frames captured — skipping video compilation.")
            return
        log.info("Compiling %d frames → %s ...", len(pngs), video_path)
        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            clip = ImageSequenceClip([str(p) for p in pngs], fps=sim_fps)
            clip.write_videofile(
                str(video_path),
                codec="libx264",
                audio=False,
                threads=4,
                preset="veryfast",
                ffmpeg_params=[
                    "-profile:v", "main",
                    "-level", "4.0",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-crf", "23",
                ],
            )
            log.info("Video saved: %s", video_path)
        except Exception as e:
            log.error("Failed to compile video: %s", e)

    atexit.register(compile_video)

    # ------------------------------------------------------------------
    # Load tracker configs
    # ------------------------------------------------------------------
    ckpt_path = Path(args.tracker_checkpoint)
    resolved_path = ckpt_path.parent / "resolved_configs_inference.pt"
    configs = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config = configs["robot"]
    simulator_config = configs["simulator"]
    terrain_config = configs.get("terrain")
    scene_lib_config = configs["scene_lib"]
    motion_lib_config = configs["motion_lib"]
    env_config = configs["env"]
    agent_config = configs["agent"]

    # Make robot asset path absolute
    asset_root = getattr(robot_config.asset, "asset_root", "")
    if asset_root and not os.path.isabs(asset_root):
        import protomotions
        proto_root = Path(protomotions.__file__).parent.parent
        robot_config.asset.asset_root = str(proto_root / asset_root)

    # Switch simulator if needed
    current_sim = simulator_config._target_.split(".")[-3]
    if args.simulator != current_sim:
        from protomotions.simulator.factory import update_simulator_config_for_test
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    simulator_config.num_envs = args.num_envs
    simulator_config.headless = args.headless
    motion_lib_config.motion_file = motion_path

    if hasattr(env_config, "max_episode_length"):
        env_config.max_episode_length = max(args.episode_length, max_steps + 100)

    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------
    fabric_config = FabricConfig(devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric = Fabric(**asdict(fabric_config))
    fabric.launch()
    device = fabric.device

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=device,
        **simulator_extra_params,
    )

    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config, env=env, fabric=fabric,
        root_dir=ckpt_path.parent,
    )
    agent.setup()
    agent.load(args.tracker_checkpoint, load_env=False)

    simulator = components["simulator"]

    # ------------------------------------------------------------------
    # Simulation loop with frame capture
    # ------------------------------------------------------------------
    agent.eval()
    done_indices = None
    log.info("Starting simulation loop (%d steps) ...", max_steps)

    try:
        for step in range(max_steps):
            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)

            model_outs = agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))

            obs, rewards, dones, terminated, extras = env.step(actions)
            obs = agent.add_agent_info_to_obs(obs)

            # Capture frame
            if not args.headless:
                frame_path = str(frames_dir / f"{step:06d}.png")
                simulator._write_viewport_to_file(frame_path)

            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

            if step % 100 == 0:
                log.info("  step %d / %d", step, max_steps)

    except KeyboardInterrupt:
        log.info("Interrupted at step %d — compiling video from captured frames...", step)

    # Compile video (also registered with atexit as fallback)
    compile_video()
    log.info("Done.")
    return 0


# ===================================================================
# Diffusion-only fallback
# ===================================================================

def run_diffusion_only(args):
    """Run standalone diffusion generation without ProtoMotions tracker.

    Useful for verifying the diffusion pipeline independently.
    """
    log.info("=" * 60)
    log.info("Running DIFFUSION-ONLY mode (no tracker)")
    log.info("=" * 60)

    provider = init_diffusion(args)
    rotation_solver = init_rotation_solver(args)

    num_seconds = args.episode_length / 30.0
    log.info("Generating %.1f seconds of motion for: '%s'", num_seconds, args.prompt)

    positions, hml_raw = provider.generate_standalone(
        text_prompt=args.prompt,
        num_seconds=num_seconds,
    )

    log.info("  positions shape : %s", list(positions.shape))
    log.info("  hml_raw shape   : %s", list(hml_raw.shape))
    log.info(
        "  positions range : [%.3f, %.3f]",
        positions.min().item(),
        positions.max().item(),
    )

    # Extract rotations
    local_rots, dof_pos, consistency = rotation_solver.solve(
        positions, hml_raw=hml_raw
    )
    log.info("  local_rots shape: %s", list(local_rots.shape))
    if dof_pos is not None:
        log.info("  dof_pos shape   : %s", list(dof_pos.shape))
    if consistency is not None:
        log.info("  FK consistency  : %.4f m", consistency)

    # Build RobotState
    builder = init_robot_state_builder(rotation_solver)
    builder.build(positions, hml_raw)
    state = builder.get_state_at_frames(list(range(min(5, positions.shape[1]))))
    log.info(
        "  RobotState sample: pos=%s, vel=%s",
        list(state["rigid_body_pos"].shape) if state["rigid_body_pos"] is not None else None,
        list(state["rigid_body_vel"].shape) if state["rigid_body_vel"] is not None else None,
    )

    log.info("Diffusion-only run complete.")
    return positions, hml_raw


# ===================================================================
# Main
# ===================================================================

def main():
    global args
    args = parser.parse_args()

    log.info("CLoSD-IsaacLab pipeline")
    log.info("  prompt             : %s", args.prompt)
    log.info("  rotation_mode      : %s", args.rotation_mode)
    log.info("  episode_length     : %d", args.episode_length)
    log.info("  guidance           : %.1f", args.guidance)
    log.info("  num_envs           : %d", args.num_envs)
    log.info("  headless           : %s", args.headless)
    log.info("  diffusion_only     : %s", args.diffusion_only)

    # ----- Diffusion-only shortcut -----
    if args.diffusion_only:
        run_diffusion_only(args)
        return

    # ===== Full pipeline =====
    # Architecture: generate .motion file from diffusion, then use ProtoMotions'
    # standard inference pipeline (same as verify_tracking.py). This ensures
    # FK-consistent reference data that the tracker can actually track.

    # Phase 1: Diffusion model
    motion_provider = init_diffusion(args)

    # Phase 2: Rotation solver (for local rotation extraction from HML 6D)
    rotation_solver = init_rotation_solver(args)

    # Phase 3: Generate diffusion motion → .motion file
    motion_path = generate_motion_file(
        motion_provider, args.prompt, "outputs/closd_pipeline", rotation_solver
    )

    # Phase 4: Run tracker with .motion file via standard ProtoMotions pipeline
    rc = run_with_motion_file(motion_path, args)
    if rc != 0:
        log.error("Tracker exited with code %d", rc)


if __name__ == "__main__":
    main()
