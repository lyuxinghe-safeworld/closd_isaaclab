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
        --record-frames 300

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
        "--record-frames",
        type=int,
        default=0,
        help="Number of frames to record (0 = no recording).",
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
    """Initialize the RotationSolver."""
    from closd_isaaclab.diffusion.rotation_solver import RotationSolver

    log.info("Initializing RotationSolver (mode=%s) ...", args.rotation_mode)
    solver = RotationSolver(mode=args.rotation_mode, device="cuda")
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

def run_closed_loop(env, agent, closd_manager, args):
    """Run the closed-loop CLoSD pipeline.

    Steps per sim step:
      1. Agent selects action from policy (conditioned on ref motion from CLoSDMotionLib)
      2. Env steps physics
      3. CLoSDMotionManager.post_physics_step() advances time, updates pose buffer,
         and triggers diffusion replanning when the horizon expires
    """
    log.info("=" * 60)
    log.info("Starting closed-loop CLoSD pipeline")
    log.info("  prompt         : %s", args.prompt)
    log.info("  episode_length : %d steps", args.episode_length)
    log.info("  num_envs       : %d", args.num_envs)
    log.info("=" * 60)

    # TODO: Wire closd_manager.get_body_positions_fn to the simulator's
    # body position query. This depends on the specific simulator API:
    #   - IsaacLab: env.simulator.get_body_positions()
    #   - IsaacGym: env.simulator.rigid_body_pos
    # The exact method name needs to be identified from the env/simulator instance.
    #
    # Example wiring:
    #   closd_manager.get_body_positions_fn = lambda: env.simulator.get_body_positions()

    # TODO: Replace the env's motion_manager with closd_manager so that
    # the mimic env uses CLoSD-generated reference motions. The exact attribute
    # name depends on the env class. Candidates:
    #   env.motion_manager = closd_manager
    #   env.mimic_motion_manager = closd_manager
    # Inspect env.__dict__ or env.__class__.__mro__ to find the right attribute.

    log.info(
        "NOTE: The closed-loop wiring requires iterative debugging. "
        "Running agent.evaluator.simple_test_policy() as the inference entry point."
    )

    # This runs the standard ProtoMotions inference loop, which will query
    # closd_motion_lib for reference states. With proper wiring, the
    # CLoSDMotionManager's post_physics_step hook will trigger replanning.
    try:
        agent.evaluator.simple_test_policy(collect_metrics=True)
    except Exception as e:
        log.error("Inference loop failed: %s", e)
        log.error(
            "This is expected during initial integration. "
            "See TODO markers for wiring that needs refinement."
        )
        raise


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

    # Phase 1: Diffusion model
    motion_provider = init_diffusion(args)

    # Phase 2: Rotation solver + RobotStateBuilder
    rotation_solver = init_rotation_solver(args)
    robot_state_builder = init_robot_state_builder(rotation_solver)

    # Phase 3: Load tracker configs
    tracker_configs = load_tracker_configs(args.tracker_checkpoint)

    # Phase 4: CLoSDMotionLib + CLoSDMotionManager
    closd_motion_lib = init_closd_motion_lib(robot_state_builder, device="cuda")
    closd_manager = init_closd_motion_manager(
        motion_lib=closd_motion_lib,
        motion_provider=motion_provider,
        robot_state_builder=robot_state_builder,
        text_prompt=args.prompt,
        num_envs=args.num_envs,
        env_dt=1.0 / 30.0,
        pred_len_20fps=motion_provider.pred_len,
        device="cuda",
    )

    # Phase 5: ProtoMotions env + agent
    log.info("Building ProtoMotions environment and agent ...")
    env, agent, simulator, device = build_protomotions_env_and_agent(
        tracker_configs=tracker_configs,
        closd_motion_lib=closd_motion_lib,
        args=args,
    )

    # Phase 6: Run closed-loop
    run_closed_loop(env, agent, closd_manager, args)


if __name__ == "__main__":
    main()
