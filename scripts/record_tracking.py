#!/usr/bin/env python3
"""Record ProtoMotions tracker playing an offline motion in Isaac Lab.

Automatically starts recording on launch and stops after --record-frames.
Requires DISPLAY to be set (TurboVNC).

Usage:
    export DISPLAY=:1
    python scripts/record_tracking.py --record-frames 300

Output: video saved to ProtoMotions/viewer_recordings/
"""
import argparse
import os
import sys
import time
import threading
from pathlib import Path

# Must set PYTHONPATH before imports
PROTO_ROOT = Path.home() / "code" / "ProtoMotions"
sys.path.insert(0, str(PROTO_ROOT))

# IsaacLab must be imported before torch
from protomotions.utils.simulator_imports import import_simulator_before_torch
AppLauncher = import_simulator_before_torch("isaaclab")

import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(PROTO_ROOT / "data/pretrained_models/motion_tracker/smpl/last.ckpt"))
    parser.add_argument("--motion-file", default=str(PROTO_ROOT / "examples/data/smpl_humanoid_sit_armchair.motion"))
    parser.add_argument("--record-frames", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    resolved_path = checkpoint.parent / "resolved_configs_inference.pt"
    log.info("Loading configs from %s", resolved_path)
    configs = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config = configs["robot"]
    simulator_config = configs["simulator"]
    terrain_config = configs.get("terrain")
    scene_lib_config = configs["scene_lib"]
    motion_lib_config = configs["motion_lib"]
    env_config = configs["env"]
    agent_config = configs["agent"]

    # Overrides
    simulator_config.num_envs = args.num_envs
    simulator_config.headless = False  # Need viewer for recording
    motion_lib_config.motion_file = args.motion_file

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # Fabric
    from dataclasses import asdict
    from lightning.fabric import Fabric
    from protomotions.utils.fabric_config import FabricConfig

    fabric_config = FabricConfig(devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric = Fabric(**asdict(fabric_config))
    fabric.launch()
    device = fabric.device

    # AppLauncher
    app_launcher = AppLauncher({"headless": False, "device": str(device)})

    # Friction
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    terrain_config, simulator_config = convert_friction_for_simulator(terrain_config, simulator_config)

    # Build components
    from protomotions.utils.component_builder import build_all_components
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=device,
        simulation_app=app_launcher.app,
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # Create env
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.utils.hydra_replacement import get_class

    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # Create agent
    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent
    )
    agent.setup()
    agent.load(str(checkpoint), load_env=False)

    # Auto-start recording
    log.info("Auto-starting video recording for %d frames...", args.record_frames)
    simulator._toggle_video_record()

    # Run policy for N frames then stop recording
    record_frames = args.record_frames
    step_count = 0

    obs, extras = env.reset()
    while step_count < record_frames:
        obs_td = agent.obs_dict_to_tensordict(obs)
        model_outs = agent.model(obs_td)
        actions = model_outs["mean_action"]
        obs, rewards, dones, terminated, extras = env.step(actions)

        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_indices) > 0:
            obs, extras = env.reset(done_indices)

        step_count += 1
        if step_count % 100 == 0:
            log.info("  Step %d / %d", step_count, record_frames)

    # Stop recording — this triggers video compilation
    log.info("Stopping recording and compiling video...")
    simulator._toggle_video_record()

    # Give it a moment to finalize
    time.sleep(2)

    log.info("Done! Check ProtoMotions/viewer_recordings/ for the output video.")


if __name__ == "__main__":
    main()
