#!/usr/bin/env python3
"""Generate motion from text via DiP diffusion, convert to .motion, play in Isaac Lab and record.

End-to-end pipeline:
  1. DiP generates HumanML3D motion from text prompt
  2. Convert to ProtoMotions .motion format (positions, rotations, DOFs, velocities)
  3. Play via ProtoMotions tracker in Isaac Lab and record video

Usage:
    cd ~/code/ProtoMotions
    python ~/code/closd_isaaclab/scripts/generate_and_record.py \
        --prompt "a person is dancing" \
        --num-seconds 6 \
        --record-frames 200
"""
import argparse
import os
import sys
import time
from pathlib import Path

PROTO_ROOT = Path.home() / "code" / "ProtoMotions"
CLOSD_ROOT = Path.home() / "code" / "CLoSD"
STANDALONE_ROOT = Path.home() / "code" / "CLoSD_t2m_standalone"
PROJECT_ROOT = Path.home() / "code" / "closd_isaaclab"

for p in [str(CLOSD_ROOT), str(PROTO_ROOT), str(STANDALONE_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# IsaacLab must be imported before torch
from protomotions.utils.simulator_imports import import_simulator_before_torch
AppLauncher = import_simulator_before_torch("isaaclab")

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


def generate_diffusion_motion(prompt, num_seconds, guidance, device):
    """Generate motion from text via DiP diffusion."""
    from closd_isaaclab.diffusion.motion_provider import DiffusionMotionProvider

    provider = DiffusionMotionProvider(
        model_path=str(CLOSD_ROOT / "closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000200000.pt"),
        mean_path=str(STANDALONE_ROOT / "standalone_t2m/assets/t2m_mean.npy"),
        std_path=str(STANDALONE_ROOT / "standalone_t2m/assets/t2m_std.npy"),
        device=device,
        guidance=guidance,
    )

    log.info("Generating %.1fs of motion for: '%s'", num_seconds, prompt)
    positions, hml_raw = provider.generate_standalone(prompt, num_seconds)
    log.info("  Generated: positions %s, hml_raw %s", list(positions.shape), list(hml_raw.shape))
    return positions, hml_raw


def convert_to_motion_file(positions_smpl, hml_raw, output_path, device):
    """Convert diffusion output to ProtoMotions .motion format.

    ProtoMotions .motion files contain:
      - rigid_body_pos: [T, num_bodies, 3]
      - rigid_body_rot: [T, num_bodies, 4]  (xyzw)
      - rigid_body_vel: [T, num_bodies, 3]
      - rigid_body_ang_vel: [T, num_bodies, 3]
      - dof_pos: [T, num_dofs]
      - dof_vel: [T, num_dofs]
      - rigid_body_contacts: [T, num_bodies]
      - fps: float
    """
    from closd_isaaclab.utils.coord_transform import CoordTransform
    from closd_isaaclab.utils.fps_convert import fps_convert
    from closd_isaaclab.diffusion.rotation_solver import RotationSolver, cont6d_to_matrix, wxyz_quat_to_matrix
    from closd_isaaclab.diffusion.hml_conversion import recover_root_rot_pos

    ct = CoordTransform()
    T_20fps = positions_smpl.shape[1]

    # positions_smpl is [1, T, 22, 3] in SMPL space at 20fps
    # Convert to Isaac Lab space (24 joints, mujoco order)
    pos_isaac_20 = ct.smpl_to_isaac(positions_smpl[0].to(device))  # [T, 24, 3]

    # Upsample to 30fps
    pos_isaac_30 = fps_convert(pos_isaac_20.unsqueeze(0), 20, 30)[0]  # [T_30, 24, 3]
    T_30 = pos_isaac_30.shape[0]

    # Compute velocities via finite differencing (30fps)
    dt = 1.0 / 30.0
    vel = torch.zeros_like(pos_isaac_30)
    if T_30 >= 3:
        vel[1:-1] = (pos_isaac_30[2:] - pos_isaac_30[:-2]) / (2 * dt)
        vel[0] = (pos_isaac_30[1] - pos_isaac_30[0]) / dt
        vel[-1] = (pos_isaac_30[-1] - pos_isaac_30[-2]) / dt

    # Extract rotations from HML 6D (dims 67-192, local/parent-relative)
    hml = hml_raw[0].to(device)  # [T_20, 263]
    rot_6d = hml[:, 67:193].reshape(T_20fps, 21, 6)

    from closd_isaaclab.diffusion.rotation_solver import cont6d_to_matrix
    local_rot_21 = cont6d_to_matrix(rot_6d)  # [T, 21, 3, 3]

    # Get root rotation
    r_rot_quat, _ = recover_root_rot_pos(hml.unsqueeze(0))  # [1, T, 4] wxyz
    root_mat = wxyz_quat_to_matrix(r_rot_quat[0])  # [T, 3, 3]

    # Assemble 24-joint local rotations: root + 21 joints + 2 hands (identity)
    identity = torch.eye(3, device=device).expand(T_20fps, 1, 3, 3)
    local_rot_24 = torch.cat([
        root_mat.unsqueeze(1),  # root
        local_rot_21,           # 21 joints
        identity.expand(T_20fps, 2, 3, 3),  # 2 hands
    ], dim=1)  # [T_20, 24, 3, 3]

    # Convert local rotation matrices to quaternions (xyzw for COMMON format)
    def mat_to_quat_xyzw(mat):
        """Batch rotation matrix to xyzw quaternion."""
        from protomotions.utils.rotations import matrix_to_quaternion
        q = matrix_to_quaternion(mat, w_last=True)  # xyzw
        return q

    rot_quat_20 = mat_to_quat_xyzw(local_rot_24.reshape(-1, 3, 3)).reshape(T_20fps, 24, 4)

    # For the .motion file we need GLOBAL rotations, not local.
    # ProtoMotions stores global rigid body rotations.
    # We need FK to get global rotations from local ones.
    # For now, use identity rotations as placeholder — the tracker primarily
    # uses positions for observation building.
    # TODO: Use ProtoMotions FK to compute global rotations
    global_rot = torch.zeros(T_20fps, 24, 4, device=device)
    global_rot[..., 3] = 1.0  # identity xyzw

    # Upsample rotations to 30fps (simple repeat for quaternions)
    # Better: SLERP, but repeat is acceptable for initial testing
    rot_30 = fps_convert(global_rot.unsqueeze(0), 20, 30)[0]  # [T_30, 24, 4]
    # Renormalize after interpolation
    rot_30 = rot_30 / (rot_30.norm(dim=-1, keepdim=True) + 1e-8)

    # Angular velocity (zeros for now)
    ang_vel = torch.zeros(T_30, 24, 3, device=device)

    # DOF positions from local rotations -> exp_map
    # For each 3-DOF joint, the local rotation matrix encodes the joint angles
    # Convert to axis-angle (exp_map) representation
    # Convert local rotation matrices to exp_map (axis-angle) for dof_pos
    from protomotions.utils.rotations import quat_to_exp_map, matrix_to_quaternion
    local_rot_joints = local_rot_24[:, 1:, :, :]  # exclude root, [T, 23, 3, 3]
    joint_quats = matrix_to_quaternion(local_rot_joints.reshape(-1, 3, 3), w_last=True)  # xyzw
    dof_pos_20 = quat_to_exp_map(joint_quats, w_last=True).reshape(T_20fps, 69)
    dof_pos_30 = fps_convert(dof_pos_20.unsqueeze(0), 20, 30)[0]  # [T_30, 69]

    # DOF velocity
    dof_vel = torch.zeros_like(dof_pos_30)
    if T_30 >= 3:
        dof_vel[1:-1] = (dof_pos_30[2:] - dof_pos_30[:-2]) / (2 * dt)
        dof_vel[0] = (dof_pos_30[1] - dof_pos_30[0]) / dt
        dof_vel[-1] = (dof_pos_30[-1] - dof_pos_30[-2]) / dt

    # Contacts from HML (dims 259-262)
    foot_contacts_20 = hml[:, 259:263]  # [T_20, 4]
    contacts_20 = torch.zeros(T_20fps, 24, device=device)
    contacts_20[:, 3] = foot_contacts_20[:, 0]   # L_Ankle
    contacts_20[:, 4] = foot_contacts_20[:, 1]   # L_Toe
    contacts_20[:, 7] = foot_contacts_20[:, 2]   # R_Ankle
    contacts_20[:, 8] = foot_contacts_20[:, 3]   # R_Toe
    contacts_30 = fps_convert(contacts_20.unsqueeze(0), 20, 30)[0]
    contacts_30 = (contacts_30 > 0.5).float()  # binarize after interpolation

    # Build .motion dict
    motion_data = {
        "rigid_body_pos": pos_isaac_30.cpu(),
        "rigid_body_rot": rot_30.cpu(),
        "rigid_body_vel": vel.cpu(),
        "rigid_body_ang_vel": ang_vel.cpu(),
        "dof_pos": dof_pos_30.cpu(),
        "dof_vel": dof_vel.cpu(),
        "rigid_body_contacts": contacts_30.cpu(),
        "fps": 30,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(motion_data, output_path)
    log.info("Saved .motion file: %s (%d frames at 30fps = %.1fs)",
             output_path, T_30, T_30 / 30.0)
    return output_path


def record_in_isaaclab(motion_file, checkpoint, record_frames, num_envs):
    """Play motion in Isaac Lab tracker and record video."""
    from dataclasses import asdict
    from lightning.fabric import Fabric
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components

    checkpoint = Path(checkpoint)
    configs = torch.load(checkpoint.parent / "resolved_configs_inference.pt", map_location="cpu", weights_only=False)

    robot_config = configs["robot"]
    simulator_config = configs["simulator"]
    terrain_config = configs.get("terrain")
    scene_lib_config = configs["scene_lib"]
    motion_lib_config = configs["motion_lib"]
    env_config = configs["env"]
    agent_config = configs["agent"]

    simulator_config.num_envs = num_envs
    simulator_config.headless = False
    motion_lib_config.motion_file = motion_file

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    fabric_config = FabricConfig(devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric = Fabric(**asdict(fabric_config))
    fabric.launch()
    device = fabric.device

    app_launcher = AppLauncher({"headless": False, "device": str(device)})

    terrain_config, simulator_config = convert_friction_for_simulator(terrain_config, simulator_config)

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=device,
        simulation_app=app_launcher.app,
    )

    from protomotions.envs.base_env.env import BaseEnv
    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config, robot_config=robot_config, device=device,
        terrain=components["terrain"], scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"], simulator=components["simulator"],
    )

    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent)
    agent.setup()
    agent.load(str(checkpoint), load_env=False)

    simulator = components["simulator"]

    # Auto-start recording
    log.info("Recording %d frames in Isaac Lab...", record_frames)
    simulator._toggle_video_record()

    obs, extras = env.reset()
    for step in range(record_frames):
        obs_td = agent.obs_dict_to_tensordict(obs)
        model_outs = agent.model(obs_td)
        actions = model_outs["mean_action"]
        obs, rewards, dones, terminated, extras = env.step(actions)

        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_indices) > 0:
            obs, extras = env.reset(done_indices)

        if (step + 1) % 100 == 0:
            log.info("  Step %d / %d", step + 1, record_frames)

    # Stop recording
    simulator._toggle_video_record()
    time.sleep(2)

    # Find the recorded frames and compile
    rendering_dir = Path("output/renderings")
    if rendering_dir.exists():
        subdirs = sorted(rendering_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if subdirs:
            frame_dir = subdirs[0]
            frames = sorted(frame_dir.glob("*.png"))
            if frames:
                output_mp4 = str(PROJECT_ROOT / "outputs" / "dancing_isaaclab.mp4")
                os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y", "-framerate", "30",
                    "-i", str(frame_dir / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-preset", "veryfast",
                    output_mp4,
                ], check=True)
                log.info("Video saved to: %s (%d frames)", output_mp4, len(frames))
                return output_mp4

    log.warning("No frames found for video compilation")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a person is dancing")
    parser.add_argument("--num-seconds", type=float, default=6.0)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--record-frames", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--checkpoint", default=str(PROTO_ROOT / "data/pretrained_models/motion_tracker/smpl/last.ckpt"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Generate motion
    positions, hml_raw = generate_diffusion_motion(args.prompt, args.num_seconds, args.guidance, device)

    # Step 2: Convert to .motion
    motion_file = str(PROJECT_ROOT / "outputs" / "generated_motion.motion")
    convert_to_motion_file(positions, hml_raw, motion_file, device)

    # Step 3: Record in Isaac Lab
    video_path = record_in_isaaclab(motion_file, args.checkpoint, args.record_frames, args.num_envs)

    if video_path:
        log.info("="*60)
        log.info("Done! Video: %s", video_path)
        log.info("="*60)


if __name__ == "__main__":
    main()
