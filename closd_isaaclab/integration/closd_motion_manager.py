"""CLoSDMotionManager: closed-loop diffusion motion manager with sim feedback.

Extends MimicMotionManager to maintain a sliding window pose buffer, trigger
diffusion replanning when the planning horizon is exhausted, and feed sim state
back to the diffusion model as the next prefix.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from torch import Tensor

from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

logger = logging.getLogger(__name__)


class CLoSDMotionManager(MimicMotionManager):
    """Closed-loop diffusion motion manager.

    Maintains a sliding-window pose buffer of simulator body positions, triggers
    diffusion replanning when the planning horizon expires, and updates the
    RobotStateBuilder cache with new horizon data.

    Parameters
    ----------
    config : MimicMotionManagerConfig
        Motion manager configuration (passed through to MimicMotionManager).
    num_envs : int
        Number of parallel simulation environments.
    env_dt : float
        Simulation timestep in seconds.
    device : str or torch.device
        PyTorch device.
    motion_lib : CLoSDMotionLib
        Duck-typed motion library that serves diffusion-generated states.
    motion_provider : DiffusionMotionProvider
        Wraps the DiP diffusion model; called during replanning.
    robot_state_builder : RobotStateBuilder
        Builds and caches RobotState-compatible data from diffusion output.
    text_prompt : str
        Natural-language motion description forwarded to the diffusion model.
    pred_len_20fps : int
        Number of frames predicted per diffusion step at 20 fps (DiP's native rate).
        The equivalent horizon at 30 fps is derived automatically.
    context_len_30fps : int
        Length (in 30 fps frames) of the sliding-window pose buffer used as
        context / prefix for the next diffusion call.
    get_body_positions_fn : callable, optional
        Callable ``() -> Tensor`` that returns the current body positions from
        the simulator with shape ``[num_envs, 24, 3]``.  Required for
        post_physics_step to update the pose buffer.
    """

    def __init__(
        self,
        config: MimicMotionManagerConfig,
        num_envs: int,
        env_dt: float,
        device,
        motion_lib,
        motion_provider,
        robot_state_builder,
        text_prompt: str = "",
        pred_len_20fps: int = 40,
        context_len_30fps: int = 30,
        get_body_positions_fn: Optional[Callable[[], Tensor]] = None,
    ):
        super().__init__(config, num_envs, env_dt, device, motion_lib)

        self.motion_provider = motion_provider
        self.robot_state_builder = robot_state_builder
        self.text_prompt = text_prompt

        # Derived planning horizon at 30 fps (sim rate)
        self.planning_horizon_30fps: int = int(pred_len_20fps * 30 / 20)

        # Sliding-window pose buffer: [num_envs, context_len_30fps, 24, 3]
        self.context_len_30fps = context_len_30fps
        self.pose_buffer = torch.zeros(
            num_envs, context_len_30fps, 24, 3,
            dtype=torch.float32,
            device=device,
        )

        # Most recent reconstruction data from hml_conversion (per-env alignment)
        self.recon_data: Optional[dict] = None

        # Counts sim steps since the last replan; triggers replan when it
        # reaches a multiple of planning_horizon_30fps
        self.frame_counter: int = 0

        # Callback to fetch current body positions from the simulator
        self.get_body_positions_fn = get_body_positions_fn

    # ------------------------------------------------------------------
    # Core simulation loop hooks
    # ------------------------------------------------------------------

    def post_physics_step(self):
        """Advance motion time and update pose buffer; trigger replanning if needed."""
        # 1. Advance motion_times (inherited behaviour)
        super().post_physics_step()

        # 2. Update pose buffer with current simulator body positions
        if self.get_body_positions_fn is not None:
            current_positions = self.get_body_positions_fn()
            # current_positions: [num_envs, 24, 3]
            self._append_to_pose_buffer(current_positions)

        # 3. Advance frame counter
        self.frame_counter += 1

        # 4. Replan when a full horizon has elapsed
        if self.frame_counter % self.planning_horizon_30fps == 0:
            self._replan()

    def _append_to_pose_buffer(self, new_positions: Tensor) -> None:
        """Slide the pose buffer forward by one frame.

        Drops the oldest frame and appends *new_positions* at the end.

        Parameters
        ----------
        new_positions : Tensor
            Shape [num_envs, 24, 3] — body positions from the current sim step.
        """
        # Drop oldest frame, shift left, append new frame at the end
        # pose_buffer[:, :-1] <- pose_buffer[:, 1:]
        self.pose_buffer[:, :-1, :, :] = self.pose_buffer[:, 1:, :, :]
        self.pose_buffer[:, -1, :, :] = new_positions

    # ------------------------------------------------------------------
    # Replanning
    # ------------------------------------------------------------------

    def _replan(self) -> None:
        """Call the diffusion model and update the motion library cache.

        Steps
        -----
        1. Call ``motion_provider.generate_next_horizon`` with the current pose
           buffer, previous reconstruction data, and the text prompt.
        2. Check the output for NaN values; if found, log a warning and keep
           the previous horizon (skip the update).
        3. Call ``robot_state_builder.build`` to populate the motion cache.
        4. Store the updated ``recon_data`` for the next iteration.
        5. Reset ``motion_times`` to 0 so the environment replays from the
           start of the new horizon.
        6. Update ``motion_lib.motion_lengths`` so that ``get_done_tracks``
           returns the correct boundary.
        """
        # 1. Generate next horizon from diffusion model
        positions_isaac, hml_raw, recon_data_new = (
            self.motion_provider.generate_next_horizon(
                self.pose_buffer,
                self.recon_data,
                self.text_prompt,
            )
        )

        # 2. NaN check — keep last valid horizon if generation is degenerate
        if torch.isnan(positions_isaac).any() or torch.isnan(hml_raw).any():
            logger.warning(
                "CLoSDMotionManager._replan: NaN detected in diffusion output — "
                "keeping previous horizon."
            )
            return

        # 3. Update robot state builder cache
        self.robot_state_builder.build(positions_isaac, hml_raw)

        # 4. Store reconstruction data for next call
        self.recon_data = recon_data_new

        # 5. Reset motion times to replay from beginning of new horizon
        self.motion_times[:] = 0.0

        # 6. Update motion_lib so MimicMotionManager.get_done_tracks() uses the
        #    correct horizon length.
        horizon_duration = self.planning_horizon_30fps * self.env_dt
        self.motion_lib.motion_lengths = torch.tensor(
            [horizon_duration],
            dtype=torch.float32,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Reset / environment reset
    # ------------------------------------------------------------------

    def sample_motions(
        self, env_ids: torch.Tensor, new_motion_ids: Optional[torch.Tensor] = None
    ) -> None:
        """Reset environments and trigger an initial diffusion replan.

        On reset:
          1. Fill ``pose_buffer[env_ids]`` with the current body pose repeated
             along the context window dimension.
          2. Clear ``recon_data`` so the next replan starts fresh.
          3. Reset ``frame_counter`` to 0.
          4. Reset ``motion_times[env_ids]`` to 0.
          5. Trigger an initial ``_replan()`` to populate the motion cache.

        Parameters
        ----------
        env_ids : Tensor
            Indices of environments being reset.
        new_motion_ids : Tensor, optional
            Ignored (CLoSD uses a single diffusion-generated motion).
        """
        # 1. Seed pose_buffer for reset environments with the current body pose
        if self.get_body_positions_fn is not None:
            current_positions = self.get_body_positions_fn()  # [num_envs, 24, 3]
            # Repeat current pose over the whole context window
            # current_positions[env_ids]: [len(env_ids), 24, 3]
            seed_poses = current_positions[env_ids]  # [len_ids, 24, 3]
            # Expand to [len_ids, context_len_30fps, 24, 3]
            seed_poses_expanded = seed_poses.unsqueeze(1).expand(
                -1, self.context_len_30fps, -1, -1
            )
            self.pose_buffer[env_ids] = seed_poses_expanded

        # 2. Clear reconstruction data — next replan starts from scratch
        self.recon_data = None

        # 3. Reset frame counter
        self.frame_counter = 0

        # 4. Reset motion times for the specified environments
        self.motion_times[env_ids] = 0.0

        # 5. Trigger initial replan so the motion cache is populated
        self._replan()
