"""CLoSDMotionLib: Duck-types MotionLib for MimicControl.get_context() and MimicMotionManager.

This module provides a drop-in replacement for ProtoMotions' MotionLib that serves
diffusion-generated motion states instead of pre-loaded .motion files.

ProtoMotions interface duck-typed:
  - num_motions() -> int
  - motion_lengths property -> tensor
  - motion_weights property -> tensor
  - motion_file attribute -> str
  - get_motion_state(motion_ids, motion_times, **kwargs) -> RobotState
  - get_motion_length(motion_ids) -> tensor
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)


class CLoSDMotionLib:
    """Duck-types MotionLib for MimicControl.get_context() and MimicMotionManager.

    Serves diffusion-generated motion states on demand, rather than pre-loaded
    .motion files. The underlying RobotStateBuilder is updated externally (by
    CLoSDMotionManager) whenever the diffusion planner produces a new horizon.

    Parameters
    ----------
    robot_state_builder : RobotStateBuilder
        Builder that caches the most recent diffusion-generated motion horizon.
    device : str
        PyTorch device string (default "cuda").
    horizon_duration : float
        Nominal length of one planning horizon in seconds (default 2.0).
        Used as the single motion length returned by motion_lengths.
    """

    def __init__(
        self,
        robot_state_builder,
        device: str = "cuda",
        horizon_duration: float = 2.0,
    ) -> None:
        self._builder = robot_state_builder
        self.device = device
        self._horizon_duration = horizon_duration

        # Stub attributes expected by MimicControl / MimicMotionManager
        self.motion_file: str = "closd_diffusion"
        self._motion_weights = torch.ones(1, dtype=torch.float32, device=device)
        self._motion_lengths = torch.tensor(
            [horizon_duration], dtype=torch.float32, device=device
        )

    # ------------------------------------------------------------------
    # MotionLib interface: properties and simple methods
    # ------------------------------------------------------------------

    @property
    def motion_lengths(self) -> Tensor:
        """Tensor of shape [1] containing the horizon duration in seconds."""
        return self._motion_lengths

    @motion_lengths.setter
    def motion_lengths(self, value: Tensor) -> None:
        """Allow CLoSDMotionManager (or tests) to update the horizon duration."""
        self._motion_lengths = value

    @property
    def motion_weights(self) -> Tensor:
        """Tensor of shape [1] with uniform weight (required by MimicMotionManager)."""
        return self._motion_weights

    def num_motions(self) -> int:
        """Return the number of motions (always 1 for diffusion-based lib)."""
        return 1

    def smooth_contacts(self, window_size: int) -> None:
        """No-op: diffusion-generated contacts don't need smoothing."""
        pass

    def get_motion_length(self, motion_ids) -> Tensor:
        """Return motion length(s) for the given motion_ids.

        Parameters
        ----------
        motion_ids : tensor or None
            Indices into motion_lengths. None returns all lengths.

        Returns
        -------
        Tensor
            Length(s) in seconds.
        """
        if motion_ids is None:
            return self._motion_lengths
        return self._motion_lengths[motion_ids]

    # ------------------------------------------------------------------
    # Core method: get_motion_state
    # ------------------------------------------------------------------

    def get_motion_state(
        self,
        motion_ids,
        motion_times: Tensor,
        **kwargs,
    ) -> RobotState:
        """Convert time stamps to frame indices and return the cached RobotState.

        Delegates to ``robot_state_builder.get_state_at_frames()``. If the
        builder has not yet been populated (``build()`` not called), returns a
        zero-filled RobotState so that the system can safely start up.

        Parameters
        ----------
        motion_ids : tensor
            Motion IDs (currently ignored since only one motion exists).
        motion_times : Tensor
            Shape [N], time within the motion horizon (seconds).
        **kwargs
            Ignored; present for API compatibility.

        Returns
        -------
        RobotState
            State at the requested times, with ``state_conversion=COMMON``.
        """
        dt = self._builder.dt
        num_bodies = self._builder.num_bodies

        # If builder hasn't been populated yet, return a zero-filled state
        if self._builder._positions is None:
            return self._make_zero_state(len(motion_times), num_bodies)

        # Determine the maximum valid frame index
        max_frame = self._builder._positions.shape[1] - 1  # T - 1

        # Convert times to integer frame indices
        frame_indices = (motion_times / dt).long().clamp(0, max_frame)

        # Retrieve state dict from builder
        # frame_indices is [N]; get_state_at_frames expects a list/tensor
        state_dict = self._builder.get_state_at_frames(frame_indices)

        # state_dict values have shape [bs, N, ...]; we need [N, ...]
        # The builder uses batch size (bs) on dim 0. We collapse by selecting
        # the first batch element and returning per-env (index along dim 1 -> dim 0).
        rigid_body_pos = self._squeeze_batch(state_dict["rigid_body_pos"])
        rigid_body_vel = self._squeeze_batch(state_dict["rigid_body_vel"])
        dof_pos = self._squeeze_batch(state_dict.get("dof_pos"))
        dof_vel = self._squeeze_batch(state_dict.get("dof_vel"))
        rigid_body_contacts = self._squeeze_batch(state_dict.get("rigid_body_contacts"))

        robot_state = RobotState(
            state_conversion=StateConversion.COMMON,
            rigid_body_pos=rigid_body_pos,
            rigid_body_vel=rigid_body_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            rigid_body_contacts=rigid_body_contacts,
        )
        return robot_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _squeeze_batch(self, tensor: Optional[Tensor]) -> Optional[Tensor]:
        """Collapse the batch dimension (dim 0) of a [bs, N, ...] tensor.

        When the builder produces data with shape [bs, N, ...], we need to
        return [N, ...] for MimicControl. We take the first batch element
        (index 0 along dim 0) since in the CLoSD pipeline bs==1 during
        reference-motion queries.

        Parameters
        ----------
        tensor : Tensor or None
            Input tensor with at least 2 dimensions.

        Returns
        -------
        Tensor or None
            Shape [N, ...], or None if input is None.
        """
        if tensor is None:
            return None
        if tensor.dim() >= 2:
            return tensor[0]  # [bs=1, N, ...] -> [N, ...]
        return tensor

    def _make_zero_state(self, n: int, num_bodies: int) -> RobotState:
        """Create a zero-filled RobotState for the not-yet-built case.

        Parameters
        ----------
        n : int
            Number of environments / time steps.
        num_bodies : int
            Number of rigid bodies.

        Returns
        -------
        RobotState
            Zero-filled state with state_conversion=COMMON.
        """
        dev = self.device
        return RobotState(
            state_conversion=StateConversion.COMMON,
            rigid_body_pos=torch.zeros(n, num_bodies, 3, device=dev),
            rigid_body_vel=torch.zeros(n, num_bodies, 3, device=dev),
            rigid_body_contacts=torch.zeros(n, num_bodies, device=dev),
        )
