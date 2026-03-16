"""DiP (Diffusion Planner) motion provider for standalone and closed-loop generation.

Wraps CLoSD's DiP model via the CLoSD_t2m_standalone package, providing:
- generate_standalone: open-loop generation from a text prompt
- generate_next_horizon: closed-loop generation with simulator feedback
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import torch

from closd_isaaclab.diffusion.hml_conversion import HMLConversion

if TYPE_CHECKING:
    from standalone_t2m.config import LoadedModelBundle


class DiffusionMotionProvider:
    """Wraps the DiP diffusion model for motion generation.

    Parameters
    ----------
    model_path : str or Path
        Path to the DiP checkpoint (.pt file).
    mean_path : str or Path
        Path to HumanML3D mean .npy file.
    std_path : str or Path
        Path to HumanML3D std .npy file.
    device : str
        Torch device string.
    guidance : float
        Classifier-free guidance scale.
    context_len : int
        Number of context frames for autoregressive generation.
    pred_len : int
        Number of predicted frames per diffusion step.
    """

    def __init__(
        self,
        model_path: str = "/home/lyuxinghe/code/CLoSD/closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000200000.pt",
        mean_path: str = "/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_mean.npy",
        std_path: str = "/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_std.npy",
        device: str = "cuda",
        guidance: float = 5.0,
        context_len: int = 20,
        pred_len: int = 40,
    ):
        self.device = device
        self.guidance = guidance
        self.context_len = context_len
        self.pred_len = pred_len

        # Load mean/std as tensors
        self.mean = torch.from_numpy(np.load(mean_path)).float()
        self.std = torch.from_numpy(np.load(std_path)).float()

        # HML conversion utility
        self.hml_conversion = HMLConversion(self.mean, self.std, device=device)

        # Build checkpoint bundle and load model (import here to avoid
        # pulling in heavy vendor deps at module-import time)
        from standalone_t2m.checkpoint import CheckpointBundle
        from standalone_t2m.config import build_model_and_diffusion

        model_path = Path(model_path)
        bundle = CheckpointBundle(
            model_path=model_path,
            args_path=model_path.parent / "args.json",
            mean_path=Path(mean_path),
            std_path=Path(std_path),
        )
        self.model_bundle: LoadedModelBundle = build_model_and_diffusion(bundle)

        # Override context/pred lens if they differ from args
        self.context_len = self.model_bundle.context_len
        self.pred_len = self.model_bundle.pred_len

        # Text embedding cache to avoid re-encoding the same prompt
        self._text_embed_cache: Dict[str, torch.Tensor] = {}

    def generate_standalone(
        self,
        text_prompt: str,
        num_seconds: float = 8.0,
        prefix_mode: str = "standing",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Open-loop generation without simulator feedback.

        Parameters
        ----------
        text_prompt : str
            Natural-language motion description.
        num_seconds : float
            Duration of the generated motion in seconds.
        prefix_mode : str
            How to initialize the prefix. Currently only "standing" is supported.

        Returns
        -------
        positions : torch.Tensor
            [1, T, 22, 3] joint positions in XYZ (HumanML convention).
        hml_raw : torch.Tensor
            [1, T, 263] unnormalized HumanML3D features.
        """
        from standalone_t2m.decode import decode_to_xyz
        from standalone_t2m.generation import compose_output_motion, generate_motion
        from standalone_t2m.prefix.standing import build_standing_prefix

        # 1. Build prefix
        if prefix_mode == "standing":
            prefix = build_standing_prefix(self.context_len).to(self.model_bundle.device)
        else:
            raise ValueError(f"Unknown prefix_mode: {prefix_mode}")

        # 2. Compute target frames at 20 fps (HumanML3D rate)
        target_frames = int(num_seconds * 20)

        # 3. Generate motion autoregressively
        generated = generate_motion(
            self.model_bundle,
            prompt=text_prompt,
            target_frames=target_frames,
            guidance=self.guidance,
            prefix=prefix,
        )

        # 4. Compose full motion (prefix + generated)
        full_motion = compose_output_motion(prefix, generated)

        # 5. Decode to XYZ joint positions
        positions = decode_to_xyz(
            full_motion,
            self.mean.cpu(),
            self.std.cpu(),
        )  # [1, T, 22, 3]

        # 6. Get raw (unnormalized) HML features
        # full_motion shape: [1, 263, 1, T] -> squeeze feat dim, permute to [1, T, 263]
        hml_norm = full_motion.squeeze(2).permute(0, 2, 1)  # [1, T, 263]
        mean_dev = self.mean.to(hml_norm.device)
        std_dev = self.std.to(hml_norm.device)
        hml_raw = hml_norm * std_dev.unsqueeze(0).unsqueeze(0) + mean_dev.unsqueeze(0).unsqueeze(0)

        return positions, hml_raw

    def generate_next_horizon(
        self,
        pose_buffer_isaac: torch.Tensor,
        recon_data: Optional[Dict[str, torch.Tensor]],
        text_prompt: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Closed-loop generation with simulator state feedback.

        This method converts the simulator's current pose buffer into HML space,
        uses it as a prefix for the diffusion model, generates the next horizon,
        and converts back to Isaac Lab coordinates.

        Parameters
        ----------
        pose_buffer_isaac : torch.Tensor
            [bs, T_30fps, 24, 3] recent joint positions from the simulator.
        recon_data : dict or None
            Previous reconstruction data for alignment. If None, it will be
            computed from the pose buffer.
        text_prompt : str
            Natural-language motion description.

        Returns
        -------
        positions_isaac : torch.Tensor
            [bs, T_30fps, 24, 3] predicted positions in Isaac Lab coordinates.
        hml_raw : torch.Tensor
            [bs, T_20fps, 263] unnormalized HumanML3D features.
        recon_data_new : dict
            Updated reconstruction data for the next call.

        Notes
        -----
        This is a skeleton implementation documenting the intended closed-loop flow.
        It may need debugging during integration with the full simulator pipeline.
        """
        # 1. Convert simulator poses to HML prefix
        hml_prefix, recon_data_from_sim = self.hml_conversion.pose_to_hml(pose_buffer_isaac)
        # hml_prefix: [bs, T_20fps, 263]

        if recon_data is None:
            recon_data = recon_data_from_sim

        # 2. Reshape HML prefix for MDM input: [bs, T, 263] -> [bs, 263, 1, T]
        hml_prefix_mdm = hml_prefix.permute(0, 2, 1).unsqueeze(2)
        # Take last context_len frames as prefix
        if hml_prefix_mdm.shape[-1] > self.context_len:
            hml_prefix_mdm = hml_prefix_mdm[..., -self.context_len:]

        bs = hml_prefix_mdm.shape[0]

        # 3. Build model kwargs for inpainting-style generation
        model_kwargs = {
            "y": {
                "text": [text_prompt] * bs,
                "prefix": hml_prefix_mdm.to(self.model_bundle.device),
                "mask": torch.ones(
                    bs, 1, 1, self.pred_len,
                    dtype=torch.bool,
                    device=self.model_bundle.device,
                ),
            }
        }

        # Add guidance scale
        if self.guidance != 1.0:
            model_kwargs["y"]["scale"] = torch.full(
                (bs,),
                float(self.guidance),
                device=self.model_bundle.device,
                dtype=hml_prefix_mdm.dtype,
            )

        # 4. Run diffusion sampling
        with torch.no_grad():
            sample = self.model_bundle.sample_fn(
                self.model_bundle.model,
                (bs, self.model_bundle.model.njoints, self.model_bundle.model.nfeats, self.pred_len),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

        # 5. Extract prediction and unnormalize
        # sample shape: [bs, 263, 1, pred_len]
        sample_hml = sample.squeeze(2).permute(0, 2, 1)  # [bs, pred_len, 263]
        mean_dev = self.mean.to(sample_hml.device)
        std_dev = self.std.to(sample_hml.device)
        hml_raw = sample_hml * std_dev.unsqueeze(0).unsqueeze(0) + mean_dev.unsqueeze(0).unsqueeze(0)

        # 6. Convert back to Isaac Lab coordinates
        # Use the full sample (context + prediction) for hml_to_pose
        full_hml = torch.cat([hml_prefix[..., -self.context_len:, :], sample_hml], dim=1)
        full_hml_norm = (full_hml - mean_dev.unsqueeze(0).unsqueeze(0)) / std_dev.unsqueeze(0).unsqueeze(0)

        positions_isaac = self.hml_conversion.hml_to_pose(
            full_hml_norm,
            recon_data_from_sim,
            sim_at_hml_idx=self.context_len - 1,
        )

        # 7. Prepare new recon_data for next iteration
        recon_data_new = recon_data_from_sim

        return positions_isaac, hml_raw, recon_data_new
