import torch
import torch.nn.functional as F


def fps_convert(data: torch.Tensor, src_fps: int, tgt_fps: int, mode: str = "bicubic") -> torch.Tensor:
    """Convert motion data between frame rates via interpolation.

    Args:
        data: [batch, src_frames, ...] motion data with arbitrary trailing dims.
        src_fps: Source frame rate.
        tgt_fps: Target frame rate.
        mode: Interpolation mode ('bicubic', 'linear', 'nearest').

    Returns:
        [batch, tgt_frames, ...] resampled data where tgt_frames = int(src_frames * tgt_fps / src_fps).
    """
    if src_fps == tgt_fps:
        return data

    B = data.shape[0]
    src_frames = data.shape[1]
    trailing_shape = data.shape[2:]

    tgt_frames = int(src_frames * tgt_fps / src_fps)

    # Flatten trailing dims: [B, T_src, D]
    D = 1
    for s in trailing_shape:
        D *= s
    x = data.reshape(B, src_frames, D)

    # Permute to [B, D, T_src] for F.interpolate
    x = x.permute(0, 2, 1)

    if mode == "bicubic":
        # Unsqueeze to [B, D, 1, T_src], interpolate, squeeze back
        x = x.unsqueeze(2)
        x = F.interpolate(x, size=(1, tgt_frames), mode="bicubic", align_corners=False)
        x = x.squeeze(2)
    else:
        interp_mode = "linear" if mode == "linear" else mode
        x = F.interpolate(x, size=tgt_frames, mode=interp_mode, align_corners=False)

    # Permute back to [B, T_tgt, D]
    x = x.permute(0, 2, 1)

    # Reshape to original trailing dims
    x = x.reshape(B, tgt_frames, *trailing_shape)

    return x
