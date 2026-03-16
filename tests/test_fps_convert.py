import pytest
import torch
from closd_isaaclab.utils.fps_convert import fps_convert


def test_30fps_to_20fps_reduces_frames():
    """30fps -> 20fps: 30 frames -> 20 frames (2/3 reduction)."""
    data = torch.randn(2, 30, 63)
    result = fps_convert(data, src_fps=30, tgt_fps=20)
    assert result.shape == (2, 20, 63), f"Expected (2, 20, 63), got {result.shape}"


def test_20fps_to_30fps_increases_frames():
    """20fps -> 30fps: 20 frames -> 30 frames (3/2 increase)."""
    data = torch.randn(2, 20, 63)
    result = fps_convert(data, src_fps=20, tgt_fps=30)
    assert result.shape == (2, 30, 63), f"Expected (2, 30, 63), got {result.shape}"


def test_same_fps_returns_identical():
    """Same src and tgt fps returns the exact same data (identity)."""
    data = torch.randn(4, 25, 63)
    result = fps_convert(data, src_fps=30, tgt_fps=30)
    assert result is data or torch.equal(result, data), "Expected identical data for same fps"


def test_preserves_batch_dimension():
    """Batch dimension is preserved across conversion."""
    for batch_size in [1, 4, 8]:
        data = torch.randn(batch_size, 30, 63)
        result = fps_convert(data, src_fps=30, tgt_fps=20)
        assert result.shape[0] == batch_size, f"Expected batch size {batch_size}, got {result.shape[0]}"


def test_preserves_trailing_dims():
    """Trailing dimensions beyond T are preserved after conversion."""
    # Single trailing dim
    data = torch.randn(2, 30, 63)
    result = fps_convert(data, src_fps=30, tgt_fps=20)
    assert result.shape == (2, 20, 63)

    # Multiple trailing dims
    data2 = torch.randn(2, 30, 21, 3)
    result2 = fps_convert(data2, src_fps=30, tgt_fps=20)
    assert result2.shape == (2, 20, 21, 3), f"Expected (2, 20, 21, 3), got {result2.shape}"


def test_round_trip_30_20_30_close():
    """30->20->30 round-trip should be close to original (atol=0.3 due to interpolation).

    Uses smooth motion-like data (cumulative sum of small increments) since bicubic
    interpolation is designed for smooth signals; high-frequency random noise would not
    satisfy this tolerance.
    """
    torch.manual_seed(0)
    # Simulate smooth motion data: cumsum of small steps, as real motion signals are smooth
    data = torch.cumsum(torch.randn(2, 30, 63) * 0.1, dim=1)
    data_20 = fps_convert(data, src_fps=30, tgt_fps=20)
    data_30 = fps_convert(data_20, src_fps=20, tgt_fps=30)
    assert data_30.shape == data.shape, f"Expected shape {data.shape}, got {data_30.shape}"
    assert torch.allclose(data, data_30, atol=0.3), "Round-trip 30->20->30 not close enough (atol=0.3)"
