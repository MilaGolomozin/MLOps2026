import pytest
import torch
from torch import nn

from vdm_pokemon.model import LinearGammaSchedule, VDM


class DummyNoiseModel(nn.Module):
    """Provide a minimal model that returns a tensor with the same shape as the input."""

    def __init__(self, channels: int) -> None:
        """Initialize the model with a small convolution layer."""
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Return a simple convolution output."""
        _ = gamma
        return self.conv(x)


def build_vdm(channels: int = 1, gamma_min: float = 0.0, gamma_max: float = 0.0) -> VDM:
    """Create a small VDM instance for tests.

    Args:
        channels: Number of input channels.
        gamma_min: Minimum gamma value.
        gamma_max: Maximum gamma value.

    Returns:
        VDM: A VDM instance for tests.
    """
    model = DummyNoiseModel(channels)
    image_shape = (channels, 2, 2)
    return VDM(model=model, image_shape=image_shape, gamma_min=gamma_min, gamma_max=gamma_max)


def test_linear_gamma_schedule_endpoints() -> None:
    """Test that the schedule returns the bounds for the time range."""
    schedule = LinearGammaSchedule(gamma_min=0.1, gamma_max=1.1)
    times = torch.tensor([0.0, 1.0])
    result = schedule(times)
    expected = torch.tensor([0.1, 1.1])
    assert torch.allclose(result, expected)


def test_vdm_sample_q_t_0_uses_noise() -> None:
    """Test that the forward diffusion uses the provided noise."""
    vdm = build_vdm(channels=1, gamma_min=0.0, gamma_max=0.0)
    x = torch.ones(2, 1, 2, 2)
    noise = torch.full_like(x, 2.0)
    times = torch.tensor([0.25, 0.75])
    result, gamma_t = vdm.sample_q_t_0(x=x, times=times, noise=noise)

    gamma_expected = vdm.gamma(times)[:, None, None, None]
    alpha = torch.sqrt(torch.sigmoid(-gamma_expected))
    sigma = torch.sqrt(torch.sigmoid(gamma_expected))
    expected = alpha * x + sigma * noise

    assert result.shape == x.shape
    assert gamma_t.shape == gamma_expected.shape
    assert torch.allclose(result, expected)


def test_vdm_data_decode_shape_and_finite() -> None:
    """Test that data decoding returns finite log probabilities with the expected shape."""
    vdm = build_vdm(channels=1, gamma_min=0.1, gamma_max=1.0)
    z_0_rescaled = torch.zeros(1, 1, 2, 2)
    gamma_0 = torch.tensor(0.5)
    log_probs = vdm.data_decode(z_0_rescaled, gamma_0)
    assert log_probs.shape == (1, 4, 256)
    assert torch.isfinite(log_probs).all()


def test_vdm_forward_returns_loss_and_metrics_placeholder() -> None:
    """Test that the forward method returns a scalar loss and metrics."""
    pytest.skip("This test will be completed after the loss contract is finalized.")


def test_vdm_sample_clips_range_placeholder() -> None:
    """Test that sampling clips values to the expected range when requested."""
    pytest.skip("This test will be completed after the sampling interface is finalized.")


def test_vdm_data_logprob_matches_decode_placeholder() -> None:
    """Test that data logprob aligns with the decoded probabilities."""
    pytest.skip("This test will be completed after the data logprob contract is finalized.")
