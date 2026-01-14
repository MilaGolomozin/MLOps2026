import pytest
import torch

from vdm_pokemon.unet import FiLMBlock, FourierFeatures, TimeEmbedding, UNet


def test_fourier_features_output_shape() -> None:
    """Test that Fourier features expand the channel dimension as expected."""
    features = FourierFeatures()
    x = torch.zeros(2, 3, 4, 4)
    output = features(x)
    expected_channels = x.shape[1] * features.num_features
    assert output.shape == (x.shape[0], expected_channels, x.shape[2], x.shape[3])


def test_unet_forward_output_shape() -> None:
    """Test that the UNet returns a tensor with the same spatial shape as the input."""
    model = UNet(in_channels=3, out_channels=3, time_emb_dim=32, base_channels=8)
    x = torch.zeros(2, 3, 32, 32)
    gamma_t = torch.zeros(2)
    output = model(x, gamma_t)
    assert output.shape == x.shape


def test_time_embedding_output_shape_placeholder() -> None:
    """Test that the time embedding returns the expected dimension."""
    _ = TimeEmbedding(32)
    pytest.skip("This test will be completed after the embedding contract is finalized.")


def test_film_block_output_shape_placeholder() -> None:
    """Test that the FiLM block preserves the expected tensor shape."""
    _ = FiLMBlock(2, 2, 8)
    pytest.skip("This test will be completed after the FiLM contract is finalized.")
