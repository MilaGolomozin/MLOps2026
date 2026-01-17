import torch

from vdm_pokemon.unet import FiLMBlock, FourierFeatures, TimeEmbedding, UNet


def test_fourier_features_output_shape() -> None:
    """This test verifies that Fourier features expand the channel dimension as expected."""
    features = FourierFeatures()
    x = torch.zeros(2, 3, 4, 4)
    output = features(x)
    expected_channels = x.shape[1] * features.num_features
    assert output.shape == (x.shape[0], expected_channels, x.shape[2], x.shape[3])


def test_unet_forward_output_shape() -> None:
    """This test verifies that the UNet returns a tensor with the same spatial shape as the input."""
    model = UNet(in_channels=3, out_channels=3, time_emb_dim=32, base_channels=8)
    x = torch.zeros(2, 3, 32, 32)
    gamma_t = torch.zeros(2)
    output = model(x, gamma_t)
    assert output.shape == x.shape


def test_time_embedding_output_shape() -> None:
    """This test verifies that the time embedding returns the expected dimension."""
    embedding = TimeEmbedding(32)
    gamma = torch.zeros(2)
    output = embedding(gamma)
    assert output.shape == (2, 32)
    assert torch.isfinite(output).all()


def test_film_block_output_shape() -> None:
    """This test verifies that the FiLM block returns the expected spatial shape."""
    block = FiLMBlock(2, 4, 8)
    x = torch.zeros(2, 2, 8, 8)
    time_emb = torch.zeros(2, 8)
    output = block(x, time_emb)
    assert output.shape == (2, 4, 8, 8)


def test_unet_forward_accepts_four_dimensional_gamma() -> None:
    """This test confirms that the UNet accepts a four dimensional gamma input."""
    model = UNet(in_channels=3, out_channels=3, time_emb_dim=32, base_channels=8)
    x = torch.zeros(1, 3, 32, 32)
    gamma_t = torch.zeros(1, 1, 1, 1)
    output = model(x, gamma_t)
    assert output.shape == x.shape
