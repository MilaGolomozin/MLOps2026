import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from vdm_pokemon import api

client = TestClient(api.app)


def test_root_returns_message() -> None:
    """Verify that the root endpoint returns a message."""
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert "message" in payload
    assert payload["message"].startswith("VDM")


def test_generate_returns_png(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that the generate endpoint returns a png image."""

    def fake_sample(batch_size: int, n_sample_steps: int, clip_samples: bool) -> torch.Tensor:
        """Return a deterministic batch of images."""
        _ = n_sample_steps
        _ = clip_samples
        return torch.zeros((batch_size, 3, 64, 64))

    monkeypatch.setattr(api.vdm, "sample", fake_sample)

    response = client.post("/generate", json={"batch_size": 1, "n_sample_steps": 1})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    image = Image.open(io.BytesIO(response.content))
    assert image.size == (64, 64)
