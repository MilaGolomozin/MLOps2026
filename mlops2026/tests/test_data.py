from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader

from vdm_pokemon.data import get_pokemon_dataloaders


def write_rgb_image(path: Path, size: int, color: tuple[int, int, int]) -> None:
    """This helper writes a simple RGB image to disk for tests."""
    image = Image.new("RGB", (size, size), color=color)
    image.save(path)


def create_image_folder(
    root: Path,
    class_counts: dict[str, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    """This helper creates a basic image folder structure for tests."""
    for class_name, count in class_counts.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            write_rgb_image(class_dir / f"img_{index}.png", size, color)


def test_get_pokemon_dataloaders_split_sizes(tmp_path: Path) -> None:
    """This test verifies that the data loader split sizes match the requested ratio."""
    create_image_folder(tmp_path, {"class_a": 6, "class_b": 4}, size=32, color=(0, 0, 0))
    train_loader, val_loader = get_pokemon_dataloaders(
        data_dir=str(tmp_path),
        image_size=32,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        val_split=0.2,
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert len(train_loader.dataset) == 8
    assert len(val_loader.dataset) == 2


def test_get_pokemon_dataloaders_normalization_values(tmp_path: Path) -> None:
    """This test confirms that the normalization produces the expected values for a white image."""
    create_image_folder(tmp_path, {"class_a": 1}, size=32, color=(255, 255, 255))
    train_loader, _ = get_pokemon_dataloaders(
        data_dir=str(tmp_path),
        image_size=32,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        val_split=0.0,
    )

    batch, _ = next(iter(train_loader))
    assert torch.allclose(batch, torch.ones_like(batch))


def test_get_pokemon_dataloaders_output_shape(tmp_path: Path) -> None:
    """This test verifies that resized samples have the expected shape."""
    create_image_folder(tmp_path, {"class_a": 3}, size=40, color=(10, 10, 10))
    train_loader, _ = get_pokemon_dataloaders(
        data_dir=str(tmp_path),
        image_size=16,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        val_split=0.0,
    )

    batch, _ = next(iter(train_loader))
    assert batch.shape == (2, 3, 16, 16)
