# from pathlib import Path

# import typer
# from torch.utils.data import Dataset


# class MyDataset(Dataset):
#     """My custom dataset."""

#     def __init__(self, data_path: Path) -> None:
#         self.data_path = data_path

#     def __len__(self) -> int:
#         """Return the length of the dataset."""

#     def __getitem__(self, index: int):
#         """Return a given sample from the dataset."""

#     def preprocess(self, output_folder: Path) -> None:
#         """Preprocess the raw data and save it to the output folder."""

# def preprocess(data_path: Path, output_folder: Path) -> None:
#     print("Preprocessing data...")
#     dataset = MyDataset(data_path)
#     dataset.preprocess(output_folder)


# if __name__ == "__main__":
#     typer.run(preprocess)


# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms


# def get_cifar10_dataloaders(
#     data_dir="./data",
#     batch_size=64,
#     num_workers=4,
#     pin_memory=True
# ):
#     """
#     Returns train and validation dataloaders for CIFAR-10.
#     """

#     transform = transforms.Compose([
#         transforms.ToTensor(),  # [0,1]
#     ])

#     train_dataset = datasets.CIFAR10(
#         root=data_dir,
#         train=True,
#         download=True,
#         transform=transform
#     )

#     val_dataset = datasets.CIFAR10(
#         root=data_dir,
#         train=False,
#         download=True,
#         transform=transform
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=pin_memory
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=pin_memory
#     )

#     return train_loader, val_loader

import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_pokemon_dataloaders(
    data_dir,
    image_size=32,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    val_split=0.1
):
    """
    Returns train and validation dataloaders for image-only VDM training.
    """

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),              # [0, 1]
        transforms.Normalize(               # [-1, 1] (important for diffusion)
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    # ImageFolder treats subfolders as "classes"
    # Labels will be ignored for diffusion
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader
