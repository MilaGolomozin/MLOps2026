from vdm_pokemon.model import VDM
from vdm_pokemon.data import get_cifar10_dataloaders
from unet import UNet
import hydra
from omegaconf import DictConfig, OmegaConf


import math
import torch
import torch.optim as optim
from torchvision.utils import make_grid
from ema_pytorch import EMA
import wandb

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
def evaluate_vdm(vdm, dataloader, device):
    vdm.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.enable_grad():
        for x, _ in dataloader:
            x = x.to(device)
            loss, _ = vdm(x)
            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Initialize W&B using Hydra config
    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    cfg = wandb.config  # keep using cfg.* as before, but now cfg comes from Hydra

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = get_cifar10_dataloaders(
        batch_size=cfg.batch_size
    )

    image_shape = (3, cfg.image_size, cfg.image_size)

    # Model
    model = UNet(in_channels=3).to(device)

    vdm = VDM(
        model=model,
        image_shape=image_shape,
        gamma_min=cfg.gamma_min,
        gamma_max=cfg.gamma_max,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=0.001,
        eps=1e-8
    )

    ema = EMA(model, beta=0.9999)

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            loss, metrics = vdm(x)
            loss.backward()
            optimizer.step()
            ema.update()

            wandb.log({
                "train/loss_batch": loss.item(),
                **{f"train/{k}": v for k, v in metrics.items()},
            })

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        wandb.log({"train/loss_epoch": avg_loss}, step=epoch)

        # ---------------------------------------------------------
        # Evaluation
        # ---------------------------------------------------------
        val_loss = evaluate_vdm(vdm, val_loader, device)
        wandb.log({"val/loss": val_loss}, step=epoch)

    # ---------------------------------------------------------
    # Final sampling (as in your original code)
    # ---------------------------------------------------------
    model.eval()
    with torch.no_grad():
        samples = vdm.sample(
            num_samples=16,
            n_sample_steps=50,
            clip_samples=True
        )
        samples = samples.clamp(-1, 1)
        samples = (samples + 1) / 2

        grid = make_grid(samples.cpu(), nrow=4)
        wandb.log({
            "final/samples_grid": wandb.Image(grid)
        })

    run.finish()


if __name__ == "__main__":
    main()