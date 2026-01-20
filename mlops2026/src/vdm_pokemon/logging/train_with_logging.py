from model import VDM
from data import get_pokemon_dataloaders
from unet import UNet

import math
import torch
import torch.optim as optim
from torchvision.utils import make_grid
from ema_pytorch import EMA
import wandb

import sys
from loguru import logger

#setup the log file
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("vdm_training.log", level="DEBUG", rotation="50 MB")

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
def evaluate_vdm(vdm, dataloader, device):
    vdm.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():        #with torch.enable_grad():
        for x, _ in dataloader:
            x = x.to(device)
            loss, _ = vdm(x)
            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def main():
    run = wandb.init(
        project="MLOPS2026",
        config={
            "dataset": "Pokemon",
            "epochs": 10,
            "learning_rate": 5e-4,
            "batch_size": 64,
            "image_size": 32,
            "model": "UNet",
            "gamma_min": -13.3,
            "gamma_max": 5.0,
        },
    )
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = get_pokemon_dataloaders(
        data_dir=f"/zhome/07/3/219372/.cache/kagglehub/datasets/yehongjiang/pokemon-sprites-images/versions/1",
        batch_size=cfg.batch_size
    )

    image_shape = (3, 64, 64)

    # Model
    model = UNet(in_channels=3).to(device)

    #ema = EMA(model, beta=0.9999)

    #ema.load_state_dict(torch.load("vdm_ema.pth"))
    #ema_model = ema.ema_model

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

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            loss, metrics = vdm(x)
            
            #Mathematical Stability 
            if torch.isnan(loss) or torch.isinf(loss):
                logger.critical(f"Numerical instability at Epoch {epoch+1}, Batch {batch_idx}!")
                logger.error(f"Loss: {loss.item()} | Gamma Min: {cfg.gamma_min} | Gamma Max: {cfg.gamma_max}")
                run.finish()
                sys.exit(1) 

            loss.backward()
            optimizer.step()
            ema.update()

            #Progress Heartbeat
            # Log every 50 batches to the .log file
            if batch_idx % 50 == 0:
                logger.debug(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

            wandb.log({
                "train/loss_batch": loss.item(),
                **{f"train/{k}": v for k, v in metrics.items()},
            })
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Completed Epoch {epoch+1}/{cfg.epochs} | Avg Loss: {avg_loss:.4f}")

        # ----------------------------------
        # Validation (EMA model)
        # ----------------------------------
        ema_model = ema.ema_model
        vdm_ema = VDM(
            model=ema_model,
            image_shape=image_shape,
            gamma_min=cfg.gamma_min,
            gamma_max=cfg.gamma_max,
        ).to(device)

        val_elbo = evaluate_vdm(vdm_ema, val_loader, device)
        print(f"→ Validation ELBO (EMA): {val_elbo:.4f}")

        wandb.log({
            "train/loss_epoch": avg_loss,
            "val/elbo": val_elbo,
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
        })

    # ---------------------------------------------------------
    # Final Sampling
    # ---------------------------------------------------------
    vdm_ema.eval()
    with torch.no_grad():
        samples = vdm_ema.sample(
            batch_size=16,
            n_sample_steps=250,
            clip_samples=True
        )
        samples = samples.clamp(-1, 1)
        samples = (samples + 1) / 2

        grid = make_grid(samples.cpu(), nrow=4)
        wandb.log({
            "final/samples_grid": wandb.Image(grid)
        })
    #Artifact Saving (REPLACES YOUR OLD LINES) ---
    logger.info("Training finished. Saving EMA model...")
    try:
        torch.save(vdm_ema.model.state_dict(), "vdm_ema.pth")
        logger.success("Model saved successfully as 'vdm_ema.pth'")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    run.finish()
    logger.info("WandB run closed. HPC job exiting.")

if __name__ == "__main__":
    main()
