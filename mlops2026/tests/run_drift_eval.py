# import torch
# import wandb
# from src.vdm_pokemon.model import VDM
# from src.vdm_pokemon.unet import UNet
# from src.vdm_pokemon.data import get_pokemon_dataloaders


# def gaussian_noise(x, sigma):
#     return x + sigma * torch.randn_like(x)

# def brightness_shift(x, delta):
#     return torch.clamp(x + delta, -1, 1)

# def blur(x):
#     return torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)

# def channel_dropout(x, p=0.3):
#     mask = torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) > p
#     return x * mask


# @torch.no_grad()
# def evaluate_under_drift(vdm, dataloader, transform, severity, device):
#     vdm.eval()
#     losses = []

#     for x, _ in dataloader:
#         x = x.to(device)
#         x_drifted = transform(x, severity)
#         loss, _ = vdm(x_drifted)
#         losses.append(loss.item())

#     return sum(losses) / len(losses)


import torch
import wandb
from torchvision.utils import make_grid

from src.vdm_pokemon.model import VDM
from src.vdm_pokemon.unet import UNet
from src.vdm_pokemon.data import get_pokemon_dataloaders


# ---------------------------------------------------------
# Drift transforms
# ---------------------------------------------------------
def gaussian_noise(x, sigma):
    return x + sigma * torch.randn_like(x)

def brightness_shift(x, delta):
    return torch.clamp(x + delta, -1, 1)

def blur(x, _):
    return torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)

def channel_dropout(x, p):
    mask = torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) > p
    return x * mask


# ---------------------------------------------------------
# Evaluation under drift
# ---------------------------------------------------------
@torch.no_grad()
def evaluate_under_drift(vdm, dataloader, transform, severity, device):
    vdm.eval()
    losses = []

    for x, _ in dataloader:
        x = x.to(device)
        x_drifted = transform(x, severity)
        loss, _ = vdm(x_drifted)
        losses.append(loss.item())

    return sum(losses) / len(losses)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    wandb.init(
        project="MLOPS2026",
        name="vdm_drift_evaluation"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Data
    # -----------------------------
    _, val_loader = get_pokemon_dataloaders(
        data_dir="/zhome/68/a/168414/kagglehub/datasets/yehongjiang/pokemon-sprites-images",
        batch_size=64
    )

    # -----------------------------
    # Load EMA model
    # -----------------------------
    model = UNet(in_channels=3).to(device)
    model.load_state_dict(torch.load("/zhome/68/a/168414/MLOps/mlops2026/src/vdm_pokemon/vdm_ema.pth", map_location=device))

    vdm = VDM(
        model=model,
        image_shape=(3, 64, 64),
        gamma_min=-13.3,
        gamma_max=5.0,
    ).to(device)

    # -----------------------------
    # Drift setup
    # -----------------------------
    drift_tests = {
        "gaussian_noise": gaussian_noise,
        "brightness_shift": brightness_shift,
        "blur": blur,
        "channel_dropout": channel_dropout,
    }

    severities = [0.0, 0.05, 0.1, 0.2, 0.4]

    # -----------------------------
    # Quantitative drift evaluation
    # -----------------------------
    for drift_name, drift_fn in drift_tests.items():
        for s in severities:
            elbo = evaluate_under_drift(
                vdm,
                val_loader,
                drift_fn,
                s,
                device
            )

            wandb.log({
                "drift/type": drift_name,
                "drift/severity": s,
                "drift/elbo": elbo
            })

            print(f"[{drift_name}] severity={s:.2f} → ELBO={elbo:.4f}")

    # -----------------------------
    # Qualitative sample robustness
    # -----------------------------
    x, _ = next(iter(val_loader))
    x = x.to(device)

    for s in [0.0, 0.2, 0.4]:
        x_drifted = gaussian_noise(x, s)

        with torch.no_grad():
            samples = vdm.sample(
                batch_size=16,
                n_sample_steps=250,
                clip_samples=True
            )

        samples = samples.clamp(-1, 1)
        samples = (samples + 1) / 2

        grid = make_grid(samples.cpu(), nrow=4)

        wandb.log({
            f"samples/gaussian_noise_{s}": wandb.Image(grid)
        })

    wandb.finish()


if __name__ == "__main__":
    main()

