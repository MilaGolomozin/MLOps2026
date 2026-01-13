import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------------
# Fourier features applied to latent
# ----------------------------------
class FourierFeatures(nn.Module):
    def __init__(self, first=7.0, last=8.0, step=1.0): # paper lists nmin = 7, and nmax = 8
        super().__init__()
        self.freqs_exponent = torch.arange(first, last + 1e-8, step)

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        assert len(x.shape) >= 2

        # Compute (2pi * 2^n) for n in freqs.
        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
        freqs = 2.0**freqs_exponent * math.pi  # (F, ) // match the paper pi*2^n
        freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

        # Compute (2pi * 2^n * x) for n in freqs.
        features = freqs * x.unsqueeze(1)  # (B, F, X1, X2, ...)
        features = features.flatten(1, 2)  # (B, F * C, X1, X2, ...)

        # Output features are cos and sin of above. Shape (B, 2 * F * C, H, W).
        return torch.cat([features.sin(), features.cos()], dim=1)
    

# ----------------------------------
# Sinusoidal time embedding for gamma_t
# ----------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, gamma):
        # gamma: [B,1,1,1] or [B]
        if len(gamma.shape) == 4:
            gamma = gamma.view(gamma.shape[0])
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=gamma.device) * -emb)
        emb = gamma[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # [B, dim]

# ----------------------------------
# FiLM modulation block
# ----------------------------------
class FiLMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        # Linear layer to generate gamma and beta for FiLM
        self.time_mlp = nn.Linear(time_emb_dim, out_channels*2)

    def forward(self, x, time_emb):
        # time_emb: [B, time_emb_dim]
        gamma_beta = self.time_mlp(time_emb)  # [B, 2*out_channels]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = x * (1 + gamma) + beta  # FiLM
        return x

# ----------------------------------
# DoubleConv block
# ----------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.block1 = FiLMBlock(in_channels, out_channels, time_emb_dim)
        self.block2 = FiLMBlock(out_channels, out_channels, time_emb_dim)

    def forward(self, x, time_emb):
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        return x

# ----------------------------------
# UNet for diffusion
# ----------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128, base_channels=64):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # Condition time embedding MLP
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True)
        )

        # Fourier features on latent inputs
        self.fourier_features = FourierFeatures() # remove this line to disable Fourier features

        # Encoder
        # Concatenate Fourier features to input for the channel dimension 
        # Fourier features will not be used
        total_input_ch = in_channels * (1 + self.fourier_features.num_features) 
        self.enc1 = DoubleConv(total_input_ch, base_channels, time_emb_dim) # change this line to be in_channels to remove fourier features
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels*2, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels*2, base_channels*4, time_emb_dim)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels*4, base_channels*8, time_emb_dim)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels*8, base_channels*4, time_emb_dim)

        self.upconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels*4, base_channels*2, time_emb_dim)

        self.upconv1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels*2, base_channels, time_emb_dim)

        # Output
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, gamma_t):
        """
        x: [B, C, H, W] input image
        gamma_t: [B,1,1,1] or [B] diffusion time embedding
        """
        time_emb = self.time_mlp(gamma_t)  # [B, time_emb_dim]

        # Concatenate Fourier features to input along channel dimension
        x = torch.cat([x, self.fourier_features(x)], dim=1) # remove this line to disable Fourier features

        # Encoder
        e1 = self.enc1(x, time_emb)
        e2 = self.enc2(self.pool1(e1), time_emb)
        e3 = self.enc3(self.pool2(e2), time_emb)

        # Bottleneck
        b = self.bottleneck(self.pool3(e3), time_emb)

        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3, time_emb)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, time_emb)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, time_emb)

        out = self.output_conv(d1)
        return out
