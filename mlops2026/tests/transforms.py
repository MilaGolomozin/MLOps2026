import torch

def gaussian_noise(x, sigma):
    return x + sigma * torch.randn_like(x)

def brightness_shift(x, delta):
    return torch.clamp(x + delta, -1, 1)

def blur(x):
    return torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)

def channel_dropout(x, p=0.3):
    mask = torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) > p
    return x * mask
