import torch
import time
from collections import OrderedDict
from loguru import logger
from model import VDM
from unet import UNet

def benchmark_model(model, device, name, batch_size=16):
    
    dummy_input = torch.randn(batch_size, 3, 64, 64).to(device)
    dummy_gamma = torch.tensor([[[[1.0]]]]).to(device)
    
    
    logger.info(f"warm up")
    for _ in range(15):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _ = model(dummy_input, dummy_gamma)
    
    
    torch.cuda.synchronize() 
    start_time = time.time()
    
    iters = 100
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for _ in range(iters):
                _ = model(dummy_input, dummy_gamma)
    
    torch.cuda.synchronize() 
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iters
    logger.success(f"[{name}] Avg Inference: {avg_time*1000:.2f} ms")
    return avg_time

def run_comparison():
    device = torch.device("cuda")
    logger.info("comparing base vs optimized model")

    #load original
    orig_unet = UNet(in_channels=3).to(device)
    orig_unet.load_state_dict(torch.load("vdm_ema.pth", map_location=device))
    orig_unet.eval()
    
    #load optimized
    opt_unet = UNet(in_channels=3).to(device)
    state_dict = torch.load("vdm_gpu_optimized.pt", map_location=device)

    #
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "") 
        new_state_dict[name] = v
    
    opt_unet.load_state_dict(new_state_dict)
    
    
    opt_unet = torch.compile(opt_unet, mode="reduce-overhead")
    opt_unet.eval()

    #run
    t_orig = benchmark_model(orig_unet, device, "Baseline")
    t_opt = benchmark_model(opt_unet, device, "Optimized")
    
    print("\n" + "="*30)
    print(f"SPEEDUP: {t_orig / t_opt:.2f}x faster")
    print("="*30)

if __name__ == "__main__":
    run_comparison()