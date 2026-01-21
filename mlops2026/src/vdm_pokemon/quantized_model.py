import torch
import torch.nn.utils.prune as prune
from loguru import logger


from model import VDM
from unet import UNet

def run_gpu_optimization():
    #note: the vdm model is exetremely heavy, thus training was done on the cluster given using hpc
    #the corresponding changes have been made to be able to run on gpu
    #for example in the quantization part the GPU equivalent of quantization was used, using 16-bit (FP16) 
    #instead of 8-bit, which can be processed at high speeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting GPU Optimization on {torch.cuda.get_device_name(0)}")

    #model reconstruction
    inner_unet = UNet(in_channels=3).to(device)
    vdm_wrapper = VDM(
        model=inner_unet, 
        image_shape=(3, 64, 64), 
        gamma_min=-13.3, 
        gamma_max=5.0
    ).to(device)

    #load weights
    vdm_wrapper.model.load_state_dict(torch.load("vdm_ema.pth"))
    vdm_wrapper.eval()

    #pruning phase
    #remove the 10% smallest weights
    logger.info("global unstructured pruning...")
    parameters_to_prune = [
        (module, "weight") for module in vdm_wrapper.model.modules() 
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
    ]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.10,
    )
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    
    
    logger.info("Compiling...")
    #compile
    vdm_wrapper.model = torch.compile(vdm_wrapper.model, mode="reduce-overhead")

    #quantization
    logger.info("warming up GPU kernels with autocast (Mixed Precision)...")
    with torch.no_grad():
        #autocast allows the model to use 16-bit precision 
        with torch.cuda.amp.autocast(): 
            for _ in range(10):
                dummy_x = torch.randn(1, 3, 64, 64).to(device)
                dummy_gamma = torch.tensor([[[[1.0]]]]).to(device)
                _ = vdm_wrapper.model(dummy_x, dummy_gamma)

    #save 
    torch.save(vdm_wrapper.model.state_dict(), "vdm_gpu_optimized.pt")
    logger.success("artifact saved: vdm_gpu_optimized.pt")

if __name__ == "__main__":
    run_gpu_optimization()
