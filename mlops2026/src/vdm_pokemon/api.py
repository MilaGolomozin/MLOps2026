# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Optional
# import torch
# from model import VDM
# from unet import UNet
# from torchvision.utils import save_image
# import io
# import base64
# from PIL import Image

# # ---------------------------
# # FastAPI app
# # ---------------------------
# app = FastAPI(title="VDM Pokémon Inference API")

# # ---------------------------
# # Input schema
# # ---------------------------
# class InferenceRequest(BaseModel):
#     prompt: Optional[str] = "pikachu"  # or other inputs if your model uses text
#     batch_size: Optional[int] = 1
#     n_sample_steps: Optional[int] = 50

# # ---------------------------
# # Load model
# # ---------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# image_shape = (3, 32, 32)

# # Initialize model architecture
# unet_model = UNet(in_channels=3).to(device)

# vdm = VDM(
#     model=unet_model,
#     image_shape=image_shape,
#     gamma_min=-13.3,
#     gamma_max=5.0,
# ).to(device)

# # Load trained weights
# vdm.model.load_state_dict(torch.load("vdm_ema.pth", map_location=device))
# vdm.eval()

# # ---------------------------
# # Helper function: tensor -> base64 image
# # ---------------------------
# def tensor_to_base64(img_tensor):
#     img_tensor = (img_tensor.clamp(-1,1) + 1) / 2  # scale to [0,1]
#     img_pil = Image.fromarray((img_tensor.permute(1,2,0).cpu().numpy()*255).astype("uint8"))
#     buffered = io.BytesIO()
#     img_pil.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode()

# # ---------------------------
# # Inference endpoint
# # ---------------------------
# @app.post("/generate")
# def generate(req: InferenceRequest):
#     with torch.no_grad():
#         samples = vdm.sample(
#             batch_size=req.batch_size,
#             n_sample_steps=req.n_sample_steps,
#             clip_samples=True
#         )
    
#     # Convert first sample to base64 for easy JSON response
#     sample_img = samples[0]
#     img_base64 = tensor_to_base64(sample_img)

#     return {"image_base64": img_base64, "batch_size": req.batch_size, "steps": req.n_sample_steps}

# # ---------------------------
# # Health check
# # ---------------------------
# @app.get("/")
# def root():
#     return {"message": "VDM Pokémon Inference API is running"}

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import torch
from model import VDM
from unet import UNet
from torchvision.utils import make_grid
from PIL import Image
import io
import base64

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="VDM Pokémon Inference API")

# ---------------------------
# Input schema
# ---------------------------
class InferenceRequest(BaseModel):
    batch_size: Optional[int] = 1
    n_sample_steps: Optional[int] = 50

# ---------------------------
# Load model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_shape = (3, 32, 32)

# Initialize UNet and VDM
unet_model = UNet(in_channels=3).to(device)
vdm = VDM(
    model=unet_model,
    image_shape=image_shape,
    gamma_min=-13.3,
    gamma_max=5.0
).to(device)

# Load trained EMA model weights
vdm.model.load_state_dict(torch.load("vdm_ema.pth", map_location=device))
vdm.eval()

# ---------------------------
# Helper: tensor -> base64 image
# ---------------------------
def tensor_to_base64(img_tensor):
    img_tensor = (img_tensor.clamp(-1,1) + 1) / 2  # scale to [0,1]
    img_pil = Image.fromarray((img_tensor.permute(1,2,0).cpu().numpy()*255).astype("uint8"))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ---------------------------
# Health check
# ---------------------------
@app.get("/")
def root():
    return {"message": "VDM Pokémon Inference API is running"}

# ---------------------------
# Inference endpoint
# ---------------------------
@app.post("/generate")
def generate(req: InferenceRequest):
    with torch.no_grad():
        samples = vdm.sample(
            batch_size=req.batch_size,
            n_sample_steps=req.n_sample_steps,
            clip_samples=True
        )

    # Convert each sample to base64
    images_base64 = [tensor_to_base64(img) for img in samples]

    return {
        "batch_size": req.batch_size,
        "n_sample_steps": req.n_sample_steps,
        "images": images_base64
    }

# from fastapi import FastAPI
# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int):
#     return {"item_id": item_id}