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

import io
import base64
import time
from typing import Optional

import torch
import psutil
from PIL import Image
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from model import VDM
from unet import UNet

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="VDM Pokémon Inference API")

# ---------------------------
# Metrics definitions
# ---------------------------

# HTTP level metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests in seconds",
    ["endpoint"],
)

# System metrics
CPU_USAGE = Gauge(
    "system_cpu_percent",
    "System-wide CPU usage percentage",
)

MEMORY_USAGE = Gauge(
    "system_memory_percent",
    "System-wide memory usage percentage",
)


# ---------------------------
# Metrics middleware
# ---------------------------
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response: Response = await call_next(request)
    process_time = time.time() - start_time

    endpoint = request.url.path

    # Avoid polluting metrics with the metrics endpoint itself (optional)
    if endpoint != "/metrics":
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(process_time)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            http_status=response.status_code,
        ).inc()

        # Update system metrics
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)

    return response


# ---------------------------
# Input schema
# ---------------------------
class InferenceRequest(BaseModel):
    """Define the request body for image generation."""
    batch_size: Optional[int] = 1
    n_sample_steps: Optional[int] = 250


# ---------------------------
# Load model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_shape = (3, 128, 128)

# Initialize UNet and VDM
unet_model = UNet(in_channels=3).to(device)
vdm = VDM(
    model=unet_model,
    image_shape=image_shape,
    gamma_min=-13.3,
    gamma_max=5.0
).to(device)

weights_path = Path(os.getenv("VDM_WEIGHTS_PATH", "vdm_ema.pth"))
if weights_path.is_file():
    vdm.model.load_state_dict(torch.load(weights_path, map_location=device))
vdm.eval()


# ---------------------------
# Helper: tensor -> base64 image (currently unused but kept)
# ---------------------------
def tensor_to_base64(img_tensor):
    img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2  # scale to [0,1]
    img_pil = Image.fromarray(
        (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    )
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# ---------------------------
# Health check
# ---------------------------
@app.get("/")
def root() -> dict[str, str]:
    """Return a health check message."""
    return {"message": "VDM Pokémon Inference API is running"}


# ---------------------------
# Metrics endpoint
# ---------------------------
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---------------------------
# Inference endpoint
# ---------------------------
@app.post("/generate")
def generate(req: InferenceRequest) -> Response:
    """Generate a png image from the model."""
    with torch.no_grad():
        samples = vdm.sample(
            batch_size=1,                     # force single image
            n_sample_steps=req.n_sample_steps,
            clip_samples=True,
        )

    img = samples[0].clamp(-1, 1)
    img = (img + 1) / 2  # [-1,1] → [0,1]
    img = (img * 255).byte().cpu()
    img = img.permute(1, 2, 0).numpy()

    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )
# @app.post("/generate")
# def generate(req: InferenceRequest):
#     with torch.no_grad():
#         samples = vdm.sample(
#             batch_size=req.batch_size,
#             n_sample_steps=req.n_sample_steps,
#             clip_samples=True
#         )

#     # Convert each sample to base64
#     images_base64 = [tensor_to_base64(img) for img in samples]

#     return {
#         "batch_size": req.batch_size,
#         "n_sample_steps": req.n_sample_steps,
#         "images": images_base64
#     }

# from fastapi import FastAPI
# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int):
#     return {"item_id": item_id}
