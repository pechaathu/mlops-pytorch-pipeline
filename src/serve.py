"""FastAPI model-serving app: /health and /predict endpoints."""

import io
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

# CIFAR-10 class names, index-aligned with the model outputs.
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Same normalization used for validation data.
PREPROCESS = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
])

app = FastAPI(title="CIFAR-10 Classifier")

# Module-level holders, populated at startup.
model = None
device = torch.device("cpu")


def _checkpoint_path() -> str:
    """Prefer the container path, fall back to the local repo path."""
    container = Path("/app/checkpoints/classifier_v1.pt")
    if container.exists():
        return str(container)
    return os.environ.get("CHECKPOINT_PATH", "checkpoints/classifier_v1.pt")


@app.on_event("startup")
def load_model():
    """Load the checkpoint once when the server starts."""
    global model
    ckpt = torch.load(_checkpoint_path(), map_location=device)
    model = get_model(
        architecture=ckpt.get("architecture", "resnet18"),
        num_classes=ckpt.get("num_classes", 10),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()


@app.get("/health")
def health():
    """Return 200 with status when the model is loaded, 503 otherwise."""
    if model is None:
        return {"status": "loading"}, 503
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Accept an image file, return per-class probabilities."""
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    return {
        "predicted_class": CLASSES[int(probs.argmax())],
        "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))},
    }
