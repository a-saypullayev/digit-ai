
import os
import json
import time
import threading
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from fastapi import FastAPI, HTTPException, Header, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
MODEL_PATH = BASE / "digit_model.pt"
DATA_PATH = BASE / "training_data.jsonl"
DEVICE = torch.device("cpu")

# Neural network: 324 inputs -> 128 hidden -> 9 outputs
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(324, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.layers(x)


model = DigitNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
model_lock = threading.Lock()

if MODEL_PATH.exists():
    try:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        print("Loaded saved neural network.")
    except Exception as exc:
        print("Could not load model:", exc)


app = FastAPI(title="Handwritten Digit Neural Network")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


class ImageRequest(BaseModel):
    pixels: list[float] = Field(..., min_length=324, max_length=324)


class TrainRequest(BaseModel):
    pixels: list[float] = Field(..., min_length=324, max_length=324)
    label: int = Field(..., ge=1, le=9)
    epochs: int = Field(default=5, ge=1, le=30)


def validate_pixels(pixels):
    if len(pixels) != 324:
        raise HTTPException(status_code=400, detail="Expected 324 pixels.")
    if any(not 0 <= float(v) <= 1 for v in pixels):
        raise HTTPException(status_code=400, detail="Pixels must be 0 to 1.")


def load_training_data():
    xs, ys = [], []

    if DATA_PATH.exists():
        with DATA_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if len(item["pixels"]) == 324 and 1 <= int(item["label"]) <= 9:
                        xs.append(item["pixels"])
                        ys.append(int(item["label"]) - 1)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue

    return xs, ys


def run_prediction(pixels):
    x = torch.tensor(pixels, dtype=torch.float32).reshape(1, 324)

    with model_lock:
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
            confidence, index = torch.max(probs, dim=0)

    return {
        "digit": int(index.item()) + 1,
        "confidence": round(float(confidence.item()), 4),
        "probabilities": {
            str(i + 1): round(float(probs[i].item()), 4)
            for i in range(9)
        }
    }


# Store incoming transmitter groups
groups = {}
groups_lock = threading.Lock()

last_prediction = None
last_prediction_time = None


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    xs, _ = load_training_data()
    return {
        "status": "ok",
        "device": str(DEVICE),
        "training_examples": len(xs),
        "architecture": "324 -> 128 -> 9",
        "classes": "1-9"
    }


@app.post("/predict")
def predict(req: ImageRequest):
    validate_pixels(req.pixels)
    return run_prediction(req.pixels)


@app.post("/receive/{group_id}")
def receive(
    group_id: int,
    value: str = Form(...)
):
    global last_prediction, last_prediction_time

    if not 1 <= group_id <= 41:
        raise HTTPException(status_code=400, detail="Group must be 1 to 41.")

    if len(value) != 8 or any(bit not in "01" for bit in value):
        raise HTTPException(
            status_code=400,
            detail="Value must contain exactly 8 binary digits."
        )

    # Last group uses only its first 4 bits.
    if group_id == 41:
        value = value[:4]

    with groups_lock:
        groups[group_id] = value

        received = len(groups)

        if received < 41:
            return {
                "status": "receiving",
                "received_groups": received,
                "remaining": 41 - received
            }

        # Assemble in transmitter order.
        binary = "".join(groups[i] for i in range(1, 42))

        if len(binary) != 324:
            raise HTTPException(status_code=400, detail="Incorrect image size.")

        pixels = [float(bit) for bit in binary]

        # Reset for the next drawing.
        groups.clear()

    result = run_prediction(pixels)

    last_prediction = result
    last_prediction_time = time.time()

    print("Build a Logic prediction:", result["digit"])

    return {
        "status": "complete",
        "digit": result["digit"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"]
    }


@app.get("/result")
def result():
    if last_prediction is None:
        return {"status": "waiting", "message": "No complete image received yet."}

    return {
        "status": "complete",
        **last_prediction,
        "timestamp": last_prediction_time
    }


@app.post("/train")
def train(req: TrainRequest, x_training_key: str | None = Header(default=None)):
    expected_key = os.getenv("TRAINING_KEY")

    if expected_key and x_training_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid training key.")

    validate_pixels(req.pixels)

    with DATA_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "pixels": req.pixels,
            "label": req.label
        }) + "\n")

    xs, ys = load_training_data()

    x_tensor = torch.tensor(xs, dtype=torch.float32)
    y_tensor = torch.tensor(ys, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=min(32, len(xs)),
        shuffle=True
    )

    with model_lock:
        model.train()
        total_loss = 0.0

        for _ in range(req.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

        torch.save(model.state_dict(), MODEL_PATH)

    return {
        "message": "Trained and saved.",
        "label": req.label,
        "examples": len(xs),
        "epochs": req.epochs,
        "average_batch_loss": round(
            total_loss / max(1, req.epochs * len(loader)), 4
        )
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )
