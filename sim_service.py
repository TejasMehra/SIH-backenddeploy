"""
sim_service.py — Local FastAPI service for image <-> description similarity.

Usage (local):
  1. Create venv and install deps (see README commands below).
  2. Run:
       uvicorn sim_service:app --host 127.0.0.1 --port 8000 --reload

Endpoint:
  POST /analyze
    - multipart form fields:
        * description (text)
        * image (file)
    - returns JSON:
        { "similarity_confidence": 0.73, "verdict": "RELATED", "details": {...} }

Behavior:
  - Attempts to use Hugging Face CLIP (transformers + torch). If unavailable, falls
    back to a small color-histogram heuristic so the API still works.
  - Loads the CLIP model lazily on the first request to reduce startup time.
"""

import io
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="sim-service-local", version="1.0")

# Allow local testing from browsers/tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Model globals (loaded lazily)
MODEL = None
PROCESSOR = None
DEVICE = "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

def load_clip_model():
    """Lazy-load CLIP model (transformers). If import or load fails, MODEL stays None."""
    global MODEL, PROCESSOR, DEVICE
    if MODEL is not None:
        return
    try:
        import torch   # type: ignore
        from transformers import CLIPProcessor, CLIPModel  # type: ignore
    except Exception as e:
        print("CLIP imports failed (transformers/torch missing?):", e)
        MODEL = None
        PROCESSOR = None
        return

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model {MODEL_NAME} on {DEVICE} ... (this may take a while)")
    MODEL = CLIPModel.from_pretrained(MODEL_NAME)
    PROCESSOR = CLIPProcessor.from_pretrained(MODEL_NAME)
    MODEL.to(DEVICE)
    print("CLIP model loaded.")

def compute_clip_similarity_bytes(image_bytes: bytes, text: str):
    """
    Use CLIP model to compute similarity.
    Returns: (confidence_in_0_1, details_dict)
    """
    try:
        import torch   # type: ignore
        from PIL import Image  # type: ignore
        from io import BytesIO
    except Exception as e:
        raise RuntimeError("Missing runtime deps for CLIP: " + str(e))

    load_clip_model()
    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("CLIP model not available on this server (failed to import or load).")

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = PROCESSOR(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_emb = MODEL.get_image_features(**{k:inputs[k] for k in ["pixel_values"]})
        txt_emb = MODEL.get_text_features(**{k:inputs[k] for k in ["input_ids", "attention_mask"]})
    img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
    sim = (img_emb @ txt_emb.T).cpu().numpy()[0][0]  # -1..1
    conf = float((sim + 1.0) / 2.0)  # map to 0..1
    return conf, {"cosine_raw": float(sim)}

def fallback_similarity_bytes(image_bytes: bytes, text: str):
    """Simple histogram/color heuristic fallback (works without torch/transformers)."""
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return 0.5, {"note": "no-deps-fallback"}

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((256,256))
    arr = (np.array(image) / 255.0).astype("float32")
    mean_rgb = arr.mean(axis=(0,1))
    t = text.lower()
    if any(w in t for w in ("oil", "sheen", "stain", "pollut", "slick", "tar", "black", "brown")):
        darkness = 1.0 - mean_rgb.mean()
        brownish = max(0.0, (mean_rgb[0] - mean_rgb[2]))  # R - B
        score = 0.35 * darkness + 0.65 * brownish
    elif any(w in t for w in ("clear", "blue", "clean", "sunset", "scenic", "calm")):
        score = float(mean_rgb[2])  # bluish
    else:
        score = 0.5
    score = max(0.0, min(1.0, float(score)))
    return score, {"mean_rgb": [float(x) for x in mean_rgb.tolist()]}

@app.post("/analyze")
async def analyze(image: UploadFile = File(...), description: str = Form(...)):
    """
    Analyze an image vs a description. Returns similarity confidence (0..1).
    """
    if not description or description.strip() == "":
        raise HTTPException(status_code=400, detail="description is required")
    if image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="uploaded file must be an image")

    data = await image.read()
    max_bytes = 10 * 1024 * 1024  # 10 MB default local limit
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"image too large (max {max_bytes} bytes)")

    # Try CLIP
    try:
        conf, details = compute_clip_similarity_bytes(data, description)
        verdict = "RELATED" if conf >= 0.55 else ("POSSIBLY_RELATED" if conf >= 0.45 else "NOT_RELATED")
        return JSONResponse({"similarity_confidence": round(conf, 4), "verdict": verdict, "details": details})
    except Exception as e:
        # fallback
        tb = traceback.format_exc()
        print("CLIP not available or failed:", e)
        print(tb)
        conf, details = fallback_similarity_bytes(data, description)
        verdict = "RELATED" if conf >= 0.55 else ("POSSIBLY_RELATED" if conf >= 0.45 else "NOT_RELATED")
        return JSONResponse({"similarity_confidence": round(conf, 4), "verdict": verdict, "details": details, "note": "Used fallback heuristic because CLIP not available."})

@app.get("/")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}
