# sim_service.py
# FastAPI service that computes image<->text similarity using CLIP.
# POST /analyze accepts multipart form fields:
#   - image: file
#   - description: text
#
# Returns JSON with similarity_confidence (0..1) and verdict.

import io
import os
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="sim-service", version="1.0")

# Allow CORS from anywhere by default (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Load model lazily on first request to reduce startup time for lightweight deploys
MODEL = None
PROCESSOR = None
DEVICE = "cpu"
MODEL_NAME = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")

def load_clip_model():
    global MODEL, PROCESSOR, DEVICE
    if MODEL is not None:
        return
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
    except Exception as e:
        # we'll fall back later
        print("CLIP imports failed:", e)
        MODEL = None
        PROCESSOR = None
        return

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model {MODEL_NAME} on {DEVICE} ...")
    MODEL = CLIPModel.from_pretrained(MODEL_NAME)
    PROCESSOR = CLIPProcessor.from_pretrained(MODEL_NAME)
    MODEL.to(DEVICE)
    print("Model loaded.")

def compute_clip_similarity_bytes(image_bytes: bytes, text: str):
    """
    Returns confidence in [0,1] and details dict.
    """
    try:
        import torch
        from PIL import Image
        from io import BytesIO
    except Exception as e:
        raise RuntimeError("Missing runtime dependencies for CLIP: " + str(e))

    # Ensure model is loaded
    load_clip_model()
    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("CLIP model not available on this server.")

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = PROCESSOR(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_emb = MODEL.get_image_features(**{k:inputs[k] for k in ["pixel_values"]})
        txt_emb = MODEL.get_text_features(**{k:inputs[k] for k in ["input_ids", "attention_mask"]})
    # normalize and compute cosine
    img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
    sim = (img_emb @ txt_emb.T).cpu().numpy()[0][0]  # -1..1
    conf = float((sim + 1.0) / 2.0)  # map to 0..1
    return conf, {"cosine_raw": float(sim)}

def fallback_similarity_bytes(image_bytes: bytes, text: str):
    # trivial color-histogram fallback used when CLIP not installed
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
    Analyzes the image against the given description and returns similarity.
    """
    # Basic input checks
    if not description or description.strip() == "":
        raise HTTPException(status_code=400, detail="description is required")
    if image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="uploaded file must be an image")

    # Read image bytes (limit size to avoid memory blowups)
    data = await image.read()
    max_bytes = int(os.environ.get("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))  # default 5MB
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"image too large (max {max_bytes} bytes)")

    # Attempt CLIP similarity, otherwise fallback
    try:
        conf, details = compute_clip_similarity_bytes(data, description)
        verdict = "RELATED" if conf >= 0.55 else ("POSSIBLY_RELATED" if conf >= 0.45 else "NOT_RELATED")
        return JSONResponse({
            "similarity_confidence": round(conf, 4),
            "verdict": verdict,
            "details": details
        })
    except Exception as e:
        # fallback
        tb = traceback.format_exc()
        print("CLIP failed or not available:", str(e))
        print(tb)
        conf, details = fallback_similarity_bytes(data, description)
        verdict = "RELATED" if conf >= 0.55 else ("POSSIBLY_RELATED" if conf >= 0.45 else "NOT_RELATED")
        return JSONResponse({
            "similarity_confidence": round(conf, 4),
            "verdict": verdict,
            "details": details,
            "note": "Used fallback heuristic because CLIP was not available or failed."
        })

@app.get("/")
def health():
    return {"status": "ok", "model": ("loaded" if MODEL is not None else "not-loaded")}
