# predownload_model.py
from transformers import CLIPModel, CLIPProcessor
MODEL_NAME = "openai/clip-vit-base-patch32"
print("Pre-downloading", MODEL_NAME)
CLIPProcessor.from_pretrained(MODEL_NAME)
CLIPModel.from_pretrained(MODEL_NAME)
print("Done pre-downloading")
