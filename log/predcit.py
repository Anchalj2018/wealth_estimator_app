# embedding_and_upload.py
import os
import glob
import pinecone
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from dotenv import load_dotenv
import json

# === Load environment variables ===
load_dotenv()

# === Init FastAPI ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load CLIP ===
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Normalize function ===
def normalize(vec):
    return vec / np.linalg.norm(vec)

# === Init Pinecone ===
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
index_name = "wealth-estimator"
index = pinecone.Index(index_name)

# === Load metadata ===
with open("app/metadata.json") as f:
    metadata_lookup = json.load(f)

# === Get embedding for uploaded image ===
def get_embedding(image):
    img = Image.open(image).convert("RGB")
    inputs = processor(images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs).squeeze().numpy()
    return normalize(embedding)

# === Predict endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    embedding = get_embedding(file.file)
    result = index.query(vector=embedding.tolist(), top_k=3)
    top_matches = []
    net_worths = []
    for match in result["matches"]:
        match_id = match["id"]
        score = match["score"]
        meta = metadata_lookup.get(match_id, {})
        net = meta.get("net_worth", 0)
        top_matches.append({"id": match_id, "score": score, "net_worth": net})
        net_worths.append(net)
    estimated_wealth = sum(net_worths) / len(net_worths) if net_worths else 0
    return {"estimated_wealth": estimated_wealth, "matches": top_matches}

# === (Optional) Reuse this for uploading ===
def compute_embeddings(image_folder):
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder, "*.png"))
    ids = []
    embeddings = []
    for i, path in enumerate(tqdm(image_paths)):
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs).squeeze().numpy()
        embedding = normalize(embedding)
        embeddings.append(embedding.tolist())
        ids.append(os.path.basename(path))  # Use filename as ID
    return ids, embeddings

def upload_to_pinecone(ids, embeddings):
    vectors = list(zip(ids, embeddings))
    index.upsert(vectors)

# === Optional: script mode to upload ===
if __name__ == "__main__":
    ids, embs = compute_embeddings("./data")
    upload_to_pinecone(ids, embs)
