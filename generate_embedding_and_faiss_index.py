
import os
import uuid
import pickle
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

# ---------------------------
# Config / paths
# ---------------------------
DATA_DIR = "data"
EMBEDDINGS_DIR = "embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(EMBEDDINGS_DIR, "metadata.pkl")
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss_index.index")

# ---------------------------
# Sample image DB
# ---------------------------
IMAGE_DB = [
    {"path": "pic1.jpg", "label": "Wealthy", "networth": 500000},
    {"path": "pic2.jpg", "label": "Not Wealthy", "networth": 6000},
    {"path": "pic3.jpg", "label": "Wealthy", "networth": 700000},
    {"path": "pic4.jpg", "label": "Not Wealthy", "networth": 5000},
    {"path": "pic5.jpg", "label": "Wealthy", "networth": 600000},
    {"path": "pic6.jpg", "label": "Not Wealthy", "networth": 2000},
]

# ---------------------------
# Load CLIP model once
# ---------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_clip_embedding(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)  # normalize for cosine
    return embedding[0].cpu().numpy().astype("float32")  # NumPy array, ready for FAISS


def load_and_resize_image(path, size=(224, 224)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BICUBIC)
    return img

# ---------------------------
# Generate embeddings & metadata
# ---------------------------
embedding_list = []
metadata_list = []

for item in IMAGE_DB:
    uid = str(uuid.uuid4())
    img_path = os.path.join(DATA_DIR, item["path"])
    img=load_and_resize_image(img_path)
    emb = compute_clip_embedding(img)

    embedding_list.append(emb)

    metadata_list.append({
        "id": uid,
        "filename": item["path"],
        "label": item["label"],
        "networth": item["networth"]
    })

# Save embeddings as NumPy array
embeddings_array = np.stack(embedding_list)
np.save(EMBEDDINGS_FILE, embeddings_array)

# Save metadata
with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadata_list, f)

# ---------------------------
# Build FAISS index
# ---------------------------
dim = embeddings_array.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity using inner product
faiss.normalize_L2(embeddings_array)  # Important for cosine similarity
index.add(embeddings_array)


# Save FAISS index
faiss.write_index(index, FAISS_INDEX_FILE)

print("Embeddings, metadata, and FAISS index saved successfully.")




