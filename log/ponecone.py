# embedding_and_upload.py
import os
import glob
import pinecone
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

# === Step 1: Load CLIP ===
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Step 2: Normalize function ===
def normalize(vec):
    return vec / np.linalg.norm(vec)

# === Step 3: Load images and compute embeddings ===
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

# === Step 4: Upload to Pinecone ===
def upload_to_pinecone(ids, embeddings, index_name):
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=512, metric="cosine")
    index = pinecone.Index(index_name)
    vectors = list(zip(ids, embeddings))
    index.upsert(vectors)

if __name__ == "__main__":
    import torch
    ids, embs = compute_embeddings("./data")  # Folder with your images
    upload_to_pinecone(ids, embs, index_name="wealth-estimator")
