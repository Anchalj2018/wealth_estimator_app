# from app.database import embedding_store
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import pickle 
import faiss
import numpy as np
import os

clip_model=None
clip_processor=None
metadata=None
index=None



def load_clip_model(device="cpu"):
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise e  # crash app if critical

    return clip_model, clip_processor

def load_files(base_path):
    ## Load index and inmut embeddings.
    index_path = os.path.join(base_path, "faiss_index.index")
    metadata_path = os.path.join(base_path, "metadata.pkl")

    index = faiss.read_index(index_path)

    ##load metadata
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    return index, metadata


#compute image embedding
def compute_image_embedding(image: Image.Image,clip_model,clip_processor) -> torch.Tensor:

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")

    
    inputs = clip_processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
        emb = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)  # normalize for cosine similarity

    # return embeddings[0]  # shape: (512,)

    return emb.cpu().numpy().astype("float32")


#load image
def load_and_resize_image(imgpath, size=(224, 224)):
    img = Image.open(imgpath).convert("RGB")
    img = img.resize(size, Image.BICUBIC)
    return img



#compute similarity score and return estimated results for wealth
def get_top_k_with_estimation_faiss(query_emb: np.ndarray, index, metadata, k=3):
    
    
    query = query_emb.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query)

    sim_scores, indices = index.search(query, k)
    top_k = []

    for i, idx in enumerate(indices[0]):
        meta = metadata[idx]
        top_k.append({
            "net_wealth": meta["networth"],
            "wealth_level": "Wealthy" if meta["networth"] >= 1_000_000 else "Not Wealthy",
            "similarity_score": float(sim_scores[0][i])
        })

    estimated = np.mean([x["net_wealth"] for x in top_k])
    mean_similarity = np.mean([x["similarity_score"] for x in top_k])
    label = "Wealthy" if estimated >= 50000 else "Not Wealthy"

    return {

        "estimated_net_worth_canadian": round(float(estimated), 2),
        "estimated_wealth_level": label,
        "top_3_matches": top_k
    }