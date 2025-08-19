

from fastapi import FastAPI, UploadFile, File, HTTPException ,status
from fastapi.responses import JSONResponse
from app.helper import compute_image_embedding, get_top_k_with_estimation_faiss,load_clip_model,load_files
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pydantic import BaseModel
from typing import List
import logging
import os

# --- Pydantic response model ---
class Match(BaseModel):

    net_wealth: float
    wealth_level: str
    similarity_score: float


class EstimationResponse(BaseModel):
    estimated_net_worth_canadian: float
    estimated_wealth_level: str
    top_3_matches: List[Match]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_ROOT, "log")
LOG_FILE = os.path.join(LOG_PATH, "model.log")        
os.makedirs("log", exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Logger initialized and ready.")  # Test log entry



# --- App Init ---
app = FastAPI(
    title="Wealth Estimation API",
    description="Estimates a person's net worth from an image and returns top 3 visually similar profiles.",
    version="1.0.0"
)


# --- Predict Endpoint ---
@app.post("/predict", response_model=EstimationResponse)
async def predict(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png","image/jpg"]:
            return JSONResponse(
                content={"message": "Only JPEG or PNG images are supported."},
                status_code=status.HTTP_400_BAD_REQUEST
            )
        size=(224, 224)
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(size, Image.BICUBIC)


        query_embedding = compute_image_embedding(image, app.state.clip_model, app.state.clip_processor)

        results = get_top_k_with_estimation_faiss(query_embedding, app.state.index_file, app.state.meta)

        # return results

        return JSONResponse(
                content={ "results": results},
                status_code=status.HTTP_200_OK
        )

    except Exception as e:
        logger.error(str(e))
        return JSONResponse(
            content={"message": "Unexpected error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# --- Root Endpoint ---
@app.get("/")
def root():
    return {"message": "Wealth Estimator API is running."}


# --- Startup Event ---
FILE_DIR = "embeddings"

@app.on_event("startup")
def load_components():
    app.state.clip_model, app.state.clip_processor = load_clip_model(device="cpu")
    app.state.index_file, app.state.meta = load_files(FILE_DIR)






    
