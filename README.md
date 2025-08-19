# Wealth Estimator Vision

##### A FastAPI-based application that estimates net worth using vector search with FAISS and embedding logic.
## 🧠 Project Structure
wealth_estimator_vision/
├wealth_estimator_image_app/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entrypoint with uvicorn.run()
│   ├── helper.py                # Embedding + FAISS logic
│   ├── faiss_index.index        # FAISS index file
│   ├── embedding_store.pkl      # Metadata (e.g., net worth, labels, etc.)
├── requirements.txt
├── Dockerfile
├── generate_faiss_index.py
├── Makefile
├──database_embeddings.py
|─ .gitignore
├── README.md
---
### Assumptions
📝 Assumptions & Key Points
- This project uses FAISS and precomputed image embeddings for similarity search.
- Only image files in .png or .jpeg/.jpg format are supported.
- Embeddings were generated using a pre-trained CLIP model to convert images into vector representations.
- For simplicity, the FAISS index (faiss_index.index) and embedding metadata (embedding_store.pkl) are precomputed and loaded.

### 🚀 Run the App with Docker
#### 🔨 Build the Docker image

From the root of the project directory build and run the docker as  follows:
```bash
docker build -t wealth-estimator-app .
docker run -p 8000:8000 wealth-estimator-app
```
### 🐍 Run the App Locally Without Docker
You can also run the app locally without Docker:
From the project root directory, create virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> ⚠️ **Note**: If you run the app locally (not in Docker), make sure to set the `PYTHONPATH` so that imports like `app.helper` work correctly.
On macOS/Linux:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 
```
### 🔍 After Running the App (Docker or Local)
Once the app is running, you can verify it's working by visiting:
- 🧪 API Base URL: http://localhost:8000
- 📘 Swagger UI (interactive docs): http://localhost:8000/docs
- use /predict endpoint
- If you deployed this on Render or another platform, replace localhost with the live domain.

### 🧪 Main API Endpoint
To perform a similarity search and get predicted net worth, use the following endpoint:

- 🔗 **POST /predict**
Send a request with an image file (in `.png` or `.jpeg` format) to this endpoint. The server will return the most similar result from the embedding store based on FAISS vector similarity.
You can test it directly in the Swagger UI at:
[http://localhost:8000/docs](http://localhost:8000/docs)
