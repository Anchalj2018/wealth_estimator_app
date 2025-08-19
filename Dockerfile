
# Use a slim Python base image
FROM python:3.9-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=.


# Set working directory
WORKDIR /app

# Install system-level dependencies (FAISS/OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY embeddings ./embeddings

# Expose FastAPI default port
EXPOSE 8000

# Run FastAPI via uvicorn (production-style)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port","${PORT}"]
