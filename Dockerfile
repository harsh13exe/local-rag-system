FROM python:3.11-slim

WORKDIR /app

# Install system deps needed for pypdf/chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY ingest.py .

# documents/ and chroma_db/ are mounted as volumes at runtime (see docker-compose.yml),
# not baked into the image, so you can update your docs without rebuilding.

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
