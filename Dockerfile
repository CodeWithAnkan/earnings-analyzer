FROM python:3.11-slim

# HF Spaces runs on port 7860
EXPOSE 7860

WORKDIR /app

# System dependencies for psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies — CPU-only torch keeps image size down
COPY requirements_hf.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements_hf.txt

# Pre-download FinBERT into the image so it never downloads at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('ProsusAI/finbert')"

# Copy app code
COPY . .

# Start FastAPI on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
