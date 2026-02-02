# ────────────────────────────────────────────────────────────────
# Dockerfile for BDD100K Data Analysis (Bosch Assignment)
# ────────────────────────────────────────────────────────────────
# Supports:
#   - Batch analysis: python src/analyze.py --labels-train ... --labels-val ...
#   - Interactive dashboard: streamlit run src/dashboard.py
#
# Build:    docker build -t bdd-analysis .
# Run batch: docker run --rm -v $(pwd)/data:/data:ro -v $(pwd)/output:/output bdd-analysis python src/analyze.py ...
# Run dashboard: docker run -it --rm -p 8501:8501 -v $(pwd)/output:/output bdd-analysis
# ────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── Install system dependencies (for matplotlib, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory
WORKDIR /app

# ── Copy and install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy analysis code
COPY src/ ./src/

# ── Create output directory (will be mounted anyway, but good practice)
RUN mkdir -p /output /data

# ── Expose Streamlit default port
EXPOSE 8501

# ── Default command = start Streamlit dashboard
#    (override with python src/analyze.py ... when doing batch analysis)
CMD ["streamlit", "run", "src/dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]