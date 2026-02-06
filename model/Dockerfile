FROM python:3.10-slim
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-
# FROM pytorch/pytorch:2.1.0-cpu-py3.10
# FROM --platform=linux/amd64 pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime


ENV DEBIAN_FRONTEND=noninteractive
ENV FIFTYONE_DISABLE_SERVICES=1

# System deps
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluate ./evaluate
COPY training ./training
# COPY /Users/chandagroverkamra/Documents/Bosch/ADAS_Assignment/data_bdd ./data
COPY Dashboard ./dashboard

# Default command (can be overridden)
CMD ["bash"]
