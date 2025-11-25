FROM python:3.12-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Workdir inside the container
WORKDIR /app

# Copy only requirements first for better Docker layer caching
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code & config (so image is self-contained)
COPY src ./src
COPY config ./config

# Environment settings
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "src.run_analysis", "--config", "config/config.yaml"]
