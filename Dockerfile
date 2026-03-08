# Dockerfile for Hugging Face Spaces
# ===================================
# Deploy for free at: https://huggingface.co/spaces

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=tiny
ENV FLASK_ENV=production
ENV FLASK_DEBUG=0

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "app.app:create_app()"]

