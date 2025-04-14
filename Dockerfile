# syntax=docker/dockerfile:1

FROM rapidsai/rapidsai-core:23.12-cuda11.8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy code
COPY . /app

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev build-essential \
        git curl wget libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Run the app using gunicorn and UNIX socket
CMD ["gunicorn", "app.main:app", "--bind", "unix:/app/loan-prediction.sock", "--workers", "1", "--threads", "2"]
