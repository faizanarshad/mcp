FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p templates static/css static/js

# Expose ports for different services
EXPOSE 5000 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=src/web_interface.py

# Default command (Discord bot)
CMD ["python", "src/diabetes_discord_bot.py"]

# Alternative commands for different services:
# Web Interface: CMD ["python", "src/web_interface.py"]
# API Server: CMD ["python", "src/api_server.py"]
# Mobile App: CMD ["python", "src/mobile_app.py"] 