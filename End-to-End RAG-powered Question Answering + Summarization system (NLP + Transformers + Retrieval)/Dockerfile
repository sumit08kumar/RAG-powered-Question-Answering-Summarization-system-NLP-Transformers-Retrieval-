FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY demo/ ./demo/

# Create directories for data
RUN mkdir -p /app/data /app/vector_store

# Expose ports
EXPOSE 8000 8501

# Copy startup script
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app/src
ENV TRANSFORMERS_CACHE=/app/data/transformers_cache

# Start both services
CMD ["./docker-entrypoint.sh"]

