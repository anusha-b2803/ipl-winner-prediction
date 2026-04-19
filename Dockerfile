FROM python:3.12-slim

# Install system dependencies for build
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layering
COPY api/requirements.txt ./api/requirements.txt
COPY scraper/requirements.txt ./scraper/requirements.txt

RUN pip install --no-cache-dir -r api/requirements.txt
RUN pip install --no-cache-dir -r scraper/requirements.txt

# Copy the entire project
COPY . .

# Set environment paths
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Start the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
