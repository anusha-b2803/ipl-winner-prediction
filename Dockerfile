# STAGE 1: Build Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# Set API URL to relative path for monolith
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build

# STAGE 2: Python Runner
FROM python:3.12-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY api/requirements.txt ./api/
COPY scraper/requirements.txt ./scraper/
RUN pip install --no-cache-dir -r api/requirements.txt
RUN pip install --no-cache-dir -r scraper/requirements.txt

# Copy source and built frontend
COPY . .
COPY --from=frontend-builder /build/frontend/out ./static

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
