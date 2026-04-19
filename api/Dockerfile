FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy sibling packages
COPY ../pipeline /pipeline
COPY ../rag /rag
COPY ../scraper /scraper

ENV PYTHONPATH=/app:/pipeline:/rag:/scraper

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
