# IPL Prophet: Predictive Intelligence System

IPL Prophet is a high-capacity RAG (Retrieval-Augmented Generation) system designed to predict tournament outcomes and provide lead-analyst level insights for the Indian Premier League. By combining seventeen seasons of historical match data with real-time scraping and advanced transformer-based reasoning, the system delivers evidence-based forecasts with professional-grade depth.

## Core Features

- **Hybrid RAG Engine**: Integrates semantic vector search (Qdrant) for match narratives and structured SQL queries (SQLite) for point tables and head-to-head records.
- **Lead Analyst Persona**: High-fidelity AI responses mimicking professional cricket commentators, delivering structured reports with headlines, key statistics, and strategic insights.
- **Standardized Forecasting**: Tournament predictions identifying the Top 5 contenders with calculated winning probabilities and detailed reasoning.
- **Automated Data Sync**: An integrated pipeline that synchronizes live season data from official sources every 12 hours.
- **Robust Team Mapping**: Advanced alias resolution supporting all major abbreviations (MI, CSK, RCB, etc.) to ensure data integrity during complex queries.

## Architecture

The system follows a modular intelligence-first architecture:

1. **Scraper Service**: Async implementation that pulls live standings and match results from ESPNcricinfo.
2. **Ingestion Pipeline**: Processes raw JSON data into localized vector embeddings (BAAI/bge-small-en-v1.5) and structured SQL records.
3. **Intelligence Layer**: Utilizes Meta-Llama-3.1-70B via Hugging Face Inference API for reasoning, grounded by the hybrid retrieval context.
4. **Backend API**: FastAPI-based server handling predictions, statistics retrieval, and RAG-powered Q&A.
5. **Frontend Dashboard**: Professional analytics interface for real-time interaction.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Node.js 20 or higher
- Hugging Face API Key (with access to Llama-3.1-70B)

### Installation

1. Clone the repository and navigate to the project root.
2. Install Python dependencies:
   ```bash
   pip install -r api/requirements.txt
   ```
3. Configure environment variables in a `.env` file:
   ```env
   HUGGINGFACE_API_KEY=your_key_here
   HF_MODEL=meta-llama/Llama-3.1-70B-Instruct
   SQLITE_DB_PATH=./data/ipl_stats.db
   QDRANT_PATH=./data/vector_db
   ```
4. Initialize the data pipeline:
   ```bash
   python pipeline/sync_pipeline.py
   ```
5. Start the API server:
   ```bash
   python api/main.py
   ```

## API Documentation

### Key Endpoints

- **GET `/predict/{year}`**: Generates a professional prediction report for the specified tournament year.
- **POST `/ask`**: Accepts natural language questions and returns RAG-grounded analyst insights.
- **GET `/stats/{year}`**: Retrieves the full points table and recent momentum for a specific season.
- **GET `/stats/titles/all`**: Returns a summary of all champion titles held by franchises.

Complete documentation is available at the `/docs` endpoint of the running server.

## License

This project is licensed under the MIT License.
