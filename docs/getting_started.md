# Getting Started

This guide walks you through setting up the Unified XAI Governance Engine for development and production workloads.

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.10+
- Docker and Docker Compose
- Make (optional, but recommended for task execution)

## Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/beingaditya18/Unified-XAI-Engine-Dashboard.git
   cd Unified-XAI-Engine-Dashboard
   ```

2. **Establish Environment:**
   Create a `.env` file in the root directory (using `.env.example` as a starting point):
   ```bash
   cp .env.example .env
   ```

3. **Install dependencies:**
   ```bash
   make install
   ```

4. **Run model training pipeline:**
   This pre-processes the UCI Adult Census data, serializes encoders, and trains both the XGBoost and Neural Network models:
   ```bash
   make train
   ```

5. **Start service components:**
   - **FastAPI backend:** `make run-api` (Port 8000)
   - **Streamlit interface:** `make run-ui` (Port 8501)

## Docker Orchestration

To run the complete FastAPI + Streamlit backend-frontend stack inside containers:

```bash
# Build and run containerized services
docker compose up -d --build
```
- Streamlit application: [http://localhost:8501](http://localhost:8501)
- FastAPI documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
