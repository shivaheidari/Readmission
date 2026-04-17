# Readmission Prediction and Explanation APIs

This project provides a microservices-based architecture for predicting hospital readmissions and generating insights/explanations for those predictions. The application is containerized using Docker and orchestrated with Docker Compose.

## Project Structure

The project consists of two main services:

*   **Prediction API** (`prediction_service`): Handles the machine learning predictions for hospital readmissions.
    *   Source: `./services/prediction`
    *   Exposed Port: `8000` (mapped to container port `8000`)
*   **Explanation API** (`explanation_service`): Generates explainability metrics and insights for the model's predictions.
    *   Source: `./services/explanation`
    *   Exposed Port: `8001` (mapped to container port `8000`)

## Prerequisites

Before running the project, ensure you have the following installed on your machine:
*   Docker
*   Docker Compose

## Configuration

The `explanation_api` service relies on environment variables to function correctly. 

1. Create a `.env` file in the root directory of the project (next to `docker-compose.yml`).
2. Add the necessary environment variables to this file. *(Update this section with the specific variables your app needs, e.g., `MODEL_PATH`, `DB_URI`, or `API_KEY`)*.

## Running the Project

To start both services, open your terminal in the root directory of the project and run:

```bash
docker-compose up --build
```

This will build the Docker images and start the containers. 

Once the containers are running, you can access the APIs at:
*   **Prediction API:** `http://localhost:8000`
*   **Explanation API:** `http://localhost:8001`

To run the services in detached mode (in the background), use:
```bash
docker-compose up -d --build
```

To stop the running services:
```bash
docker-compose down
```

## Development & Live Reloading

The `api.py` files for both services are mounted as volumes in the `docker-compose.yml` file:
*   `./services/prediction/api.py:/app/api.py`
*   `./services/explanation/api.py:/app/api.py`

This means that if you make changes to either `api.py` file locally, the updates will immediately reflect inside the running Docker containers. If your Python web framework (e.g., FastAPI with Uvicorn, or Flask) is configured with live-reloading, the API will update automatically without needing to rebuild the Docker images.