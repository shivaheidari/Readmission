# Readmission Risk Prediction Service

## Overview
This microservice provides a REST API to predict 30-day hospital readmission risk using a fine-tuned ClinicalBERT model on patient discharge notes.

## Key Features
- Exposes an API to submit clinical notes for readmission risk prediction.
- Uses a fine-tuned transformer-based model for inference.
- Returns predicted class and probability scores.

## Setup

### Prerequisites
- Python 3.8+
- GPU recommended for inference (optional)

### Installation
pip install -r requirements.txt

## Running the Service

### Locally
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

### Docker
docker build -t prediction-service .
docker run -p 8000:8000 prediction-service

## Model Artifacts
Place your trained model files in the `model/` directory inside this service.

## API Endpoints

### POST `/v1/predict`
- **Input:** JSON with clinical note text `{ "text": "<patient discharge note>" }`
- **Output:** JSON containing
  - `prediction`: Risk class ("High Risk" / "Low Risk")
  - `risk_score`: Probability score

Example request:
{
"text": "Patient notes go here ..."
}

