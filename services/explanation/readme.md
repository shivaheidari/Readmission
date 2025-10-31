# Clinical Explanation Service

## Overview
This microservice provides interpretability and narrative explanations for patient readmission risk predictions. It leverages SHAP for feature attribution and the Gemini LLM model for generating human-readable summaries of the prediction results.

## Key Features
- Clean and preprocess clinical notes.
- Predict readmission risk using a fine-tuned ClinicalBERT model.
- Generate SHAP explanations to highlight important predictive features.
- Produce concise narrative explanations using Gemini generative AI.
  
## Setup

### Prerequisites
- Python 3.8+
- GPU recommended for faster inference (optional)
- Access to Google Gemini API (set `API_KEY` environment variable)

### Installation
- pip install -r requirements.txt



### Environment Variables
- `API_KEY`: Your Gemini API key for narrative generation.

## Running the Service

### Locally
- uvicorn api:app --host 0.0.0.0 --port 8001 --reload
### Docker
- docker build -t explanation-service .
- docker run -p 8001:8001 explanation-service


## API Endpoints

### POST `/v1/explain`
- **Input:** JSON with clinical note text `{ "text": "<patient discharge note>" }`
- **Output:** JSON containing
  - `prediction`: Risk class ("High Risk" / "Low Risk")
  - `risk_score`: Probability score
  - `narrative_summary`: Human-readable explanation
  - `quantitative_explanation`: List of feature impacts from SHAP

Example request:
{
"text": "Patient notes go here ..."
}