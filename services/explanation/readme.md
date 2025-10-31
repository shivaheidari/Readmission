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
