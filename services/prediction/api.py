import re
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Define Data Models and Helpers ---
class InputNote(BaseModel):
    hadm_id: int
    text: str

def clean_mimic_text(text: str) -> str:
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 2. Load ML Assets at Startup ---
def load_assets():
    """Loads the fine-tuned model and tokenizer at startup."""
    print("Loading prediction model assets...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    fine_tuned_model_path = "./model/best_model"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
    model.to(device)
    
    print(f"Prediction service assets loaded on {device}.")
    return model, tokenizer, device

model, tokenizer, device = load_assets()

# --- 3. Create the FastAPI Application ---
app = FastAPI(
    title="Prediction Service",
    description="Provides fast predictions for 30-day hospital readmission.",
    version="1.0.0"
)

# --- 4. Define the Prediction Endpoint ---
@app.post("/v1/predict-readmission")
def predict_readmission(note: InputNote):
    """Accepts a clinical note and returns a FHIR-compliant RiskAssessment."""
    cleaned_text = clean_mimic_text(note.text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
    
    probabilities = torch.softmax(logits, dim=1).squeeze()
    risk_probability = probabilities[1].item()
    prediction_outcome = "High Risk" if risk_probability > 0.5 else "Low Risk"

    fhir_response = {
        "resourceType": "RiskAssessment", "status": "final",
        "subject": {"reference": f"Encounter/{note.hadm_id}"},
        "prediction": [{"outcome": {"text": prediction_outcome}, "probabilityDecimal": risk_probability}]
    }
    
    return fhir_response