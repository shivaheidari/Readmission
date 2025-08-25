from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Define Helper Functions and Data Models ---

class InputNote(BaseModel):
    hadm_id: int
    text: str

def clean_mimic_text(text: str) -> str:
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
    text = re.sub(r'\_+', '', text)
    text = re.sub(r'\-+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 2. Load Your Model and Tokenizer at Startup ---

def load_model():
    """Loads the tokenizer from the Hub and the fine-tuned model locally."""
    model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    fine_tuned_model_path = "./model/best_model"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer from: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print(f"Loading model from: {fine_tuned_model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
    model.to(device) # Move model to GPU if available
    
    return model, tokenizer, device

model, tokenizer, device = load_model()

# --- 3. Create the FastAPI Application ---

app = FastAPI(
    title="Readmission Risk Prediction API",
    description="An API to predict 30-day hospital readmission from clinical notes.",
    version="1.0.0"
)

# --- 4. Define the Prediction Endpoint ---

@app.post("/v1/predict-readmission")
def predict_readmission(note: InputNote):
    """Accepts a clinical note and returns a FHIR-compliant RiskAssessment."""
    
    # <<< PREDICTION LOGIC STARTS HERE >>>
    
    # 1. Clean and prepare the input text
    cleaned_text = clean_mimic_text(note.text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True).to(device)

    # 2. Make prediction (no gradient calculation needed)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # 3. Get probabilities from logits
    probabilities = torch.softmax(logits, dim=1).squeeze()
    risk_probability = probabilities[1].item() # Probability of the positive class (readmission)

    # 4. Determine outcome based on a 0.5 threshold
    prediction_outcome = "High Risk" if risk_probability > 0.5 else "Low Risk"
    
    # <<< PREDICTION LOGIC ENDS HERE >>>

    # --- Construct the FHIR RiskAssessment Resource ---
    fhir_response = {
        "resourceType": "RiskAssessment",
        "status": "final",
        "subject": { "reference": f"Encounter/{note.hadm_id}" },
        "prediction": [{
            "outcome": { "text": prediction_outcome },
            "probabilityDecimal": risk_probability
        }],
        "interpretation": { 
            "text": f"Model prediction based on analysis of clinical text. Risk score of {risk_probability:.2f}."
        }
    }
    
    return fhir_response