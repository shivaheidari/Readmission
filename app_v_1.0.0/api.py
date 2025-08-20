from fastapi import FastAPI
from pydantic import BaseModel
import torch 
from transformers import AutoTokenizer

# --- Input and Output Data Models ---

class InputNote(BaseModel):
    hadm_id: int
    text: str


def load_model():
    """Loads the fine-tuned model and tokenizer."""
    print("Loading model and tokenizer...")
    model = "results_weighted_30/best_model" 
    model_checkpoint = "emilyalsentzer/Bio_ClinicalBert"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)
    print("Model loaded.")
    return model, tokenizer

model, tokenizer = load_model()

# --- 3. Create the FastAPI Application ---

app = FastAPI(
    title="Readmission Risk Prediction API",
    description="An API to predict 30-day hospital readmission from clinical notes.",
    version="1.0.0"
)

# --- 4. Define the Prediction Endpoint ---

@app.post("/v1/predict-readmission")
def predict_readmission(note: InputNote):
    """
    Accepts a clinical note and returns a FHIR-compliant RiskAssessment.
    """
    # --- Prediction Logic (Placeholder) ---
    risk_probability = 0.65  
    prediction_outcome = "High Risk"
    interpretation_text = "Model prediction based on terms related to chronic comorbidities."

    # --- Construct the FHIR RiskAssessment Resource ---
    fhir_response = {
        "resourceType": "RiskAssessment",
        "status": "final",
        "subject": {
            "reference": f"Encounter/{note.hadm_id}"
        },
        "prediction": [
            {
                "outcome": {
                    "text": prediction_outcome
                },
                "probabilityDecimal": risk_probability,
                "qualitativeRisk": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                            "code": "high",
                            "display": "High risk"
                        }
                    ]
                }
            }
        ],
        "interpretation": {
            "text": interpretation_text
        }
    }
    
    return fhir_response