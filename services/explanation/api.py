from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .explainer import ExplanationService
import os
from dotenv import load_dotenv

load_dotenv()

class InputNote(BaseModel):
    text: str

API_KEY = os.getenv("GEMINI_API_KEY")
print(API_KEY)
model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
fine_tuned_model_path = "./model_artifacts/best_model"


explainer_service = ExplanationService(fine_tuned_model_path, model_checkpoint, API_KEY)

app = FastAPI(title="Clinical Explanation Service", version="1.0.0")

@app.post("/v1/explain")
def get_explanation(data: InputNote):
    try:
        result = explainer_service.predict_and_explain(data.text)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
