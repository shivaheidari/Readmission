import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from dotenv import load_dotenv
import google.generativeai as genai

# --- 1. Load Environment Variables and Configure LLM ---
load_dotenv()
API_KEY = os.getenv("API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("Warning: API_KEY not found in .env file. Narrative generation will fail.")

# --- 2. Define Helper Functions and Data Models ---
class InputNote(BaseModel):
    text: str

def clean_mimic_text(text: str) -> str:
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. Load ML Assets at Startup ---
def load_assets():
    """Loads all necessary models, tokenizers, and clients at startup."""
    print("Loading ML assets...")
    device = 0 if torch.cuda.is_available() else -1
    
    model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    # This path assumes the model is in a 'model' subfolder relative to this script
    fine_tuned_model_path = "./model/best_model" 

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
    
    # Create a pipeline for SHAP, specifying the device
    pred_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, return_all_scores=True)
    explainer = shap.Explainer(pred_pipeline)
    
    # Create the Gemini model instance
    llm_model = genai.GenerativeModel('gemini-1.5-flash') if API_KEY else None
    
    print("ML assets loaded successfully.")
    return model, tokenizer, explainer, llm_model, device

model, tokenizer, explainer, llm_model, device = load_assets()

# --- 4. Create the FastAPI Application ---
app = FastAPI(
    title="Clinical Explanation Service",
    description="Generates SHAP and narrative explanations for readmission risk predictions.",
    version="1.0.0"
)

