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
    print("Warning: API_KEY not found. Narrative generation will fail.")

# --- 2. Define Data Models and Helpers ---


class InputNote(BaseModel):
    text: str

def clean_mimic_text(text: str) -> str:
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. Load ML Assets at Startup ---
def load_assets():
    """Loads all necessary models, tokenizers, and clients at startup."""
    print("Loading explanation service assets...")
    device = 0 if torch.cuda.is_available() else -1
    
    model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    fine_tuned_model_path = "./model/best_model" 

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
    
    pred_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, return_all_scores=True)
    explainer = shap.Explainer(pred_pipeline)
    llm_model = genai.GenerativeModel('gemini-1.5-flash') if API_KEY else None
    
    print("Explanation service assets loaded.")
    return model, tokenizer, explainer, llm_model, device

model, tokenizer, explainer, llm_model, device = load_assets()

# --- 4. Create the FastAPI Application ---
app = FastAPI(
    title="Clinical Explanation Service",
    description="Generates SHAP and narrative explanations for readmission risk predictions.",
    version="1.0.0"
)

# --- 5. Define the Explanation Endpoint ---
@app.post("/v1/explain")
def get_explanation(data: InputNote):
    if not llm_model:
        raise HTTPException(status_code=500, detail="LLM client not initialized. Check API key.")
        
    cleaned_text = clean_mimic_text(data.text)
    
    # Get the initial prediction for context
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1).squeeze()
    risk_probability = probabilities[1].item()
    predicted_class = "High Risk" if risk_probability > 0.5 else "Low Risk"

    # Get SHAP values
    shap_values = explainer([cleaned_text])
    positive_class_explanation = shap_values[0, :, "LABEL_1"]
    words = positive_class_explanation.data
    impacts = positive_class_explanation.values
    
    explanation_data = [{"word": str(word), "impact": round(float(impact), 4)} for word, impact in zip(words, impacts) if word is not None]
    explanation_data.sort(key=lambda x: x['impact'], reverse=True)
    top_positive_words = [item['word'] for item in explanation_data[:5] if item['impact'] > 0]

    # Generate Narrative with Gemini
    prompt = (
        "You are a clinical AI assistant. A machine learning model analyzed a patient's discharge note. "
        "Explain the model's prediction to a clinician in 2-3 concise, professional sentences.\n\n"
        f"The model's prediction is '{predicted_class}' with a risk score of {risk_probability:.2f}.\n"
        f"The primary factors increasing this risk were mentions of: {', '.join(top_positive_words)}.\n\n"
        "Provide a summary of these findings:"
    )
    response = llm_model.generate_content(prompt)
    narrative_summary = response.text.strip()

    # Return the enriched response
    return {
        "prediction": predicted_class,
        "risk_score": risk_probability,
        "narrative_summary": narrative_summary,
        "quantitative_explanation": explanation_data
    }
