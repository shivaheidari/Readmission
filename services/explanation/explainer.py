import re
import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import google.generativeai as genai


class ExplanationService:
    def __init__(self, fine_tuned_model_path, model_checkpoint, llm_api_key=None):
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
        
        self.pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer,
                                 device=self.device, return_all_scores=True)
        self.explainer = shap.Explainer(self.pipeline)
        
        # Configure Gemini
        if llm_api_key:
            genai.configure(api_key=llm_api_key)
            self.llm_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.llm_model = None

    @staticmethod
    def clean_mimic_text(text):
        text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict_and_explain(self, raw_text):
        if not self.llm_model:
            raise RuntimeError("LLM client not initialized! Check API key.")
        
        cleaned_text = self.clean_mimic_text(raw_text)

        # Prediction
        inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).squeeze()
        risk_probability = probabilities[1].item()
        predicted_class = "High Risk" if risk_probability > 0.5 else "Low Risk"

        # SHAP explanation
        shap_values = self.explainer([cleaned_text])
        positive_class_explanation = shap_values[0, :, "LABEL_1"]
        words = positive_class_explanation.data
        impacts = positive_class_explanation.values
        explanation_data = [{"word": str(word), "impact": round(float(impact), 4)} for word, impact in zip(words, impacts) if word]
        explanation_data.sort(key=lambda x: x['impact'], reverse=True)
        top_positive_words = [item['word'] for item in explanation_data[:5] if item['impact'] > 0]

        # Narrative generation using Gemini
        prompt = (
            "You are a clinical AI assistant. A machine learning model analyzed a patient's discharge note. "
            "Explain the model's prediction to a clinician in 2-3 concise, professional sentences.\n\n"
            f"The model's prediction is '{predicted_class}' with a risk score of {risk_probability:.2f}.\n"
            f"The primary factors increasing this risk were mentions of: {', '.join(top_positive_words)}.\n\n"
            "Provide a summary of these findings:"
        )
        response = self.llm_model.generate_content(prompt)
        narrative_summary = response.text.strip()

        return {
            "prediction": predicted_class,
            "risk_score": risk_probability,
            "narrative_summary": narrative_summary,
            "quantitative_explanation": explanation_data
        }
