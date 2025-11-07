import re
import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import google.generativeai as genai


class ExplanationService:
    def __init__(self, fine_tuned_model_path, model_checkpoint, llm_api_key=None):
        # Use torch.device for PyTorch operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
        self.model.to(self.device)

        # Pipeline expects device as int: 0 for GPU, -1 for CPU
        pipeline_device = 0 if torch.cuda.is_available() else -1

        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipeline_device,
            return_all_scores=True
        )
        self.explainer = shap.Explainer(self.pipeline)

        # Configure Gemini LLM
        if llm_api_key:
            genai.configure(api_key=llm_api_key)
            try:
                # Use updated, valid model name - change this if needed!
                self.llm_model = genai.GenerativeModel('gemini-flash-latest')
            except Exception as e:
                print("LLM model initialization failed:", e)
                # Optionally list available models
                try:
                    models = genai.Client().models.list()
                    print("Available models:", [model.name for model in models])
                except Exception:
                    pass
                self.llm_model = None
        else:
            self.llm_model = None

    @staticmethod
    def clean_mimic_text(text):
        text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict_and_explain(self, raw_text):
        if not self.llm_model:
            # Optionally handle no LLM client case gracefully
            print("Warning: LLM model not initialized; skipping narrative generation.")

        cleaned_text = self.clean_mimic_text(raw_text)

        # Prepare inputs on correct device
        inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=1).squeeze()
        risk_probability = probabilities[1].item()
        predicted_class = "High Risk" if risk_probability > 0.5 else "Low Risk"

        # SHAP explanation
        shap_values = self.explainer([cleaned_text])
        positive_class_expl = shap_values[0, :, "LABEL_1"]
        words = positive_class_expl.data
        impacts = positive_class_expl.values

        explanation_data = [
            {"word": str(word), "impact": round(float(impact), 4)}
            for word, impact in zip(words, impacts)
            if word is not None
        ]
        explanation_data.sort(key=lambda x: x['impact'], reverse=True)
        top_positive_words = [item['word'] for item in explanation_data[:5] if item['impact'] > 0]

        narrative_summary = ""
        if self.llm_model:
            prompt = (
                "You are a clinical AI assistant. A machine learning model analyzed a patient's discharge note. "
                f"Explain the model's prediction to a clinician in 2-3 concise, professional sentences.\n\n"
                f"The model's prediction is '{predicted_class}' with a risk score of {risk_probability:.2f}.\n"
                f"The primary factors increasing this risk were mentions of: {', '.join(top_positive_words)}.\n\n"
                "Provide a summary of these findings:"
            )
            try:
                response = self.llm_model.generate_content(prompt)
                narrative_summary = response.text.strip()
            except Exception as e:
                print("Error generating narrative summary:", e)

        return {
            "prediction": predicted_class,
            "risk_score": risk_probability,
            "narrative_summary": narrative_summary,
            "quantitative_explanation": explanation_data
        }
