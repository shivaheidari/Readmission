from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap

model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
fine_tuned_model_path = "./model_artifacts/best_model"

# Try using the same checkpoint for both if model fine-tuned from this
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)

my_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)
explainer = shap.Explainer(my_pipeline)

# Test a prediction/explanation
text = "Patient discharged after heart failure."
print(my_pipeline(text))
print(explainer([text]))
