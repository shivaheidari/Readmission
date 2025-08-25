import pandas as pd
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

def clean_mimic_text(text):
    text = str(text)
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
    text = re.sub(r'\_+', '', text)
    text = re.sub(r'\-+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Interpretability Logic ---

# 1. Define model paths
model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
fine_tuned_model_path = "./results_run2/best_model" # Or your latest best checkpoint

# 2. Load your fine-tuned model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path, use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)
print("Done.")

# 3. Create a prediction pipeline for SHAP
pred_pipeline = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer, 
    return_all_scores=True
)

# 4. Create a SHAP Explainer
explainer = shap.Explainer(pred_pipeline)

# 5. Load some sample notes from your test set to explain
print("Loading sample data...")
test_df = pd.read_parquet('test_dataset.parquet')
test_df.columns = test_df.columns.str.lower()
test_df['text'] = test_df['text'].apply(clean_mimic_text)
sample_texts = test_df['text'].head(5).tolist()
print("Done.")

# 6. Generate SHAP values
print("Generating SHAP explanations...")
shap_values = explainer(sample_texts)
print("Done.")

# 7. Save plots for each explanation as HTML files
for i in range(len(sample_texts)):
    # The output from the pipeline for binary classification returns a list of two dictionaries.
    # We want the explanation for the positive class ("LABEL_1" is the default name).
    shap.plots.text(shap_values[i, :, "LABEL_1"], display=False, show=False)
    
    filename = f"shap_explanation_{i}.html"
    shap.save_html(filename, shap.plots.text(shap_values[i, :, "LABEL_1"], show=False))
    print(f"Saved SHAP plot to {filename}")