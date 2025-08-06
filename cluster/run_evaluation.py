import pandas as pd
import torch
import re
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification


def clean_mimic_text(text):
    """Performs targeted cleaning of MIMIC-IV clinical notes."""
    text = str(text) # Ensure text is a string
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
    text = re.sub(r'\_+', '', text)
    text = re.sub(r'\-+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class ReadmissionDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for readmission prediction."""
    def __init__(self, dataframe, tokenizer):
        self.df = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'].iloc[idx]
        label = self.df['readmitted_within_30_days'].iloc[idx]
        cleaned_text = clean_mimic_text(text)
        encodings = self.tokenizer(
            cleaned_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {key: val.squeeze() for key, val in encodings.items()}
        item['labels'] = torch.tensor(label)
        return item

def compute_metrics(eval_preds):
    """Computes and returns a dictionary of evaluation metrics."""
    predictions, labels = eval_preds
    probs = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)
    preds = np.argmax(predictions, axis=1)
    
    return {
        'roc_auc': roc_auc_score(labels, probs[:, 1]),
        'f1': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'accuracy': accuracy_score(labels, preds)
    }


if __name__ == "__main__":
    print("--- Starting Final Evaluation on the Test Set ---")

    model_path = "./results_run2/best_model" 
    
    print(f"Loading model from: {model_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)

   
    test_df = pd.read_parquet('test_dataset.parquet')
    test_df.columns = test_df.columns.str.lower() # Standardize column names
    test_dataset = ReadmissionDataset(test_df, tokenizer)

 
    eval_trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

  
    results = eval_trainer.predict(test_dataset)

    print("\n--- Test Set Performance ---")
   
    for key, value in results.metrics.items():
        print(f"{key}: {value:.4f}")