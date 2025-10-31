import pandas as pd
import torch
import sys
import re
import numpy as np
import transformers
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch import nn

# --- 1. Helper Functions and Classes 

def clean_mimic_text(text):
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
    text = re.sub(r'\_+', '', text)
    text = re.sub(r'\-+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class ReadmissionDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer):
        self.df = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'].iloc[idx]
        label = self.df['readmitted_within_30_days'].iloc[idx]
        cleaned_text = clean_mimic_text(text)
        encodings = self.tokenizer(cleaned_text, padding="max_length", truncation=True, return_tensors="pt")
        item = {key: val.squeeze() for key, val in encodings.items()}
        item['labels'] = torch.tensor(label)
        return item

def compute_metrics(eval_preds):
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

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        #  penalize errors on the positive class (1) more heavily.
        weights = torch.tensor([1.0, 10.0]).to(self.args.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# --- 2. Main Training Logic ---

# Define model checkpoint
model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# Load datasets
train_df = pd.read_parquet('../Data/train_dataset.parquet')
train_df.columns = train_df.columns.str.lower()
val_df = pd.read_parquet('../Data/validation_dataset.parquet')
val_df.columns = val_df.columns.str.lower()
train_dataset = ReadmissionDataset(train_df, tokenizer)
val_dataset = ReadmissionDataset(val_df, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    
    # Use the new parameter names
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    
    load_best_model_at_end=True,
)

# Create the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics,
# )

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,

)
# --- 3. Start Training ---
print("Starting model training...")
trainer.train()
print("Training complete.")

# Save the final best model
trainer.save_model("./results/best_model")
print("Final model saved.")