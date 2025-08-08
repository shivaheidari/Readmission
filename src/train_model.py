# src/train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm # For a nice progress bar

# --- Configuration ---

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "MIMIC"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT" 
MODEL_OUTPUT_PATH = "models/clinical_bert_readmission.pt" 

# Hyperparameters
MAX_LENGTH = 512
BATCH_SIZE = 8 
EPOCHS = 3 
LEARNING_RATE = 2e-5

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"), 
        logging.StreamHandler() 
    ]
)

# --- Data Loading Class and Functions ---

class AdmissionDataset(Dataset):
    """PyTorch Dataset for the readmission task."""
    def __init__(self, data, tokenizer, max_length):
       self.data = data
       self.tokenizer = tokenizer
       self.max_length = max_length
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        text = record["concatenated_notes"]
        label = torch.tensor(record["readmission"], dtype=torch.long) 
        
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            "input_ids": tokens['input_ids'].squeeze(0),
            "attention_mask": tokens['attention_mask'].squeeze(0),
            "labels": label
        }

def get_balanced_data_from_mongo(mongo_uri, db_name):
    """Connects to MongoDB and returns a balanced list of records."""
    logging.info("Connecting to MongoDB...")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    
    logging.info("Fetching readmitted and not-readmitted records...")
    records_admitted = list(db["readmitted_concated"].find({}))
    records_noreadmited = list(db["no_readmitted_concated"].find({}))
    
    # Balancing the data by undersampling the majority class
    size_admitted = len(records_admitted)
    sample_noreadmited = random.sample(records_noreadmited, size_admitted)
    
    combined_list = records_admitted + sample_noreadmited
    random.shuffle(combined_list)
    
    logging.info(f"Created a balanced dataset with {len(combined_list)} records.")
    return combined_list

# --- Main Training and Evaluation Script ---

def main():
    """Main function to run the training pipeline."""
    
    # --- 1. Setup Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- 2. Load Data ---
    data = get_balanced_data_from_mongo(MONGO_URI, DB_NAME)
    
    # --- 3. Initialize Tokenizer and Model ---
    logging.info(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    logging.info(f"Loading model: {MODEL_NAME}")
    # We use AutoModelForSequenceClassification which already includes a classification head.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # --- 4. Create Datasets and DataLoaders ---
    dataset = AdmissionDataset(data, tokenizer, MAX_LENGTH)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    # --- 5. Initialize Optimizer and Loss Function ---
    # FIX: Initialize optimizer on the correct model's parameters.
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # --- 6. Training Loop ---
    for epoch in range(EPOCHS):
        logging.info(f"--- Starting Epoch {epoch+1}/{EPOCHS} ---")
        model.train()
        total_loss = 0
        
        # Use tqdm for a progress bar over batches
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # The model returns a loss when labels are provided
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        logging.info(f"Average Training Loss: {avg_train_loss:.4f}")

    # --- 7. Evaluation ---
    logging.info("--- Starting Evaluation ---")
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            _, predictions = torch.max(logits, dim=1)
            
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
    accuracy = correct_predictions / total_predictions
    logging.info(f"** Final Test Accuracy: {accuracy:.4f} **")

    # --- 8. Save the Fine-Tuned Model ---
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    logging.info(f"Model saved to {MODEL_OUTPUT_PATH}")


if __name__ == '__main__':
    main()