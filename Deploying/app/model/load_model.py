import torch
from transformers import AutoModelForAudioClassification,BertTokenizer, AutoTokenizer
import os
import pickle


MODEL_DIR = "Readmission/Deploying/app/00_34_bert_custom/data.pkl"

def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    pass


model = torch.load(MODEL_DIR)
model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "this is a test"
inputs = tokenizer(text, return_tensor="pt", padding=True, truncatin=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)

print("predictions", predictions)
