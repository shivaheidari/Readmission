import torch
from transformers import AutoModelForAudioClassification, AutoTokenizer
import os

model_DIR = "../../../../models/00_34_bert_custom"
if os.path.exists(model_DIR):
    print("yes")

# def load_model():
#     tokenizer = Atu
#     return model, tokenizer