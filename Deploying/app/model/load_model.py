import torch
from transformers import AutoModelForAudioClassification, AutoTokenizer
import os

MODEL_DIR = "../00_34_bert_custom"
if os.path.exists(MODEL_DIR):
    print("yes")

# def load_model():
#     tokenizer = Atu
#     return model, tokenizer