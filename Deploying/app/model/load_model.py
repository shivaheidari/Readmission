import torch
from transformers import AutoModelForAudioClassification,BertTokenizer, AutoTokenizer, AutoModel
import os
import pickle
from torch import nn




class CustomModel(nn.Module):
    def __init__(self, input_size, base_model):
        super(CustomModel, self).__init__()
        self.model = base_model
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token_output = last_hidden_state[:, 0, :]
        return self.classifier(cls_token_output)


def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=True)
    print("checkpoints loaded")
    base_model = AutoModel.from_pretrained(checkpoint["base_model_name"])
    input_size = checkpoint["input_size"]
    model = CustomModel(input_size, base_model)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    print("Model loaded and ready for inference.")
    return model
