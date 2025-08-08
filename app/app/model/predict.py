import torch
from model.load_model import load_model
from transformers import AutoTokenizer

def predict(model, tokenizer, text):

    #tokenize input

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        #print("------logits------",logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
    return {"probabilities": probs.tolist(), "predicted_class": predicted_class}



if __name__ == "__main__":

    "--------------------data----------------"
    "******************running main in predict file*************"
    model_path = "Readmission/Deploying/app/00_66_bert_custom_dict"
    test_file = "Readmission/Deploying/app/model/test.txt"

    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    with open(test_file, "r") as file:
        content = file.read()
    
    results = predict(model, tokenizer, content)
    print(f"Predicted Class: {results['predicted_class']}")


