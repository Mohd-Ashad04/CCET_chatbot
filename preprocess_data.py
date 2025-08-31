import json
import torch
from transformers import DistilBertTokenizer

# Load dataset
with open("ashad_dataset.json", encoding="utf-8") as file:
    data = json.load(file)


# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

X_texts = []
y_labels = []
label_dict = {}

for i, intent in enumerate(data["intents"]):
    label_dict[intent["intent"]] = i
    for pattern in intent["text"]:
        X_texts.append(pattern)
        y_labels.append(i)

# Tokenize input texts
tokens = tokenizer(X_texts, padding=True, truncation=True, return_tensors="pt")

# Save tokenized data
torch.save({
    "input_ids": tokens["input_ids"],
    "attention_mask": tokens["attention_mask"],
    "labels": torch.tensor(y_labels)
}, "chatbot_data.pt")

print("✅ Data Preprocessed & Saved Successfully!")
