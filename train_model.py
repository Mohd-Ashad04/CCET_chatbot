import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Load dataset
with open("janvi_dataset.json", encoding="utf-8") as file:
    data = json.load(file)

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Prepare Data
X_texts = []
y_labels = []
label_dict = {}

for i, intent in enumerate(data["intents"]):
    label_dict[intent["intent"]] = i
    for pattern in intent["text"]:
        X_texts.append(pattern)
        y_labels.append(i)

tokens = tokenizer(X_texts, padding=True, truncation=True, return_tensors="pt")

# Convert to PyTorch tensors
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]
labels = torch.tensor(y_labels)

# Dataset Class
class ChatbotDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# Create Dataset
dataset = ChatbotDataset(input_ids, attention_mask, labels)

# Split Dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load Model
num_labels = len(set(y_labels))
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Train Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_function = CrossEntropyLoss()

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_function(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Save Model
model.save_pretrained("distilbert_chatbot")
tokenizer.save_pretrained("distilbert_chatbot")
print("✅ Model Trained & Saved Successfully!")
