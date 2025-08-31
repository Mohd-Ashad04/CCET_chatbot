from flask import Flask, render_template, request, jsonify
import json
import random
import requests
import sqlite3
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import nltk
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

load_dotenv()


# Load intents from JSON
with open(r'E:\folder2\ashad_dataset.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Load Pretrained DistilBERT Model and Tokenizer
MODEL_PATH = "distilbert_chatbot"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = os.getenv("API_URL")


# Function to initialize database
def init_db():
    conn = sqlite3.connect("chatbot_memory.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            bot_response TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def classify_intent(user_message, confidence_threshold=0.5):
    """Classifies user input into an intent using DistilBERT and checks confidence."""
    tokens = tokenizer(user_message, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens)
    
    probabilities = F.softmax(outputs.logits, dim=1)  # Convert logits to probabilities
    confidence, predicted_label = torch.max(probabilities, dim=1)  # Get highest probability

    # Only return intent if confidence is high enough
    if confidence.item() >= confidence_threshold:
        for intent in intents["intents"]:
            if predicted_label.item() == intents["intents"].index(intent):
                return intent["intent"], random.choice(intent["responses"]), confidence.item()

    return None, None, confidence.item()  # No intent matched

def get_ai_response(user_message):
    """Fetch response from Gemini AI when intent confidence is low."""
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY  # Use correct header key for Gemini API
    }

    full_prompt = f"Answer the following question conversationally:\n{user_message}"

    data = {
        "contents": [
            {
                "parts": [{"text": full_prompt}]
            }
        ]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()

        # If Gemini gives valid response, return it (you can fine-tune this logic)
        if "candidates" in result and result["candidates"]:
            ai_text = result["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if ai_text.strip():
                return "I'm sorry, this chatbot only provides information related to the college (CCET)."
        return "I'm sorry, this chatbot only provides information related to the college (CCET)."

    except requests.exceptions.RequestException as e:
        print(f"Gemini API error: {e}")
        return "I'm sorry, this chatbot only provides information related to the college (CCET)."



def store_conversation(user_message, bot_response):
    """Store chat history in the database."""
    conn = sqlite3.connect("chatbot_memory.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (user_message, bot_response) VALUES (?, ?)", (user_message, bot_response))
    conn.commit()
    conn.close()

def retrieve_past_conversation(limit=5):
    """Retrieve the last few messages from the conversation history."""
    conn = sqlite3.connect("chatbot_memory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_message, bot_response FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
    past_conversations = cursor.fetchall()
    conn.close()
    return past_conversations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': "Please enter a message."})

    # Retrieve last conversation if requested
    if "previous conversation" in user_message.lower():
        past_conversations = retrieve_past_conversation()
        formatted_history = "\n".join([f"User: {conv[0]}\nBot: {conv[1]}" for conv in past_conversations])
        return jsonify({'response': formatted_history or "No previous conversations found."})

    # Classify intent using DistilBERT
    intent, response, confidence = classify_intent(user_message)

    if response is None or confidence < 0.5:
        # If confidence is too low, fetch response from Gemini AI
        response = get_ai_response(user_message)

    # Store conversation in DB
    store_conversation(user_message, response)

    return jsonify({'response': response, 'confidence': round(confidence, 2)})

if __name__ == '__main__':
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    app.run(debug=True)
