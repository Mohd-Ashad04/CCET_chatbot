# 🤖 CCET Chatbot  

A **Conversational AI Chatbot** built with **DistilBERT**, designed to help students, faculty, and visitors interact with the **Chandigarh College of Engineering & Technology (CCET)** ecosystem. This chatbot can answer queries, guide users, and improve accessibility through **NLP-powered conversation**.  

---

## 🚀 Features  

- 🔍 **Natural Language Understanding** (powered by DistilBERT)  
- 🎓 **College-specific Knowledge Base** (custom dataset trained on CCET FAQs & info)  
- 💬 **Interactive Web UI** (Flask + HTML + CSS + JS frontend)  
- ⚡ **Real-time Responses** without page reload  
- 🛠️ **Easily Extensible** — add new Q/A pairs to dataset and retrain  
- 📊 **Lightweight model** (fast inference, optimized for deployment)  

---

## 📸 Project Preview  

Here’s the chatbot homepage:  

![CCET Chatbot Screenshot](https://github.com/Mohd-Ashad04/CCET_chatbot/blob/486d97dcede1b604f0518c4eb8ee06f3af546fcd/ccet%20chat%20bot%20.jpg)  



---

## 🏗️ Project Structure  

```bash
CCET_chatbot/
│── app.py                 # Flask web app entry point
│── preprocess_data.py      # Preprocessing script for dataset
│── train_model.py          # Script to train chatbot model
│── ashad_dataset.json      # Custom dataset for chatbot
│── chatbot_data.pt         # Saved model state (weights)
│── requirements.txt        # Dependencies
│── static/                 # CSS, JS, Images
│── templates/              # HTML frontend (Jinja2 templates)
│── distilbert_chatbot/     # Model configs + tokenizer
│── README.md               # Project documentation
│── .gitignore              # Ignored files (venv, large models, etc.)
