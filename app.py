from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer
from pymongo import MongoClient
import os

app = Flask(__name__)

# MongoDB Configuration
MONGO_URI = "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/stance_feedback"
client = MongoClient(MONGO_URI)
db = client["stance_db"]
feedback_collection = db["feedback"]

# Load trained RLHF stance detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("ppo_trained_model.pth", map_location=device)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Stance prediction function
def predict_stance(tweet, target):
    input_text = f"Does the tweet '{tweet}' express Favor, Against, or Neutral stance toward '{target}'?"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    logits = model(**inputs)
    predicted_label = torch.argmax(logits.logits if hasattr(logits, "logits") else logits, dim=-1).item()
    stance_map = {0: "NEUTRAL", 1: "AGAINST", 2: "FAVOR"}
    return stance_map[predicted_label]

@app.route('/predict_stance', methods=['POST'])
def get_prediction():
    data = request.json
    tweet = data.get("tweet")
    target = data.get("target")

    if not tweet or not target:
        return jsonify({"error": "Tweet and Target required"}), 400

    stance = predict_stance(tweet, target)
    return jsonify({"predicted_stance": stance}), 200

@app.route('/store_feedback', methods=['POST'])
def store_feedback():
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400
    
    # Insert feedback into MongoDB
    feedback_collection.insert_one(data)
    
    return jsonify({"message": "Feedback stored successfully"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
