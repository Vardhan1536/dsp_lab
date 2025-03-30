from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import torch.nn as nn

from fastapi.middleware.cors import CORSMiddleware

# FastAPI app initialization
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains (change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Define PPOPolicyModel (same as in Kaggle)
class PPOPolicyModel(nn.Module):
    def __init__(self, base_model):
        super(PPOPolicyModel, self).__init__()
        self.base_model = base_model
        self.policy_head = nn.Linear(768, 3)  # Favor, Against, Neutral

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.policy_head(pooled_output)
        return logits

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "stance_db")  # Default: stance_db
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "feedback")  # Default: feedback

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
feedback_collection = db[COLLECTION_NAME]

# Model Loading
MODEL_NAME = "princeton-nlp/sup-simcse-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
simcse_model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the PPO Policy Model
model = PPOPolicyModel(simcse_model).to(device)
model.load_state_dict(torch.load("ppo_trained_model.pth", map_location=device))
model.eval()

# Request Body Models
class StanceRequest(BaseModel):
    tweet: str
    target: str

class FeedbackRequest(BaseModel):
    tweet: str
    target: str
    predicted_stance: str
    human_stance: str

# Prediction function
def predict_stance(tweet: str, target: str) -> str:
    input_text = f"Tweet: {tweet} Target: {target}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits = model(inputs.input_ids, inputs.attention_mask)
    
    predicted_label = torch.argmax(logits, dim=-1).item()
    stance_map = {0: "FAVOR", 1: "AGAINST", 2: "NEUTRAL"}
    return stance_map[predicted_label]

@app.post("/predict_stance")
def get_prediction(request: StanceRequest):
    stance = predict_stance(request.tweet, request.target)
    return {"predicted_stance": stance}

@app.post("/store_feedback")
def store_feedback(request: FeedbackRequest):
    feedback_collection.insert_one(request.dict())
    return {"message": "Feedback stored successfully"}
