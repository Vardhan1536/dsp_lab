from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from pymongo import MongoClient
import os
from dotenv import load_dotenv
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

# Define the SimCSE model (directly using your pretrained model)
class SimCSEStanceModel(nn.Module):
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, 128)  # 128-dim contrastive space

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.projection(outputs.pooler_output)
        return nn.functional.normalize(embeddings, dim=-1)  # Normalize embeddings

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "stance_db")  # Default: stance_db
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "feedback")  # Default: feedback

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
feedback_collection = db[COLLECTION_NAME]

# Tokenizer & Model Loading
MODEL_NAME = "roberta-base"  # Adjust to your SimCSE model name
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
simcse_model = SimCSEStanceModel(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your pretrained SimCSE model
simcse_model.load_state_dict(torch.load("simcse_stance_model.pth", map_location=device))
simcse_model.to(device)
simcse_model.eval()

# Request Body Models
class StanceRequest(BaseModel):
    tweet: str
    target: str

class FeedbackRequest(BaseModel):
    tweet: str
    target: str
    predicted_stance: str
    human_stance: str

def predict_stance(tweet: str, target: str) -> str:
    input_text = f"Tweet: {tweet} Target: {target}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        embeddings = simcse_model(inputs.input_ids, inputs.attention_mask)
    
    # Assuming embeddings are normalized and we classify based on cosine similarity or clustering
    stance_map = {0: "FAVOR", 1: "AGAINST", 2: "NEUTRAL"}

    predicted_label = torch.argmax(embeddings, dim=-1).item()  # Assuming we want to classify as 0, 1, 2

    # Ensure the predicted label is within the expected range
    if predicted_label not in stance_map:
        # If the predicted label is outside the valid range, return a default stance (e.g., "NEUTRAL")
        return "NEUTRAL"

    return stance_map[predicted_label]


@app.post("/predict_stance")
def get_prediction(request: StanceRequest):
    stance = predict_stance(request.tweet, request.target)
    return {"predicted_stance": stance}

@app.post("/store_feedback")
def store_feedback(request: FeedbackRequest):
    feedback_collection.insert_one(request.dict())
    return {"message": "Feedback stored successfully"}
