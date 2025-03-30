from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import os
import random
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

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "stance_db")  # Default: stance_db
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "feedback")  # Default: feedback

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
feedback_collection = db[COLLECTION_NAME]

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
    stance_map = ["FAVOR", "AGAINST", "NEUTRAL"]
    return random.choice(stance_map)  # Randomly select a stance

@app.post("/predict_stance")
def get_prediction(request: StanceRequest):
    stance = predict_stance(request.tweet, request.target)
    return {"predicted_stance": stance}

@app.post("/store_feedback")
def store_feedback(request: FeedbackRequest):
    feedback_collection.insert_one(request.dict())
    return {"message": "Feedback stored successfully"}
