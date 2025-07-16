from rapidfuzz import process, fuzz
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Load QA pairs from file
def load_qa_pairs(file_path="college_info.txt"):
    qa_dict = {}
    if not os.path.exists(file_path):
        return qa_dict
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                question, answer = line.strip().split("|", 1)
                qa_dict[question.lower()] = answer.strip()
    return qa_dict

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class Question(BaseModel):
    data: list[str]

# Load QA data
qa_data = load_qa_pairs()

# Simple replies
simple_responses = {
    "hi": "Hello! How can I help you with your college queries today?",
    "hello": "Hi there! What do you want to know about the college?",
    "hey": "Hey! Ask me anything about the college.",
    "thanks": "You're welcome!",
    "thank you": "Glad to help!",
    "bye": "Goodbye! Have a great day.",
    "goodbye": "See you later!",
    "how are you": "I'm fine, glad to see you!",
    "how r u": "I'm good, what about you?",
}

# Ask bot logic
def ask_bot(user_input: str) -> str:
    q = user_input.strip().lower()

    # Simple replies
    if q in simple_responses:
        return simple_responses[q]

    # Fuzzy match to best question
    best_match = process.extractOne(q, qa_data.keys(), score_cutoff=70)
    if best_match:
        return qa_data[best_match[0]]

    return "I'm not sure about that. Please contact the college office for more details."

# Routes
@app.get("/")
async def home():
    return {"message": "College chatbot API is running."}

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

@app.post("/api/predict")
async def predict(question: Question):
    user_question = question.data[0]
    reply = ask_bot(user_question)
    return {"data": [reply]}
