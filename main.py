import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

# Load environment variables (Create a .env file in your folder with GROQ_API_KEY=your_key)
load_dotenv()

app = FastAPI(title="TruthLens LLM Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq Client securely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment variables.")
groq_client = Groq(api_key=GROQ_API_KEY)

# --- Pydantic Data Models (Matches your frontend exactly) ---
class BiasedItem(BaseModel):
    sentence: str
    explanation: str
    confidence: float

class BiasResponse(BaseModel):
    is_hyperpartisan: bool
    overall_confidence: float
    biased_items: List[BiasedItem]

class ArticleRequest(BaseModel):
    text: str

# --- The Core Endpoint ---
@app.post("/analyze", response_model=BiasResponse)
def analyze_article(request: ArticleRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    # Limit the text to roughly the first 1500 words to stay safely within context windows 
    # and keep API costs/latency very low. Bias is usually established early in an article.
    words = request.text.split()[:1500]
    truncated_text = " ".join(words)

    # Prompt Engineering: The magic happens here
    system_prompt = """You are an expert in media literacy and algorithmic bias detection. 
    Analyze the provided news article text for hyperpartisan, manipulative, or emotionally loaded language.
    Respond ONLY with a valid JSON object using this exact schema:
    {
        "is_hyperpartisan": true/false,
        "overall_confidence": float between 0.0 and 1.0,
        "biased_items": [
            {
                "sentence": "THE EXACT SENTENCE FROM THE TEXT",
                "explanation": "One short sentence explaining why it is manipulative or biased.",
                "confidence": float between 0.0 and 1.0
            }
        ]
    }
    
    STRICT RULES:
    1. Extract a maximum of 5 biased items.
    2. The "sentence" MUST be an exact, word-for-word substring from the provided text. Do not alter a single character, punctuation mark, or capitalization, otherwise the frontend highlighting will fail.
    3. If the text is completely neutral and objective, return "is_hyperpartisan": false and an empty list [] for "biased_items".
    """

    try:
        # Requesting structured JSON directly from Llama-3
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": truncated_text}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}, 
            temperature=0.1 # Kept very low so the model extracts exact quotes rather than "creative" paraphrasing
        )
        
        # Parse the JSON returned by the LLM
        result_str = chat_completion.choices[0].message.content
        result_json = json.loads(result_str)
        
        return BiasResponse(
            is_hyperpartisan=result_json.get("is_hyperpartisan", False),
            overall_confidence=result_json.get("overall_confidence", 0.0),
            biased_items=result_json.get("biased_items", [])
        )
        
    except Exception as e:
        print(f"Error during LLM analysis: {str(e)}")
        #If a fail happens, the tront end displays the error
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "TruthLens LLM API is running seamlessly!"}