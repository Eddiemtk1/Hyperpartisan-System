import os
import json
import asyncio  
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="TruthLens LLM Backend")

EXTENSION_ID = "dkfgjjeofponpepkplgjembijdcabgce"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"chrome-extension://{EXTENSION_ID}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq securely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment variables.")
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

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

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

# --- Helper: Async API Caller ---
async def analyze_chunk(chunk: str, system_prompt: str):
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}, 
            temperature=0.1 
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Chunk failed: {str(e)}")
        # If one chunk fails (e.g., rate limit), return empty neutral data so other chunks can still succeed
        return {"is_hyperpartisan": False, "overall_confidence": 0.0, "biased_items": []}

# --- The Core Endpoint ---
@app.post("/analyze", response_model=BiasResponse)
async def analyze_article(request: ArticleRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        # 1. Break text into sliding-window chunks
        chunks = chunk_text(request.text, chunk_size=800, overlap=150)
        
        # Optional safety: Limit to 3 chunks (~2,100 words) so free-tier Groq APIs don't rate-limit you
        chunks = chunks[:3]

        # Prompt Engineering:
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

        # 2. Fire all API requests concurrently using asyncio! 
        tasks = [analyze_chunk(chunk, system_prompt) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        # 3. Aggregate the results from all chunks
        all_items = []
        total_confidence = 0.0
        partisan_flags = 0

        for res in results:
            if res.get("is_hyperpartisan"):
                partisan_flags += 1
            total_confidence += res.get("overall_confidence", 0.0)
            all_items.extend(res.get("biased_items", []))

        # Remove duplicates (in case the sliding window caught the same sentence twice in the overlap)
        unique_items = {item["sentence"]: item for item in all_items}.values()
        final_items = list(unique_items)[:5] # Keep max 5 for a clean UI

        # Determine final verdict based on aggregated data
        final_is_hyperpartisan = partisan_flags > 0 and len(final_items) > 0
        final_confidence = (total_confidence / len(results)) if len(results) > 0 else 0.0
        
        # Failsafe logic
        if not final_items:
            final_is_hyperpartisan = False
            final_confidence = 0.0

        return BiasResponse(
            is_hyperpartisan=final_is_hyperpartisan,
            overall_confidence=round(final_confidence, 2),
            biased_items=final_items
        )
            
    except Exception as e:
        print(f"Error during LLM analysis: {str(e)}")
        # FIXED: This except is now properly matched to a "try" block above!
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "TruthLens LLM API is running seamlessly!"}