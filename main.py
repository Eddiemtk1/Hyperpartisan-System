#Install dependencies pip install -r requirements.txt
#To run use uvicorn main:app --reload
import os
import json
import difflib
import re  
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from dotenv import load_dotenv

#Load environment variables
load_dotenv()

app = FastAPI(title="TruthLens LLM Backend")

EXTENSION_ID = os.getenv("EXTENSION_ID")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"chrome-extension://{EXTENSION_ID}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment variables.")
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

#Pydantic Data Models
class BiasedItem(BaseModel):
    sentence: str
    explanation: str
    confidence: float
    bias_type: str

class BiasResponse(BaseModel):
    is_hyperpartisan: bool
    overall_confidence: float
    biased_items: List[BiasedItem]

class ArticleRequest(BaseModel):
    text: str

# --- HELPER: SMART CHRONOLOGICAL INDEXER ---
def get_chronological_index(full_text: str, quote: str) -> int:
    # 1. Try an exact string match first
    idx = full_text.find(quote)
    if idx != -1:
        return idx
        
    # 2. If the AI hallucinated punctuation (like missing inner quotes), 
    # strip all punctuation from both strings and try to find the relative position!
    clean_text = re.sub(r'[^\w\s]', '', full_text.lower())
    clean_quote = re.sub(r'[^\w\s]', '', quote.lower())
    
    idx = clean_text.find(clean_quote)
    if idx != -1:
        return idx
        
    # 3. Absolute failsafe (push to bottom)
    return 999999

@app.post("/analyze", response_model=BiasResponse)
async def analyze_article(request: ArticleRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        # Prompt for bias analysis - Updated for full document processing
        system_prompt = """You are an expert in media literacy and algorithmic bias detection. 
        Analyse the provided full news article text for hyperpartisan, manipulative, or emotionally loaded language.
        Respond ONLY with a valid JSON object using this exact schema:
        {
            "is_hyperpartisan": true/false,
            "overall_confidence": float between 0.0 and 1.0,
            "biased_items": [
                {
                    "sentence": "THE EXACT SENTENCE FROM THE TEXT",
                    "bias_type": "Category of bias (e.g., Ad Hominem, Emotional Language, Fear-Mongering, Loaded Words, False Equivalence)",
                    "explanation": "One short sentence explaining why it is manipulative or biased.",
                    "confidence": float between 0.0 and 1.0
                }
            ]
        }
        
        STRICT RULES:
        1. Extract a maximum of 5 biased items from across the entire article.
        2. The "sentence" MUST be an exact, word-for-word substring from the provided text.
        3. If neutral, return "is_hyperpartisan": false and an empty list [].
        """

        # Process the entire article in one go leveraging the large context window
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}, 
            temperature=0.1 
        )
        
        # Parse the single JSON response
        res = json.loads(chat_completion.choices[0].message.content)
        
        all_items = res.get("biased_items", [])
        final_is_hyperpartisan = res.get("is_hyperpartisan", False)
        final_confidence = res.get("overall_confidence", 0.0)

        # Split original text for difflib matching
        original_sentences = [s.strip() for s in re.split(r'(?<=[.!?])(?:\'|"|”|’)?\s+|\n+', request.text) if len(s.strip()) > 10]
        
        for item in all_items:
            llm_sentence = item.get("sentence", "")
            # difflib finds the closest matching real sentence in the text so even if the model hallucinates, there won't be any issues
            matches = difflib.get_close_matches(llm_sentence, original_sentences, n=1, cutoff=0.7)
            if matches:
                # Replace the LLM's hallucinated sentence with the EXACT original sentence!
                item["sentence"] = matches[0]

        # Remove exact duplicates based on the matched sentence
        unique_items = {item["sentence"]: item for item in all_items}.values()
        final_items = list(unique_items)

        # Sort the extracted quotes based on their actual character index in the original text!
        # If a quote somehow isn't found (-1), we push it to the very bottom (index 999999).
        final_items.sort(key=lambda x: get_chronological_index(request.text, x["sentence"]))
        
        # Now keep the top 5 (which are now guaranteed to be in reading order)
        final_items = final_items[:5]
        
        # Final validation checks
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
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "TruthLens LLM API is running seamlessly!"}