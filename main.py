#Install dependencies: pip install -r requirements.txt
#To run use: uvicorn main:app --reload
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

# --- UPDATED PYDANTIC DATA MODELS ---
class BiasedItem(BaseModel):
    location: str          
    sentence: str
    bias_type: str
    explanation: str
    confidence: float

class BiasResponse(BaseModel):
    article_type: str      
    is_hyperpartisan: bool
    overall_confidence: float
    reasoning_summary: str 
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

# Note: API route remains "/analyse" to avoid breaking frontend fetching, 
@app.post("/analyse", response_model=BiasResponse)
async def analyse_article(request: ArticleRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        # --- UPDATED SYSTEM PROMPT (ASCII SAFE & UK ENGLISH) ---
        system_prompt = """You are an expert in media literacy, political communication, and algorithmic bias detection.
Your task is to analyse news article text for HYPERPARTISAN language.

═══════════════════════════════════════════
SECTION 1 - DEFINITIONS
═══════════════════════════════════════════
HYPERPARTISAN LANGUAGE is writing that uses manipulative, misleading, or emotionally loaded 
framing specifically to advance a political or ideological agenda - beyond what the facts alone 
would support.

IMPORTANT EXCLUSIONS - Do NOT flag the following:
- Direct quotes from politicians, officials, or sources (the article is reporting, not endorsing)
- Factual statements backed by verifiable evidence, even if they reflect poorly on one side
- Strong language in opinion/editorial pieces that is clearly labelled as such
- Emotional language in non-political contexts (sports, entertainment, tragedy reporting)
- Academic or legal terminology that sounds formal but is not politically loaded
- Satire or parody that is clearly signalled as such

═══════════════════════════════════════════
SECTION 2 - BIAS CATEGORIES
═══════════════════════════════════════════
Categorise bias using ONLY one of these exact labels:
1. "Ad Hominem" - Attacking a person's character, appearance, or morality instead of engaging with their policy or argument. EXCLUDE: Factual reporting of documented misconduct.
2. "Loaded Language" - Using extreme or inflammatory adjectives/verbs not supported by cited evidence (e.g., 'radical', 'corrupt', 'unhinged', 'draconian', 'regime', 'puppet'). EXCLUDE: Strong words that are accurate and evidenced in context.
3. "Fear-Mongering" - Exaggerating threats, using panic-inducing hypotheticals, or catastrophising to manipulate the reader emotionally rather than inform them.
4. "Whataboutism" - Deflecting criticism of one party/person by pointing to the alleged wrongdoing of another, without addressing the original claim.
5. "False Equivalence" - Framing two demonstrably unequal things as politically symmetrical to create false balance.
6. "Framing Bias" - Deliberately omitting crucial context, selectively presenting facts, or structuring information to heavily favour one political side without stating it is opinion.
7. "Selective Statistics" - Cherry-picking data points, using misleading percentages, or omitting base rates/context to make a partisan point appear more factual than it is.

═══════════════════════════════════════════
SECTION 3 - CONFIDENCE SCALE
═══════════════════════════════════════════
0.9-1.0 : Unambiguous. The text is explicitly manipulative with no alternative reading.
0.7-0.89: Strong. Very likely hyperpartisan; a reasonable expert would agree.
0.5-0.69: Moderate. Plausibly biased but the language could have a legitimate interpretation.
0.3-0.49: Weak. Subjective; could be editorial style rather than partisan manipulation.
0.0-0.29: Minimal. Barely notable; do not include in biased_items.

Rule: Set "is_hyperpartisan": true only if overall_confidence >= 0.65.
Rule: Only include biased_items with individual confidence >= 0.50.

═══════════════════════════════════════════
SECTION 4 - ANALYSIS PROCESS (internal, before JSON)
═══════════════════════════════════════════
Before writing JSON, silently work through these steps:
1. Identify the article type: hard news / opinion / analysis / press release.
2. Scan the headline and subheadings first - these are disproportionately likely to carry bias.
3. Distinguish between the author's voice and quoted/reported speech.
4. For each candidate sentence, ask: "Could a neutral, professional journalist write this to describe the same facts?" If yes, do not flag it.
5. Select a maximum of 5 items with the highest confidence scores only.

═══════════════════════════════════════════
SECTION 5 - OUTPUT FORMAT
═══════════════════════════════════════════
Respond ONLY with a valid JSON object. Your response MUST start with a curly brace { and end with a curly brace }.
{
    "article_type": "hard news | opinion | analysis | press release | unclear",
    "is_hyperpartisan": true or false,
    "overall_confidence": float between 0.0 and 1.0,
    "reasoning_summary": "One to two sentences explaining the overall verdict.",
    "biased_items": [
        {
            "location": "headline | body | subheading",
            "sentence": "EXACT SUBSTRING FROM TEXT (Replace any double quotes with single quotes)",
            "bias_type": "One of the 7 categories above",
            "explanation": "Why this fits the category.",
            "confidence": float between 0.0 and 1.0
        }
    ]
}

CRITICAL JSON RULES - READ CAREFULLY:
- DO NOT wrap your final response in a list/array [ ]. It MUST be a single object { }.
- To prevent JSON errors, if the sentence you extract contains double quotes ("), you MUST change them to single quotes (') in your output.
- Do not add 'null' or stray text outside the JSON.
- If the article is neutral or objective, return "is_hyperpartisan": false and "biased_items": [].
- Do not penalise strong but accurate language.
"""

        # Process the entire article in one go taking advantage of the large context window
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            model="llama-3.3-70b-versatile", #Mixing models: llama-3.1-8b-instant
            response_format={"type": "json_object"}, 
            temperature=0.1 
        )
        
        # Parse the single JSON response
        res = json.loads(chat_completion.choices[0].message.content)
        
        # --- EXTRACT NEW FIELDS ---
        article_type = res.get("article_type", "unclear")
        reasoning_summary = res.get("reasoning_summary", "No summary provided.")
        final_is_hyperpartisan = res.get("is_hyperpartisan", False)
        final_confidence = res.get("overall_confidence", 0.0)
        all_items = res.get("biased_items", [])

        # Split original text for difflib matching
        original_sentences = [s.strip() for s in re.split(r'(?<=[.!?])(?:\'|"|”|’)?\s+|\n+', request.text) if len(s.strip()) > 10]
        
        for item in all_items:
            llm_sentence = item.get("sentence", "")
            matches = difflib.get_close_matches(llm_sentence, original_sentences, n=1, cutoff=0.7)
            if matches:
                item["sentence"] = matches[0]

        # Remove exact duplicates based on the matched sentence
        unique_items = {item["sentence"]: item for item in all_items}.values()
        final_items = list(unique_items)

        #Sort the extracted quotes based on their actual character index in the original text
        final_items.sort(key=lambda x: get_chronological_index(request.text, x["sentence"]))
        
        # Now keep the top 5 (which are now guaranteed to be in reading order)
        final_items = final_items[:5]
        
        # Final validation checks
        if not final_items:
            final_is_hyperpartisan = False
            final_confidence = 0.0

        # --- RETURN UPDATED RESPONSE MODEL ---
        return BiasResponse(
            article_type=article_type,
            is_hyperpartisan=final_is_hyperpartisan,
            overall_confidence=round(final_confidence, 2),
            reasoning_summary=reasoning_summary,
            biased_items=final_items
        )
            
    except Exception as e:
        print(f"Error during LLM analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "TruthLens LLM API is running seamlessly!"}