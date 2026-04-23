# Install dependencies: pip install -r requirements.txt
# To run use: uvicorn main:app --reload
import os
import json
import difflib
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from env
load_dotenv()

app = FastAPI(title="TruthLens LLM Backend")

#Only allows the extension to call this API
EXTENSION_ID = os.getenv("EXTENSION_ID")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"chrome-extension://{EXTENSION_ID}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OPENROUTER API
#API key in env as to not hardcode it
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not found in environment variables.")

llm_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

#Pydantic classes
class BiasedItem(BaseModel):
    location: str
    sentence: str
    is_quote_or_reported_speech: bool
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
    title: str = ""
    text: str


#Chronological index
def get_chronological_index(full_text: str, quote: str) -> int:
    idx = full_text.find(quote)
    if idx != -1:
        return idx

    #strips all punctuation from both strings and tries to find the relative position
    clean_text = re.sub(r"[^\w\s]", "", full_text.lower())
    clean_quote = re.sub(r"[^\w\s]", "", quote.lower())

    idx = clean_text.find(clean_quote)
    if idx != -1:
        return idx
    return 999999


#/analyse endpoint
@app.post("/analyse", response_model=BiasResponse)
async def analyse_article(request: ArticleRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

#Here is my prompt, it went through many iteration until it achieved a satisfactory level
    try:
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

Rule: Set "is_hyperpartisan": true only if overall_confidence >= 0.50.
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
SECTION 4.5 - EXAMPLES (CRITICAL FOR CALIBRATION)
═══════════════════════════════════════════
Below are examples of how you should classify specific sentences. Pay close attention to objective reporting versus manipulative framing.

EXAMPLE 1 (OBJECTIVE REPORTING - DO NOT FLAG):
Text: "Critics, including UN human rights chief Volker Türk, have described the new law as discriminatory. Türk also said its application would 'constitute a war crime'."
Reasoning: The journalist is neutrally reporting what a public figure said. The journalist is NOT making the claim themselves. 
Action: Do NOT extract this. If the whole article is like this, return "is_hyperpartisan": false.

EXAMPLE 2 (OBJECTIVE REPORTING - DO NOT FLAG):
Text: "Amnesty International urged Israeli authorities to repeal the new law."
Reasoning: "Repeal" is a standard, factual legal term describing the organization's action. It is not loaded language.
Action: Do NOT extract this.

EXAMPLE 3 (HYPERPARTISAN - FLAG THIS):
Text: "France already has the largest Muslim population in Europe, leading to serious cultural, societal, and even security problems."
Reasoning: The author jumps from a demographic statistic to a massive, unsupported negative conclusion ("serious cultural... problems") without evidence.
Action: Extract this and flag as "Loaded Language" and "Fear-Mongering". Change double quotes to single quotes in your JSON.

EXAMPLE 4 (OBJECTIVE REPORTING - LEGAL/CONFLICT - DO NOT FLAG):
Text: "The Ex-UVA dean's lawyer blasted the magazine's editor, calling the article a 'reckless' piece of journalism."
Reasoning: Words like 'blasted' or 'reckless' are strong, but the journalist is factually describing a legal defamation dispute and quoting the lawyer's official stance. This is standard conflict reporting, not partisan manipulation.
Action: Do NOT extract this. 

EXAMPLE 5 (HYPERPARTISAN - WEAPONISED OPINION - FLAG THIS):
Text: "If we truly want to tear down white supremacy, we must start with Planned Parenthood, an organization built on eradicating minorities."
Reasoning: Even though this might be labeled as 'opinion', it utilizes extreme False Equivalence and Fear-Mongering. It jumps from a historical claim to a massive, inflammatory accusation ("eradicating minorities") to advance a political agenda.
Action: Extract this and flag as "False Equivalence" and "Loaded Language".

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
            "is_quote_or_reported_speech": true or false,
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

        #combine title and text if the title exists
        article_content = request.text
        if request.title:
            article_content = f"HEADLINE: {request.title}\n\nBODY TEXT:\n{request.text}"

        #Process the entire article in one push
        chat_completion = await llm_client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:nitro",  
            # Models are from OpenRouter
            # openai/gpt-4o-mini, meta-llama/llama-3.3-70b-instruct, deepseek/deepseek-v3.2, google/gemini-2.5-flash
            # qwen/qwen-2.5-72b-instruct, meta-llama/llama-3.1-8b-instruct, x-ai/grok-4.1-fast
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": article_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2000,
            extra_body={"provider": {"sort": "throughput"}},
        )

        #Post processing
        res = json.loads(chat_completion.choices[0].message.content)

        article_type = res.get("article_type", "unclear")
        reasoning_summary = res.get("reasoning_summary", "No summary provided.")
        final_is_hyperpartisan = res.get("is_hyperpartisan", False)

        try:
            final_confidence = float(res.get("overall_confidence", 0.0))
        except (ValueError, TypeError):
            final_confidence = 0.0

        all_items = res.get("biased_items", [])

        #BACKEND GUARDRAIL: Force extension to display false if confidence dropsbelow 50%
        if final_is_hyperpartisan and final_confidence < 0.50:
            final_is_hyperpartisan = False
            all_items = [] 

        #Match each sentnce back to the original in article
        original_sentences = [
            s.strip()
            for s in re.split(r'(?<=[.!?])(?:\'|"|”|’)?\s+|\n+', request.text)
            if len(s.strip()) > 10
        ]

        for item in all_items:
            llm_sentence = item.get("sentence", "")
            matches = difflib.get_close_matches(
                llm_sentence, original_sentences, n=1, cutoff=0.7
            )
            if matches:
                item["sentence"] = matches[0]

        #Remove exact duplicates based on the matched sentence
        unique_items = {item["sentence"]: item for item in all_items}.values()
        final_items = list(unique_items)

        #Sort the extracted quotes based on where they appear in the original text
        final_items.sort(
            key=lambda x: get_chronological_index(request.text, x["sentence"])
        )

        #Keep the top 5 results
        final_items = final_items[:5]

        #If no results remai after filtering, force non-hyperpartisan
        if not final_items:
            final_is_hyperpartisan = False
            final_confidence = 0.0

        return BiasResponse(
            article_type=article_type,
            is_hyperpartisan=final_is_hyperpartisan,
            overall_confidence=round(final_confidence, 2),
            reasoning_summary=reasoning_summary,
            biased_items=final_items,
        )

    except Exception as e:
        print(f"Error during LLM analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "TruthLens LLM API is running seamlessly!"}
