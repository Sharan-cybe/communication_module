import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GRAMMAR_PROMPT = """You are an English speaking coach evaluating an interview candidate's grammar.

Candidate's spoken text:
\"\"\"{text}\"\"\"

Evaluate sentence structure, article usage, tense, subject-verb agreement, and prepositions.

Do NOT penalize Indian-English features:
  - "I am having" instead of "I have"
  - "only" used as emphasis
  - Occasional article omission

Scoring:
0 = poor (errors hinder understanding)
1 = average (some errors, still comprehensible)
2 = excellent (grammatically accurate)

Return ONLY valid JSON, no markdown, no preamble:
{{"score": <0|1|2>, "mistakes": [{{"original": "<wrong phrase>", "corrected": "<correct phrase>"}}], "note": "<one sentence overall assessment>"}}

Return at most 3 mistakes. If no mistakes, return empty array."""


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def evaluate_grammar(text: str, max_retries: int = 2) -> dict:
    if not text or not text.strip():
        return {"score": 1, "mistakes": [], "note": "No transcript to evaluate"}

    prompt = GRAMMAR_PROMPT.format(text=text.strip())

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )
            data = _extract_json(response.choices[0].message.content)

            score = int(data.get("score", 1))
            if score not in (0, 1, 2):
                score = 1

            # Normalise mistakes — accept both string list and object list
            raw_mistakes = data.get("mistakes", [])
            mistakes = []
            for m in raw_mistakes[:3]:
                if isinstance(m, dict):
                    mistakes.append({
                        "original":  m.get("original", ""),
                        "corrected": m.get("corrected", ""),
                    })
                elif isinstance(m, str):
                    mistakes.append({"original": m, "corrected": ""})

            note = data.get("note", "")
            if not note:
                note = ("Grammatically strong" if score == 2
                        else "Some grammatical issues present" if score == 1
                        else "Multiple grammatical errors detected")

            return {"score": score, "mistakes": mistakes, "note": note}

        except Exception as e:
            if attempt == max_retries:
                print(f"GRAMMAR ERROR: {e}")
                return {"score": 1, "mistakes": [], "note": "Evaluation failed"}

    return {"score": 1, "mistakes": [], "note": "Evaluation failed"}