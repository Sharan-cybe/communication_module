import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GRAMMAR_PROMPT = """You are an English speaking coach evaluating an interview candidate's grammar.

Candidate's spoken text:
\"\"\"
{text}
\"\"\"

Evaluate:
- Sentence structure and correctness
- Article usage (a/an/the)
- Tense consistency
- Subject-verb agreement
- Preposition usage

Note: The candidate may have an Indian-English accent. Do NOT penalize:
  • "I am having" instead of "I have" (Indian English habitual aspect)
  • "only" or "itself" used as emphasis ("He told only")
  • Omission of articles in some contexts (common in South Asian English)
These are accent features, not errors.

Scoring:
0 = poor (multiple grammatical errors that hinder understanding)
1 = average (some errors but overall comprehensible)
2 = excellent (grammatically accurate throughout)

Return ONLY valid JSON — no explanation, no markdown fences, no preamble:
{{"score": <0|1|2>, "mistakes": ["<error 1>", "<error 2>"]}}"""


def _extract_json(text: str) -> dict:
    """Robustly extract JSON from an LLM response that may have markdown fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find the first {...} block
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def evaluate_grammar(text: str, max_retries: int = 2) -> dict:
    """
    Evaluate grammar using Groq LLM.

    Improvements over v1:
    ─────────────────────
    1. Explicit Indian-English accent allowances in the prompt
    2. Robust JSON extraction (strips markdown fences, searches for JSON block)
    3. Retry logic: up to 2 retries on parse failure
    4. Score is validated to be 0/1/2 (not some other value)
    """
    if not text or not text.strip():
        return {"score": 1, "mistakes": ["No transcript to evaluate"]}

    prompt = GRAMMAR_PROMPT.format(text=text.strip())

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,   # low temp for consistent structured output
                max_tokens=300,
            )
            content = response.choices[0].message.content
            data = _extract_json(content)

            score = int(data.get("score", 1))
            if score not in (0, 1, 2):
                score = 1

            mistakes = data.get("mistakes", [])
            if not isinstance(mistakes, list):
                mistakes = [str(mistakes)]

            return {"score": score, "mistakes": mistakes[:5]}   # cap at 5

        except Exception as e:
            if attempt == max_retries:
                print(f"GRAMMAR ERROR (attempt {attempt}): {e}")
                return {"score": 1, "mistakes": ["Evaluation failed"]}

    return {"score": 1, "mistakes": ["Evaluation failed"]}