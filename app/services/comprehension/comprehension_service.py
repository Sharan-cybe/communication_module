import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

COMPREHENSION_PROMPT = """You are an interview evaluator assessing how well a candidate answered a question.

Question asked:
\"\"\"
{question}
\"\"\"

Candidate's answer:
\"\"\"
{answer}
\"\"\"

Evaluate:
1. Did the candidate directly address the question?
2. Did they cover all key parts of the question?
3. Is the response relevant and focused (not off-topic)?
4. Is the answer structured and clear?

Note: Evaluate content quality only — do not penalize for accent or grammar.

Scoring:
0 = irrelevant or completely missed the question
1 = partially answered (missed key parts or went off-topic)
2 = fully answered with clarity and structure

Return ONLY valid JSON — no markdown, no preamble:
{{"score": <0|1|2>, "reason": "<one sentence explanation>"}}"""


def _extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM response."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def evaluate_comprehension(question: str, answer: str, max_retries: int = 2) -> dict:
    """
    Evaluate comprehension using Groq LLM.

    Improvements over v1:
    ─────────────────────
    1. Explicitly separates content evaluation from grammar/accent evaluation
    2. Robust JSON extraction with fallback regex
    3. Retry logic on parse failure
    4. Score validated to be 0/1/2
    """
    if not question or not answer or not answer.strip():
        return {"score": 1, "reason": "No answer provided to evaluate"}

    prompt = COMPREHENSION_PROMPT.format(
        question=question.strip(),
        answer=answer.strip(),
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            content = response.choices[0].message.content
            data = _extract_json(content)

            score = int(data.get("score", 1))
            if score not in (0, 1, 2):
                score = 1

            reason = data.get("reason", "No reason provided")
            return {"score": score, "reason": str(reason)}

        except Exception as e:
            if attempt == max_retries:
                print(f"COMPREHENSION ERROR (attempt {attempt}): {e}")
                return {"score": 1, "reason": "Evaluation failed"}

    return {"score": 1, "reason": "Evaluation failed"}