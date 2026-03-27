import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

COMPREHENSION_PROMPT = """You are an interview evaluator.

Question asked:
\"\"\"{question}\"\"\"

Candidate's answer:
\"\"\"{answer}\"\"\"

Evaluate content only — ignore grammar and accent.

Did they:
1. Directly address the question?
2. Cover all key parts?
3. Stay relevant and focused?

Scoring:
0 = completely missed the question
1 = partially answered (key parts missing or off-topic)
2 = fully answered with clarity

Return ONLY valid JSON, no markdown:
{{"score": <0|1|2>, "note": "<one clear sentence on what was missing or well done>"}}"""


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


def evaluate_comprehension(question: str, answer: str, max_retries: int = 2) -> dict:
    if not question or not answer or not answer.strip():
        return {"score": 1, "note": "No answer provided to evaluate"}

    prompt = COMPREHENSION_PROMPT.format(
        question=question.strip(), answer=answer.strip()
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            data = _extract_json(response.choices[0].message.content)

            score = int(data.get("score", 1))
            if score not in (0, 1, 2):
                score = 1

            note = data.get("note", "")
            if not note:
                note = ("Fully answered the question" if score == 2
                        else "Partially answered the question; key parts missing" if score == 1
                        else "Did not address the question")

            return {"score": score, "note": note}

        except Exception as e:
            if attempt == max_retries:
                print(f"COMPREHENSION ERROR: {e}")
                return {"score": 1, "note": "Evaluation failed"}

    return {"score": 1, "note": "Evaluation failed"}