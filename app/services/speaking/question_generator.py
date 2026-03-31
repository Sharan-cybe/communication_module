import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

QUESTION_PROMPT = """You are an English communication assessment designer.

Generate exactly 2 general speaking questions for an interview-style English assessment.

Requirements:
- Questions should be general, open-ended, and suitable for any candidate
- Topics can include: hobbies, technology, travel, daily routine, future plans, 
  teamwork, problem-solving, favorite experiences, communication, learning
- Questions should encourage the candidate to speak for about 1 minute
- Do NOT ask personal/sensitive questions (religion, politics, salary, health)
- Questions must be different from each other
- Keep the language simple and clear

Return ONLY valid JSON, no markdown, no preamble:
{{"questions": ["<question 1>", "<question 2>"]}}
"""


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


def generate_speaking_questions(max_retries: int = 2) -> list[str]:
    """Generate 2 dynamic speaking questions using Groq/Llama."""

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": QUESTION_PROMPT}],
                temperature=0.2,
                max_tokens=300,
            )
            data = _extract_json(response.choices[0].message.content)
            questions = data.get("questions", [])

            if isinstance(questions, list) and len(questions) >= 2:
                return questions[:2]

        except Exception as e:
            if attempt == max_retries:
                print(f"QUESTION GENERATION ERROR: {e}")

    # Fallback questions if LLM fails
    return [
        "Describe a hobby or activity you enjoy and explain why it is important to you.",
        "How has technology changed the way people communicate in everyday life?",
    ]
