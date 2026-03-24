from groq import Groq
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluate_comprehension(question, answer):
    prompt = f"""
You are an interview evaluator.

Question:
{question}

Answer:
{answer}

Evaluate:
- Did the candidate answer the question directly?
- Did they cover all parts?
- Is the response relevant?

Scoring:
0 = irrelevant
1 = partially answered
2 = fully answered with clarity

Also give reason.

Return STRICT JSON:
{{
  "score": number,
  "reason": "..."
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
        return {"score": int(data.get("score", 1)), "reason": data.get("reason", "No reason provided")}
    except:
        return {"score": 1, "reason": "Failed to evaluate comprehension"}