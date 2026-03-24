from groq import Groq
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluate_grammar(text):
    prompt = f"""
You are an English speaking evaluator.

Evaluate the grammar quality of the candidate.

Text:
{text}

Consider:
- sentence correctness
- article usage
- tense
- phrasing

Scoring:
0 = poor (many errors)
1 = average (some mistakes)
2 = excellent (no mistakes)

Also give 2–3 examples of mistakes.

Return STRICT JSON:
{{
  "score": number,
  "mistakes": ["...","..."]
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
        return {"score": int(data.get("score", 1)), "mistakes": data.get("mistakes", [])}
    except:
        return {"score": 1, "mistakes": ["Failed to evaluate grammar"]}