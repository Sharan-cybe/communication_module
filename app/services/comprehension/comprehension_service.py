from groq import Groq
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluate_comprehension(question, answer):
    prompt = f"""
Question: {question}
Answer: {answer}

Score:
0 = wrong
1 = partial
2 = correct

Return JSON:
{{"score": number}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
        return {"score": int(data.get("score", 1))}
    except:
        return {"score": 1}