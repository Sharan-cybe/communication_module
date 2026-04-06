import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GRAMMAR_PROMPT = """You are an expert English language assessor specializing in evaluating spoken interview responses. Your task is to analyze the grammar of a candidate's transcribed speech with high precision.

Candidate's transcribed speech:
\"\"\"{text}\"\"\"

---

## YOUR EVALUATION FRAMEWORK

### Step 1 — Spoken English Allowances (do NOT penalize these)
These are natural features of fluent spoken English or accepted Indian-English variants:
- Sentence-initial fillers: "So", "Well", "You know", "Like", "Basically", "Right"
- Trailing conjunctions: ending with "and", "but", "so" mid-thought
- Contracted forms: "gonna", "wanna", "kinda" in informal contexts
- Indian-English constructions: "I am having 5 years experience", "itself", "only" as emphasis, "prepone"
- Minor repetitions or false starts: "I — I worked on..." (transcription artifact)
- Omission of articles in lists: "I worked on frontend, backend, and database"

### Step 2 — Errors to ACTIVELY detect and flag
Scan for these spoken grammar error categories:

**[TENSE]** — Tense inconsistency or incorrect tense
  Examples:
  - ❌ "I was working there and then I go to the meeting" → ✅ "I was working there and then I went to the meeting"
  - ❌ "I have joined the company in 2019" → ✅ "I joined the company in 2019"
  - ❌ "Yesterday I am debugging the issue" → ✅ "Yesterday I was debugging the issue"

**[SVA]** — Subject-verb agreement breakdown
  Examples:
  - ❌ "The results was incorrect" → ✅ "The results were incorrect"
  - ❌ "Each of the modules have a bug" → ✅ "Each of the modules has a bug"
  - ❌ "Our team are working on it" (in American context) → ✅ "Our team is working on it"

**[ARTICLE]** — Missing, extra, or wrong article (a / an / the)
  Examples:
  - ❌ "I gave presentation to client" → ✅ "I gave a presentation to the client"
  - ❌ "We used an SQL query" → ✅ "We used a SQL query"
  - ❌ "I am working in startup" → ✅ "I am working in a startup"

**[PREP]** — Wrong or missing preposition
  Examples:
  - ❌ "I am good in Python" → ✅ "I am good at Python"
  - ❌ "I discussed about the issue" → ✅ "I discussed the issue"
  - ❌ "We depend on this since 2020" → ✅ "We have depended on this since 2020"

**[WF]** — Wrong word form (noun/verb/adjective confusion)
  Examples:
  - ❌ "I did a analyse of the data" → ✅ "I did an analysis of the data"
  - ❌ "The system become more efficiently" → ✅ "The system became more efficient"

**[CONC]** — Faulty concord or pronoun–antecedent mismatch
  Examples:
  - ❌ "The team, they has finished" → ✅ "The team has finished"
  - ❌ "Everyone submitted their report on time" — ✅ ACCEPTABLE (singular 'they')

**[STRUCT]** — Broken or incoherent sentence structure
  Examples:
  - ❌ "The reason because I chose this approach was it is faster"
    → ✅ "The reason I chose this approach was that it is faster"
  - ❌ "Although I tried but I failed" → ✅ "Although I tried, I failed" OR "I tried but I failed"

---

### Step 3 — Scoring Rubric

Score **2 — Excellent**:
- Zero or one negligible error; grammar does not distract at all
- Tenses are consistent; sentences are well-formed

Score **1 — Average**:
- 2–4 noticeable errors across different categories
- Meaning is still clear despite the errors

Score **0 — Poor**:
- 5+ errors, OR errors that obscure meaning / confuse the listener
- Multiple tense collapses or broken sentence structure throughout

---

## OUTPUT RULES
- Return ONLY a valid JSON object — no markdown fences, no explanation, no preamble
- List at most 3 mistakes (pick the most impactful ones)
- Each mistake must have: the error category tag, the exact original phrase, and the corrected phrase
- The note must be one crisp sentence summarising grammar quality

Required JSON schema:
{{"score": <0|1|2>, "mistakes": [{{"category": "<TAG>", "original": "<wrong phrase>", "corrected": "<corrected phrase>"}}], "note": "<one sentence assessment>"}}

If there are no mistakes, return: {{"score": 2, "mistakes": [], "note": "Grammar is accurate and well-controlled throughout."}}"""

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