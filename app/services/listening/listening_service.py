"""
listening_service.py
─────────────────────
Evaluates all 4 clip responses together in a single call per task type.

Parameters:
  1. listening_accuracy   — correct content captured?      (keyword + LLM)
  2. retention            — how complete was the recall?   (coverage ratio + LLM)
  3. sentence_reconstruction — grammatical structure?      (edit-distance + LLM)

Key improvements over v1:
  - Accuracy: keyword hit-rate as a deterministic signal BEFORE LLM call
    → LLM can't hallucinate a 2 if key facts are clearly missing
  - Retention: token coverage ratio computed independently from LLM
    → gives a numeric anchor; LLM only adjusts ±1 based on context
  - Sentence reconstruction: normalised Levenshtein distance computed first
    → LLM refines edge cases; distance gives ground truth
  - QnA: both questions evaluated in ONE LLM call (halves API usage)
  - All LLM prompts include the numeric signals so the model reasons
    from data, not memory
"""

import re
import json
import os
import statistics
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _tokens(text: str) -> set:
    STOP = {"a","an","the","and","or","but","in","on","at","to","for","of",
            "is","are","was","were","be","i","you","we","it","this","that",
            "have","has","will","please","all","can","your"}
    return {w for w in _clean(text).split() if w and w not in STOP}


def _llm(prompt: str, max_tokens: int = 400) -> dict:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            raw     = resp.choices[0].message.content
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if m:
                    return json.loads(m.group())
        except Exception as e:
            if attempt == 2:
                print(f"LLM ERROR: {e}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Parameter 1 — Listening Accuracy
# Signal: keyword hit-rate → anchors the LLM score
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_hit_rate(key_facts: list, response: str) -> float:
    """What fraction of key_facts appear in the response text?"""
    if not key_facts:
        return 0.75   # no facts defined — neutral
    resp_lower = _clean(response)
    hits = sum(1 for kf in key_facts if _clean(kf) in resp_lower)
    return round(hits / len(key_facts), 2)


ACCURACY_REPEAT_PROMPT = """You are a listening accuracy evaluator.

The candidate was asked to REPEAT this sentence:
Reference: "{reference}"

They said:
Response: "{response}"

Keyword hit rate (computed): {hit_rate:.0%} of key facts present in their response.
Key facts that must appear: {key_facts}

Based on the keyword hit rate AND the overall content accuracy, score:
0 = < 40% key facts captured OR major factual errors (wrong numbers/names)
1 = 40-79% key facts captured OR 1-2 minor substitutions
2 = ≥ 80% key facts captured AND all critical details correct

Return ONLY valid JSON:
{{"score": <0|1|2>, "note": "<what was correct or wrong, max 15 words>"}}"""

ACCURACY_QNA_PROMPT = """You are a listening comprehension evaluator.

Audio passage:
"{reference}"

Question 1: {q1}
Answer 1: "{a1}"
Q1 key facts: {kf1}
Q1 keyword hit: {h1:.0%}

Question 2: {q2}
Answer 2: "{a2}"
Q2 key facts: {kf2}
Q2 keyword hit: {h2:.0%}

Score each answer independently using the keyword hit rate as anchor:
0 = < 40% key facts OR factually wrong
1 = 40-79% OR partially correct
2 = ≥ 80% AND factually correct

Return ONLY valid JSON:
{{"q1": {{"score": <0|1|2>, "note": "<max 12 words>"}}, "q2": {{"score": <0|1|2>, "note": "<max 12 words>"}}}}"""


def evaluate_accuracy_repeat(reference: str, response: str, key_facts: list) -> dict:
    hit_rate = _keyword_hit_rate(key_facts, response)
    data = _llm(ACCURACY_REPEAT_PROMPT.format(
        reference=reference, response=response,
        hit_rate=hit_rate, key_facts=key_facts,
    ))
    score = max(0, min(2, int(data.get("score", 1))))
    # Hard override: hit_rate < 0.3 → cap at 1; hit_rate > 0.85 → floor at 1
    if hit_rate < 0.30 and score == 2:
        score = 1
    if hit_rate > 0.85 and score == 0:
        score = 1
    return {"score": score, "keyword_hit_rate": hit_rate, "note": data.get("note", "")}


def evaluate_accuracy_qna(
    reference: str,
    q1: str, a1: str, kf1: list,
    q2: str, a2: str, kf2: list,
) -> tuple:
    h1 = _keyword_hit_rate(kf1, a1)
    h2 = _keyword_hit_rate(kf2, a2)
    data = _llm(ACCURACY_QNA_PROMPT.format(
        reference=reference,
        q1=q1, a1=a1, kf1=kf1, h1=h1,
        q2=q2, a2=a2, kf2=kf2, h2=h2,
    ))
    def _parse(raw: dict, hit: float) -> dict:
        score = max(0, min(2, int(raw.get("score", 1))))
        if hit < 0.30 and score == 2: score = 1
        if hit > 0.85 and score == 0: score = 1
        return {"score": score, "keyword_hit_rate": hit, "note": raw.get("note", "")}
    r1 = _parse(data.get("q1", {}), h1)
    r2 = _parse(data.get("q2", {}), h2)
    return r1, r2


# ─────────────────────────────────────────────────────────────────────────────
# Parameter 2 — Retention
# Signal: token coverage ratio = content tokens recalled / content tokens in reference
# ─────────────────────────────────────────────────────────────────────────────

def _token_coverage(reference: str, response: str) -> float:
    ref_tokens  = _tokens(reference)
    resp_tokens = _tokens(response)
    if not ref_tokens:
        return 0.75
    overlap = ref_tokens & resp_tokens
    return round(len(overlap) / len(ref_tokens), 2)


RETENTION_PROMPT = """You are a memory retention evaluator.

The candidate heard:
"{reference}"

They recalled:
"{response}"

Token coverage ratio (computed): {coverage:.0%}
(This measures what fraction of meaningful words from the original appeared in their response.)

Score based on coverage and completeness of key details:
0 = < 40% coverage — major portions missing
1 = 40-74% coverage — roughly half recalled
2 = ≥ 75% coverage — full or near-full recall

Return ONLY valid JSON:
{{"score": <0|1|2>, "note": "<what was recalled well or missed, max 15 words>"}}"""


def evaluate_retention(reference: str, response: str) -> dict:
    coverage = _token_coverage(reference, response)
    data = _llm(RETENTION_PROMPT.format(
        reference=reference, response=response, coverage=coverage
    ))
    score = max(0, min(2, int(data.get("score", 1))))
    # Hard anchors based on coverage
    if coverage < 0.35 and score == 2: score = 1
    if coverage > 0.75 and score == 0: score = 1
    return {"score": score, "coverage_ratio": coverage, "note": data.get("note", "")}





# ─────────────────────────────────────────────────────────────────────────────
# Parameter 4 — Sentence Reconstruction (REPEAT only)
# Signal: normalised edit distance between reference and response tokens
# ─────────────────────────────────────────────────────────────────────────────

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def _structure_similarity(reference: str, response: str) -> float:
    """Normalised token edit distance (1 = identical, 0 = completely different)."""
    ref  = _clean(reference).split()
    resp = _clean(response).split()
    if not ref:
        return 0.75
    dist = _edit_distance(ref, resp)
    return round(max(0.0, 1.0 - dist / max(len(ref), 1)), 2)


RECONSTRUCTION_PROMPT = """You are a sentence structure evaluator.

Reference sentence: "{reference}"
Candidate response: "{response}"

Structure similarity score (computed): {similarity:.0%}
(1.0 = word-for-word match, 0 = completely different word order/structure)

Evaluate STRUCTURAL accuracy — word order, grammatical form, key sentence components.
Do NOT penalize for accent-related word substitutions.

Score:
0 = < 50% similarity — broken structure or sentence fragments
1 = 50-79% similarity — mostly correct but notable structural issues
2 = ≥ 80% similarity — well-structured, matches original pattern

Return ONLY valid JSON:
{{"score": <0|1|2>, "note": "<structural assessment, max 15 words>"}}"""


def evaluate_sentence_reconstruction(reference: str, response: str) -> dict:
    similarity = _structure_similarity(reference, response)
    data = _llm(RECONSTRUCTION_PROMPT.format(
        reference=reference, response=response, similarity=similarity
    ))
    score = max(0, min(2, int(data.get("score", 1))))
    if similarity < 0.45 and score == 2: score = 1
    if similarity > 0.80 and score == 0: score = 1
    return {"score": score, "structure_similarity": similarity, "note": data.get("note", "")}


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator — evaluate ALL 4 clip responses together
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_responses(session_clips: list, clip_responses: list) -> list:
    """
    Evaluate all 4 clip responses in a single function call.

    session_clips   : list of ListeningClip objects for this session
    clip_responses  : list of dicts, one per clip:
        {
          "clip_id": str,
          "transcript": str,          # Whisper transcript of candidate
          "whisper_segments": list,   # from transcribe_audio
          "whisper_words": list,      # from transcribe_audio
          # QnA only:
          "answer_q1": str,           # transcript for question 1
          "answer_q2": str,           # transcript for question 2
          "segments_q1": list,
          "words_q1": list,
          "segments_q2": list,
          "words_q2": list,
        }

    Returns list of result dicts, one per clip.
    """
    # Build a lookup from clip_id → clip definition
    clip_map = {c.clip_id: c for c in session_clips}

    results = []
    for resp in clip_responses:
        clip_id = resp["clip_id"]
        clip    = clip_map.get(clip_id)
        if not clip:
            results.append({"clip_id": clip_id, "error": "Clip not found in session"})
            continue

        result = {"clip_id": clip_id, "task_type": clip.task_type}

        if clip.task_type == "REPEAT":
            transcript = resp.get("transcript", "")
            segments   = resp.get("whisper_segments", [])
            words      = resp.get("whisper_words", [])

            result["transcript"] = transcript
            result["listening_accuracy"]      = evaluate_accuracy_repeat(
                clip.reference_text, transcript, clip.key_facts
            )
            result["retention"]               = evaluate_retention(
                clip.reference_text, transcript
            )
            result["sentence_reconstruction"] = evaluate_sentence_reconstruction(
                clip.reference_text, transcript
            )

        elif clip.task_type == "QnA":
            a1 = resp.get("answer_q1", "")
            a2 = resp.get("answer_q2", "")
            s1 = resp.get("segments_q1", [])
            w1 = resp.get("words_q1", [])
            s2 = resp.get("segments_q2", [])
            w2 = resp.get("words_q2", [])

            kf = clip.key_facts   # [[q1 facts], [q2 facts]]
            kf1 = kf[0] if len(kf) > 0 else []
            kf2 = kf[1] if len(kf) > 1 else []

            acc_q1, acc_q2 = evaluate_accuracy_qna(
                clip.reference_text,
                clip.questions[0], a1, kf1,
                clip.questions[1], a2, kf2,
            )
            ret_q1 = evaluate_retention(clip.reference_text, a1)
            ret_q2 = evaluate_retention(clip.reference_text, a2)

            # Average Q1 + Q2 for each parameter
            def _avg_score(d1, d2, extra_key=None):
                s = round((d1["score"] + d2["score"]) / 2, 2)
                out = {"score": s,
                       "q1": d1, "q2": d2}
                return out

            result["answers"] = {
                "q1": {"question": clip.questions[0], "transcript": a1},
                "q2": {"question": clip.questions[1], "transcript": a2},
            }
            result["listening_accuracy"] = _avg_score(acc_q1, acc_q2)
            result["retention"]          = _avg_score(ret_q1, ret_q2)


        results.append(result)

    return results