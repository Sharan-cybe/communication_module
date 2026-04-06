"""
listening_service.py
─────────────────────
Evaluates all clip responses together in a single call per task type.

Parameters evaluated:
  1. listening_accuracy       — correct content captured?      (keyword + LLM)
  2. retention                — how complete was the recall?   (coverage ratio + LLM)
  3. sentence_reconstruction  — grammatical structure?         (edit-distance + LLM)

NOTE: pronunciation_imitation has been removed entirely.
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


def _is_empty(text: str) -> bool:
    """Returns True if the transcript is blank or too short to evaluate."""
    return not text or len(text.strip()) < 3


# ─────────────────────────────────────────────────────────────────────────────
# Clip-repeat detection  (QnA clips only)
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(text_a: str, text_b: str) -> float:
    a = _tokens(text_a)
    b = _tokens(text_b)
    if not a and not b:
        return 0.0
    return round(len(a & b) / max(len(a | b), 1), 3)


def _is_clip_repeat(reference: str, response: str, threshold: float = 0.50) -> bool:
    """
    Returns True if the candidate's response is too similar to the reference
    passage — meaning they repeated the clip instead of answering the question.

    Threshold rationale:
      Genuine answer  → Jaccard 0.15–0.35
      Paraphrase      → Jaccard 0.20–0.40
      Partial repeat  → Jaccard 0.55–0.80  ← flagged
      Full repeat     → Jaccard 0.90–1.00  ← flagged
    """
    return _jaccard(reference, response) > threshold


# Penalty dicts applied when a clip repeat or empty response is detected
CLIP_REPEAT_PENALTY = {
    "score": 0,
    "keyword_hit_rate": 0.0,
    "note": "Candidate repeated the audio clip instead of answering the question",
    "flagged_as_repeat": True,
}

EMPTY_RESPONSE_PENALTY = {
    "score": 0,
    "note": "No response provided",
    "flagged_as_empty": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM call helper
# ─────────────────────────────────────────────────────────────────────────────

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
0 = < 40% key facts OR factually wrong OR no answer given
1 = 40-79% OR partially correct
2 = ≥ 80% AND factually correct

Return ONLY valid JSON:
{{"q1": {{"score": <0|1|2>, "note": "<max 12 words>"}}, "q2": {{"score": <0|1|2>, "note": "<max 12 words>"}}}}"""


def evaluate_accuracy_repeat(reference: str, response: str, key_facts: list) -> dict:
    # Empty response → hard 0, skip LLM
    if _is_empty(response):
        return {**EMPTY_RESPONSE_PENALTY, "keyword_hit_rate": 0.0}

    hit_rate = _keyword_hit_rate(key_facts, response)
    data = _llm(ACCURACY_REPEAT_PROMPT.format(
        reference=reference, response=response,
        hit_rate=hit_rate, key_facts=key_facts,
    ))
    score = max(0, min(2, int(data.get("score", 1))))

    # Hard overrides: LLM can't hallucinate a high score on bad hit rates
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
    # Handle empties before any LLM call
    empty_q1 = _is_empty(a1)
    empty_q2 = _is_empty(a2)

    if empty_q1 and empty_q2:
        r1 = {**EMPTY_RESPONSE_PENALTY, "keyword_hit_rate": 0.0}
        r2 = {**EMPTY_RESPONSE_PENALTY, "keyword_hit_rate": 0.0}
        return r1, r2

    h1 = 0.0 if empty_q1 else _keyword_hit_rate(kf1, a1)
    h2 = 0.0 if empty_q2 else _keyword_hit_rate(kf2, a2)

    data = _llm(ACCURACY_QNA_PROMPT.format(
        reference=reference,
        q1=q1, a1=(a1 if not empty_q1 else "[no answer]"), kf1=kf1, h1=h1,
        q2=q2, a2=(a2 if not empty_q2 else "[no answer]"), kf2=kf2, h2=h2,
    ))

    def _parse(raw: dict, hit: float, is_empty: bool) -> dict:
        if is_empty:
            return {**EMPTY_RESPONSE_PENALTY, "keyword_hit_rate": 0.0}
        score = max(0, min(2, int(raw.get("score", 1))))
        if hit < 0.30 and score == 2: score = 1
        if hit > 0.85 and score == 0: score = 1
        return {"score": score, "keyword_hit_rate": hit, "note": raw.get("note", "")}

    r1 = _parse(data.get("q1", {}), h1, empty_q1)
    r2 = _parse(data.get("q2", {}), h2, empty_q2)
    return r1, r2


# ─────────────────────────────────────────────────────────────────────────────
# Parameter 2 — Retention
#
# REPEAT clips: token coverage ratio (reference passage vs. response)
# QnA clips:    key-fact coverage  (key_facts recalled vs. total key_facts)
#               A good answer won't repeat the passage verbatim — it extracts
#               specific facts, so measuring against key_facts is semantically
#               correct rather than measuring against the full reference text.
# ─────────────────────────────────────────────────────────────────────────────

def _token_coverage(reference: str, response: str) -> float:
    """Fraction of meaningful reference tokens that appear in response."""
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


def evaluate_retention_repeat(reference: str, response: str) -> dict:
    """Retention for REPEAT clips — measures passage token coverage."""
    if _is_empty(response):
        return {**EMPTY_RESPONSE_PENALTY, "coverage_ratio": 0.0}

    coverage = _token_coverage(reference, response)
    data = _llm(RETENTION_PROMPT.format(
        reference=reference, response=response, coverage=coverage
    ))
    score = max(0, min(2, int(data.get("score", 1))))

    # Hard anchors
    if coverage < 0.35 and score == 2: score = 1
    if coverage > 0.75 and score == 0: score = 1

    return {"score": score, "coverage_ratio": coverage, "note": data.get("note", "")}


def evaluate_retention_qna(key_facts: list, response: str) -> dict:
    """
    Retention for QnA clips — measures key-fact recall, NOT passage repetition.
    A correct concise answer ("The meeting was on Monday") should score high
    even though it has low token overlap with a long passage.
    """
    if _is_empty(response):
        return {**EMPTY_RESPONSE_PENALTY, "coverage_ratio": 0.0}

    if not key_facts:
        return {"score": 1, "coverage_ratio": 0.75, "note": "No key facts defined"}

    resp_lower = _clean(response)
    hits = [kf for kf in key_facts if _clean(kf) in resp_lower]
    coverage = round(len(hits) / len(key_facts), 2)

    score = 2 if coverage >= 0.75 else (1 if coverage >= 0.40 else 0)

    return {
        "score": score,
        "coverage_ratio": coverage,
        "facts_recalled": hits,
        "note": f"Recalled {len(hits)}/{len(key_facts)} key facts",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parameter 3 — Sentence Reconstruction (REPEAT clips only)
# Signal: normalised Levenshtein distance between reference and response tokens
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
    if _is_empty(response):
        return {**EMPTY_RESPONSE_PENALTY, "structure_similarity": 0.0}

    similarity = _structure_similarity(reference, response)
    data = _llm(RECONSTRUCTION_PROMPT.format(
        reference=reference, response=response, similarity=similarity
    ))
    score = max(0, min(2, int(data.get("score", 1))))

    if similarity < 0.45 and score == 2: score = 1
    if similarity > 0.80 and score == 0: score = 1

    return {"score": score, "structure_similarity": similarity, "note": data.get("note", "")}


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator — evaluate ALL clip responses together
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_responses(session_clips: list, clip_responses: list) -> list:
    """
    Evaluate all clip responses in a single function call.

    session_clips   : list of ListeningClip objects for this session
    clip_responses  : list of dicts, one per clip:

        For REPEAT clips:
        {
          "clip_id": str,
          "transcript": str,          # Whisper transcript of candidate
          "whisper_segments": list,   # from transcribe_audio (kept for compat, unused)
          "whisper_words": list,      # from transcribe_audio (kept for compat, unused)
        }

        For QnA clips:
        {
          "clip_id": str,
          "answer_q1": str,           # transcript for question 1
          "answer_q2": str,           # transcript for question 2
        }

    Returns list of result dicts, one per clip.
    """
    clip_map = {c.clip_id: c for c in session_clips}

    results = []
    for resp in clip_responses:
        clip_id = resp["clip_id"]
        clip    = clip_map.get(clip_id)
        if not clip:
            results.append({"clip_id": clip_id, "error": "Clip not found in session"})
            continue

        result = {"clip_id": clip_id, "task_type": clip.task_type}

        # ── REPEAT clips ──────────────────────────────────────────────────────
        if clip.task_type == "REPEAT":
            transcript = resp.get("transcript", "")
            result["transcript"] = transcript

            if _is_empty(transcript):
                print(f"[{clip_id}] REPEAT — empty transcript, all params scored 0")
                result["listening_accuracy"]      = {**EMPTY_RESPONSE_PENALTY, "keyword_hit_rate": 0.0}
                result["retention"]               = {**EMPTY_RESPONSE_PENALTY, "coverage_ratio": 0.0}
                result["sentence_reconstruction"] = {**EMPTY_RESPONSE_PENALTY, "structure_similarity": 0.0}
            else:
                result["listening_accuracy"]      = evaluate_accuracy_repeat(
                    clip.reference_text, transcript, clip.key_facts
                )
                result["retention"]               = evaluate_retention_repeat(
                    clip.reference_text, transcript
                )
                result["sentence_reconstruction"] = evaluate_sentence_reconstruction(
                    clip.reference_text, transcript
                )

            print(
                f"[{clip_id}] REPEAT | "
                f"accuracy={result['listening_accuracy']['score']} "
                f"retention={result['retention']['score']} "
                f"reconstruction={result['sentence_reconstruction']['score']}"
            )

        # ── QnA clips ─────────────────────────────────────────────────────────
        elif clip.task_type == "QnA":
            a1 = resp.get("answer_q1", "")
            a2 = resp.get("answer_q2", "")

            kf  = clip.key_facts
            kf1 = kf[0] if len(kf) > 0 else []
            kf2 = kf[1] if len(kf) > 1 else []

            # ── Clip-repeat detection ─────────────────────────────────────────
            repeat_q1 = (not _is_empty(a1)) and _is_clip_repeat(clip.reference_text, a1)
            repeat_q2 = (not _is_empty(a2)) and _is_clip_repeat(clip.reference_text, a2)

            if repeat_q1:
                print(f"[{clip_id}] Q1 flagged as clip repeat "
                      f"(jaccard={_jaccard(clip.reference_text, a1):.2f})")
            if repeat_q2:
                print(f"[{clip_id}] Q2 flagged as clip repeat "
                      f"(jaccard={_jaccard(clip.reference_text, a2):.2f})")

            # ── Listening Accuracy ────────────────────────────────────────────
            if repeat_q1 and repeat_q2:
                acc_q1 = dict(CLIP_REPEAT_PENALTY)
                acc_q2 = dict(CLIP_REPEAT_PENALTY)
            elif repeat_q1:
                acc_q1 = dict(CLIP_REPEAT_PENALTY)
                _, acc_q2 = evaluate_accuracy_qna(
                    clip.reference_text,
                    clip.questions[0], a1, kf1,
                    clip.questions[1], a2, kf2,
                )
            elif repeat_q2:
                acc_q1, _ = evaluate_accuracy_qna(
                    clip.reference_text,
                    clip.questions[0], a1, kf1,
                    clip.questions[1], a2, kf2,
                )
                acc_q2 = dict(CLIP_REPEAT_PENALTY)
            else:
                acc_q1, acc_q2 = evaluate_accuracy_qna(
                    clip.reference_text,
                    clip.questions[0], a1, kf1,
                    clip.questions[1], a2, kf2,
                )

            # ── Retention ─────────────────────────────────────────────────────
            # Clip repeat → force 0 (repeating passage gives fake 100% coverage)
            # Empty answer → force 0
            # Normal answer → evaluate against key_facts (NOT full passage)
            REPEAT_RETENTION = {
                "score": 0,
                "coverage_ratio": 1.0,
                "facts_recalled": [],
                "note": "Repeated audio clip instead of answering",
                "flagged_as_repeat": True,
            }

            if repeat_q1:
                ret_q1 = dict(REPEAT_RETENTION)
            else:
                ret_q1 = evaluate_retention_qna(kf1, a1)

            if repeat_q2:
                ret_q2 = dict(REPEAT_RETENTION)
            else:
                ret_q2 = evaluate_retention_qna(kf2, a2)

            # ── Average Q1+Q2 ─────────────────────────────────────────────────
            def _avg_score(d1: dict, d2: dict) -> dict:
                return {
                    "score": round((d1["score"] + d2["score"]) / 2, 2),
                    "q1": d1,
                    "q2": d2,
                }

            result["answers"] = {
                "q1": {
                    "question": clip.questions[0],
                    "transcript": a1,
                    "flagged_as_repeat": repeat_q1,
                    "flagged_as_empty": _is_empty(a1),
                },
                "q2": {
                    "question": clip.questions[1],
                    "transcript": a2,
                    "flagged_as_repeat": repeat_q2,
                    "flagged_as_empty": _is_empty(a2),
                },
            }
            result["listening_accuracy"] = _avg_score(acc_q1, acc_q2)
            result["retention"]          = _avg_score(ret_q1, ret_q2)

            print(
                f"[{clip_id}] QnA | "
                f"accuracy={result['listening_accuracy']['score']} "
                f"retention={result['retention']['score']} "
                f"repeat_q1={repeat_q1} repeat_q2={repeat_q2}"
            )

        results.append(result)

    return results