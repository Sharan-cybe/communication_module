import re
import statistics


# ─────────────────────────────────────────────────────────────────────────────
# Pronunciation evaluation — Whisper audio signals ONLY
#
# We removed word-hit-rate entirely. Here's why:
#   • Interview questions are open-ended ("Tell me about yourself")
#   • The candidate will NEVER repeat the question words back
#   • Comparing question words vs answer words always gives 0% match
#   • This was incorrectly dragging pronunciation scores to 0
#
# What we measure instead (all from Whisper audio analysis):
#
#   Signal 1 — avg_logprob per segment       (40%)
#     How confident Whisper was in what it heard.
#     Low logprob = unclear audio = poor pronunciation.
#
#   Signal 2 — per-word probability mean     (35%)
#     Average probability Whisper assigned to each transcribed word.
#     Low mean = many words were hard to decode.
#
#   Signal 3 — articulation consistency      (15%)
#     Std deviation of per-word probabilities.
#     High std = inconsistent — some words clear, others garbled.
#
#   Signal 4 — no_speech_prob penalty        (10%)
#     Whisper's estimate that a segment had no intelligible speech.
#     High value during active speech = very unclear pronunciation.
# ─────────────────────────────────────────────────────────────────────────────


def _signal_segment_confidence(segments: list) -> tuple:
    """
    Returns (avg_confidence, worst_confidence) — both 0 to 1.
    Derived from avg_logprob per Whisper segment.
    """
    if not segments:
        return 0.75, 0.75

    logprobs = [s["avg_logprob"] for s in segments if "avg_logprob" in s]
    if not logprobs:
        return 0.75, 0.75

    avg_lp   = sum(logprobs) / len(logprobs)
    worst_lp = min(logprobs)

    avg_conf   = max(0.0, min(1.0, 1.0 + avg_lp))
    worst_conf = max(0.0, min(1.0, 1.0 + worst_lp))

    return round(avg_conf, 3), round(worst_conf, 3)


def _signal_word_clarity(words: list) -> tuple:
    """
    Returns (mean_prob, consistency_score) from per-word Whisper probabilities.

    mean_prob        — average confidence across all spoken words (0-1)
    consistency_score — penalises high variance + high weak-word ratio (0-1)
    """
    if not words:
        return 0.80, 0.75

    probs = [
        w["probability"]
        for w in words
        if "probability" in w and w.get("word", "").strip()
    ]

    if not probs:
        return 0.80, 0.75

    mean_p = statistics.mean(probs)
    std_p  = statistics.stdev(probs) if len(probs) > 1 else 0.0

    # Words Whisper was uncertain about (prob < 0.70)
    weak_ratio = sum(1 for p in probs if p < 0.70) / len(probs)

    # Consistency: high mean + low std + few weak words = near 1.0
    consistency = max(0.0, mean_p - std_p * 0.5 - weak_ratio * 0.3)

    return round(mean_p, 3), round(consistency, 3)


def _signal_no_speech_penalty(segments: list) -> float:
    """
    Returns a penalty value (0 to 0.3).
    High no_speech_prob during active speech = unintelligible audio.
    """
    if not segments:
        return 0.0

    ns_probs = [s.get("no_speech_prob", 0.0) for s in segments]
    avg_ns   = sum(ns_probs) / len(ns_probs)
    high_ns  = sum(1 for p in ns_probs if p > 0.4)

    penalty = min(0.3, avg_ns * 0.5 + (high_ns / max(len(ns_probs), 1)) * 0.2)
    return round(penalty, 3)


def evaluate_pronunciation(
    expected_text:    str,         # kept for API compatibility — not used in scoring
    spoken_text:      str,         # kept for API compatibility — not used in scoring
    whisper_segments: list = None,
    whisper_words:    list = None,
) -> dict:
    """
    Score pronunciation 0-2 using only Whisper's audio confidence signals.
    expected_text and spoken_text are intentionally ignored — word matching
    is not appropriate for open-ended interview questions.
    """
    segments = whisper_segments or []
    words    = whisper_words    or []

    seg_conf, worst_conf = _signal_segment_confidence(segments)
    mean_prob, consistency = _signal_word_clarity(words)
    ns_penalty = _signal_no_speech_penalty(segments)

    # Weighted composite (sums to 1.0 before penalty)
    composite = (
        seg_conf    * 0.40 +
        mean_prob   * 0.35 +
        consistency * 0.15
    ) - ns_penalty * 0.10

    composite = round(max(0.0, min(1.0, composite)), 2)

    # Map to 0-2
    if composite >= 0.80:
        score = 2
    elif composite >= 0.60:
        score = 1
    else:
        score = 0

    # Hard cap: one completely unintelligible segment → max score 1
    if worst_conf < 0.40 and score == 2:
        score = 1

    # Note
    if score == 2:
        note = "Speech is clear and understandable"
    elif score == 1:
        if consistency < 0.65:
            note = "Mostly clear with some inconsistent articulation"
        else:
            note = "Understandable with mild accent variation"
    else:
        note = "Pronunciation needs significant improvement"

    return {
        "score":       score,
        "clarity":     seg_conf,      # Whisper segment-level confidence  0-1
        "consistency": consistency,   # per-word articulation consistency 0-1
        "confidence":  composite,     # overall weighted composite        0-1
        "note":        note,
    }