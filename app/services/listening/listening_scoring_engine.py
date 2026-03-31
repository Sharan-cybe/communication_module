"""
listening_scoring_engine.py  v2
────────────────────────────────
Fix: normalise weights only over parameters that actually have data.

Bug that caused 0/10 overall:
  - sentence_reconstruction only appears in REPEAT clips
  - If a session's REPEAT clips had errors, those params were missing
  - used_w was still counted as 1.0 (full weight sum) → denominator wrong
  - normalized = tiny_sum / (1.0 * 2.0) * 10 → near 0

Fix: used_w = sum of weights for params that actually have scores.
  If only accuracy + retention available: used_w = 0.35 + 0.30 = 0.65
  normalized = sum / (0.65 * 2.0) * 10 → correct proportional score
"""

PARAM_WEIGHTS = {
    "listening_accuracy":      0.33,
    "retention":               0.33,
    "sentence_reconstruction": 0.34,
}
MAX_SUB = 2.0


def _verdict(score_10: int) -> str:
    if score_10 >= 9:   return "Excellent listening and comprehension skills"
    elif score_10 >= 7: return "Good listening ability with minor gaps"
    elif score_10 >= 5: return "Moderate listening — several areas to improve"
    elif score_10 >= 3: return "Below average listening comprehension"
    else:               return "Significant listening difficulties identified"


def _strengths(avgs: dict) -> list:
    out = []
    if avgs.get("listening_accuracy", 0) >= 1.5:
        out.append("Accurately captures spoken content and key details")
    elif avgs.get("listening_accuracy", 0) >= 1.0:
        out.append("Generally captures key information from audio")
    if avgs.get("retention", 0) >= 1.5:
        out.append("Strong ability to retain and recall full sentences")
    elif avgs.get("retention", 0) >= 1.0:
        out.append("Retains most of the spoken content")
    if avgs.get("sentence_reconstruction", 0) >= 1.5:
        out.append("Maintains accurate sentence structure when repeating")
    return out[:4] if out else ["Attempted all listening tasks"]


def _improvements(avgs: dict) -> list:
    out = []
    if avgs.get("listening_accuracy", 2) < 1.0:
        out.append("Focus on capturing key words, numbers, and names accurately")
    elif avgs.get("listening_accuracy", 2) < 1.5:
        out.append("Pay closer attention to specific details in the audio")
    if avgs.get("retention", 2) < 1.0:
        out.append("Practise recalling complete sentences rather than fragments")
    elif avgs.get("retention", 2) < 1.5:
        out.append("Work on retaining the full content of longer passages")
    if avgs.get("sentence_reconstruction", 2) < 1.0:
        out.append("Maintain correct word order when reconstructing sentences")
    return out[:4] if out else ["Continue practising listening exercises"]


def _safe_score(val) -> float | None:
    """Extract a numeric score safely from a parameter result dict."""
    if not isinstance(val, dict):
        return None
    raw = val.get("score")
    if raw is None:
        return None
    try:
        return max(0.0, min(2.0, float(raw)))
    except (TypeError, ValueError):
        return None


def aggregate_listening_scores(clip_results: list) -> dict:
    """
    Aggregate scores across all clip results.

    Key fix: used_w tracks only the weights of parameters that have actual
    data, so the normalisation denominator is always correct regardless of
    which clip types responded successfully.
    """
    param_scores: dict[str, list] = {p: [] for p in PARAM_WEIGHTS}

    for clip in clip_results:
        if not isinstance(clip, dict) or "error" in clip:
            continue
        for param in param_scores:
            s = _safe_score(clip.get(param))
            if s is not None:
                param_scores[param].append(s)

    avgs: dict[str, float] = {}
    for param, scores in param_scores.items():
        if scores:
            avgs[param] = round(sum(scores) / len(scores), 3)

    # ── Weighted sum with dynamic denominator ─────────────────────────────────
    weighted_sum = 0.0
    used_w       = 0.0
    for param, weight in PARAM_WEIGHTS.items():
        if param in avgs:
            weighted_sum += avgs[param] * weight
            used_w       += weight          # only count weights with actual data

    if used_w == 0:
        return {
            "listening_score":    0.0,
            "listening_score_10": 0,
            "summary": {"verdict": "No data", "strengths": [], "improvements": []},
            "parameters": {},
            "clip_details": clip_results,
        }

    # Normalise over the weights actually used (not full 1.0)
    normalized = (weighted_sum / (used_w * MAX_SUB)) * 10
    score_10   = int(round(normalized))

    param_summary = {}
    for param, scores in param_scores.items():
        if scores:
            param_summary[param] = {
                "avg_score":   round(sum(scores) / len(scores), 2),
                "clip_scores": [round(s, 2) for s in scores],
            }

    return {
        "listening_score":    round(weighted_sum, 2),
        "listening_score_10": score_10,
        "summary": {
            "verdict":      _verdict(score_10),
            "strengths":    _strengths(avgs),
            "improvements": _improvements(avgs),
        },
        "parameters":  param_summary,
        "clip_details": clip_results,
    }