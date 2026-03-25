def aggregate_scores(**kwargs) -> dict:
    """
    Aggregate all module scores into a final weighted score.

    All sub-modules return scores on a 0-2 scale.
    Final score is normalized to 0-10 for readability.

    Weights:
    ─────────
    pronunciation  0.20 — accent-tolerant phoneme accuracy
    fluency        0.20 — WPM + fillers + pauses
    tone           0.20 — pitch variation + energy + rate variation
    grammar        0.20 — grammatical accuracy (Indian-English aware)
    comprehension  0.20 — question relevance and completeness
    """
    WEIGHTS = {
        "pronunciation": 0.20,
        "fluency":       0.20,
        "tone":          0.20,
        "grammar":       0.20,
        "comprehension": 0.20,
    }

    MAX_SUB_SCORE = 2.0   # each module scores 0-2

    weighted_sum = 0.0
    used_weights = 0.0

    for key, weight in WEIGHTS.items():
        if key in kwargs:
            raw_score = kwargs[key].get("score", 1)
            # Clamp to valid range
            clamped = max(0.0, min(float(raw_score), MAX_SUB_SCORE))
            weighted_sum += clamped * weight
            used_weights += weight

    if used_weights == 0:
        return {"final_score": 0.0, "final_score_10": 0.0, "details": kwargs}

    # Normalize: max weighted_sum = 2.0 (all 100% weights × max sub-score 2)
    # Scale to 0-10
    normalized = (weighted_sum / (used_weights * MAX_SUB_SCORE)) * 10

    return {
        "final_score": round(weighted_sum, 2),    # 0-2 scale (backward compat)
        "final_score_10": round(normalized, 1),   # 0-10 scale (user-friendly)
        "details": kwargs,
    }