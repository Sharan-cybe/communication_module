WEIGHTS = {
    "pronunciation": 0.20,
    "fluency":       0.20,
    "tone":          0.20,
    "grammar":       0.20,
    "comprehension": 0.20,
}

MAX_SUB = 2.0


# ── Verdict ───────────────────────────────────────────────────────────────────

def _verdict(score_10: float) -> str:
    if score_10 >= 8.5:
        return "Excellent communication skills"
    elif score_10 >= 7.0:
        return "Good communication with minor improvements needed"
    elif score_10 >= 5.5:
        return "Moderate communication — several areas to improve"
    elif score_10 >= 4.0:
        return "Below average — focused improvement required"
    else:
        return "Significant communication challenges identified"


# ── Strengths ─────────────────────────────────────────────────────────────────

def _strengths(details: dict) -> list:
    out = []

    p = details.get("pronunciation", {})
    if p.get("score", 0) >= 2:
        out.append("Clear pronunciation")
    elif p.get("score", 0) == 1 and p.get("clarity", 0) >= 0.75:
        out.append("Generally clear pronunciation")

    f = details.get("fluency", {})
    if f.get("score", 0) >= 2:
        wpm = f.get("wpm", 0)
        if 120 <= wpm <= 155:
            out.append("Fluent speaking pace")
        else:
            out.append("Natural speaking rhythm")
    if f.get("score", 0) >= 1 and f.get("filler_rate", 99) < 3:
        out.append("Minimal use of filler words")

    t = details.get("tone", {})
    if t.get("score", 0) >= 2:
        out.append("Good energy and engagement")
    elif t.get("score", 0) == 1 and t.get("pitch_variation", 0) >= 20:
        out.append("Some vocal variety present")

    g = details.get("grammar", {})
    if g.get("score", 0) >= 2:
        out.append("Strong grammar and sentence structure")
    elif g.get("score", 0) == 1 and len(g.get("mistakes", [])) <= 1:
        out.append("Mostly correct grammar")

    c = details.get("comprehension", {})
    if c.get("score", 0) >= 2:
        out.append("Answered the question thoroughly")
    elif c.get("score", 0) == 1:
        out.append("Attempted to address the question")

    return out[:4] if out else ["Attempted all parts of the evaluation"]


# ── Improvements ──────────────────────────────────────────────────────────────

def _improvements(details: dict) -> list:
    out = []

    p = details.get("pronunciation", {})
    if p.get("score", 2) == 0:
        out.append("Work on pronunciation clarity and articulation")
    elif p.get("score", 2) == 1 and p.get("consistency", 1) < 0.65:
        out.append("Improve consistency of articulation across words")

    f = details.get("fluency", {})
    if f.get("score", 2) == 0:
        out.append("Adjust speaking pace — aim for 120–150 WPM")
    elif f.get("score", 2) == 1:
        if f.get("filler_rate", 0) > 5:
            out.append("Reduce filler words (um, uh, basically, so)")
        wpm = f.get("wpm", 130)
        if wpm < 110:
            out.append("Speak with more confidence and pace")
        elif wpm > 165:
            out.append("Slow down slightly for better clarity")
    if f.get("pauses", {}).get("count", 0) > 4:
        out.append("Reduce long pauses between sentences")

    t = details.get("tone", {})
    if t.get("score", 2) == 0:
        out.append("Add more vocal energy and expression")
    elif t.get("score", 2) == 1:
        if t.get("pitch_variation", 99) < 20:
            out.append("Vary pitch to sound less monotone")
        elif t.get("energy_variation", 0) > 0.8:
            out.append("Maintain consistent volume throughout")

    g = details.get("grammar", {})
    mistakes = g.get("mistakes", [])
    if g.get("score", 2) == 0:
        out.append("Focus on basic sentence structure and grammar")
    elif g.get("score", 2) == 1 and mistakes:
        first = mistakes[0]
        if isinstance(first, dict) and first.get("corrected"):
            out.append(f"Fix grammar — e.g. use \"{first['corrected']}\" not \"{first['original']}\"")
        else:
            out.append("Fix minor grammar mistakes")

    c = details.get("comprehension", {})
    if c.get("score", 2) == 0:
        out.append("Make sure to directly answer the question asked")
    elif c.get("score", 2) == 1:
        note = c.get("note", "")
        if note and len(note) < 80:
            out.append(note)
        else:
            out.append("Cover all parts of the question more thoroughly")

    return out[:4] if out else ["Continue practising interview responses"]


# ── Main aggregator ───────────────────────────────────────────────────────────

def aggregate_scores(**kwargs) -> dict:
    weighted_sum  = 0.0
    used_weights  = 0.0

    for key, weight in WEIGHTS.items():
        if key in kwargs:
            raw    = kwargs[key].get("score", 1)
            clamped = max(0.0, min(float(raw), MAX_SUB))
            weighted_sum += clamped * weight
            used_weights += weight

    if used_weights == 0:
        return {"final_score": 0.0, "final_score_10": 0,
                "summary": {"verdict": "No data", "strengths": [], "improvements": []},
                "details": kwargs}

    normalized   = (weighted_sum / (used_weights * MAX_SUB)) * 10
    final_score  = round(weighted_sum, 2)
    score_10     = round(normalized, 1)

    # Build clean details — only the fields matching the target format
    details = {}
    for key in WEIGHTS:
        if key in kwargs:
            details[key] = kwargs[key]

    summary = {
        "verdict":      _verdict(score_10),
        "strengths":    _strengths(details),
        "improvements": _improvements(details),
    }

    return {
        "final_score":    final_score,
        "final_score_10": int(round(score_10)),   # clean integer e.g. 7
        "summary":        summary,
        "details":        details,
    }