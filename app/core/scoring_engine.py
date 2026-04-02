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
    # Filter out system error notes — don't show backend errors to the user
    SYSTEM_NOTES = {"no answer provided to evaluate", "no speech detected to evaluate",
                    "evaluation failed", "could not evaluate comprehension",
                    "no question provided for comprehension check"}
    c_note = c.get("note", "").lower()
    if c.get("score", 2) == 0:
        if c_note not in SYSTEM_NOTES:
            out.append("Make sure to directly answer the question asked")
    elif c.get("score", 2) == 1:
        raw_note = c.get("note", "")
        if raw_note and raw_note.lower() not in SYSTEM_NOTES and len(raw_note) < 80:
            out.append(raw_note)
        elif raw_note.lower() not in SYSTEM_NOTES:
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

    # Override if it's completely empty audio / no speech
    if final_score == 0.0 and any("No speech detected" in str(kw.get("note", "")) for kw in kwargs.values()):
        summary = {
            "verdict":      "No speech detected",
            "strengths":    [],
            "improvements": ["Please ensure your microphone is working and speak clearly into it."],
        }

    return {
        "final_score":    final_score,
        "final_score_10": int(round(score_10)),   # clean integer e.g. 7
        "summary":        summary,
        "details":        details,
    }


# ── Session aggregator (Speaking) ─────────────────────────────────────────────

def aggregate_speaking_session(results: list) -> dict:
    if not results:
        return {"final_score": 0.0, "final_score_10": 0, "summary": {"verdict": "No data", "strengths": [], "improvements": []}, "details": {}}
        
    param_scores = {
        "pronunciation": {"score": [], "clarity": [], "consistency": []},
        "fluency": {"score": [], "wpm": [], "filler_rate": [], "pauses_count": []},
        "tone": {"score": [], "pitch_variation": [], "energy_variation": []},
        "grammar": {"score": [], "mistakes_count": []},
        "comprehension": {"score": []}
    }
    
    for r in results:
        details = r.get("details", {})
        
        p = details.get("pronunciation", {})
        if "score" in p: param_scores["pronunciation"]["score"].append(p["score"])
        if "clarity" in p: param_scores["pronunciation"]["clarity"].append(p["clarity"])
        if "consistency" in p: param_scores["pronunciation"]["consistency"].append(p["consistency"])
        
        f = details.get("fluency", {})
        if "score" in f: param_scores["fluency"]["score"].append(f["score"])
        if "wpm" in f: param_scores["fluency"]["wpm"].append(f["wpm"])
        if "filler_rate" in f: param_scores["fluency"]["filler_rate"].append(f["filler_rate"])
        if "pauses" in f and isinstance(f["pauses"], dict) and "count" in f["pauses"]: 
            param_scores["fluency"]["pauses_count"].append(f["pauses"]["count"])
        
        t = details.get("tone", {})
        if "score" in t: param_scores["tone"]["score"].append(t["score"])
        if "pitch_variation" in t: param_scores["tone"]["pitch_variation"].append(t["pitch_variation"])
        if "energy_variation" in t: param_scores["tone"]["energy_variation"].append(t["energy_variation"])
        
        g = details.get("grammar", {})
        if "score" in g: param_scores["grammar"]["score"].append(g["score"])
        if "mistakes" in g: param_scores["grammar"]["mistakes_count"].append(len(g["mistakes"]))
        
        c = details.get("comprehension", {})
        if "score" in c: param_scores["comprehension"]["score"].append(c["score"])
        if "note" in c and "notes" not in param_scores["comprehension"]:
            # just keep the first useful note if any
            param_scores["comprehension"]["notes"] = c["note"]
            
    def avg(lst): return sum(lst) / len(lst) if lst else 0
    
    agg_details = {
        "pronunciation": {
            "score": avg(param_scores["pronunciation"]["score"]),
            "clarity": avg(param_scores["pronunciation"]["clarity"]),
            "consistency": avg(param_scores["pronunciation"]["consistency"])
        },
        "fluency": {
            "score": avg(param_scores["fluency"]["score"]),
            "wpm": avg(param_scores["fluency"]["wpm"]),
            "filler_rate": avg(param_scores["fluency"]["filler_rate"]),
            "pauses": {"count": avg(param_scores["fluency"]["pauses_count"])}
        },
        "tone": {
            "score": avg(param_scores["tone"]["score"]),
            "pitch_variation": avg(param_scores["tone"]["pitch_variation"]),
            "energy_variation": avg(param_scores["tone"]["energy_variation"])
        },
        "grammar": {
            "score": avg(param_scores["grammar"]["score"]),
            "mistakes": [{"original": "", "corrected": ""}] if avg(param_scores["grammar"]["mistakes_count"]) > 0 else []
        },
        "comprehension": {
            "score": avg(param_scores["comprehension"]["score"]),
            "note": param_scores["comprehension"].get("notes", "")
        }
    }
    
    return aggregate_scores(**agg_details)