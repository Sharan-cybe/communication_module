def aggregate_scores(**kwargs):

    weights = {
        "fluency": 0.2,
        "tone": 0.2,
        "grammar": 0.3,
        "comprehension": 0.3
    }

    total = 0

    for key, value in kwargs.items():
        score = value.get("score", 1)
        total += score * weights.get(key, 0)

    return {
        "final_score": round(total, 2),
        "details": kwargs
    }