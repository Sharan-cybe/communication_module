def score_pronunciation(error_rate: float) -> int:
    """
    Score pronunciation 0-2 with Indian-accent tolerant thresholds.

    Threshold rationale:
    ────────────────────
    • Indian English has systematic phoneme substitutions that are NOT errors
      in spoken English comprehensibility (e.g. /t/ for /θ/).
    • After normalization those are already collapsed, but residual accent
      variation still raises the raw error rate vs a US-English baseline.
    • Thresholds are therefore shifted ~0.1 more lenient than a US-only scorer.

    0 = poor  (error rate > 0.55)  — many unintelligible words
    1 = okay  (error rate 0.30-0.55) — accent present but understandable
    2 = good  (error rate ≤ 0.30)   — clear, well-articulated speech
    """
    if error_rate > 0.55:
        return 0
    elif error_rate > 0.30:
        return 1
    else:
        return 2