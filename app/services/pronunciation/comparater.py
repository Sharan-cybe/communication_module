def phoneme_edit_distance(p1: list, p2: list) -> int:
    """Standard dynamic-programming Levenshtein on phoneme lists."""
    m, n = len(p1), len(p2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p1[i - 1] == p2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[-1][-1]


def compare_phonemes(expected: list, spoken: list) -> dict:
    """
    Compare expected vs spoken phoneme sequences and return a rich result dict.

    Changes from v1:
    ─────────────────
    • Returns a dict (not just a float) with extra diagnostic fields.
    • Uses length-normalized error rate capped at 1.0.
    • Adds a length ratio penalty: if the spoken sequence is much shorter
      than expected (candidate spoke too little), the score is penalized
      proportionally — pure silence or very short answers don't get a free pass.
    • Adds `phoneme_match_rate` for display in the response payload.
    """
    distance = phoneme_edit_distance(expected, spoken)

    max_len = max(len(expected), 1)
    raw_error_rate = distance / max_len

    # Length-ratio penalty: if spoken is less than 50% of expected length,
    # add a proportional penalty (max +0.3 on top of edit distance error)
    len_ratio = len(spoken) / max(len(expected), 1)
    if len_ratio < 0.5:
        length_penalty = (0.5 - len_ratio) * 0.6   # up to +0.3 penalty
    else:
        length_penalty = 0.0

    final_error_rate = min(raw_error_rate + length_penalty, 1.0)
    match_rate = round(1.0 - final_error_rate, 3)

    return {
        "error_rate": round(final_error_rate, 3),
        "phoneme_match_rate": match_rate,
        "raw_edit_distance": distance,
        "expected_len": len(expected),
        "spoken_len": len(spoken),
    }