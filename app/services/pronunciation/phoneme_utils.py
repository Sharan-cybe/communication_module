import pronouncing

# ─── Indian-accent phoneme normalization ───────────────────────────────────────
#
# Indian English has well-documented systematic differences from US English:
#   • Retroflex stops: /t/ and /d/ are often retroflex (written T, D)
#   • No /æ/ – replaced by /a/ or /ɛ/ (so "cat" sounds like "cut")
#   • /v/ and /w/ merge (both realized as /v/)
#   • /ɒ/ → /o/ (British "lot" vowel doesn't exist in Indian English)
#   • Vowel length distinction lost for short /ɪ/ vs long /iː/
#   • /θ/ and /ð/ → /t/ and /d/ (no dental fricatives)
#   • Schwa (/ə/) → /ʌ/ or /a/ (no reduction in unstressed syllables)
#
# Strategy: normalize BOTH expected and spoken phonemes through the same table,
# so that an Indian speaker saying "t" for /θ/ is NOT penalized.
# ──────────────────────────────────────────────────────────────────────────────

INDIAN_ACCENT_NORMALIZATION = {
    # Dental fricatives → stops (most common Indian accent substitution)
    "TH":  "T",   # voiceless "th" → /t/
    "DH":  "D",   # voiced "th"   → /d/

    # Vowel normalizations
    "AE":  "EH",  # /æ/ (trap) → /ɛ/ (dress)
    "AA":  "AH",  # /ɑ/ (lot)  → /ʌ/ (strut)
    "AO":  "AH",  # /ɔ/ (thought) → /ʌ/

    # /w/ and /v/ merge → use V for both
    "W":   "V",

    # Vowel length collapse (short/long pairs treated the same)
    "IY":  "IH",  # /iː/ (fleece) → /ɪ/ (kit)
    "UW":  "UH",  # /uː/ (goose)  → /ʊ/ (foot)

    # Rhotic vowels: Indian English is often non-rhotic in some dialects
    # but fully rhotic in others – normalize ER to AH to be safe
    "ER":  "AH",

    # Stress markers: strip them (Indian English has different stress patterns)
    # These are handled in get_expected_phonemes() by stripping digits
}


def _normalize_phoneme(p: str) -> str:
    """Strip stress digits, then apply Indian accent normalization table."""
    # CMU phones carry stress: AH0, AH1, AH2 → strip the digit
    base = p.rstrip("012")
    return INDIAN_ACCENT_NORMALIZATION.get(base, base)


def get_expected_phonemes(text: str) -> list[str]:
    """
    Convert expected (reference) text to a normalized phoneme list.
    Uses CMU Pronouncing Dictionary.  Stress markers and Indian-accent
    variants are normalized so the comparison is accent-tolerant.
    """
    words = text.lower().split()
    phonemes = []

    for word in words:
        phones = pronouncing.phones_for_word(word)
        if phones:
            raw = phones[0].split()
            phonemes.extend(_normalize_phoneme(p) for p in raw)
        else:
            # Unknown word: use character-level fallback so it's still
            # comparable (better than a hard UNK that always adds cost)
            phonemes.append(word.upper()[:4])   # max 4-char stub

    return phonemes