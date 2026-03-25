from app.services.pronunciation.phoneme_utils import get_expected_phonemes
from app.services.pronunciation.phonemizer_utils import get_spoken_phonemes
from app.services.pronunciation.comparater import compare_phonemes
from app.services.pronunciation.scorer import score_pronunciation


def evaluate_pronunciation(expected_text: str, spoken_text: str) -> dict:
    """
    Evaluate pronunciation with Indian-accent aware phoneme comparison.

    Returns
    ───────
    score           : 0-2
    error_rate      : normalized edit distance (accent-normalized)
    phoneme_match_rate: 1 - error_rate (easier to read)
    expected_phonemes : first 10 normalized reference phones
    spoken_phonemes   : first 10 normalized spoken phones
    """
    if not expected_text or not spoken_text:
        return {
            "score": 1,
            "error_rate": 0.5,
            "phoneme_match_rate": 0.5,
            "expected_phonemes": [],
            "spoken_phonemes": [],
            "note": "Missing input text",
        }

    expected_phonemes = get_expected_phonemes(expected_text)
    spoken_phonemes = get_spoken_phonemes(spoken_text)

    comparison = compare_phonemes(expected_phonemes, spoken_phonemes)
    error_rate = comparison["error_rate"]

    score = score_pronunciation(error_rate)

    return {
        "score": score,
        "error_rate": error_rate,
        "phoneme_match_rate": comparison["phoneme_match_rate"],
        "expected_phonemes": expected_phonemes[:10],
        "spoken_phonemes": spoken_phonemes[:10],
        "stats": {
            "expected_len": comparison["expected_len"],
            "spoken_len": comparison["spoken_len"],
            "raw_edit_distance": comparison["raw_edit_distance"],
        },
    }