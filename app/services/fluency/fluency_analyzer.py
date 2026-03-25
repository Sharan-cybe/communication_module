import librosa
import numpy as np
import tempfile
import os
import re

# ─── Filler words common in Indian English ────────────────────────────────────
# Indian candidates often use "basically", "actually", "only" as fillers
FILLER_WORDS = {
    "um", "uh", "hmm", "er", "ah",             # universal
    "like", "you know", "so",                   # casual fillers
    "basically", "actually", "only", "itself",  # Indian-English specific
    "na", "no", "right",                        # tag-question fillers
}


def _count_fillers(transcript: str) -> dict:
    """Count filler words/phrases in the transcript."""
    lower = transcript.lower()
    counts = {}
    total = 0
    for filler in FILLER_WORDS:
        # word boundary match
        pattern = r'\b' + re.escape(filler) + r'\b'
        matches = re.findall(pattern, lower)
        if matches:
            counts[filler] = len(matches)
            total += len(matches)
    return {"total": total, "breakdown": counts}


def _analyze_pauses(timestamps: list) -> dict:
    """
    Detect pauses from Whisper word-level timestamps.
    A pause is a gap between consecutive words > 0.5s.
    Returns stats on pause count, total pause time, avg pause duration.
    """
    if not timestamps or len(timestamps) < 2:
        return {"pause_count": 0, "total_pause_time": 0.0, "avg_pause": 0.0, "long_pauses": 0}

    pauses = []
    for i in range(1, len(timestamps)):
        prev_end = timestamps[i - 1].get("end", 0)
        curr_start = timestamps[i].get("start", 0)
        gap = curr_start - prev_end
        if gap > 0.5:   # pauses > 500ms
            pauses.append(gap)

    long_pauses = sum(1 for p in pauses if p > 2.0)  # > 2s = hesitation

    return {
        "pause_count": len(pauses),
        "total_pause_time": round(sum(pauses), 2),
        "avg_pause": round(np.mean(pauses), 2) if pauses else 0.0,
        "long_pauses": long_pauses,
    }


def _get_duration_from_audio(audio_file) -> float:
    """Load audio file and return duration in seconds."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(audio_file.file.read())
        temp_path = temp.name
    try:
        y, sr = librosa.load(temp_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _score_fluency(wpm: float, filler_rate: float, pause_penalty: float) -> int:
    """
    Multi-factor fluency scoring.

    WPM ranges calibrated for Indian English speakers:
      • Indian conversational speech: 100–160 WPM
      • Interview target range: 120–150 WPM
      • Below 100 = too slow / hesitant
      • Above 170 = too fast / unclear

    Filler rate = fillers per 100 words
    Pause penalty = weighted pause score (0-1)
    """
    # WPM sub-score
    if wpm < 90 or wpm > 180:
        wpm_score = 0
    elif wpm < 110 or wpm > 165:
        wpm_score = 1
    else:
        wpm_score = 2

    # Filler penalty: > 5 fillers per 100 words = noticeable
    if filler_rate > 10:
        filler_deduction = 2
    elif filler_rate > 5:
        filler_deduction = 1
    else:
        filler_deduction = 0

    raw = wpm_score - filler_deduction
    # Apply pause penalty
    if pause_penalty > 0.5:
        raw = max(0, raw - 1)

    return max(0, min(2, raw))


def analyze_fluency(transcript: str, timestamps: list, audio_file=None) -> dict:
    """
    Enhanced fluency analyzer.

    New in this version:
    ─────────────────────
    1. Filler word detection (including Indian-English specific fillers)
    2. Pause analysis from Whisper timestamps (pause count, total pause time)
    3. Multi-factor score: WPM + filler rate + pause pattern
    4. Richer return payload for frontend display
    """
    try:
        words = [w for w in transcript.split() if w.strip()]
        total_words = len(words)

        # ── Duration ────────────────────────────────────────────────────────
        if timestamps:
            # Use the last segment's end time
            last_seg = timestamps[-1]
            duration = last_seg.get("end", 0) or last_seg.get("start", 1)
        elif audio_file is not None:
            duration = _get_duration_from_audio(audio_file)
        else:
            duration = 1.0  # fallback

        if duration <= 0:
            duration = 1.0

        # ── WPM ─────────────────────────────────────────────────────────────
        wpm = (total_words / duration) * 60

        # ── Filler words ─────────────────────────────────────────────────────
        filler_data = _count_fillers(transcript)
        filler_rate = (filler_data["total"] / max(total_words, 1)) * 100  # per 100 words

        # ── Pause analysis ───────────────────────────────────────────────────
        pause_data = _analyze_pauses(timestamps)
        # Pause penalty: long hesitation pauses hurt more
        pause_penalty = min(1.0, (pause_data["long_pauses"] * 0.3) + (pause_data["avg_pause"] * 0.1))

        # ── Score ─────────────────────────────────────────────────────────────
        score = _score_fluency(wpm, filler_rate, pause_penalty)

        return {
            "score": score,
            "wpm": round(wpm, 1),
            "duration_secs": round(duration, 1),
            "total_words": total_words,
            "fillers": filler_data,
            "filler_rate_per_100": round(filler_rate, 1),
            "pauses": pause_data,
        }

    except Exception as e:
        print(f"FLUENCY ERROR: {e}")
        return {"score": 1, "error": str(e)}