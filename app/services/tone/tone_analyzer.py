import librosa
import numpy as np
import tempfile
import os
from pydub import AudioSegment


# ─── Scoring thresholds ───────────────────────────────────────────────────────
#
# Original thresholds (pitch_std < 15 → 0, < 40 → 1, else → 2) were calibrated
# for US speakers.
#
# Indian English speakers naturally have HIGHER pitch variation (~10-15% more
# than US speakers) due to tonal stress patterns from South Asian languages.
# Thresholds are adjusted upward accordingly.
#
# Additionally, we now score 4 sub-dimensions:
#   1. pitch_variation      — tonal expressiveness
#   2. energy_variation     — loudness dynamics (confidence indicator)
#   3. speech_rate_variation— pace variation (monotone = uniform rate)
#   4. voiced_ratio         — ratio of voiced frames (too low = breathy/nervous)
#
# ─────────────────────────────────────────────────────────────────────────────

def _load_clean_audio(audio_file) -> tuple:
    """Save, convert to 16kHz mono, and load with librosa. Returns (y, sr, paths)."""
    raw_path = None
    clean_path = None

    audio_file.file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_file.file.read())
        raw_path = f.name

    clean_path = raw_path + "_clean.wav"
    audio = AudioSegment.from_file(raw_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(clean_path, format="wav")

    y, sr = librosa.load(clean_path, sr=16000)
    y = librosa.util.normalize(y)

    return y, sr, raw_path, clean_path


def _extract_pitch_features(y: np.ndarray, sr: int) -> dict:
    """Extract pitch (F0) statistics using pyin."""
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    pitch_vals = f0[~np.isnan(f0)]
    voiced_ratio = np.sum(voiced_flag) / max(len(voiced_flag), 1)

    return {
        "pitch_std": float(np.std(pitch_vals)) if len(pitch_vals) > 0 else 0.0,
        "pitch_mean": float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0,
        "pitch_range": float(np.ptp(pitch_vals)) if len(pitch_vals) > 0 else 0.0,
        "voiced_ratio": float(voiced_ratio),
    }


def _extract_energy_features(y: np.ndarray) -> dict:
    """Extract RMS energy statistics."""
    energy = librosa.feature.rms(y=y)[0]
    return {
        "energy_std": float(np.std(energy)),
        "energy_mean": float(np.mean(energy)),
        "energy_cv": float(np.std(energy) / max(np.mean(energy), 1e-6)),  # coeff of variation
    }


def _extract_rate_variation(y: np.ndarray, sr: int) -> dict:
    """
    Estimate speech rate variation using onset strength envelope.
    High std of onset envelope → varied speaking rate (natural prosody).
    Low std → monotone / robotic delivery.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return {
        "onset_std": float(np.std(onset_env)),
        "onset_mean": float(np.mean(onset_env)),
    }


def _score_tone(pitch_std: float, energy_cv: float, onset_std: float, voiced_ratio: float) -> int:
    """
    Multi-dimensional tone scoring (0-2).

    Indian-accent calibrated thresholds:
    ─────────────────────────────────────
    pitch_std:
      • < 20 Hz  → monotone (0)
      • 20-55 Hz → moderate variation (1)
      • > 55 Hz  → expressive (2)
      (Original was < 15/< 40 — raised for Indian speech patterns)

    energy_cv (coefficient of variation):
      • < 0.3  → flat/uniform delivery
      • 0.3-0.6 → good dynamics
      • > 0.6  → too variable or noisy

    onset_std:
      • < 1.0  → uniform rate (monotone)
      • > 1.0  → natural rate variation

    voiced_ratio:
      • < 0.3  → too much silence / breathy
      • > 0.3  → engaged speaker
    """
    # Pitch sub-score
    if pitch_std < 20:
        pitch_score = 0
    elif pitch_std < 55:
        pitch_score = 1
    else:
        pitch_score = 2

    # Energy dynamics bonus
    if 0.3 <= energy_cv <= 0.7:
        energy_bonus = 1
    else:
        energy_bonus = 0

    # Rate variation bonus
    rate_bonus = 1 if onset_std > 1.0 else 0

    # Voiced ratio: penalize very low (candidate barely speaking)
    voice_penalty = 1 if voiced_ratio < 0.25 else 0

    # Weighted combination
    raw = pitch_score + (energy_bonus * 0.5) + (rate_bonus * 0.5) - voice_penalty
    return max(0, min(2, round(raw)))


def analyze_tone(audio_file) -> dict:
    """
    Enhanced tone analyzer.

    New in this version:
    ─────────────────────
    1. Multi-dimensional analysis: pitch + energy + rate variation + voicing
    2. Indian-accent calibrated thresholds (pitch_std raised ~15 Hz)
    3. energy_cv (coefficient of variation) instead of raw energy_std
    4. voiced_ratio: penalizes candidates who barely speak
    5. Richer return payload
    """
    raw_path = clean_path = None

    try:
        y, sr, raw_path, clean_path = _load_clean_audio(audio_file)

        if len(y) == 0:
            raise ValueError("Empty audio after conversion")

        pitch = _extract_pitch_features(y, sr)
        energy = _extract_energy_features(y)
        rate = _extract_rate_variation(y, sr)

        score = _score_tone(
            pitch_std=pitch["pitch_std"],
            energy_cv=energy["energy_cv"],
            onset_std=rate["onset_std"],
            voiced_ratio=pitch["voiced_ratio"],
        )

        return {
            "score": score,
            "pitch": {
                "std_hz": round(pitch["pitch_std"], 2),
                "mean_hz": round(pitch["pitch_mean"], 2),
                "range_hz": round(pitch["pitch_range"], 2),
            },
            "energy": {
                "variation_cv": round(energy["energy_cv"], 3),
                "mean_rms": round(energy["energy_mean"], 4),
            },
            "rate_variation": round(rate["onset_std"], 3),
            "voiced_ratio": round(pitch["voiced_ratio"], 3),
        }

    except Exception as e:
        print(f"TONE ERROR: {e}")
        return {"score": 1, "error": str(e)}

    finally:
        for path in [raw_path, clean_path]:
            if path and os.path.exists(path):
                os.remove(path)