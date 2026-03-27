import librosa
import numpy as np
import tempfile
import os
from pydub import AudioSegment


def _load_clean_audio(audio_file):
    audio_file.file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_file.file.read())
        raw = f.name
    clean = raw + "_clean.wav"
    seg   = AudioSegment.from_file(raw).set_frame_rate(16000).set_channels(1)
    seg.export(clean, format="wav")
    y, sr = librosa.load(clean, sr=16000)
    return librosa.util.normalize(y), sr, raw, clean


def analyze_tone(audio_file) -> dict:
    raw = clean = None
    try:
        y, sr, raw, clean = _load_clean_audio(audio_file)

        if len(y) == 0:
            raise ValueError("Empty audio")

        # Pitch
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        pitch_vals    = f0[~np.isnan(f0)]
        pitch_std     = float(np.std(pitch_vals))   if len(pitch_vals) > 0 else 0.0

        # Energy
        energy        = librosa.feature.rms(y=y)[0]
        energy_cv     = float(np.std(energy) / max(np.mean(energy), 1e-6))

        # Rate variation
        onset_std     = float(np.std(librosa.onset.onset_strength(y=y, sr=sr)))

        # Voiced ratio
        voiced_ratio  = float(np.sum(voiced_flag) / max(len(voiced_flag), 1))

        # Score
        pitch_score   = 0 if pitch_std < 20 else (1 if pitch_std < 55 else 2)
        energy_bonus  = 1 if 0.3 <= energy_cv <= 0.7 else 0
        rate_bonus    = 1 if onset_std > 1.0 else 0
        voice_penalty = 1 if voiced_ratio < 0.25 else 0

        raw_score = pitch_score + energy_bonus * 0.5 + rate_bonus * 0.5 - voice_penalty
        score     = max(0, min(2, round(raw_score)))

        if score == 2:
            note = "Good vocal variation and engagement"
        elif score == 1:
            if pitch_std < 20:
                note = "Voice sounds monotone — try varying your pitch"
            else:
                note = "Moderate vocal expression — can be more engaging"
        else:
            note = "Tone is flat and lacks energy"

        return {
            "score":            score,
            "pitch_variation":  round(pitch_std, 2),
            "energy_variation": round(energy_cv, 2),
            "note":             note,
        }

    except Exception as e:
        print(f"TONE ERROR: {e}")
        return {"score": 1, "pitch_variation": 0.0, "energy_variation": 0.0,
                "note": "Could not evaluate tone"}
    finally:
        for p in [raw, clean]:
            if p and os.path.exists(p):
                os.remove(p)