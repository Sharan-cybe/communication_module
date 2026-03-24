import librosa
import numpy as np
import tempfile
import os
from pydub import AudioSegment


def analyze_tone(audio_file):
    raw_path = None
    clean_path = None

    try:
        audio_file.file.seek(0)

        # 🔴 Save raw file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(audio_file.file.read())
            raw_path = temp.name

        # ✅ Convert to clean WAV (16kHz mono)
        clean_path = raw_path + "_clean.wav"

        audio = AudioSegment.from_file(raw_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(clean_path, format="wav")

        # ✅ Load clean audio
        y, sr = librosa.load(clean_path, sr=16000)

        if len(y) == 0:
            raise Exception("Empty audio after conversion")

        # Normalize
        y = librosa.util.normalize(y)

        # 🎵 Pitch
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )

        pitch_vals = f0[~np.isnan(f0)]
        energy = librosa.feature.rms(y=y)[0]

        pitch_std = np.std(pitch_vals) if len(pitch_vals) > 0 else 0
        energy_std = np.std(energy) if len(energy) > 0 else 0

        # 🎯 Scoring
        if pitch_std < 15:
            score = 0
        elif pitch_std < 40:
            score = 1
        else:
            score = 2

        return {
            "score": score,
            "pitch_variation": float(pitch_std),
            "energy_variation": float(energy_std)
        }

    except Exception as e:
        print("TONE ERROR:", str(e))
        return {"score": 1, "error": str(e)}

    finally:
        for path in [raw_path, clean_path]:
            if path and os.path.exists(path):
                os.remove(path)