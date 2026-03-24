import librosa
import tempfile
import os

def analyze_fluency(transcript, timestamps, audio_file=None):
    temp_path = None

    try:
        words = transcript.split()
        total_words = len(words)

        if timestamps:
            duration = timestamps[-1]["end"]
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                temp.write(audio_file.file.read())
                temp_path = temp.name

            y, sr = librosa.load(temp_path)
            duration = librosa.get_duration(y=y, sr=sr)

        wpm = (total_words / duration) * 60 if duration > 0 else 0

        if wpm < 90:
            score = 0
        elif wpm < 140:
            score = 1
        else:
            score = 2

        return {
            "score": score,
            "wpm": round(wpm, 2),
            "duration": round(duration, 2)
        }

    except Exception as e:
        print("FLUENCY ERROR:", e)
        return {"score": 1}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)