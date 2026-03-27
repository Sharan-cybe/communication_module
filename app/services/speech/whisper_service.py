from groq import Groq
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def transcribe_audio(audio_file) -> dict:
    temp_file_path = None

    try:
        audio_file.file.seek(0)
        data = audio_file.file.read()

        if not data:
            raise Exception("Empty audio file")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(data)
            temp_file_path = temp.name

        with open(temp_file_path, "rb") as f:
            response = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3",
                response_format="verbose_json",
                # ── word_timestamps gives per-word probabilities ──────────────
                # This feeds Signal 3 (articulation consistency) in Option B.
                # Groq supports this on whisper-large-v3.
                timestamp_granularities=["segment", "word"],
            )

        print("Whisper response:", response)

        text = response.text
        response_dict = response.model_dump()

        segments   = response_dict.get("segments", [])
        words      = response_dict.get("words", [])     # per-word data

        return {
            "text":       text,
            "timestamps": segments,   # segment-level (used by fluency + tone)
            "words":      words,      # word-level    (used by pronunciation)
        }

    except Exception as e:
        print("WHISPER ERROR:", e)
        return {"text": "", "timestamps": [], "words": []}

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)