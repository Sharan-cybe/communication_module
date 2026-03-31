from groq import Groq
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


async def transcribe_audio(audio_file) -> dict:
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
            )

        print("Whisper response:", response)

        response_dict = response.model_dump()
        text      = response_dict.get("text", "")
        segments  = response_dict.get("segments", [])

        # ── Extract per-word probabilities from segments ──────────────────────
        # Groq doesn't support timestamp_granularities="word" directly.
        # Instead we extract word-level data from the segment tokens if present,
        # or build synthetic per-word entries from segment avg_logprob so that
        # pronunciation Signal 3 (articulation consistency) still works.
        words = response_dict.get("words", [])
        if not words and segments:
            # Fallback: assign each word the avg_logprob of its segment
            # converted to a probability (0-1) so the signal has data to work on
            for seg in segments:
                seg_prob = max(0.0, min(1.0, 1.0 + seg.get("avg_logprob", -0.3)))
                seg_text = seg.get("text", "")
                for w in seg_text.split():
                    words.append({
                        "word":        w.strip(),
                        "probability": round(seg_prob, 3),
                        "start":       seg.get("start", 0),
                        "end":         seg.get("end", 0),
                    })

        return {
            "text":       text,
            "timestamps": segments,  # segment-level (fluency + pronunciation S1)
            "words":      words,     # word-level    (pronunciation S2+S3)
        }

    except Exception as e:
        print("WHISPER ERROR:", e)
        return {"text": "", "timestamps": [], "words": []}

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)