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

        text     = response_dict.get("text", "").strip()
        segments = response_dict.get("segments", [])

        # ── No-speech filter ──────────────────────────────────────────────────
        # Whisper hallucinates filler phrases ("Thank you.", "you", ".")
        # on silence or background noise. Guard against this using two signals:
        #
        #   1. avg no_speech_prob across ALL segments > 0.5
        #      → Whisper itself says there was likely no speech
        #
        #   2. ALL segments individually have no_speech_prob > 0.5
        #      → every segment is noise, even if the average looks borderline
        #
        # Either condition alone is enough to treat the audio as silent.

        KNOWN_HALLUCINATIONS = {
            "thank you.", "you", ".", "thanks", "thank you",
            "bye", "bye.", "goodbye", "goodbye.", "yes", "yes.",
            "no", "no.", "okay", "okay.", "ok", "ok.",
        }

        if segments:
            avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
            all_no_speech = all(s.get("no_speech_prob", 0) > 0.5 for s in segments)

            if avg_no_speech > 0.5 or all_no_speech:
                print(f"WHISPER FILTER: no_speech_prob={avg_no_speech:.2f} → treating as silence")
                return {"text": "", "timestamps": [], "words": []}

        # ── Hallucination phrase filter ───────────────────────────────────────
        # Even if no_speech_prob is borderline, catch known filler phrases
        # that Whisper commonly hallucinates on short silent clips
        if text.lower() in KNOWN_HALLUCINATIONS:
            print(f"WHISPER FILTER: hallucinated phrase detected → '{text}' → treating as silence")
            return {"text": "", "timestamps": [], "words": []}

        # ── Word-level probability fallback ───────────────────────────────────
        words = response_dict.get("words", [])
        if not words and segments:
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
            "timestamps": segments,
            "words":      words,
        }

    except Exception as e:
        print("WHISPER ERROR:", e)
        return {"text": "", "timestamps": [], "words": []}

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)