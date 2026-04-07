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
                temperature=0.0,
                language="en",
                prompt="Umm, let me think... like, uh... this is a verbatim transcript. Include all filler words like um, uh, hmm, ah, and er. Do not hallucinate 'Thanks for watching' or 'Thank you.'",
            )

        print("Whisper response:", response)

        response_dict = response.model_dump()
        text      = response_dict.get("text", "").strip()
        segments  = response_dict.get("segments", [])

        # ── Hallucination & No-Speech Filter ──────────────────────────────────
        # Whisper often hallucinates on background noise or silence.
        
        KNOWN_HALLUCINATIONS = {
            "thank you.", "you", ".", "thanks", "thank you",
            "bye", "bye.", "goodbye", "goodbye.", "yes", "yes.",
            "no", "no.", "okay", "okay.", "ok", "ok.",
            "thanks for watching", "thanks for watching.", "thank you for watching.",
            "thank you for watching", "please subscribe", "subscribe."
        }
        
        if segments:
            # Signal 1: High no_speech probability across the board
            avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
            all_no_speech = all(s.get("no_speech_prob", 0) > 0.5 for s in segments)
            
            # Signal 2: Extremely poor log probability (means the model is very uncertain)
            avg_logprob = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
            
            # If the model is confident there's no speech OR uncertain of the generation
            if avg_no_speech > 0.45 or all_no_speech or (avg_logprob < -1.0 and avg_no_speech > 0.3):
                print(f"WHISPER FILTER: no_speech_prob={avg_no_speech:.2f}, logprob={avg_logprob:.2f} → treating as silence")
                return {"text": "", "timestamps": [], "words": []}

        if text.lower().strip() in KNOWN_HALLUCINATIONS:
            print(f"WHISPER FILTER: hallucinated phrase detected → '{text}' → treating as silence")
            return {"text": "", "timestamps": [], "words": []}

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