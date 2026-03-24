from groq import Groq
from openai import OpenAI
import tempfile
import os
import time
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def transcribe_audio(audio_file):
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
                response_format="verbose_json"
            )

        print("Whisper response:", response)

        text = response.text

        # timestamps (may or may not exist)
        response_dict = response.model_dump()
        timestamps = response_dict.get("segments", [])

        return {
            "text": text,
            "timestamps": timestamps
        }

    except Exception as e:
        print("WHISPER ERROR:", e)
        return {"text": "", "timestamps": []}

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)