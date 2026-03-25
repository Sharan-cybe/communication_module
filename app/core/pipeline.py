from app.services.speech.whisper_service import transcribe_audio
from app.services.fluency.fluency_analyzer import analyze_fluency
from app.services.tone.tone_analyzer import analyze_tone
from app.services.grammar.llama_service import evaluate_grammar
from app.services.comprehension.comprehension_service import evaluate_comprehension
from app.services.pronunciation.pronunciation_service import evaluate_pronunciation
from app.core.scoring_engine import aggregate_scores


async def run_pipeline(audio_file, question: str) -> dict:
    """
    Main evaluation pipeline.

    Each module is wrapped in its own try/except so a failure in one
    module (e.g. eSpeak not installed) does not crash the entire pipeline.
    Failed modules return a neutral score of 1 with an error note.
    """

    # ── Transcription ─────────────────────────────────────────────────────────
    audio_file.file.seek(0)
    transcript_data = transcribe_audio(audio_file)
    transcript = transcript_data.get("text", "")
    timestamps = transcript_data.get("timestamps", [])
    print(f"TRANSCRIPT: {transcript[:120]}...")

    # ── Pronunciation ─────────────────────────────────────────────────────────
    try:
        audio_file.file.seek(0)
        pronunciation = evaluate_pronunciation(
            expected_text=question,
            spoken_text=transcript,
        )
    except Exception as e:
        print(f"PRONUNCIATION PIPELINE ERROR: {e}")
        pronunciation = {"score": 1, "error": str(e)}
    print(f"PRONUNCIATION: {pronunciation}")

    # ── Fluency ───────────────────────────────────────────────────────────────
    try:
        audio_file.file.seek(0)
        fluency = analyze_fluency(transcript, timestamps, audio_file)
    except Exception as e:
        print(f"FLUENCY PIPELINE ERROR: {e}")
        fluency = {"score": 1, "error": str(e)}

    # ── Tone ──────────────────────────────────────────────────────────────────
    try:
        audio_file.file.seek(0)
        tone = analyze_tone(audio_file)
    except Exception as e:
        print(f"TONE PIPELINE ERROR: {e}")
        tone = {"score": 1, "error": str(e)}

    # ── Grammar ───────────────────────────────────────────────────────────────
    try:
        grammar = evaluate_grammar(transcript)
    except Exception as e:
        print(f"GRAMMAR PIPELINE ERROR: {e}")
        grammar = {"score": 1, "error": str(e)}

    # ── Comprehension ─────────────────────────────────────────────────────────
    try:
        comprehension = evaluate_comprehension(question, transcript)
    except Exception as e:
        print(f"COMPREHENSION PIPELINE ERROR: {e}")
        comprehension = {"score": 1, "error": str(e)}

    # ── Aggregate ─────────────────────────────────────────────────────────────
    final = aggregate_scores(
        pronunciation=pronunciation,
        fluency=fluency,
        tone=tone,
        grammar=grammar,
        comprehension=comprehension,
    )

    # Attach transcript to response for frontend display
    final["transcript"] = transcript

    return final