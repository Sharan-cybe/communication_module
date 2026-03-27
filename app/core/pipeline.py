from app.services.speech.whisper_service import transcribe_audio
from app.services.fluency.fluency_analyzer import analyze_fluency
from app.services.tone.tone_analyzer import analyze_tone
from app.services.grammar.llama_service import evaluate_grammar
from app.services.comprehension.comprehension_service import evaluate_comprehension
from app.services.pronunciation.pronunciation_service import evaluate_pronunciation
from app.core.scoring_engine import aggregate_scores


async def run_pipeline(audio_file, question: str) -> dict:

    # ── Transcription ─────────────────────────────────────────────────────────
    audio_file.file.seek(0)
    transcript_data = transcribe_audio(audio_file)
    transcript = transcript_data.get("text", "")
    segments   = transcript_data.get("timestamps", [])
    words      = transcript_data.get("words", [])
    print(f"TRANSCRIPT: {transcript[:120]}...")

    # ── Pronunciation ─────────────────────────────────────────────────────────
    try:
        pronunciation = evaluate_pronunciation(
            expected_text    = question,
            spoken_text      = transcript,
            whisper_segments = segments,
            whisper_words    = words,
        )
    except Exception as e:
        print(f"PRONUNCIATION ERROR: {e}")
        pronunciation = {"score": 1, "clarity": 0.75, "consistency": 0.75,
                         "confidence": 0.75, "note": "Could not evaluate pronunciation"}

    # ── Fluency ───────────────────────────────────────────────────────────────
    try:
        audio_file.file.seek(0)
        fluency = analyze_fluency(transcript, segments, audio_file)
    except Exception as e:
        print(f"FLUENCY ERROR: {e}")
        fluency = {"score": 1, "wpm": 0.0, "filler_rate": 0.0,
                   "pauses": {"count": 0, "avg_duration": 0.0},
                   "note": "Could not evaluate fluency"}

    # ── Tone ──────────────────────────────────────────────────────────────────
    try:
        audio_file.file.seek(0)
        tone = analyze_tone(audio_file)
    except Exception as e:
        print(f"TONE ERROR: {e}")
        tone = {"score": 1, "pitch_variation": 0.0, "energy_variation": 0.0,
                "note": "Could not evaluate tone"}

    # ── Grammar ───────────────────────────────────────────────────────────────
    try:
        grammar = evaluate_grammar(transcript)
    except Exception as e:
        print(f"GRAMMAR ERROR: {e}")
        grammar = {"score": 1, "mistakes": [], "note": "Could not evaluate grammar"}

    # ── Comprehension ─────────────────────────────────────────────────────────
    try:
        comprehension = evaluate_comprehension(question, transcript)
    except Exception as e:
        print(f"COMPREHENSION ERROR: {e}")
        comprehension = {"score": 1, "note": "Could not evaluate comprehension"}

    # ── Aggregate ─────────────────────────────────────────────────────────────
    final = aggregate_scores(
        pronunciation = pronunciation,
        fluency       = fluency,
        tone          = tone,
        grammar       = grammar,
        comprehension = comprehension,
    )
    final["transcript"] = transcript.strip()
    return final