import asyncio
import io

from app.services.speech.whisper_service import transcribe_audio
from app.services.fluency.fluency_analyzer import analyze_fluency
from app.services.tone.tone_analyzer import analyze_tone
from app.services.grammar.llama_service import evaluate_grammar
from app.services.comprehension.comprehension_service import evaluate_comprehension
from app.services.pronunciation.pronunciation_service import evaluate_pronunciation
from app.core.scoring_engine import aggregate_scores


class BufferedFile:
    def __init__(self, data: bytes):
        self.file = io.BytesIO(data)


async def run_pipeline(audio_file, question: str) -> dict:

    # ── Transcription ─────────────────────────────────────────────────────────
    audio_file.file.seek(0)
    audio_bytes = audio_file.file.read()
    audio_file.file.seek(0)
    
    transcript_data = await transcribe_audio(audio_file)
    transcript = transcript_data.get("text", "").strip()
    segments   = transcript_data.get("timestamps", [])
    words      = transcript_data.get("words", [])

    print(f"TRANSCRIPT ({len(transcript)} chars): {transcript[:120]}...")
    print(f"SEGMENTS: {len(segments)} | WORDS: {len(words)}")

    # Guard: if Whisper returned nothing, log clearly and return early
    if not transcript:
        print("WARNING: Whisper returned empty transcript — audio may be silent or too short")

    def _run_sync_evals():
        if not transcript or len(transcript.strip()) < 2:
            return aggregate_scores(
                pronunciation = {"score": 0, "clarity": 0.0, "consistency": 0.0, "note": "No speech detected to evaluate"},
                fluency       = {"score": 0, "wpm": 0.0, "filler_rate": 0.0, "pauses": {"count": 0, "avg_duration": 0.0}, "note": "No speech detected to evaluate"},
                tone          = {"score": 0, "pitch_variation": 0.0, "energy_variation": 0.0, "note": "No speech detected to evaluate"},
                grammar       = {"score": 0, "mistakes": [], "note": "No speech detected to evaluate"},
                comprehension = {"score": 0, "note": "No speech detected to evaluate"},
            )

        local_audio_fluency = BufferedFile(audio_bytes)
        local_audio_tone    = BufferedFile(audio_bytes)

        # ── Pronunciation ─────────────────────────────────────────────────────────
        try:
            pronunciation = evaluate_pronunciation(
                expected_text    = question,
                spoken_text      = transcript,
                whisper_segments = segments,
                whisper_words    = words,
            )
            print(f"PRONUNCIATION: score={pronunciation['score']} composite={pronunciation['composite_score']}")
        except Exception as e:
            print(f"PRONUNCIATION ERROR: {e}")
            pronunciation = {"score": 0, "clarity": 0.0, "consistency": 0.0,
                             "composite_score": 0.0, "note": "Could not evaluate pronunciation"}

        # ── Fluency ───────────────────────────────────────────────────────────────
        try:
            fluency = analyze_fluency(transcript, segments, local_audio_fluency)
            print(f"FLUENCY: score={fluency['score']} wpm={fluency.get('wpm')}")
        except Exception as e:
            print(f"FLUENCY ERROR: {e}")
            fluency = {"score": 0, "wpm": 0.0, "filler_rate": 0.0,
                       "pauses": {"count": 0, "avg_duration": 0.0},
                       "note": "Could not evaluate fluency"}

        # ── Tone ──────────────────────────────────────────────────────────────────
        try:
            tone = analyze_tone(local_audio_tone)
            print(f"TONE: score={tone['score']} pitch_std={tone.get('pitch_variation')}")
        except Exception as e:
            print(f"TONE ERROR: {e}")
            tone = {"score": 0, "pitch_variation": 0.0, "energy_variation": 0.0,
                    "note": "Could not evaluate tone"}

        # ── Grammar ───────────────────────────────────────────────────────────────
        try:
            grammar = evaluate_grammar(transcript)
            print(f"GRAMMAR: score={grammar['score']} mistakes={len(grammar.get('mistakes', []))}")
        except Exception as e:
            print(f"GRAMMAR ERROR: {e}")
            grammar = {"score": 0, "mistakes": [], "note": "Could not evaluate grammar"}

        # ── Comprehension ─────────────────────────────────────────────────────────
        try:
            if not question or not question.strip():
                comprehension = {"score": 0, "note": "No question provided for comprehension check"}
            else:
                comprehension = evaluate_comprehension(question, transcript)
            print(f"COMPREHENSION: score={comprehension['score']}")
        except Exception as e:
            print(f"COMPREHENSION ERROR: {e}")
            comprehension = {"score": 0, "note": "Could not evaluate comprehension"}

        # ── Aggregate ─────────────────────────────────────────────────────────────
        return aggregate_scores(
            pronunciation = pronunciation,
            fluency       = fluency,
            tone          = tone,
            grammar       = grammar,
            comprehension = comprehension,
        )

    # Execute all synchronous evaluation logic inside the threadpool
    final = await asyncio.to_thread(_run_sync_evals)
    final["transcript"] = transcript
    return final