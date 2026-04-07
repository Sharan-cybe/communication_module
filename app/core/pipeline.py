import asyncio
import io
from concurrent.futures import ThreadPoolExecutor

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

    if not transcript:
        print("WARNING: Whisper returned empty transcript — audio may be silent or too short")

    def _run_sync_evals():
        if not transcript or len(transcript.strip()) < 2:
            return aggregate_scores(
                pronunciation = {"score": 0, "clarity": 0.0, "consistency": 0.0, "note": "No speech detected"},
                fluency       = {"score": 0, "wpm": 0.0, "filler_rate": 0.0, "pauses": {"count": 0, "avg_duration": 0.0}, "note": "No speech detected"},
                tone          = {"score": 0, "pitch_variation": 0.0, "energy_variation": 0.0, "note": "No speech detected"},
                grammar       = {"score": 0, "mistakes": [], "note": "No speech detected"},
                comprehension = {"score": 0, "note": "No speech detected"},
            )

        local_audio_fluency = BufferedFile(audio_bytes)
        local_audio_tone    = BufferedFile(audio_bytes)

        # ── Sync evals (pure math, instant) ──────────────────────────────────
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
            pronunciation = {"score": 1, "clarity": 0.0, "consistency": 0.0,
                             "composite_score": 0.0, "note": "Could not evaluate pronunciation"}

        try:
            fluency = analyze_fluency(transcript, segments, local_audio_fluency)
            filler_words_log = fluency.get('filler_words', [])
            print(f"FLUENCY: score={fluency['score']} wpm={fluency.get('wpm')} fillers={filler_words_log}")
        except Exception as e:
            print(f"FLUENCY ERROR: {e}")
            fluency = {"score": 1, "wpm": 0.0, "filler_rate": 0.0,
                       "pauses": {"count": 0, "avg_duration": 0.0}, "note": "Could not evaluate fluency"}

        try:
            tone = analyze_tone(local_audio_tone)
            print(f"TONE: score={tone['score']} composite={tone.get('diagnostics', {}).get('composite')}")
        except Exception as e:
            print(f"TONE ERROR: {e}")
            tone = {"score": 1, "pitch_variation": 0.0, "energy_variation": 0.0,
                    "note": "Could not evaluate tone"}

        # ── LLM evals in parallel ─────────────────────────────────────────────
        # Grammar and comprehension both make separate LLM API calls.
        # Running them sequentially adds ~4-8s of pure waiting.
        # ThreadPoolExecutor lets both HTTP requests fly at the same time.
        grammar       = {"score": 1, "mistakes": [], "note": "Could not evaluate grammar"}
        comprehension = {"score": 1, "note": "Could not evaluate comprehension"}

        def _eval_grammar():
            try:
                result = evaluate_grammar(transcript)
                print(f"GRAMMAR: score={result['score']} mistakes={len(result.get('mistakes', []))}")
                for i, m in enumerate(result.get("mistakes", []), 1):
                    print(f"  [{i}] {m.get('category','GENERAL')} | ❌ \"{m.get('original')}\" → ✅ \"{m.get('corrected')}\"")
                if result.get("note"):
                    print(f"  NOTE: {result['note']}")
                return result
            except Exception as e:
                print(f"GRAMMAR ERROR: {e}")
                return {"score": 1, "mistakes": [], "note": "Could not evaluate grammar"}

        def _eval_comprehension():
            try:
                if not question or not question.strip():
                    return {"score": 1, "note": "No question provided"}
                result = evaluate_comprehension(question, transcript)
                print(f"COMPREHENSION: score={result['score']} note={result.get('note', '')}")
                return result
            except Exception as e:
                print(f"COMPREHENSION ERROR (defaulting to neutral score=1): {e}")
                return {"score": 1, "note": "Could not evaluate comprehension"}

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_grammar       = executor.submit(_eval_grammar)
            future_comprehension = executor.submit(_eval_comprehension)
            grammar       = future_grammar.result()
            comprehension = future_comprehension.result()

        # ── Aggregate ─────────────────────────────────────────────────────────
        return aggregate_scores(
            pronunciation = pronunciation,
            fluency       = fluency,
            tone          = tone,
            grammar       = grammar,
            comprehension = comprehension,
        )

    final = await asyncio.to_thread(_run_sync_evals)
    final["transcript"] = transcript
    return final