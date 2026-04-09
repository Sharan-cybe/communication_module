"""
pipeline.py  v2
────────────────
Speaking pipeline — redesigned for minimum LLM calls.

Old: per-question → 1 Whisper + 1 grammar LLM + 1 comprehension LLM
     3 questions × 3 calls = 9 LLM calls

New: per-question → 1 Whisper (unavoidable, sequential per audio file)
     All questions combined → 1 LLM call (grammar + comprehension together)
     3 Whisper + 1 LLM = 4 total calls per session

Audio-only evals (pronunciation, fluency, tone) remain per-question
but are pure CPU — zero network calls.
"""

import asyncio
import io

from app.services.speech.whisper_service import transcribe_audio
from app.services.fluency.fluency_analyzer import analyze_fluency
from app.services.tone.tone_analyzer import analyze_tone
from app.services.pronunciation.pronunciation_service import evaluate_pronunciation
from app.services.grammar.llama_service import evaluate_speaking_session
from app.core.scoring_engine import aggregate_scores


class BufferedFile:
    def __init__(self, data: bytes):
        self.file = io.BytesIO(data)


# ─────────────────────────────────────────────────────────────────────────────
# Single-question pipeline (used by /evaluate endpoint — one audio at a time)
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(audio_file, question: str) -> dict:
    """
    Evaluate a single audio response.

    LLM calls: 1 Whisper + 1 combined LLM (grammar+comprehension)
    Audio calls: pronunciation, fluency, tone (CPU only)
    Total external calls: 2
    """
    # ── Transcription ─────────────────────────────────────────────────────────
    audio_file.file.seek(0)
    audio_bytes = audio_file.file.read()
    audio_file.file.seek(0)

    transcript_data = await transcribe_audio(audio_file)
    transcript      = transcript_data.get("text", "").strip()
    segments        = transcript_data.get("timestamps", [])
    words           = transcript_data.get("words", [])

    print(f"TRANSCRIPT ({len(transcript)} chars): {transcript[:120]}…")
    print(f"SEGMENTS: {len(segments)} | WORDS: {len(words)}")

    if not transcript:
        print("WARNING: Whisper returned empty transcript — audio may be silent or too short")

    def _run_sync_evals():
        # ── No-speech fast-path ───────────────────────────────────────────────
        if not transcript or len(transcript.strip()) < 2:
            return aggregate_scores(
                pronunciation = {"score": 0, "clarity": 0.0, "consistency": 0.0,
                                 "composite_score": 0.0, "note": "No speech detected"},
                fluency       = {"score": 0, "wpm": 0.0, "filler_rate": 0.0,
                                 "pauses": {"count": 0, "avg_duration": 0.0}, "note": "No speech detected"},
                tone          = {"score": 0, "pitch_variation": 0.0, "energy_variation": 0.0,
                                 "note": "No speech detected"},
                grammar       = {"score": 0, "mistakes": [], "note": "No speech detected"},
                comprehension = {"score": 0, "relevance": 0.0, "completeness": 0.0,
                                 "note": "No speech detected"},
            )

        local_fluency = BufferedFile(audio_bytes)
        local_tone    = BufferedFile(audio_bytes)

        # ── CPU-only evals (no network) ───────────────────────────────────────
        try:
            pronunciation = evaluate_pronunciation(
                expected_text    = question,
                spoken_text      = transcript,
                whisper_segments = segments,
                whisper_words    = words,
            )
            print(f"PRONUNCIATION: score={pronunciation['score']} composite={pronunciation.get('composite_score')}")
        except Exception as e:
            print(f"PRONUNCIATION ERROR: {e}")
            pronunciation = {"score": 1, "clarity": 0.75, "consistency": 0.75,
                             "composite_score": 0.5, "note": "Could not evaluate pronunciation"}

        try:
            fluency = analyze_fluency(transcript, segments, local_fluency)
            print(f"FLUENCY: score={fluency['score']} wpm={fluency.get('wpm')} fillers={fluency.get('filler_words')}")
        except Exception as e:
            print(f"FLUENCY ERROR: {e}")
            fluency = {"score": 1, "wpm": 0.0, "filler_rate": 0.0,
                       "pauses": {"count": 0, "avg_duration": 0.0}, "note": "Could not evaluate fluency"}

        try:
            tone = analyze_tone(local_tone)
            print(f"TONE: score={tone['score']}")
        except Exception as e:
            print(f"TONE ERROR: {e}")
            tone = {"score": 1, "pitch_variation": 30.0, "energy_variation": 0.3,
                    "note": "Could not evaluate tone"}

        # ── Single combined LLM call (grammar + comprehension) ────────────────
        try:
            llm_results = evaluate_speaking_session([{"question": question, "answer": transcript}])
            grammar       = llm_results[0]["grammar"]
            comprehension = llm_results[0]["comprehension"]
        except Exception as e:
            print(f"LLM EVAL ERROR: {e}")
            grammar       = {"score": 1, "mistakes": [], "note": "Could not evaluate grammar"}
            comprehension = {"score": 1, "relevance": 0.5, "completeness": 0.5,
                             "note": "Could not evaluate comprehension"}

        return aggregate_scores(
            pronunciation = pronunciation,
            fluency       = fluency,
            tone          = tone,
            grammar       = grammar,
            comprehension = comprehension,
        )

    final              = await asyncio.to_thread(_run_sync_evals)
    final["transcript"] = transcript
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Session pipeline (used by /speaking/aggregate — all 3 answers at once)
# ─────────────────────────────────────────────────────────────────────────────

async def run_session_pipeline(
    audio_files: list,          # list of (audio_file, question_str) tuples
) -> list[dict]:
    """
    Evaluate all speaking answers in a session.

    Call budget:
      - 1 Whisper call per audio file  (unavoidable — sequential per audio)
      - 1 combined LLM call for all grammar + comprehension (new)
      - CPU-only calls for pronunciation / fluency / tone

    Total external calls: len(audio_files) Whisper + 1 LLM
    """
    # ── Step 1: Transcribe all audio files ───────────────────────────────────
    transcriptions = []
    audio_bytes_list = []
    for audio_file, question in audio_files:
        try:
            audio_file.file.seek(0)
            audio_bytes = audio_file.file.read()
            audio_bytes_list.append(audio_bytes)
            audio_file.file.seek(0)
            td = await transcribe_audio(audio_file)
            transcriptions.append({
                "question": question,
                "text":     td.get("text", "").strip(),
                "segments": td.get("timestamps", []),
                "words":    td.get("words", []),
            })
            print(f"TRANSCRIPT Q{len(transcriptions)}: {td.get('text','')[:80]}…")
        except Exception as e:
            print(f"TRANSCRIPTION ERROR: {e}")
            transcriptions.append({"question": question, "text": "", "segments": [], "words": []})
            audio_bytes_list.append(b"")

    # ── Step 2: Combined LLM call for grammar + comprehension ─────────────────
    qa_pairs = [{"question": t["question"], "answer": t["text"]} for t in transcriptions]

    try:
        llm_results = await asyncio.to_thread(evaluate_speaking_session, qa_pairs)
    except Exception as e:
        print(f"COMBINED LLM ERROR: {e}")
        llm_results = [
            {"grammar": {"score": 1, "mistakes": [], "note": "Could not evaluate"},
             "comprehension": {"score": 1, "relevance": 0.5, "completeness": 0.5, "note": "Could not evaluate"}}
            for _ in transcriptions
        ]

    # ── Step 3: CPU-only evals per question + aggregate ──────────────────────
    results = []
    for i, (td, audio_bytes) in enumerate(zip(transcriptions, audio_bytes_list)):
        transcript = td["text"]
        segments   = td["segments"]
        words      = td["words"]
        question   = td["question"]

        if not transcript or len(transcript.strip()) < 2:
            result = aggregate_scores(
                pronunciation = {"score": 0, "clarity": 0.0, "consistency": 0.0,
                                 "composite_score": 0.0, "note": "No speech detected"},
                fluency       = {"score": 0, "wpm": 0.0, "filler_rate": 0.0,
                                 "pauses": {"count": 0, "avg_duration": 0.0}, "note": "No speech detected"},
                tone          = {"score": 0, "pitch_variation": 0.0, "energy_variation": 0.0,
                                 "note": "No speech detected"},
                grammar       = {"score": 0, "mistakes": [], "note": "No speech detected"},
                comprehension = {"score": 0, "relevance": 0.0, "completeness": 0.0,
                                 "note": "No speech detected"},
            )
            result["transcript"] = transcript
            results.append(result)
            continue

        local_fluency = BufferedFile(audio_bytes)
        local_tone    = BufferedFile(audio_bytes)

        try:
            pronunciation = evaluate_pronunciation(
                expected_text=question, spoken_text=transcript,
                whisper_segments=segments, whisper_words=words,
            )
            print(f"PRONUNCIATION Q{i+1}: score={pronunciation['score']} composite={pronunciation.get('composite_score')}")
        except Exception as e:
            print(f"PRONUNCIATION ERROR Q{i+1}: {e}")
            pronunciation = {"score": 1, "clarity": 0.75, "consistency": 0.75,
                             "composite_score": 0.5, "note": "Could not evaluate pronunciation"}

        try:
            fluency = analyze_fluency(transcript, segments, local_fluency)
            print(f"FLUENCY Q{i+1}: score={fluency['score']} wpm={fluency.get('wpm')} fillers={fluency.get('filler_words')}")
        except Exception as e:
            print(f"FLUENCY ERROR Q{i+1}: {e}")
            fluency = {"score": 1, "wpm": 0.0, "filler_rate": 0.0,
                       "pauses": {"count": 0, "avg_duration": 0.0}, "note": "Could not evaluate fluency"}

        try:
            tone = analyze_tone(local_tone)
            print(f"TONE Q{i+1}: score={tone['score']}")
        except Exception as e:
            print(f"TONE ERROR Q{i+1}: {e}")
            tone = {"score": 1, "pitch_variation": 30.0, "energy_variation": 0.3,
                    "note": "Could not evaluate tone"}

        grammar       = llm_results[i]["grammar"]
        comprehension = llm_results[i]["comprehension"]

        result = aggregate_scores(
            pronunciation=pronunciation,
            fluency=fluency,
            tone=tone,
            grammar=grammar,
            comprehension=comprehension,
        )
        result["transcript"] = transcript
        results.append(result)

    return results