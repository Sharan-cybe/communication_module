from app.services.speech.whisper_service import transcribe_audio
from app.services.fluency.fluency_analyzer import analyze_fluency
from app.services.tone.tone_analyzer import analyze_tone
from app.services.grammar.llama_service import evaluate_grammar
from app.services.comprehension.comprehension_service import evaluate_comprehension
from app.core.scoring_engine import aggregate_scores


async def run_pipeline(audio_file, question):

    # 🔥 RESET before reading
    audio_file.file.seek(0)

    transcript_data = transcribe_audio(audio_file)

    transcript = transcript_data.get("text", "")
    timestamps = transcript_data.get("timestamps", [])

    print("TRANSCRIPT:", transcript)
    print("TIMESTAMPS:", timestamps)

    # 🔥 Reset again for next modules
    audio_file.file.seek(0)
    fluency = analyze_fluency(transcript, timestamps, audio_file)

    audio_file.file.seek(0)
    tone = analyze_tone(audio_file)

    grammar = evaluate_grammar(transcript)
    comprehension = evaluate_comprehension(question, transcript)

    final = aggregate_scores(
        fluency=fluency,
        tone=tone,
        grammar=grammar,
        comprehension=comprehension
    )

    return final