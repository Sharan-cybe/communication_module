"""
listening_pipeline.py
──────────────────────
Three operations exposed to the endpoint layer:

  generate_listening_clips()     → picks 4 random clips, TTS → base64 audio
  submit_all_responses(...)      → receives all audio, transcribes, evaluates all 4 together
  aggregate_session(clip_results) → final score + summary
"""

import base64
import uuid
import asyncio

from app.services.speech.whisper_service import transcribe_audio
from app.services.listening.listening_service import evaluate_all_responses
from app.services.listening.listening_scoring_engine import aggregate_listening_scores
from app.services.listening.content_bank import get_session_clips, ListeningClip
from app.services.tts.tts_service import synthesize_text


# ─────────────────────────────────────────────────────────────────────────────
# In-memory session store  {session_id → list[ListeningClip]}
# ─────────────────────────────────────────────────────────────────────────────

SESSION_STORE: dict[str, list[ListeningClip]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generate clips for a session
# ─────────────────────────────────────────────────────────────────────────────

async def generate_listening_clips() -> dict:
    """
    Pick 2 REPEAT + 2 QnA clips randomly, synthesise audio via Azure TTS,
    return base64 audio + metadata. Creates and stores a session.
    """
    clips      = get_session_clips()
    session_id = str(uuid.uuid4())
    SESSION_STORE[session_id] = clips

    output = []
    for clip in clips:
        try:
            audio_bytes = synthesize_text(clip.reference_text)
            audio_b64   = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            print(f"TTS ERROR [{clip.clip_id}]: {e}")
            audio_b64 = None

        output.append({
            "clip_id":   clip.clip_id,
            "task_type": clip.task_type,
            "audio_b64": audio_b64,
            "questions": clip.questions,
        })

    return {"session_id": session_id, "clips": output}


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Receive and evaluate responses
# ─────────────────────────────────────────────────────────────────────────────

async def submit_all_responses(
    session_id:   str,
    clip_audios:  dict,
) -> list:
    """
    Bulk submission: Transcribe all submitted audio files, then evaluate.
    """
    session_clips = SESSION_STORE.get(session_id)
    if not session_clips:
        return [{"error": "Invalid or expired session_id"}]

    clip_responses = []

    for clip in session_clips:
        cid   = clip.clip_id
        audio = clip_audios.get(cid)

        if audio is None:
            clip_responses.append({
                "clip_id": cid, "transcript": "", "whisper_segments": [], "whisper_words": []
            })
            continue

        if clip.task_type == "REPEAT":
            audio.file.seek(0)
            td = await transcribe_audio(audio)
            clip_responses.append({
                "clip_id":          cid,
                "transcript":       td.get("text", "").strip(),
                "whisper_segments": td.get("timestamps", []),
                "whisper_words":    td.get("words", []),
            })
        elif clip.task_type == "QnA":
            audio_q1 = audio.get("q1") if isinstance(audio, dict) else audio
            audio_q2 = audio.get("q2") if isinstance(audio, dict) else None

            td1 = {"text": "", "timestamps": [], "words": []}
            td2 = {"text": "", "timestamps": [], "words": []}

            if audio_q1:
                audio_q1.file.seek(0)
                td1 = await transcribe_audio(audio_q1)
            if audio_q2:
                audio_q2.file.seek(0)
                td2 = await transcribe_audio(audio_q2)

            clip_responses.append({
                "clip_id":    cid,
                "answer_q1":  td1.get("text", "").strip(),
                "answer_q2":  td2.get("text", "").strip(),
                "segments_q1": td1.get("timestamps", []),
                "words_q1":    td1.get("words", []),
                "segments_q2": td2.get("timestamps", []),
                "words_q2":    td2.get("words", []),
            })

    return await asyncio.to_thread(evaluate_all_responses, session_clips, clip_responses)


async def evaluate_clip_response(
    session_id: str,
    clip_id: str,
    audio_file,
    question_index: int = 0,
) -> dict:
    """
    Legacy evaluator (one-by-one) for the current frontend/endpoint.
    """
    session_clips = SESSION_STORE.get(session_id)
    if not session_clips: return {"error": "Invalid session_id"}

    clip = next((c for c in session_clips if c.clip_id == clip_id), None)
    if not clip: return {"error": "Invalid clip_id"}

    audio_file.file.seek(0)
    td = await transcribe_audio(audio_file)

    responses = []
    if clip.task_type == "REPEAT":
        responses.append({
            "clip_id": clip_id if 'clip_id' in locals() else clip_id,
            "transcript": td.get("text", "").strip(),
            "whisper_segments": td.get("timestamps", []),
            "whisper_words": td.get("words", []),
        })
    else:
        responses.append({
            "clip_id": clip_id,
            "answer_q1": td.get("text", "").strip(),
            "answer_q2": "",
            "segments_q1": td.get("timestamps", []),
            "words_q1": td.get("words", []),
            "segments_q2": [],
            "words_q2": [],
        })

    results = await asyncio.to_thread(evaluate_all_responses, session_clips, responses)
    return next((r for r in results if r["clip_id"] == clip_id), {"error": "Evaluation failed"})


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Aggregate final score
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_session(clip_results: list) -> dict:
    return aggregate_listening_scores(clip_results)