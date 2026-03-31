import azure.cognitiveservices.speech as speechsdk
from app.core.config import SPEECH_KEY, SPEECH_REGION, DEFAULT_VOICE


def synthesize_text(text: str, voice: str | None = None) -> bytes:
    """
    Convert text to speech using Azure Cognitive Services.
    Returns raw audio bytes (WAV format).

    text  : the text to synthesize
    voice : optional Azure voice name — falls back to DEFAULT_VOICE from config
    """
    selected_voice = voice if voice else DEFAULT_VOICE

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION,
    )
    speech_config.speech_synthesis_voice_name = selected_voice

    # audio_config=None → no speaker output, returns bytes only
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None,
    )

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    else:
        cancellation = result.cancellation_details
        raise Exception(
            f"Speech synthesis failed: {cancellation.reason} — {cancellation.error_details}"
        )