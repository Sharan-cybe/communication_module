import os
import sys

# ── Windows eSpeak NG path fix ────────────────────────────────────────────────
# phonemizer on Windows locates espeak-ng.exe by searching the system PATH.
# Setting os.environ["espeak"] is NOT recognized — it does nothing.
# The correct fix: inject the eSpeak NG install dir into PATH before importing
# phonemizer, so its subprocess call can find espeak-ng.exe.

ESPEAK_DIR = r"C:\Program Files\eSpeak NG"
ESPEAK_EXE = os.path.join(ESPEAK_DIR, "espeak-ng.exe")

if sys.platform == "win32":
    current_path = os.environ.get("PATH", "")
    if ESPEAK_DIR not in current_path:
        os.environ["PATH"] = ESPEAK_DIR + os.pathsep + current_path

    # Some phonemizer versions also check this env var directly
    os.environ["espeak"] = ESPEAK_EXE

    if not os.path.isfile(ESPEAK_EXE):
        print(f"WARNING: espeak-ng.exe not found at {ESPEAK_EXE}")
        print("Install from: https://github.com/espeak-ng/espeak-ng/releases")

# ── Import phonemizer AFTER PATH is patched ───────────────────────────────────
from phonemizer import phonemize

# ── Same normalization table used for both sides ──────────────────────────────
from app.services.pronunciation.phoneme_utils import _normalize_phoneme


def _ipa_to_arpabet_approx(ipa_str: str) -> list[str]:
    """
    Rough IPA → ARPAbet-style token mapping for eSpeak output.
    eSpeak returns IPA characters; we map them to CMU-style tokens
    so both pipelines (expected vs spoken) are in the same space.
    Only covers the phones that matter for English.
    """
    IPA_MAP = {
        "p": "P",  "b": "B",  "t": "T",  "d": "D",
        "k": "K",  "ɡ": "G",  "f": "F",  "v": "V",
        "s": "S",  "z": "Z",  "h": "HH", "m": "M",
        "n": "N",  "ŋ": "NG", "l": "L",  "r": "R",
        "w": "W",  "j": "Y",
        # Vowels (IPA → closest ARPAbet)
        "æ": "AE", "ɑ": "AA", "ɛ": "EH", "ɪ": "IH",
        "i": "IY", "ɔ": "AO", "ʊ": "UH", "u": "UW",
        "ʌ": "AH", "ə": "AH", "eɪ": "EY","aɪ": "AY",
        "ɔɪ": "OY","aʊ": "AW","oʊ": "OW",
        # Dental fricatives (Indian candidates commonly substitute these)
        "θ": "TH", "ð": "DH",
        "ʃ": "SH", "ʒ": "ZH","tʃ": "CH","dʒ": "JH",
    }

    tokens = []
    i = 0
    chars = list(ipa_str)

    while i < len(chars):
        # Try 2-char digraph first
        digraph = "".join(chars[i:i+2])
        if digraph in IPA_MAP:
            tokens.append(IPA_MAP[digraph])
            i += 2
        elif chars[i] in IPA_MAP:
            tokens.append(IPA_MAP[chars[i]])
            i += 1
        else:
            # Unknown symbol: skip (stress markers, syllable boundaries, etc.)
            i += 1

    return tokens


def get_spoken_phonemes(text: str) -> list[str]:
    """
    Convert spoken transcript to a normalized phoneme list via eSpeak.
    Falls back to raw words if eSpeak is unavailable.
    Both IPA→ARPAbet conversion AND Indian-accent normalization are applied
    so the result is directly comparable to get_expected_phonemes() output.
    """
    try:
        ipa_output = phonemize(
            text,
            language="en-us",
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            njobs=1,     # important on Windows
            with_stress=False,
        )

        raw_tokens = _ipa_to_arpabet_approx(ipa_output)

        # Apply the same Indian-accent normalization so both sides are symmetric
        return [_normalize_phoneme(p) for p in raw_tokens]

    except Exception as e:
        print(f"PHONEMIZER ERROR: {e}")
        # Graceful fallback: word stubs (same format as phoneme_utils fallback)
        return [w.upper()[:4] for w in text.lower().split()]