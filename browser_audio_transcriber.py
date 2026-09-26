"""
Browser Audio Transcriber
--------------------------
Records whatever audio your speakers are playing (a YouTube video, a Zoom
call, anything in Chrome or any other app) and turns it into a text
transcript, using:

    soundcard      -> captures "loopback" audio (what you HEAR, not your mic)
    faster-whisper -> converts that audio into text, running fully offline
    pynput         -> listens for a global hotkey (F9) to start/stop
                       recording, even while Chrome is the active window
    soundfile      -> writes the recorded audio out as a .wav file

Controls:
    F9  -> toggle recording on/off
    Esc -> quit the program

Output:
    Each recording produces two files in transcripts/:
        recording_YYYYMMDD_HHMMSS.wav   (the raw audio, kept for reference)
        transcript_YYYYMMDD_HHMMSS.txt  (the text you actually want)

See the accompanying guide for a full, line-by-line walkthrough of how
this file works and why it's built this way.
"""

import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf
from faster_whisper import WhisperModel
from pynput import keyboard

# ---------------------------------------------------------------------------
# Configuration - change these to taste
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48000            # audio quality; 48kHz matches most system audio
CHANNELS = 2                    # stereo
CHUNK_FRAMES = 1024              # audio frames pulled from the device per read
WHISPER_MODEL_SIZE = "small"    # tiny/base/small/medium/large-v3: bigger = more accurate, slower
COMPUTE_TYPE = "int8"           # int8 is fastest on CPU; use "float16" if you have a CUDA GPU
DEVICE = "cpu"                  # change to "cuda" if you have an NVIDIA GPU set up for it
OUTPUT_DIR = Path(__file__).parent / "transcripts"
TOGGLE_KEY = keyboard.Key.f9
QUIT_KEY = keyboard.Key.esc

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared state between the hotkey-listener thread and the recorder thread.
# `_state_lock` protects `is_recording` so both threads always agree on
# whether we're currently recording.
# ---------------------------------------------------------------------------
is_recording = False
_recorder_thread = None
_audio_frames = []           # list of small numpy arrays collected while recording
_state_lock = threading.Lock()
_model = None                # the Whisper model, loaded once and reused


def get_loopback_microphone():
    """
    Return a 'microphone' object that actually captures what your speakers
    are playing (system/loopback audio) instead of your physical mic.

    On Windows, every output device (speaker) has a matching loopback input
    exposed through WASAPI. `soundcard` gives you that loopback stream by
    passing the speaker's name as the microphone id with include_loopback=True.
    """
    default_speaker = sc.default_speaker()
    return sc.get_microphone(id=str(default_speaker.name), include_loopback=True)


def _record_loop():
    """
    Runs on its own background thread so it never blocks the hotkey
    listener. Repeatedly reads small chunks of audio from the loopback
    device and appends them to `_audio_frames` until `is_recording`
    becomes False.
    """
    mic = get_loopback_microphone()
    with mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as recorder:
        while True:
            with _state_lock:
                if not is_recording:
                    break
            frame = recorder.record(numframes=CHUNK_FRAMES)
            _audio_frames.append(frame)


def start_recording():
    """Flip the recording flag on and launch the background capture thread."""
    global is_recording, _recorder_thread, _audio_frames
    with _state_lock:
        if is_recording:
            return
        is_recording = True
        _audio_frames = []
    _recorder_thread = threading.Thread(target=_record_loop, daemon=True)
    _recorder_thread.start()
    print("\n[recording] started - press F9 again to stop")


def stop_recording_and_transcribe():
    """
    Flip the recording flag off, wait for the capture thread to finish,
    then stitch the collected audio together, save it, and transcribe it.
    """
    global is_recording
    with _state_lock:
        if not is_recording:
            return
        is_recording = False
    _recorder_thread.join()
    print("[recording] stopped - saving + transcribing...")

    if not _audio_frames:
        print("[warning] no audio captured - is anything actually playing?")
        return

    audio = np.concatenate(_audio_frames, axis=0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    wav_path = OUTPUT_DIR / f"recording_{timestamp}.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)

    text = transcribe(wav_path)

    txt_path = OUTPUT_DIR / f"transcript_{timestamp}.txt"
    txt_path.write_text(text, encoding="utf-8")
    print(f"[done] transcript saved to: {txt_path}")


def get_model():
    """
    Load the Whisper model once and reuse it for every recording. The
    first call downloads the model files (needs internet, ~1-2 min);
    every call after that loads instantly from the local cache.
    """
    global _model
    if _model is None:
        print("[whisper] loading model (first run downloads it)...")
        _model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def transcribe(wav_path):
    """Run Whisper over a saved .wav file and return the transcript as one string."""
    model = get_model()
    segments, _info = model.transcribe(str(wav_path), beam_size=5)
    return " ".join(segment.text.strip() for segment in segments)


def toggle_recording():
    if is_recording:
        stop_recording_and_transcribe()
    else:
        start_recording()


def on_press(key):
    """Called by pynput for every key pressed anywhere on the system."""
    if key == TOGGLE_KEY:
        toggle_recording()
    elif key == QUIT_KEY:
        if is_recording:
            stop_recording_and_transcribe()
        return False  # returning False stops the listener -> ends the program


def main():
    print("Browser Audio Transcriber")
    print("  F9  = start/stop recording")
    print("  Esc = quit")
    try:
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    except KeyboardInterrupt:
        if is_recording:
            stop_recording_and_transcribe()
    print("Goodbye!")


if __name__ == "__main__":
    main()
