# Browser Audio Transcriber

Records whatever audio is playing on your computer (a video in Chrome, a
call, anything) and saves it as a text transcript, fully offline.

Full step-by-step guide with line-by-line code explanations: see the link
shared alongside this project. This README is just the quick-start.

## Setup (Windows)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python browser_audio_transcriber.py
```

- Press **F9** to start recording whatever is playing through your speakers.
- Play your video/call in Chrome (or anywhere).
- Press **F9** again to stop. It saves the audio and transcribes it.
- Press **Esc** to quit.

Output lands in `transcripts/`:
- `recording_<timestamp>.wav` - the raw audio
- `transcript_<timestamp>.txt` - the text

## Notes

- The first run downloads the Whisper model (~500MB for the default
  "small" size) - needs internet once, then works fully offline.
- Change `WHISPER_MODEL_SIZE` in the script for a speed/accuracy tradeoff
  (`tiny`, `base`, `small`, `medium`, `large-v3`).
- macOS/Linux: loopback capture needs a small platform-specific tweak
  (see the guide's "Other platforms" section).
