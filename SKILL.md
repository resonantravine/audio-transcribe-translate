---
name: audio-transcribe-translate
description: Transcribe local audio or video files with timestamps and translate non-English speech into English using local Whisper models. Use for MP3, WAV, M4A, MP4, MOV, multilingual recordings, French-to-English transcripts, interview dialogue, or when a faster Apple Silicon transcription workflow is requested.
---

# Audio Transcribe & Translate

Use the bundled script for deterministic Whisper inference, then review the JSON before producing the user-facing transcript.

## Setup

Before the first run, verify `ffmpeg` and install the platform-appropriate dependency:

```bash
python3 -m pip install -r requirements.txt
```

## Workflow

1. Inspect the source with `ffmpeg` and report its duration.
2. Run `scripts/transcribe.py`:

```bash
python3 scripts/transcribe.py INPUT --task both --language fr --output-dir WORK_DIR
```

3. Prefer the `mlx` backend on Apple Silicon. Use `openai` only as a fallback.
4. For mixed English/French audio:
   - Run `--task translate` without forcing a language to obtain an English master transcript.
   - Run `--task transcribe --language fr` to recover French source passages.
   - Preserve original English narration as English.
   - Include both French transcription and English translation for French passages.
5. Remove empty segments and obvious repetition loops. Mark uncertain speech as `[unclear]`; never invent missing words.
6. Write a Markdown deliverable with source name, timestamp ranges, original text, English translation, and notes.

## Performance Rules

- Set `condition_on_previous_text=False` to reduce repetition loops and long stalls.
- Omit word timestamps unless the user explicitly needs word-level timing.
- Use `--model mlx-community/whisper-small-mlx` for the default fast pass.
- Use `mlx-community/whisper-large-v3-turbo` only when better accuracy justifies a larger download.
- If one region stalls or hallucinates, rerun only that interval with `--clip START,END`.
- Reuse JSON results instead of rerunning inference when only formatting or translation wording changes.

## Script

Run:

```bash
python3 scripts/transcribe.py --help
```

The script supports:

- `--backend auto|mlx|openai`
- `--task transcribe|translate|both`
- optional forced `--language`
- custom model selection
- partial processing with `--clip START,END`
- JSON and readable text outputs

## Output Quality

- Treat names, locations, dialect names, and code-switching as review targets.
- Compare transcription and translation passes when they disagree.
- State which parts are machine-generated or uncertain.
- Do not upload source media unless the user explicitly requests an online service.
