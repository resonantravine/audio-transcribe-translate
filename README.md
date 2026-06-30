# Audio Transcribe & Translate

A Codex skill and standalone Python utility for fast, private transcription and English translation with local Whisper models.

It is tuned for Apple Silicon through [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper), while retaining an OpenAI Whisper CPU fallback for other platforms.

## What it does

- Transcribes MP3, WAV, M4A, MP4, MOV, and other FFmpeg-readable media.
- Generates segment-level timestamps.
- Translates non-English speech into English.
- Supports French, Chinese, English, and multilingual recordings.
- Runs locally; source media is not uploaded.
- Supports partial reruns for difficult or corrected sections.
- Reduces repetition loops by disabling previous-text conditioning.

## Requirements

- Python 3.10 or newer
- FFmpeg available on `PATH`
- macOS with Apple Silicon for the recommended MLX backend, or a platform supported by OpenAI Whisper

## Install as a Codex skill

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
git clone https://github.com/resonantravine/audio-transcribe-translate.git \
  "${CODEX_HOME:-$HOME/.codex}/skills/audio-transcribe-translate"
python3 -m pip install -r \
  "${CODEX_HOME:-$HOME/.codex}/skills/audio-transcribe-translate/requirements.txt"
```

Restart Codex if the skill does not appear immediately.

Invoke it with a request such as:

> Use `$audio-transcribe-translate` to transcribe this recording and translate the French passages into English.

## Use the script directly

```bash
git clone https://github.com/resonantravine/audio-transcribe-translate.git
cd audio-transcribe-translate
python3 -m pip install -r requirements.txt
```

Transcribe:

```bash
python3 scripts/transcribe.py interview.wav \
  --task transcribe \
  --output-dir output
```

Transcribe French and create an English translation pass:

```bash
python3 scripts/transcribe.py interview.wav \
  --task both \
  --language fr \
  --output-dir output
```

Rerun only a difficult interval:

```bash
python3 scripts/transcribe.py interview.wav \
  --task transcribe \
  --language fr \
  --clip 120,180 \
  --output-dir output
```

Use the more accurate turbo model:

```bash
python3 scripts/transcribe.py interview.wav \
  --model mlx-community/whisper-large-v3-turbo \
  --task transcribe \
  --output-dir output
```

## Output

Each pass creates JSON and readable timestamped text:

```text
output/
├── interview.transcribe.json
├── interview.transcribe.txt
├── interview.translate.json
└── interview.translate.txt
```

Example:

```text
[00:00–00:04] Bonjour, je m'appelle Emma.
[00:04–00:08] J'apprends le chinois depuis deux mois.
```

The JSON contains Whisper's full segment metadata for later review, alignment, or Markdown formatting.

## Backends

| Platform | Default backend | Default model |
| --- | --- | --- |
| Apple Silicon | MLX | `mlx-community/whisper-small-mlx` |
| Other platforms | OpenAI Whisper CPU | `small` |

Override automatic selection with `--backend mlx` or `--backend openai`.

The first MLX run downloads model weights from Hugging Face. Later runs reuse the local cache. In a local M1 smoke test, a cached `whisper-small-mlx` model processed a 10-second clip in about 1.6 seconds. This is a smoke-test measurement, not a general benchmark; speed varies with model, audio, and hardware.

## Mixed-language recordings

Whisper assigns a language to each decoding pass, so rapid code-switching may need two passes:

1. Run `--task translate` without `--language` to create an English master transcript.
2. Run `--task transcribe --language fr` to recover French source passages.
3. Compare timestamps and preserve existing English narration unchanged.
4. Mark uncertain words rather than guessing.

The Codex skill documents this review workflow in [`SKILL.md`](SKILL.md).

## Privacy

Inference is local. The script passes a local file path to the selected Whisper implementation and does not contain upload or analytics code. Model weights may be downloaded on first use.

## Known limitations

- Speaker diarization is not included.
- Names and dialect terms usually require manual review.
- Fast language switching can confuse Whisper.
- `--task both` performs two passes and therefore takes longer.
- Word timestamps are available but slower than segment timestamps.

## License

[MIT](LICENSE)
