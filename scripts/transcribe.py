#!/usr/bin/env python3
"""Fast local Whisper transcription with MLX and OpenAI Whisper backends."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


MLX_DEFAULT_MODEL = "mlx-community/whisper-small-mlx"
OPENAI_DEFAULT_MODEL = "small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe and/or translate local media with Whisper."
    )
    parser.add_argument("input", type=Path, help="Input audio or video file")
    parser.add_argument(
        "--backend",
        choices=("auto", "mlx", "openai"),
        default="auto",
        help="Inference backend; auto prefers MLX on Apple Silicon",
    )
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate", "both"),
        default="transcribe",
    )
    parser.add_argument(
        "--language",
        help="Optional ISO language code such as fr, en, or zh",
    )
    parser.add_argument(
        "--model",
        help="MLX Hugging Face repo or OpenAI Whisper model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for JSON and text output",
    )
    parser.add_argument(
        "--clip",
        help="Optional Whisper clip timestamps, for example 120,180",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Generate slower word-level timestamps",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show Whisper decoding progress",
    )
    return parser.parse_args()


def select_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401

            return "mlx"
        except ImportError:
            pass
    return "openai"


def transcribe_mlx(
    source: Path,
    task: str,
    language: str | None,
    model: str,
    clip: str | None,
    word_timestamps: bool,
    verbose: bool,
) -> dict[str, Any]:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise SystemExit(
            "MLX backend unavailable. Install it with: pip install mlx-whisper"
        ) from exc

    options: dict[str, Any] = {
        "path_or_hf_repo": model,
        "task": task,
        "language": language,
        "verbose": verbose,
        "temperature": 0,
        "condition_on_previous_text": False,
        "word_timestamps": word_timestamps,
        "hallucination_silence_threshold": 2.0 if word_timestamps else None,
    }
    if clip:
        options["clip_timestamps"] = clip
    return mlx_whisper.transcribe(str(source), **options)


def transcribe_openai(
    source: Path,
    task: str,
    language: str | None,
    model_name: str,
    clip: str | None,
    word_timestamps: bool,
    verbose: bool,
    cached_models: dict[str, Any],
) -> dict[str, Any]:
    try:
        import whisper
    except ImportError as exc:
        raise SystemExit(
            "OpenAI Whisper backend unavailable. Install it with: "
            "pip install openai-whisper"
        ) from exc

    model = cached_models.get(model_name)
    if model is None:
        model = whisper.load_model(model_name, device="cpu")
        cached_models[model_name] = model
    options: dict[str, Any] = {
        "task": task,
        "language": language,
        "verbose": verbose,
        "temperature": 0,
        "condition_on_previous_text": False,
        "word_timestamps": word_timestamps,
        "fp16": False,
    }
    if clip:
        options["clip_timestamps"] = clip
    return model.transcribe(str(source), **options)


def timestamp(seconds: float) -> str:
    whole = max(0, round(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def write_outputs(
    result: dict[str, Any],
    output_dir: Path,
    stem: str,
    task: str,
) -> tuple[Path, Path]:
    json_path = output_dir / f"{stem}.{task}.json"
    text_path = output_dir / f"{stem}.{task}.txt"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = []
    for segment in result.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        lines.append(
            f"[{timestamp(float(segment['start']))}–"
            f"{timestamp(float(segment['end']))}] {text}"
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, text_path


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    if not source.is_file():
        print(f"Input file not found: {source}", file=sys.stderr)
        return 2

    backend = select_backend(args.backend)
    model = args.model or (
        MLX_DEFAULT_MODEL if backend == "mlx" else OPENAI_DEFAULT_MODEL
    )
    tasks = ("transcribe", "translate") if args.task == "both" else (args.task,)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_models: dict[str, Any] = {}

    print(f"backend={backend} model={model} input={source}")
    for task in tasks:
        started = time.monotonic()
        if backend == "mlx":
            result = transcribe_mlx(
                source,
                task,
                args.language,
                model,
                args.clip,
                args.word_timestamps,
                args.verbose,
            )
        else:
            result = transcribe_openai(
                source,
                task,
                args.language,
                model,
                args.clip,
                args.word_timestamps,
                args.verbose,
                cached_models,
            )
        json_path, text_path = write_outputs(
            result, output_dir, source.stem, task
        )
        elapsed = time.monotonic() - started
        print(
            f"task={task} language={result.get('language', 'unknown')} "
            f"seconds={elapsed:.1f} json={json_path} text={text_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
