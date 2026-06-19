#!/usr/bin/env python3

import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

from piper import PiperVoice


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config" / "app.json"


def write_response(payload: dict[str, Any]) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        flush=True,
    )


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path.resolve()


def main() -> int:
    config = load_config()
    tts_config = config["tts"]

    voice_name = tts_config["voice"]
    models_directory = resolve_project_path(
        tts_config["models_directory"]
    )

    model_path = models_directory / f"{voice_name}.onnx"

    if not model_path.is_file():
        write_response(
            {
                "status": "error",
                "error": f"Piper modeli bulunamadı: {model_path}",
            }
        )
        return 1

    load_started = time.perf_counter()

    try:
        voice = PiperVoice.load(str(model_path))
    except Exception as error:
        write_response(
            {
                "status": "error",
                "error": (
                    f"Piper modeli yüklenemedi: "
                    f"{type(error).__name__}: {error}"
                ),
            }
        )
        return 1

    load_seconds = time.perf_counter() - load_started

    write_response(
        {
            "status": "ready",
            "voice": voice_name,
            "model_path": str(model_path),
            "load_seconds": round(load_seconds, 3),
        }
    )

    for input_line in sys.stdin:
        input_line = input_line.strip()

        if not input_line:
            continue

        request_id = None

        try:
            request = json.loads(input_line)
            request_id = request.get("request_id")
            action = request.get("action")

            if action == "shutdown":
                write_response(
                    {
                        "status": "shutdown",
                        "request_id": request_id,
                    }
                )
                return 0

            if action != "synthesize":
                raise ValueError(
                    f"Bilinmeyen işlem: {action}"
                )

            text = str(request.get("text", "")).strip()

            if not text:
                raise ValueError(
                    "Seslendirilecek metin boş."
                )

            output_value = str(
                request.get("output", "")
            ).strip()

            if not output_value:
                raise ValueError(
                    "Çıktı dosyası belirtilmedi."
                )

            output_path = resolve_project_path(
                output_value
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            synthesis_started = time.perf_counter()

            with wave.open(
                str(output_path),
                "wb",
            ) as wav_file:
                voice.synthesize_wav(
                    text,
                    wav_file,
                )

            synthesis_seconds = (
                time.perf_counter()
                - synthesis_started
            )

            write_response(
                {
                    "status": "completed",
                    "request_id": request_id,
                    "output": str(output_path),
                    "text": text,
                    "synthesis_seconds": round(
                        synthesis_seconds,
                        3,
                    ),
                    "size_bytes": output_path.stat().st_size,
                }
            )

        except Exception as error:
            write_response(
                {
                    "status": "error",
                    "request_id": request_id,
                    "error": (
                        f"{type(error).__name__}: {error}"
                    ),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
