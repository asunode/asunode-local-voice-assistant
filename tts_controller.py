#!/usr/bin/env python3

import json
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class TTSController:
    """Persistent Piper worker controller."""

    def __init__(
        self,
        project_dir: Path,
        config: dict[str, Any],
    ) -> None:
        self.project_dir = project_dir.resolve()
        self.config = config
        self.tts_config = config["tts"]

        self.enabled = bool(
            self.tts_config.get("enabled", False)
        )

        self.voice = self.tts_config.get("voice")
        self.auto_play = bool(
            self.tts_config.get(
                "auto_play_response",
                True,
            )
        )

        python_value = Path(
            self.tts_config["python_executable"]
        )

        if not python_value.is_absolute():
            python_value = self.project_dir / python_value

        # Sanal ortamın python bağlantısını çözümleme.
        # Path.resolve() bağlantıyı /usr/bin/python3.13'e dönüştürür
        # ve .venv-tts ortamının devre dışı kalmasına neden olur.
        self.python_executable = python_value.absolute()

        self.worker_script = (
            self.project_dir / "tts_worker.py"
        ).resolve()

        self.output_directory = self._resolve_path(
            self.tts_config["output_directory"]
        )

        self.playback_command = self.tts_config.get(
            "playback_command",
            "aplay",
        )

        self.playback_device = self.tts_config.get(
            "playback_device",
            "default",
        )

        self.process: subprocess.Popen[str] | None = None
        self.stderr_file = None

        self.lock = threading.RLock()

        self.ready = False
        self.load_seconds: float | None = None
        self.last_synthesis_seconds: float | None = None
        self.last_output: Path | None = None
        self.last_error: str | None = None

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)

        if not path.is_absolute():
            path = self.project_dir / path

        return path.resolve()

    def _read_response(self) -> dict[str, Any]:
        if self.process is None:
            raise RuntimeError(
                "Piper worker süreci bulunamadı."
            )

        if self.process.stdout is None:
            raise RuntimeError(
                "Piper worker stdout bağlantısı bulunamadı."
            )

        line = self.process.stdout.readline()

        if not line:
            return_code = self.process.poll()

            raise RuntimeError(
                "Piper worker beklenmedik biçimde kapandı. "
                f"Çıkış kodu: {return_code}"
            )

        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Piper worker geçersiz JSON üretti: "
                f"{line.strip()}"
            ) from error

        return response

    def _send_request(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if self.process is None:
            raise RuntimeError(
                "Piper worker çalışmıyor."
            )

        if self.process.poll() is not None:
            raise RuntimeError(
                "Piper worker kapalı durumda."
            )

        if self.process.stdin is None:
            raise RuntimeError(
                "Piper worker stdin bağlantısı bulunamadı."
            )

        self.process.stdin.write(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
            + "\n"
        )
        self.process.stdin.flush()

        return self._read_response()

    def start(self) -> dict[str, Any]:
        if not self.enabled:
            self.ready = False

            return {
                "status": "disabled",
                "message": "TTS yapılandırmada kapalı.",
            }

        with self.lock:
            if (
                self.process is not None
                and self.process.poll() is None
                and self.ready
            ):
                return self.status()

            if not self.python_executable.is_file():
                raise RuntimeError(
                    "TTS Python çalıştırıcısı bulunamadı: "
                    f"{self.python_executable}"
                )

            if not self.worker_script.is_file():
                raise RuntimeError(
                    "TTS worker dosyası bulunamadı: "
                    f"{self.worker_script}"
                )

            self.output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            log_directory = (
                self.project_dir / "logs"
            )
            log_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            stderr_path = (
                log_directory / "tts_worker.stderr.log"
            )

            self.stderr_file = stderr_path.open(
                "a",
                encoding="utf-8",
            )

            started = time.perf_counter()

            self.process = subprocess.Popen(
                [
                    str(self.python_executable),
                    str(self.worker_script),
                ],
                cwd=str(self.project_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr_file,
                text=True,
                bufsize=1,
            )

            response = self._read_response()

            if response.get("status") != "ready":
                self.last_error = str(
                    response.get(
                        "error",
                        "Piper worker hazır olmadı.",
                    )
                )

                self.shutdown()

                raise RuntimeError(self.last_error)

            elapsed = time.perf_counter() - started

            self.ready = True
            self.load_seconds = float(
                response.get(
                    "load_seconds",
                    elapsed,
                )
            )
            self.voice = response.get(
                "voice",
                self.voice,
            )
            self.last_error = None

            return self.status()

    def synthesize(
        self,
        text: str,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        clean_text = text.strip()

        if not clean_text:
            raise ValueError(
                "Seslendirilecek metin boş."
            )

        with self.lock:
            if not self.ready:
                self.start()

            if output_path is None:
                timestamp = time.strftime(
                    "%Y%m%d_%H%M%S"
                )

                output_path = (
                    self.output_directory
                    / f"response_{timestamp}.wav"
                )

            output_path = output_path.resolve()
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            request_id = uuid.uuid4().hex

            response = self._send_request(
                {
                    "request_id": request_id,
                    "action": "synthesize",
                    "text": clean_text,
                    "output": str(output_path),
                }
            )

            if response.get("status") != "completed":
                self.last_error = str(
                    response.get(
                        "error",
                        "Piper ses üretimi başarısız.",
                    )
                )

                raise RuntimeError(self.last_error)

            self.last_synthesis_seconds = float(
                response.get(
                    "synthesis_seconds",
                    0.0,
                )
            )
            self.last_output = Path(
                response["output"]
            )
            self.last_error = None

            return {
                "status": "completed",
                "output": str(self.last_output),
                "synthesis_seconds": (
                    self.last_synthesis_seconds
                ),
                "size_bytes": response.get(
                    "size_bytes"
                ),
            }

    def play(
        self,
        audio_path: Path | None = None,
    ) -> dict[str, Any]:
        selected_path = (
            audio_path.resolve()
            if audio_path is not None
            else self.last_output
        )

        if selected_path is None:
            raise RuntimeError(
                "Çalınacak TTS ses dosyası yok."
            )

        if not selected_path.is_file():
            raise RuntimeError(
                "TTS ses dosyası bulunamadı: "
                f"{selected_path}"
            )

        command = [
            self.playback_command,
            "-q",
        ]

        if self.playback_device:
            command.extend(
                [
                    "-D",
                    self.playback_device,
                ]
            )

        command.append(str(selected_path))

        started = time.perf_counter()

        completed = subprocess.run(
            command,
            cwd=str(self.project_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
            check=False,
        )

        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            message = completed.stderr.strip()

            raise RuntimeError(
                "Ses çalınamadı: "
                f"{message or completed.returncode}"
            )

        return {
            "status": "played",
            "output": str(selected_path),
            "playback_seconds": round(
                elapsed,
                3,
            ),
        }

    def synthesize_and_play(
        self,
        text: str,
    ) -> dict[str, Any]:
        synthesis_result = self.synthesize(text)

        playback_result = self.play(
            Path(synthesis_result["output"])
        )

        return {
            **synthesis_result,
            **playback_result,
        }

    def set_auto_play(
        self,
        enabled: bool,
    ) -> dict[str, Any]:
        self.auto_play = bool(enabled)

        return self.status()

    def status(self) -> dict[str, Any]:
        process_running = (
            self.process is not None
            and self.process.poll() is None
        )

        return {
            "enabled": self.enabled,
            "ready": self.ready and process_running,
            "voice": self.voice,
            "auto_play": self.auto_play,
            "load_seconds": self.load_seconds,
            "last_synthesis_seconds": (
                self.last_synthesis_seconds
            ),
            "last_output": (
                str(self.last_output)
                if self.last_output is not None
                else None
            ),
            "last_error": self.last_error,
        }

    def shutdown(self) -> None:
        with self.lock:
            process = self.process

            if process is not None:
                if process.poll() is None:
                    try:
                        response = self._send_request(
                            {
                                "request_id": uuid.uuid4().hex,
                                "action": "shutdown",
                            }
                        )

                        if response.get("status") != "shutdown":
                            process.terminate()

                    except Exception:
                        process.terminate()

                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)

            self.process = None
            self.ready = False

            if self.stderr_file is not None:
                self.stderr_file.close()
                self.stderr_file = None
