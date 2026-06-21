#!/usr/bin/env python3

import asyncio
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import nemo.collections.asr as nemo_asr
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from local_commands import match_local_command
from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    RNNTPromptTranscribeConfig,
)


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config" / "app.json"
HISTORY_PATH = PROJECT_DIR / "data" / "history.json"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


CONFIG = load_config()

user_config = CONFIG.get("user", {})

if not isinstance(user_config, dict):
    user_config = {}

configured_user_name = user_config.get("name", "Kullanıcı")

USER_NAME = (
    configured_user_name.strip()
    if (
        isinstance(configured_user_name, str)
        and configured_user_name.strip()
    )
    else "Kullanıcı"
)

APPLICATION_NAME = CONFIG["application"]["name"]
APPLICATION_VERSION = CONFIG["application"]["version"]

AUDIO_CONFIG = CONFIG["audio"]
CAPTURE_DEVICE = AUDIO_CONFIG["capture_device"]
SAMPLE_RATE = int(AUDIO_CONFIG["sample_rate"])
CHANNELS = int(AUDIO_CONFIG["channels"])
SAMPLE_FORMAT = AUDIO_CONFIG["sample_format"]
RECORDINGS_DIR = PROJECT_DIR / AUDIO_CONFIG["recordings_directory"]

ASR_MODEL_NAME = CONFIG["asr"]["model"]
ASR_LANGUAGE = CONFIG["asr"]["language"]
ASR_DEVICE = CONFIG["asr"]["device"]

OLLAMA_CONFIG = CONFIG["ollama"]
OLLAMA_BASE_URL = OLLAMA_CONFIG["base_url"].rstrip("/")
OLLAMA_CHAT_URL = (
    OLLAMA_BASE_URL
    + OLLAMA_CONFIG["chat_endpoint"]
)
OLLAMA_MODEL = OLLAMA_CONFIG["model"]
OLLAMA_KEEP_ALIVE = OLLAMA_CONFIG["keep_alive"]
OLLAMA_TEMPERATURE = float(OLLAMA_CONFIG["temperature"])
OLLAMA_TIMEOUT = int(
    OLLAMA_CONFIG["request_timeout_seconds"]
)

UI_CONFIG = CONFIG["interface"]
UI_HOST = UI_CONFIG["host"]
UI_PORT = int(UI_CONFIG["port"])
OPEN_BROWSER = bool(UI_CONFIG["open_browser_on_start"])
HISTORY_LIMIT = int(UI_CONFIG["history_limit"])


SYSTEM_PROMPT = f"""
Sen tamamen yerel çalışan bir sesli asistan testisin.

Kullanıcının mesajı otomatik konuşma tanıma sisteminden gelir.
Metinde küçük fonetik hatalar, yanlış ekler veya kelime ayrımı
sorunları bulunabilir.

Kullanıcının adı {USER_NAME}.
Uygun durumlarda kullanıcıya adıyla hitap edebilirsin.
Kullanıcı kendi adını sorarsa adının {USER_NAME} olduğunu söyle.

Kurallar:
- Kullanıcının en olası anlamını değerlendir.
- Türkçe cevap ver.
- Cevabı mümkün olduğunca kısa ve açık tut.
- Bilmediğin bilgiyi uydurma.
- Mesaj anlaşılmıyorsa kullanıcının tekrar etmesini iste.
""".strip()


state_lock = threading.Lock()
recording_lock = threading.Lock()
history_lock = threading.Lock()

asr_model = None
recording_process: subprocess.Popen | None = None
recording_path: Path | None = None
initialization_task: asyncio.Task | None = None


runtime_state: dict[str, Any] = {
    "application_started_at": datetime.now().isoformat(
        timespec="seconds"
    ),
    "asr_status": "waiting",
    "asr_message": "Nemotron henüz yüklenmedi.",
    "asr_load_seconds": None,
    "asr_loaded_at": None,
    "asr_model_class": None,
    "asr_parameter_count": None,
    "ollama_status": "checking",
    "ollama_message": "Ollama denetleniyor.",
    "ollama_model_available": False,
    "ollama_warm": False,
    "ollama_warm_seconds": None,
    "recording_status": "idle",
    "recording_message": "Mikrofon hazır.",
    "recording_started_at": None,
    "recording_file": None,
    "last_transcription": "",
    "last_answer": "",
    "last_asr_seconds": None,
    "last_llm_seconds": None,
    "history_count": 0,
    "last_error": None,
}


def update_state(**changes: Any) -> None:
    with state_lock:
        runtime_state.update(changes)


def get_state_copy() -> dict[str, Any]:
    with state_lock:
        return dict(runtime_state)


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.is_file():
        return []

    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

    except (OSError, json.JSONDecodeError):
        pass

    return []


conversation_history = load_history()
update_state(history_count=len(conversation_history))


def save_history() -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = HISTORY_PATH.with_suffix(".json.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            conversation_history,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(HISTORY_PATH)


def append_history(item: dict[str, Any]) -> None:
    with history_lock:
        conversation_history.append(item)

        if len(conversation_history) > HISTORY_LIMIT:
            del conversation_history[:-HISTORY_LIMIT]

        save_history()
        count = len(conversation_history)

    update_state(history_count=count)


def get_history_copy() -> list[dict[str, Any]]:
    with history_lock:
        return list(conversation_history)


def clear_history() -> None:
    with history_lock:
        conversation_history.clear()
        save_history()

    update_state(history_count=0)


def check_ollama_sync() -> bool:
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/tags",
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=5,
        ) as response:
            payload = json.loads(
                response.read().decode("utf-8")
            )

        available_models = {
            item.get("name")
            for item in payload.get("models", [])
        }

        model_available = OLLAMA_MODEL in available_models

        if model_available:
            update_state(
                ollama_status="available",
                ollama_message=(
                    f"{OLLAMA_MODEL} bulundu; ısıtma bekleniyor."
                ),
                ollama_model_available=True,
                last_error=None,
            )
            return True

        update_state(
            ollama_status="warning",
            ollama_message=(
                "Ollama çalışıyor fakat model bulunamadı: "
                f"{OLLAMA_MODEL}"
            ),
            ollama_model_available=False,
        )
        return False

    except Exception as error:
        update_state(
            ollama_status="error",
            ollama_message=(
                f"Ollama bağlantısı kurulamadı: {error}"
            ),
            ollama_model_available=False,
            last_error=f"{type(error).__name__}: {error}",
        )
        return False


def warm_ollama_sync() -> None:
    if not get_state_copy()["ollama_model_available"]:
        return

    update_state(
        ollama_status="warming",
        ollama_message=(
            f"{OLLAMA_MODEL} belleğe yükleniyor."
        ),
        ollama_warm=False,
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "",
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }

    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request,
            timeout=OLLAMA_TIMEOUT,
        ) as response:
            response.read()

        elapsed = time.perf_counter() - started

        update_state(
            ollama_status="ready",
            ollama_message=(
                f"{OLLAMA_MODEL} bellekte ve sıcak."
            ),
            ollama_warm=True,
            ollama_warm_seconds=round(elapsed, 2),
            last_error=None,
        )

    except Exception as error:
        update_state(
            ollama_status="error",
            ollama_message=(
                f"Ollama modeli ısıtılamadı: {error}"
            ),
            ollama_warm=False,
            last_error=f"{type(error).__name__}: {error}",
        )


def load_asr_model_sync() -> None:
    global asr_model

    update_state(
        asr_status="loading",
        asr_message="Nemotron CPU belleğine yükleniyor.",
        last_error=None,
    )

    started = time.perf_counter()

    try:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=ASR_MODEL_NAME,
            map_location=ASR_DEVICE,
        )

        model.eval()

        if hasattr(model.decoding, "set_strip_lang_tags"):
            model.decoding.set_strip_lang_tags(True)

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        asr_model = model

        elapsed = time.perf_counter() - started

        update_state(
            asr_status="ready",
            asr_message=(
                "Nemotron bellekte ve kullanıma hazır."
            ),
            asr_load_seconds=round(elapsed, 2),
            asr_loaded_at=datetime.now().isoformat(
                timespec="seconds"
            ),
            asr_model_class=type(model).__name__,
            asr_parameter_count=parameter_count,
            last_error=None,
        )

    except Exception as error:
        update_state(
            asr_status="error",
            asr_message="Nemotron yüklenemedi.",
            last_error=f"{type(error).__name__}: {error}",
        )


async def initialize_models() -> None:
    await asyncio.to_thread(check_ollama_sync)
    await asyncio.to_thread(load_asr_model_sync)
    await asyncio.to_thread(warm_ollama_sync)


def start_recording_sync() -> dict[str, Any]:
    global recording_process
    global recording_path

    state = get_state_copy()

    if state["asr_status"] != "ready":
        raise RuntimeError(
            "Nemotron henüz kullanıma hazır değil."
        )

    if state["ollama_status"] != "ready":
        raise RuntimeError(
            "Gemma 4 henüz kullanıma hazır değil."
        )

    with recording_lock:
        if (
            recording_process is not None
            and recording_process.poll() is None
        ):
            raise RuntimeError("Kayıt zaten devam ediyor.")

        RECORDINGS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        new_recording_path = (
            RECORDINGS_DIR
            / f"voice_{timestamp}.wav"
        )

        command = [
            "arecord",
            "-q",
            "-D",
            CAPTURE_DEVICE,
            "-f",
            SAMPLE_FORMAT,
            "-r",
            str(SAMPLE_RATE),
            "-c",
            str(CHANNELS),
            "-t",
            "wav",
            str(new_recording_path),
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        time.sleep(0.25)

        if process.poll() is not None:
            raise RuntimeError(
                "arecord başlatılamadı. "
                "Mikrofon başka bir uygulama tarafından "
                "kullanılıyor olabilir."
            )

        recording_process = process
        recording_path = new_recording_path

    started_at = datetime.now().isoformat(
        timespec="seconds"
    )

    update_state(
        recording_status="recording",
        recording_message="Kayıt devam ediyor.",
        recording_started_at=started_at,
        recording_file=str(new_recording_path),
        last_error=None,
    )

    return {
        "status": "recording",
        "message": "Kayıt başladı.",
        "started_at": started_at,
        "file": str(new_recording_path),
    }


def stop_recording_sync() -> Path:
    global recording_process
    global recording_path

    with recording_lock:
        process = recording_process
        audio_path = recording_path

        if process is None or audio_path is None:
            raise RuntimeError(
                "Durdurulacak aktif kayıt bulunamadı."
            )

        if process.poll() is None:
            try:
                os.killpg(
                    process.pid,
                    signal.SIGINT,
                )
                process.wait(timeout=5)

            except subprocess.TimeoutExpired:
                process.terminate()

                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

        recording_process = None
        recording_path = None

    if not audio_path.is_file():
        raise RuntimeError(
            "Kayıt dosyası oluşturulamadı."
        )

    if audio_path.stat().st_size <= 44:
        raise RuntimeError(
            "Kayıt dosyası boş görünüyor."
        )

    update_state(
        recording_status="processing",
        recording_message=(
            "Ses Nemotron ve Gemma tarafından işleniyor."
        ),
    )

    return audio_path


def transcribe_audio_sync(
    audio_path: Path,
) -> tuple[str, float]:
    if asr_model is None:
        raise RuntimeError(
            "Nemotron modeli bellekte değil."
        )

    config = RNNTPromptTranscribeConfig(
        use_lhotse=False,
        batch_size=1,
        return_hypotheses=False,
        num_workers=0,
        verbose=False,
        target_lang=ASR_LANGUAGE,
    )

    started = time.perf_counter()

    with torch.inference_mode():
        result = asr_model.transcribe(
            audio=[str(audio_path)],
            override_config=config,
        )

    elapsed = time.perf_counter() - started

    if isinstance(result, tuple):
        result = result[0]

    if not result:
        raise RuntimeError(
            "Nemotron herhangi bir sonuç üretmedi."
        )

    first_item = result[0]

    text = (
        first_item.text
        if hasattr(first_item, "text")
        else str(first_item)
    )

    text = text.strip()

    if not text:
        raise RuntimeError(
            "Nemotron boş metin üretti."
        )

    return text, elapsed


def build_ollama_messages(
    user_text: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    recent_history = get_history_copy()[-6:]

    for item in recent_history:
        messages.append(
            {
                "role": "user",
                "content": item["transcription"],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": item["answer"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    return messages


def ask_ollama_sync(
    user_text: str,
) -> tuple[str, float]:
    local_started = time.perf_counter()
    local_answer = match_local_command(user_text, USER_NAME)

    if local_answer is not None:
        return (
            local_answer,
            time.perf_counter() - local_started,
        )

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "messages": build_ollama_messages(
            user_text
        ),
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
        },
    }

    request = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request,
            timeout=OLLAMA_TIMEOUT,
        ) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as error:
        body = error.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"Ollama HTTP hatası {error.code}: {body}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama API bağlantısı kurulamadı."
        ) from error

    elapsed = time.perf_counter() - started

    try:
        answer = response_data["message"]["content"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "Ollama beklenen yanıt biçimini üretmedi."
        ) from error

    answer = answer.strip()

    if not answer:
        raise RuntimeError(
            "Ollama boş cevap üretti."
        )

    return answer, elapsed


def process_recording_sync(
    audio_path: Path,
) -> dict[str, Any]:
    try:
        transcription, asr_seconds = (
            transcribe_audio_sync(audio_path)
        )

        update_state(
            last_transcription=transcription,
            last_asr_seconds=round(
                asr_seconds,
                2,
            ),
        )

        answer, llm_seconds = ask_ollama_sync(
            transcription
        )

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )

        history_item = {
            "id": timestamp.replace(
                ":",
                "",
            ).replace(
                "-",
                "",
            ),
            "timestamp": timestamp,
            "transcription": transcription,
            "answer": answer,
            "asr_seconds": round(
                asr_seconds,
                2,
            ),
            "llm_seconds": round(
                llm_seconds,
                2,
            ),
            "audio_file": str(audio_path),
        }

        append_history(history_item)

        update_state(
            recording_status="idle",
            recording_message="Mikrofon hazır.",
            recording_started_at=None,
            last_transcription=transcription,
            last_answer=answer,
            last_asr_seconds=round(
                asr_seconds,
                2,
            ),
            last_llm_seconds=round(
                llm_seconds,
                2,
            ),
            last_error=None,
        )

        return history_item

    except Exception as error:
        update_state(
            recording_status="idle",
            recording_message=(
                "İşlem başarısız oldu; mikrofon yeniden hazır."
            ),
            last_error=f"{type(error).__name__}: {error}",
        )
        raise


def stop_recording_if_active() -> None:
    global recording_process
    global recording_path

    with recording_lock:
        process = recording_process

        if process is not None and process.poll() is None:
            try:
                os.killpg(
                    process.pid,
                    signal.SIGINT,
                )
                process.wait(timeout=3)
            except Exception:
                process.kill()

        recording_process = None
        recording_path = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global initialization_task

    initialization_task = asyncio.create_task(
        initialize_models()
    )

    yield

    stop_recording_if_active()

    if (
        initialization_task
        and not initialization_task.done()
    ):
        initialization_task.cancel()


app = FastAPI(
    title=APPLICATION_NAME,
    version=APPLICATION_VERSION,
    lifespan=lifespan,
)


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    state = get_state_copy()

    state["torch_version"] = torch.__version__
    state["cuda_available"] = (
        torch.cuda.is_available()
    )
    state["configured_asr_model"] = (
        ASR_MODEL_NAME
    )
    state["configured_ollama_model"] = (
        OLLAMA_MODEL
    )
    state["capture_device"] = CAPTURE_DEVICE

    return state


@app.get("/api/history")
async def api_history() -> dict[str, Any]:
    return {
        "items": list(
            reversed(get_history_copy())
        )
    }


@app.post("/api/history/clear")
async def api_history_clear() -> dict[str, Any]:
    await asyncio.to_thread(clear_history)

    return {
        "status": "ok",
        "message": "Konuşma geçmişi temizlendi.",
    }


@app.post("/api/record/start")
async def api_record_start() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(
            start_recording_sync
        )

    except Exception as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error


@app.post("/api/record/stop")
async def api_record_stop() -> dict[str, Any]:
    try:
        audio_path = await asyncio.to_thread(
            stop_recording_sync
        )

        return await asyncio.to_thread(
            process_recording_sync,
            audio_path,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
  >
  <title>Asunode Local Voice Assistant</title>

  <style>
    :root {
      color-scheme: dark;
      --background: #0a1f3d;
      --panel: #102a4c;
      --panel-soft: #163657;
      --border: #2a4f78;
      --text: #e7edf6;
      --muted: #91a0b5;
      --cyan: #5ed9e7;
      --green: #5fe09b;
      --yellow: #f0c96a;
      --red: #f17b83;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
      background:
        radial-gradient(
          circle at top right,
          rgba(94, 217, 231, 0.08),
          transparent 34rem
        ),
        var(--background);
      color: var(--text);
    }

    main {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 34px 0 64px;
    }

    header {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-start;
      margin-bottom: 24px;
    }

    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: -0.04em;
    }

    header p,
    .muted {
      margin: 0;
      color: var(--muted);
    }

    .local-badge {
      border: 1px solid rgba(95, 224, 155, 0.42);
      background: rgba(95, 224, 155, 0.08);
      color: var(--green);
      border-radius: 999px;
      padding: 9px 14px;
      white-space: nowrap;
      font-weight: 800;
      font-size: 13px;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }

    .card {
      border: 1px solid var(--border);
      background:
        linear-gradient(
          145deg,
          rgba(255, 255, 255, 0.025),
          transparent
        ),
        var(--panel);
      border-radius: 20px;
      padding: 22px;
      box-shadow:
        0 18px 50px rgba(0, 0, 0, 0.18);
    }

    .card-wide {
      grid-column: 1 / -1;
    }

    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }

    .card h2 {
      margin: 0;
      font-size: 18px;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 7px 11px;
      font-weight: 800;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      background: var(--panel-soft);
      color: var(--muted);
    }

    .status::before {
      content: "";
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 14px currentColor;
    }

    .status-ready,
    .status-available {
      color: var(--green);
    }

    .status-loading,
    .status-checking,
    .status-waiting,
    .status-warming,
    .status-recording,
    .status-processing,
    .status-warning {
      color: var(--yellow);
    }

    .status-error {
      color: var(--red);
    }

    .description {
      margin: 0 0 18px;
      min-height: 44px;
      color: var(--muted);
      line-height: 1.55;
    }

    dl {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px 18px;
      margin: 0;
      font-size: 14px;
    }

    dt {
      color: var(--muted);
    }

    dd {
      margin: 0;
      text-align: right;
      overflow-wrap: anywhere;
    }

    .assistant {
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 24px;
      align-items: stretch;
    }

    .microphone-panel {
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      min-height: 310px;
      border: 1px solid var(--border);
      background: var(--panel-soft);
      border-radius: 18px;
      padding: 20px;
      text-align: center;
    }

    .microphone-button {
      width: 138px;
      height: 138px;
      border: 1px solid rgba(94, 217, 231, 0.55);
      border-radius: 50%;
      background:
        radial-gradient(
          circle,
          rgba(94, 217, 231, 0.18),
          rgba(94, 217, 231, 0.06)
        );
      color: var(--cyan);
      font: inherit;
      font-size: 17px;
      font-weight: 900;
      cursor: pointer;
      box-shadow:
        0 0 28px rgba(94, 217, 231, 0.08);
    }

    .microphone-button.recording {
      color: var(--red);
      border-color: rgba(241, 123, 131, 0.7);
      background:
        radial-gradient(
          circle,
          rgba(241, 123, 131, 0.23),
          rgba(241, 123, 131, 0.06)
        );
      animation: pulse 1.4s infinite;
    }

    .microphone-button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }

    @keyframes pulse {
      50% {
        box-shadow:
          0 0 0 16px rgba(241, 123, 131, 0);
      }
    }

    .microphone-help {
      margin: 18px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }

    .result-grid {
      display: grid;
      gap: 16px;
    }

    .result-box {
      border: 1px solid var(--border);
      background: var(--panel-soft);
      border-radius: 16px;
      padding: 18px;
      min-height: 116px;
    }

    .result-label {
      margin: 0 0 10px;
      color: var(--cyan);
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .result-text {
      margin: 0;
      font-size: 17px;
      line-height: 1.6;
      white-space: pre-wrap;
    }

    .metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .metric {
      border: 1px solid var(--border);
      background: var(--panel-soft);
      border-radius: 12px;
      padding: 9px 12px;
      color: var(--muted);
      font-size: 13px;
    }

    .history-list {
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }

    .history-item {
      border: 1px solid var(--border);
      background: var(--panel-soft);
      border-radius: 15px;
      padding: 16px;
    }

    .history-time {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
    }

    .history-user,
    .history-assistant {
      margin: 6px 0;
      line-height: 1.5;
    }

    .history-user strong {
      color: var(--cyan);
    }

    .history-assistant strong {
      color: var(--green);
    }

    .toolbar {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
    }

    button.secondary {
      border: 1px solid var(--border);
      background: var(--panel-soft);
      color: var(--muted);
      border-radius: 12px;
      min-height: 42px;
      padding: 0 14px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }

    .error {
      display: none;
      margin-top: 18px;
      border: 1px solid rgba(241, 123, 131, 0.42);
      background: rgba(241, 123, 131, 0.08);
      color: var(--red);
      border-radius: 14px;
      padding: 14px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    footer {
      margin-top: 24px;
      color: var(--muted);
      font-size: 13px;
      text-align: center;
    }

    @media (max-width: 800px) {
      header {
        flex-direction: column;
      }

      .grid {
        grid-template-columns: 1fr;
      }

      .card-wide {
        grid-column: auto;
      }

      .assistant {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>

<body>
  <main>
    <header>
      <div>
        <h1>Asunode Voice Assistant</h1>
        <p>Nemotron ASR + Gemma 4 · Yerel sesli asistan</p>
      </div>

      <div class="local-badge">
        LOCAL ONLY
      </div>
    </header>

    <section class="grid">
      <article class="card">
        <div class="card-header">
          <h2>Nemotron ASR</h2>
          <span id="asrStatus" class="status status-waiting">
            Bekleniyor
          </span>
        </div>

        <p id="asrMessage" class="description">
          Nemotron durumu okunuyor.
        </p>

        <dl>
          <dt>Model</dt>
          <dd id="asrModel">—</dd>

          <dt>Yükleme</dt>
          <dd id="asrLoadTime">—</dd>

          <dt>Parametre</dt>
          <dd id="asrParameters">—</dd>

          <dt>Aygıt</dt>
          <dd>CPU</dd>
        </dl>
      </article>

      <article class="card">
        <div class="card-header">
          <h2>Ollama / Gemma 4</h2>
          <span id="ollamaStatus" class="status status-checking">
            Denetleniyor
          </span>
        </div>

        <p id="ollamaMessage" class="description">
          Ollama durumu okunuyor.
        </p>

        <dl>
          <dt>Model</dt>
          <dd id="ollamaModel">—</dd>

          <dt>Sıcak tutma</dt>
          <dd>30 dakika</dd>

          <dt>Isıtma süresi</dt>
          <dd id="ollamaWarmTime">—</dd>

          <dt>Çalışma</dt>
          <dd>Yerel</dd>
        </dl>
      </article>

      <article class="card card-wide">
        <div class="card-header">
          <h2>Sesli asistan</h2>
          <span id="recordingStatus" class="status status-waiting">
            Hazırlanıyor
          </span>
        </div>

        <div class="assistant">
          <div class="microphone-panel">
            <button
              id="microphoneButton"
              class="microphone-button"
              type="button"
              disabled
            >
              Başlat
            </button>

            <p id="recordingMessage" class="microphone-help">
              Modeller hazırlanıyor.
            </p>
          </div>

          <div class="result-grid">
            <div class="result-box">
              <p class="result-label">Nemotron metni</p>
              <p id="transcription" class="result-text">
                Henüz konuşma kaydı yok.
              </p>
            </div>

            <div class="result-box">
              <p class="result-label">Gemma 4 cevabı</p>
              <p id="answer" class="result-text">
                Henüz cevap yok.
              </p>
            </div>

            <div class="metrics">
              <span class="metric">
                ASR:
                <strong id="asrSeconds">—</strong>
              </span>

              <span class="metric">
                LLM:
                <strong id="llmSeconds">—</strong>
              </span>

              <span class="metric">
                Mikrofon:
                <strong id="captureDevice">—</strong>
              </span>

              <span class="metric">
                TTS:
                <strong>Henüz kapalı</strong>
              </span>
            </div>
          </div>
        </div>

        <div id="errorBox" class="error"></div>
      </article>

      <article class="card card-wide">
        <div class="card-header">
          <h2>Konuşma geçmişi</h2>

          <div class="toolbar">
            <button
              id="clearHistoryButton"
              class="secondary"
              type="button"
            >
              Geçmişi temizle
            </button>
          </div>
        </div>

        <div id="historyList" class="history-list">
          <p class="muted">Henüz konuşma geçmişi yok.</p>
        </div>
      </article>
    </section>

    <footer>
      Kayıt doğrudan yerel Logitech ALSA mikrofonundan alınır.
    </footer>
  </main>

  <script>
    const statusLabels = {
      waiting: "Bekleniyor",
      loading: "Yükleniyor",
      ready: "Hazır",
      available: "Bulundu",
      warming: "Isıtılıyor",
      checking: "Denetleniyor",
      warning: "Uyarı",
      error: "Hata",
      idle: "Hazır",
      recording: "Kayıt",
      processing: "İşleniyor"
    };

    let recording = false;
    let busy = false;

    function setStatus(element, status) {
      element.className = `status status-${status}`;
      element.textContent = statusLabels[status] ?? status;
    }

    function formatParameters(value) {
      if (!value) {
        return "—";
      }

      return `${(value / 1000000).toFixed(1)} milyon`;
    }

    function setError(message) {
      const box = document.getElementById("errorBox");

      if (message) {
        box.style.display = "block";
        box.textContent = message;
      } else {
        box.style.display = "none";
        box.textContent = "";
      }
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, {
        cache: "no-store",
        ...options
      });

      let data = null;

      try {
        data = await response.json();
      } catch {
        data = null;
      }

      if (!response.ok) {
        const message =
          data?.detail
          ?? `HTTP ${response.status}`;

        throw new Error(message);
      }

      return data;
    }

    async function refreshStatus() {
      try {
        const data = await requestJson("/api/status");

        setStatus(
          document.getElementById("asrStatus"),
          data.asr_status
        );

        setStatus(
          document.getElementById("ollamaStatus"),
          data.ollama_status
        );

        setStatus(
          document.getElementById("recordingStatus"),
          data.recording_status
        );

        document.getElementById("asrMessage").textContent =
          data.asr_message;

        document.getElementById("ollamaMessage").textContent =
          data.ollama_message;

        document.getElementById("recordingMessage").textContent =
          data.recording_message;

        document.getElementById("asrModel").textContent =
          data.configured_asr_model;

        document.getElementById("ollamaModel").textContent =
          data.configured_ollama_model;

        document.getElementById("captureDevice").textContent =
          data.capture_device;

        document.getElementById("asrLoadTime").textContent =
          data.asr_load_seconds === null
            ? "—"
            : `${data.asr_load_seconds} saniye`;

        document.getElementById("ollamaWarmTime").textContent =
          data.ollama_warm_seconds === null
            ? "—"
            : `${data.ollama_warm_seconds} saniye`;

        document.getElementById("asrParameters").textContent =
          formatParameters(data.asr_parameter_count);

        if (data.last_transcription) {
          document.getElementById("transcription").textContent =
            data.last_transcription;
        }

        if (data.last_answer) {
          document.getElementById("answer").textContent =
            data.last_answer;
        }

        document.getElementById("asrSeconds").textContent =
          data.last_asr_seconds === null
            ? "—"
            : `${data.last_asr_seconds} sn`;

        document.getElementById("llmSeconds").textContent =
          data.last_llm_seconds === null
            ? "—"
            : `${data.last_llm_seconds} sn`;

        recording =
          data.recording_status === "recording";

        busy =
          data.recording_status === "processing";

        const button =
          document.getElementById("microphoneButton");

        const modelsReady =
          data.asr_status === "ready"
          && data.ollama_status === "ready";

        button.disabled =
          !modelsReady || busy;

        button.classList.toggle(
          "recording",
          recording
        );

        button.textContent =
          recording
            ? "Durdur"
            : busy
              ? "İşleniyor"
              : "Başlat";

        setError(data.last_error);
      } catch (error) {
        setError(
          `Durum servisine erişilemedi: ${error.message}`
        );
      }
    }

    async function startRecording() {
      busy = true;
      setError("");

      const button =
        document.getElementById("microphoneButton");

      button.disabled = true;
      button.textContent = "Başlatılıyor";

      try {
        await requestJson(
          "/api/record/start",
          {
            method: "POST"
          }
        );

        recording = true;
        await refreshStatus();
      } catch (error) {
        setError(error.message);
        recording = false;
      } finally {
        busy = false;
        await refreshStatus();
      }
    }

    async function stopRecording() {
      busy = true;
      setError("");

      const button =
        document.getElementById("microphoneButton");

      button.disabled = true;
      button.textContent = "İşleniyor";

      try {
        const result = await requestJson(
          "/api/record/stop",
          {
            method: "POST"
          }
        );

        document.getElementById("transcription").textContent =
          result.transcription;

        document.getElementById("answer").textContent =
          result.answer;

        document.getElementById("asrSeconds").textContent =
          `${result.asr_seconds} sn`;

        document.getElementById("llmSeconds").textContent =
          `${result.llm_seconds} sn`;

        recording = false;

        await refreshHistory();
      } catch (error) {
        setError(error.message);
      } finally {
        busy = false;
        recording = false;
        await refreshStatus();
      }
    }

    async function toggleRecording() {
      if (busy) {
        return;
      }

      if (recording) {
        await stopRecording();
      } else {
        await startRecording();
      }
    }

    async function refreshHistory() {
      try {
        const data = await requestJson("/api/history");
        const list = document.getElementById("historyList");

        if (!data.items.length) {
          list.innerHTML =
            '<p class="muted">Henüz konuşma geçmişi yok.</p>';
          return;
        }

        list.innerHTML = "";

        for (const item of data.items) {
          const article = document.createElement("article");
          article.className = "history-item";

          const time = document.createElement("div");
          time.className = "history-time";
          time.textContent =
            `${item.timestamp} · ASR ${item.asr_seconds} sn · `
            + `LLM ${item.llm_seconds} sn`;

          const user = document.createElement("p");
          user.className = "history-user";

          const userLabel = document.createElement("strong");
          userLabel.textContent = "Sen: ";

          user.appendChild(userLabel);
          user.appendChild(
            document.createTextNode(item.transcription)
          );

          const assistant = document.createElement("p");
          assistant.className = "history-assistant";

          const assistantLabel =
            document.createElement("strong");

          assistantLabel.textContent = "Gemma: ";

          assistant.appendChild(assistantLabel);
          assistant.appendChild(
            document.createTextNode(item.answer)
          );

          article.appendChild(time);
          article.appendChild(user);
          article.appendChild(assistant);
          list.appendChild(article);
        }
      } catch (error) {
        setError(error.message);
      }
    }

    async function clearHistory() {
      try {
        await requestJson(
          "/api/history/clear",
          {
            method: "POST"
          }
        );

        await refreshHistory();
      } catch (error) {
        setError(error.message);
      }
    }

    document
      .getElementById("microphoneButton")
      .addEventListener(
        "click",
        toggleRecording
      );

    document
      .getElementById("clearHistoryButton")
      .addEventListener(
        "click",
        clearHistory
      );

    refreshStatus();
    refreshHistory();

    setInterval(
      refreshStatus,
      1500
    );
  </script>
</body>
</html>
"""


def open_browser() -> None:
    webbrowser.open(
        f"http://{UI_HOST}:{UI_PORT}"
    )


def main() -> None:
    print()
    print(f"{APPLICATION_NAME} başlatılıyor.")
    print(f"Arayüz: http://{UI_HOST}:{UI_PORT}")
    print("Durdurmak için Ctrl+C kullanın.")
    print()

    if OPEN_BROWSER:
        timer = threading.Timer(
            1.5,
            open_browser,
        )
        timer.daemon = True
        timer.start()

    uvicorn.run(
        app,
        host=UI_HOST,
        port=UI_PORT,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
