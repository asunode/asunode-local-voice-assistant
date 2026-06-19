#!/usr/bin/env python3

import asyncio
import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import nemo.collections.asr as nemo_asr
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config" / "app.json"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


CONFIG = load_config()

ASR_MODEL_NAME = CONFIG["asr"]["model"]
ASR_DEVICE = CONFIG["asr"]["device"]

OLLAMA_BASE_URL = CONFIG["ollama"]["base_url"].rstrip("/")
OLLAMA_MODEL = CONFIG["ollama"]["model"]

UI_HOST = CONFIG["interface"]["host"]
UI_PORT = int(CONFIG["interface"]["port"])


state_lock = threading.Lock()

runtime_state: dict[str, Any] = {
    "application_started_at": datetime.now().isoformat(timespec="seconds"),
    "asr_status": "waiting",
    "asr_message": "Nemotron henüz yüklenmedi.",
    "asr_load_seconds": None,
    "asr_loaded_at": None,
    "asr_model_class": None,
    "asr_parameter_count": None,
    "ollama_status": "checking",
    "ollama_message": "Ollama denetleniyor.",
    "ollama_model_available": False,
    "last_error": None,
}

asr_model = None
asr_load_task: asyncio.Task | None = None


def update_state(**changes: Any) -> None:
    with state_lock:
        runtime_state.update(changes)


def get_state_copy() -> dict[str, Any]:
    with state_lock:
        return dict(runtime_state)


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
            asr_message="Nemotron bellekte ve kullanıma hazır.",
            asr_load_seconds=round(elapsed, 2),
            asr_loaded_at=datetime.now().isoformat(timespec="seconds"),
            asr_model_class=type(model).__name__,
            asr_parameter_count=parameter_count,
        )

    except Exception as error:
        update_state(
            asr_status="error",
            asr_message="Nemotron yüklenemedi.",
            last_error=f"{type(error).__name__}: {error}",
        )


def check_ollama_sync() -> None:
    url = f"{OLLAMA_BASE_URL}/api/tags"

    request = urllib.request.Request(
        url,
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))

        available_models = {
            item.get("name")
            for item in payload.get("models", [])
        }

        model_available = OLLAMA_MODEL in available_models

        if model_available:
            update_state(
                ollama_status="ready",
                ollama_message=f"{OLLAMA_MODEL} kullanıma hazır.",
                ollama_model_available=True,
            )
        else:
            update_state(
                ollama_status="warning",
                ollama_message=(
                    "Ollama çalışıyor fakat seçilen model bulunamadı: "
                    f"{OLLAMA_MODEL}"
                ),
                ollama_model_available=False,
            )

    except urllib.error.URLError as error:
        update_state(
            ollama_status="error",
            ollama_message=f"Ollama bağlantısı kurulamadı: {error}",
            ollama_model_available=False,
        )

    except Exception as error:
        update_state(
            ollama_status="error",
            ollama_message=f"Ollama denetim hatası: {error}",
            ollama_model_available=False,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_load_task

    await asyncio.to_thread(check_ollama_sync)

    asr_load_task = asyncio.create_task(
        asyncio.to_thread(load_asr_model_sync)
    )

    yield

    if asr_load_task and not asr_load_task.done():
        asr_load_task.cancel()


app = FastAPI(
    title=CONFIG["application"]["name"],
    version=CONFIG["application"]["version"],
    lifespan=lifespan,
)


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    await asyncio.to_thread(check_ollama_sync)

    state = get_state_copy()

    state["torch_version"] = torch.__version__
    state["cuda_available"] = torch.cuda.is_available()
    state["configured_asr_model"] = ASR_MODEL_NAME
    state["configured_ollama_model"] = OLLAMA_MODEL

    return state


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
      --background: #0c111b;
      --panel: #151d2b;
      --panel-soft: #1b2535;
      --border: #2b3a50;
      --text: #e7edf6;
      --muted: #91a0b5;
      --cyan: #5ed9e7;
      --green: #5fe09b;
      --yellow: #f0c96a;
      --red: #f17b83;
      --blue: #6ea8fe;
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
          transparent 32rem
        ),
        var(--background);
      color: var(--text);
    }

    main {
      width: min(1080px, calc(100% - 32px));
      margin: 0 auto;
      padding: 42px 0 64px;
    }

    header {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-start;
      margin-bottom: 28px;
    }

    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 46px);
      letter-spacing: -0.04em;
    }

    header p {
      margin: 0;
      color: var(--muted);
      font-size: 16px;
    }

    .local-badge {
      border: 1px solid rgba(95, 224, 155, 0.42);
      background: rgba(95, 224, 155, 0.08);
      color: var(--green);
      border-radius: 999px;
      padding: 9px 14px;
      white-space: nowrap;
      font-weight: 700;
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
      min-height: 190px;
      box-shadow:
        0 18px 50px rgba(0, 0, 0, 0.18);
    }

    .card-wide {
      grid-column: 1 / -1;
      min-height: unset;
    }

    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
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

    .status-ready {
      color: var(--green);
    }

    .status-loading,
    .status-checking,
    .status-waiting {
      color: var(--yellow);
    }

    .status-warning {
      color: var(--yellow);
    }

    .status-error {
      color: var(--red);
    }

    .description {
      min-height: 48px;
      margin: 0 0 18px;
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

    .assistant-area {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
    }

    button {
      border: 1px solid rgba(94, 217, 231, 0.42);
      background: rgba(94, 217, 231, 0.1);
      color: var(--cyan);
      border-radius: 14px;
      min-height: 46px;
      padding: 0 18px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }

    button:hover {
      background: rgba(94, 217, 231, 0.17);
    }

    .planned {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }

    .planned span {
      border: 1px solid var(--border);
      background: var(--panel-soft);
      color: var(--muted);
      border-radius: 12px;
      padding: 9px 12px;
      font-size: 13px;
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

    @media (max-width: 760px) {
      header {
        flex-direction: column;
      }

      .grid {
        grid-template-columns: 1fr;
      }

      .card-wide {
        grid-column: auto;
      }

      .assistant-area {
        grid-template-columns: 1fr;
      }

      button {
        width: 100%;
      }
    }
  </style>
</head>

<body>
  <main>
    <header>
      <div>
        <h1>Asunode Voice Assistant</h1>
        <p>Nemotron ASR + Gemma 4 · Yerel çalışma paneli</p>
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

          <dt>Yükleme süresi</dt>
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

          <dt>API</dt>
          <dd>127.0.0.1:11434</dd>

          <dt>Sıcak tutma</dt>
          <dd>30 dakika</dd>

          <dt>Çalışma</dt>
          <dd>Yerel</dd>
        </dl>
      </article>

      <article class="card card-wide">
        <div class="card-header">
          <h2>Sesli asistan arayüzü</h2>
        </div>

        <div class="assistant-area">
          <div>
            <p class="description">
              Sıcak çekirdek doğrulandıktan sonra mikrofon,
              konuşma metni, Gemma cevabı ve sesli yanıt bu
              panelde etkinleştirilecek.
            </p>

            <div class="planned">
              <span>Mikrofon başlat / durdur</span>
              <span>Nemotron metni</span>
              <span>Gemma 4 cevabı</span>
              <span>İşlem süreleri</span>
              <span>Konuşma geçmişi</span>
              <span>Piper TTS</span>
            </div>
          </div>

          <button id="refreshButton" type="button">
            Durumu yenile
          </button>
        </div>

        <div id="errorBox" class="error"></div>
      </article>
    </section>

    <footer>
      Panel her iki saniyede bir yerel servis durumunu yeniler.
    </footer>
  </main>

  <script>
    const statusLabels = {
      waiting: "Bekleniyor",
      loading: "Yükleniyor",
      ready: "Hazır",
      checking: "Denetleniyor",
      warning: "Uyarı",
      error: "Hata"
    };

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

    async function refreshStatus() {
      const errorBox = document.getElementById("errorBox");

      try {
        const response = await fetch("/api/status", {
          cache: "no-store"
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        const asrStatus = document.getElementById("asrStatus");
        const ollamaStatus = document.getElementById("ollamaStatus");

        setStatus(asrStatus, data.asr_status);
        setStatus(ollamaStatus, data.ollama_status);

        document.getElementById("asrMessage").textContent =
          data.asr_message;

        document.getElementById("ollamaMessage").textContent =
          data.ollama_message;

        document.getElementById("asrModel").textContent =
          data.configured_asr_model;

        document.getElementById("ollamaModel").textContent =
          data.configured_ollama_model;

        document.getElementById("asrLoadTime").textContent =
          data.asr_load_seconds === null
            ? "—"
            : `${data.asr_load_seconds} saniye`;

        document.getElementById("asrParameters").textContent =
          formatParameters(data.asr_parameter_count);

        if (data.last_error) {
          errorBox.style.display = "block";
          errorBox.textContent = data.last_error;
        } else {
          errorBox.style.display = "none";
          errorBox.textContent = "";
        }
      } catch (error) {
        errorBox.style.display = "block";
        errorBox.textContent =
          `Arayüz servisine erişilemedi: ${error.message}`;
      }
    }

    document
      .getElementById("refreshButton")
      .addEventListener("click", refreshStatus);

    refreshStatus();
    setInterval(refreshStatus, 2000);
  </script>
</body>
</html>
"""


def main() -> None:
    print()
    print("Asunode Local Voice Assistant başlatılıyor.")
    print(f"Arayüz: http://{UI_HOST}:{UI_PORT}")
    print("Durdurmak için Ctrl+C kullanın.")
    print()

    uvicorn.run(
        app,
        host=UI_HOST,
        port=UI_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
