#!/usr/bin/env python3

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import voice_assistant_ui_stage2 as base
from tts_controller import TTSController


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config" / "app.json"

tts_controller = TTSController(
    project_dir=PROJECT_DIR,
    config=base.CONFIG,
)

original_process_recording_sync = (
    base.process_recording_sync
)


base.runtime_state.update(
    {
        "tts_status": "waiting",
        "tts_message": "Piper henüz başlatılmadı.",
        "tts_ready": False,
        "tts_voice": tts_controller.voice,
        "tts_auto_play": tts_controller.auto_play,
        "tts_load_seconds": None,
        "last_tts_seconds": None,
        "last_playback_seconds": None,
        "last_tts_output": None,
        "tts_last_error": None,
    }
)


class TTSToggleRequest(BaseModel):
    enabled: bool


def persist_auto_play(enabled: bool) -> None:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    config.setdefault("tts", {})
    config["tts"]["auto_play_response"] = bool(
        enabled
    )

    temporary_path = CONFIG_PATH.with_suffix(
        ".json.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(CONFIG_PATH)


def update_tts_runtime_state(
    *,
    message: str | None = None,
    status_override: str | None = None,
) -> dict[str, Any]:
    controller_status = tts_controller.status()

    if status_override is not None:
        status_name = status_override
    elif not controller_status["enabled"]:
        status_name = "disabled"
    elif controller_status["ready"]:
        status_name = "ready"
    elif controller_status["last_error"]:
        status_name = "error"
    else:
        status_name = "loading"

    if message is None:
        if status_name == "disabled":
            message = "Piper yapılandırmada kapalı."
        elif status_name == "ready":
            if controller_status["auto_play"]:
                message = (
                    "Piper hazır; cevaplar otomatik okunacak."
                )
            else:
                message = (
                    "Piper hazır; otomatik sesli yanıt kapalı."
                )
        elif status_name == "error":
            message = (
                controller_status["last_error"]
                or "Piper hatası oluştu."
            )
        else:
            message = "Piper hazırlanıyor."

    base.update_state(
        tts_status=status_name,
        tts_message=message,
        tts_ready=controller_status["ready"],
        tts_voice=controller_status["voice"],
        tts_auto_play=controller_status[
            "auto_play"
        ],
        tts_load_seconds=controller_status[
            "load_seconds"
        ],
        last_tts_seconds=controller_status[
            "last_synthesis_seconds"
        ],
        last_tts_output=controller_status[
            "last_output"
        ],
        tts_last_error=controller_status[
            "last_error"
        ],
    )

    return controller_status


def start_tts_sync() -> None:
    base.update_state(
        tts_status="loading",
        tts_message="Piper sesi belleğe yükleniyor.",
        tts_last_error=None,
    )

    try:
        tts_controller.start()

        update_tts_runtime_state(
            message=(
                "Piper bellekte ve sesli yanıt için hazır."
            ),
            status_override="ready",
        )

    except Exception as error:
        base.update_state(
            tts_status="error",
            tts_message="Piper başlatılamadı.",
            tts_ready=False,
            tts_last_error=(
                f"{type(error).__name__}: {error}"
            ),
        )


def process_recording_sync_with_tts(
    audio_path: Path,
) -> dict[str, Any]:
    result = original_process_recording_sync(
        audio_path
    )

    if not tts_controller.auto_play:
        update_tts_runtime_state()
        return result

    base.update_state(
        recording_status="processing",
        recording_message=(
            "Gemma cevabı Piper tarafından seslendiriliyor."
        ),
    )

    try:
        tts_result = (
            tts_controller.synthesize_and_play(
                result["answer"]
            )
        )

        result["tts_seconds"] = tts_result.get(
            "synthesis_seconds"
        )
        result["playback_seconds"] = (
            tts_result.get("playback_seconds")
        )
        result["tts_output"] = tts_result.get(
            "output"
        )

        base.update_state(
            tts_status="ready",
            tts_message=(
                "Cevap hoparlörden başarıyla okundu."
            ),
            tts_ready=True,
            tts_voice=tts_controller.voice,
            tts_auto_play=(
                tts_controller.auto_play
            ),
            tts_load_seconds=(
                tts_controller.load_seconds
            ),
            last_tts_seconds=tts_result.get(
                "synthesis_seconds"
            ),
            last_playback_seconds=(
                tts_result.get("playback_seconds")
            ),
            last_tts_output=tts_result.get(
                "output"
            ),
            tts_last_error=None,
        )

    except Exception as error:
        base.update_state(
            tts_status="error",
            tts_message=(
                "Metin cevabı üretildi fakat sesli "
                "yanıt oynatılamadı."
            ),
            tts_ready=False,
            tts_last_error=(
                f"{type(error).__name__}: {error}"
            ),
        )

    finally:
        base.update_state(
            recording_status="idle",
            recording_message="Mikrofon hazır.",
        )

    return result


base.process_recording_sync = (
    process_recording_sync_with_tts
)


@asynccontextmanager
async def stage3_lifespan(app):
    base.initialization_task = asyncio.create_task(
        base.initialize_models()
    )

    tts_initialization_task = (
        asyncio.create_task(
            asyncio.to_thread(start_tts_sync)
        )
    )

    yield

    base.stop_recording_if_active()

    await asyncio.to_thread(
        tts_controller.shutdown
    )

    if (
        base.initialization_task
        and not base.initialization_task.done()
    ):
        base.initialization_task.cancel()

    if not tts_initialization_task.done():
        tts_initialization_task.cancel()


base.app.router.lifespan_context = (
    stage3_lifespan
)


@base.app.post("/api/tts/toggle")
async def api_tts_toggle(
    request: TTSToggleRequest,
) -> dict[str, Any]:
    enabled = bool(request.enabled)

    try:
        await asyncio.to_thread(
            tts_controller.set_auto_play,
            enabled,
        )

        await asyncio.to_thread(
            persist_auto_play,
            enabled,
        )

        controller_status = (
            update_tts_runtime_state()
        )

        return {
            "status": "ok",
            "enabled": enabled,
            "controller": controller_status,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@base.app.post("/api/tts/replay")
async def api_tts_replay() -> dict[str, Any]:
    state = base.get_state_copy()
    answer = str(
        state.get("last_answer", "")
    ).strip()

    if not answer:
        raise HTTPException(
            status_code=409,
            detail=(
                "Tekrar okunabilecek bir Gemma cevabı yok."
            ),
        )

    base.update_state(
        recording_status="processing",
        recording_message=(
            "Son cevap hoparlörden tekrar okunuyor."
        ),
    )

    def replay_sync() -> dict[str, Any]:
        status = tts_controller.status()

        last_output = status.get("last_output")

        if last_output:
            audio_path = Path(last_output)

            if audio_path.is_file():
                return tts_controller.play(
                    audio_path
                )

        return tts_controller.synthesize_and_play(
            answer
        )

    try:
        result = await asyncio.to_thread(
            replay_sync
        )

        update_values: dict[str, Any] = {
            "tts_status": "ready",
            "tts_message": (
                "Son cevap hoparlörden tekrar okundu."
            ),
            "tts_ready": True,
            "tts_last_error": None,
            "last_tts_output": result.get(
                "output"
            ),
            "last_playback_seconds": (
                result.get("playback_seconds")
            ),
        }

        if result.get("synthesis_seconds") is not None:
            update_values["last_tts_seconds"] = (
                result["synthesis_seconds"]
            )

        base.update_state(**update_values)

        return {
            "status": "played",
            **result,
        }

    except Exception as error:
        base.update_state(
            tts_status="error",
            tts_message=(
                "Cevap tekrar okunamadı."
            ),
            tts_last_error=(
                f"{type(error).__name__}: {error}"
            ),
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    finally:
        base.update_state(
            recording_status="idle",
            recording_message="Mikrofon hazır.",
        )


base.app.router.routes = [
    route
    for route in base.app.router.routes
    if not (
        getattr(route, "path", None) == "/"
        and "GET"
        in getattr(route, "methods", set())
    )
]


TTS_INTERFACE_EXTENSION = """
<style>
  .tts-control-panel {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    margin-top: 14px;
  }

  .tts-control-panel .tts-info {
    color: var(--muted);
    font-size: 13px;
  }

  .tts-control-panel button.active {
    border-color: rgba(95, 224, 155, 0.5);
    color: var(--green);
    background: rgba(95, 224, 155, 0.08);
  }
</style>

<script>
  function setTtsBadge(status, message) {
    const badge = document.getElementById("ttsStatus");

    const labels = {
      waiting: "Piper bekleniyor",
      loading: "Piper yükleniyor",
      ready: "Piper hazır",
      disabled: "Ses kapalı",
      error: "Piper hatası"
    };

    badge.className = `status status-${status}`;
    badge.textContent = labels[status] ?? status;
    badge.title = message ?? "";
  }

  function installTtsControls() {
    const assistantCard =
      document.querySelectorAll(".card-wide")[0];

    if (!assistantCard) {
      return;
    }

    const cardHeader =
      assistantCard.querySelector(".card-header");

    if (
      cardHeader
      && !document.getElementById("ttsStatus")
    ) {
      const badge = document.createElement("span");

      badge.id = "ttsStatus";
      badge.className =
        "status status-waiting";
      badge.textContent = "Piper bekleniyor";

      cardHeader.appendChild(badge);
    }

    const metrics =
      assistantCard.querySelector(".metrics");

    if (
      metrics
      && !document.getElementById("ttsSeconds")
    ) {
      const existingTtsMetric = Array
        .from(metrics.children)
        .find(
          (element) =>
            element.textContent
              .trim()
              .startsWith("TTS:")
        );

      if (existingTtsMetric) {
        existingTtsMetric.innerHTML =
          'TTS: <strong id="ttsSeconds">—</strong>';
      }

      const controls =
        document.createElement("div");

      controls.className = "tts-control-panel";
      controls.innerHTML = `
        <button
          id="ttsToggleButton"
          class="secondary"
          type="button"
        >
          Sesli yanıt
        </button>

        <button
          id="ttsReplayButton"
          class="secondary"
          type="button"
        >
          Cevabı tekrar oku
        </button>

        <span
          id="ttsInfo"
          class="tts-info"
        >
          Piper durumu okunuyor.
        </span>
      `;

      metrics.insertAdjacentElement(
        "afterend",
        controls
      );

      document
        .getElementById("ttsToggleButton")
        .addEventListener(
          "click",
          toggleTtsAutoPlay
        );

      document
        .getElementById("ttsReplayButton")
        .addEventListener(
          "click",
          replayLastAnswer
        );
    }
  }

  async function refreshTtsStatus() {
    try {
      const data = await requestJson(
        "/api/status"
      );

      setTtsBadge(
        data.tts_status,
        data.tts_message
      );

      const toggleButton =
        document.getElementById(
          "ttsToggleButton"
        );

      const replayButton =
        document.getElementById(
          "ttsReplayButton"
        );

      const info =
        document.getElementById("ttsInfo");

      const seconds =
        document.getElementById(
          "ttsSeconds"
        );

      if (toggleButton) {
        toggleButton.dataset.enabled =
          data.tts_auto_play
            ? "true"
            : "false";

        toggleButton.textContent =
          data.tts_auto_play
            ? "Sesli yanıt: Açık"
            : "Sesli yanıt: Kapalı";

        toggleButton.classList.toggle(
          "active",
          Boolean(data.tts_auto_play)
        );

        toggleButton.disabled =
          data.tts_status === "loading";
      }

      if (replayButton) {
        replayButton.disabled =
          !data.last_answer
          || data.recording_status !== "idle";
      }

      if (info) {
        const voice =
          data.tts_voice ?? "—";

        const load =
          data.tts_load_seconds === null
            ? "—"
            : `${data.tts_load_seconds} sn`;

        info.textContent =
          `${data.tts_message} · `
          + `Ses: ${voice} · `
          + `Yükleme: ${load}`;
      }

      if (seconds) {
        seconds.textContent =
          data.last_tts_seconds === null
            ? "—"
            : `${data.last_tts_seconds} sn`;
      }

      if (data.tts_last_error) {
        console.error(
          "Piper:",
          data.tts_last_error
        );
      }
    } catch (error) {
      console.error(
        "TTS durum hatası:",
        error
      );
    }
  }

  async function toggleTtsAutoPlay() {
    const button =
      document.getElementById(
        "ttsToggleButton"
      );

    const currentlyEnabled =
      button.dataset.enabled === "true";

    button.disabled = true;

    try {
      await requestJson(
        "/api/tts/toggle",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            enabled: !currentlyEnabled
          })
        }
      );

      await refreshTtsStatus();
    } catch (error) {
      setError(error.message);
    } finally {
      button.disabled = false;
    }
  }

  async function replayLastAnswer() {
    const button =
      document.getElementById(
        "ttsReplayButton"
      );

    button.disabled = true;
    button.textContent = "Okunuyor";

    try {
      await requestJson(
        "/api/tts/replay",
        {
          method: "POST"
        }
      );

      await refreshTtsStatus();
    } catch (error) {
      setError(error.message);
    } finally {
      button.textContent =
        "Cevabı tekrar oku";

      await refreshTtsStatus();
    }
  }

  installTtsControls();
  refreshTtsStatus();

  setInterval(
    refreshTtsStatus,
    1500
  );
</script>
"""


@base.app.get(
    "/",
    response_class=HTMLResponse,
)
async def stage3_index() -> str:
    html = await base.index()

    return html.replace(
        "</body>",
        TTS_INTERFACE_EXTENSION
        + "\n</body>",
    )


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
