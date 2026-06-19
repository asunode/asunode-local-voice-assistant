#!/usr/bin/env python3

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    RNNTPromptTranscribeConfig,
)


ASR_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
ASR_LANGUAGE = "tr-TR"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "gemma4:12b-it-qat"

SYSTEM_PROMPT = """
Sen yerel bir sesli asistan testisin.

Kullanıcı metni otomatik konuşma tanıma sisteminden gelmektedir.
Metinde küçük fonetik, kelime ayrımı veya yazım hataları olabilir.

Kurallar:
- En olası anlamı değerlendir.
- Türkçe cevap ver.
- Cevabı en fazla iki kısa cümleyle sınırla.
- Anlam yeterince açık değilse tahmin uydurma; kullanıcının tekrar etmesini iste.
""".strip()


def transcribe_audio(audio_path: Path) -> tuple[str, float, float]:
    print("Nemotron modeli CPU üzerinde yükleniyor...")
    load_started = time.perf_counter()

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=ASR_MODEL,
        map_location="cpu",
    )
    model.eval()

    if hasattr(model.decoding, "set_strip_lang_tags"):
        model.decoding.set_strip_lang_tags(True)

    load_time = time.perf_counter() - load_started

    transcribe_config = RNNTPromptTranscribeConfig(
        use_lhotse=False,
        batch_size=1,
        return_hypotheses=False,
        num_workers=0,
        verbose=False,
        target_lang=ASR_LANGUAGE,
    )

    print("Ses Türkçe olarak çözümleniyor...")
    asr_started = time.perf_counter()

    with torch.inference_mode():
        result = model.transcribe(
            audio=[str(audio_path)],
            override_config=transcribe_config,
        )

    asr_time = time.perf_counter() - asr_started

    if isinstance(result, tuple):
        result = result[0]

    if not result:
        raise RuntimeError("Nemotron herhangi bir sonuç üretmedi.")

    first_item = result[0]
    text = first_item.text if hasattr(first_item, "text") else str(first_item)
    text = text.strip()

    if not text:
        raise RuntimeError("Nemotron boş metin üretti.")

    return text, load_time, asr_time


def ask_ollama(user_text: str) -> tuple[str, float]:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "keep_alive": "10m",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
        "options": {
            "temperature": 0.2,
        },
    }

    request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"Metin Ollama modeline gönderiliyor: {OLLAMA_MODEL}")
    llm_started = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            response_data = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP hatası: {error.code}\n{error_body}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama API bağlantısı kurulamadı. "
            "Ollama servisinin çalıştığını kontrol edin."
        ) from error

    llm_time = time.perf_counter() - llm_started

    try:
        answer = response_data["message"]["content"].strip()
    except (KeyError, TypeError, AttributeError) as error:
        raise RuntimeError(
            "Ollama beklenen yanıt biçimini üretmedi:\n"
            + json.dumps(response_data, ensure_ascii=False, indent=2)
        ) from error

    if not answer:
        raise RuntimeError("Ollama boş cevap üretti.")

    return answer, llm_time


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Kullanım: {Path(sys.argv[0]).name} SES_DOSYASI.wav")
        return 2

    audio_path = Path(sys.argv[1]).expanduser().resolve()

    if not audio_path.is_file():
        print(f"Hata: Ses dosyası bulunamadı: {audio_path}")
        return 2

    try:
        transcription, load_time, asr_time = transcribe_audio(audio_path)

        print()
        print("========================================")
        print("NEMOTRON ASR METNİ")
        print("========================================")
        print(transcription)
        print("========================================")
        print(f"Model yükleme süresi: {load_time:.2f} saniye")
        print(f"Ses çözümleme süresi: {asr_time:.2f} saniye")

        answer, llm_time = ask_ollama(transcription)

        print()
        print("========================================")
        print("GEMMA 4 CEVABI")
        print("========================================")
        print(answer)
        print("========================================")
        print(f"LLM cevap süresi: {llm_time:.2f} saniye")

    except KeyboardInterrupt:
        print("\nİşlem kullanıcı tarafından durduruldu.")
        return 130

    except Exception as error:
        print()
        print(f"HATA: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
