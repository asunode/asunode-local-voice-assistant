# Asunode Local Voice Assistant

[English](README.md) | [Türkçe](README.tr.md)

> NVIDIA Nemotron ASR, Ollama/Gemma, Piper TTS, FastAPI ve ALSA ile oluşturulmuş, tamamen yerel ve yalnızca CPU üzerinde çalışan bir sesli asistan zinciri.

Asunode Local Voice Assistant, bas-konuş biçiminde çalışan tamamen yerel bir Türkçe sesli asistan prototipidir:

![Asunode Local Voice Assistant web arayüzü](docs/images/asunode-local-voice-assistant-ui.png)

```text
Logitech/ALSA mikrofon
→ NVIDIA Nemotron 3.5 ASR Streaming 0.6B
→ Ollama / Gemma 4 12B
→ Piper TTS
→ Hoparlör
```

Ses kaydı, konuşma tanıma, dil modeli ve ses sentezi yerel makinede yürütülür. Web arayüzü varsayılan olarak yalnızca `127.0.0.1` adresinde dinler.

## Özellikler

- CPU-only yerel çalışma
- Türkçe ASR ve Türkçe TTS
- Ollama üzerinden yerel Gemma 4 modeli
- FastAPI tabanlı web arayüzü
- Nemotron, Gemma ve Piper modellerini sıcak tutma
- Kayıt başlatma/durdurma
- ASR metni ve LLM cevabını gösterme
- Otomatik sesli cevap ve son cevabı tekrar okuma
- Yerel konuşma geçmişi
- Systemd kullanıcı servisi örneği
- Parlament mavisi arayüz

## Test edilen sistem

Bu proje aşağıdaki tek sistemde doğrulandı:

| Bileşen | Değer |
|---|---|
| İşletim sistemi | Ubuntu 25.10 |
| Python | 3.13.7 |
| İşlemci | Intel Core i5-11600K |
| Bellek | 32 GB |
| Çalışma biçimi | CPU-only |
| PyTorch | 2.12.1+cpu |
| TorchAudio | 2.11.0+cpu |
| NeMo | GitHub `main`, doğrulanan commit `0f378e9d8` |
| Mikrofon | Logitech USB webcam mikrofonu |
| LLM sunucusu | Ollama |
| LLM | `gemma4:12b-it-qat` |

GPU performansı ve başka işletim sistemleri test edilmemiştir. Tam ortam sürümleri [requirements/main-lock.txt](requirements/main-lock.txt) ve [requirements/tts-lock.txt](requirements/tts-lock.txt) içindedir.

## Gözlemsel ölçümler

19 Haziran 2026 tarihinde test makinesinde gözlenen yaklaşık değerler:

- Nemotron modelinin yerel önbellekten başlangıçta yüklenmesi: yaklaşık 30 saniye
- 5–10 saniyelik sesin ASR işlemi: yaklaşık 0,6–1,0 saniye
- Piper model yükleme: yaklaşık 0,8 saniye
- Piper ses üretimi: yaklaşık 0,1–0,35 saniye
- Gemma 4 CPU yanıtı: soruya göre yaklaşık 30–140 saniye

Bunlar tek makinedeki gözlemsel ölçümlerdir; standartlaştırılmış benchmark sonuçları değildir.

## Gelişim süreci

1. Logitech mikrofon ALSA ile doğrulandı.
2. 16 kHz mono WAV kaydı test edildi.
3. Python sanal ortamı ve CPU PyTorch kuruldu.
4. NeMo ASR bağımlılıkları kuruldu.
5. Nemotron modeli indirildi ve CPU üzerinde açıldı.
6. Türkçe `tr-TR` prompt testi yapıldı.
7. `Unknown prompt key: None` hatası `RNNTPromptTranscribeConfig`, `target_lang="tr-TR"` ve `use_lhotse=False` kullanılarak çözüldü.
8. Nemotron çıktısı Ollama/Gemma'ya bağlandı.
9. FastAPI arayüzü oluşturuldu.
10. Gradio 6.19 ile Hugging Face Hub/Transformers bağımlılık çatışması nedeniyle Gradio'dan vazgeçildi.
11. Piper ayrı `.venv-tts` ortamına kuruldu.
12. Modeli bellekte tutan kalıcı Piper worker geliştirildi.
13. `.venv-tts/bin/python` bağlantısında `Path.resolve()` kullanımının ortamı `/usr/bin/python` yoluna taşıması önlendi.
14. Otomatik hoparlör cevabı eklendi.
15. Systemd kullanıcı servisi hazırlandı.
16. Parlament mavisi arayüz tamamlandı.

## Kurulum

### 1. Sistem paketleri

```bash
sudo apt update
sudo apt install -y \
  ffmpeg libsndfile1 alsa-utils git \
  python3-venv python3-dev build-essential
```

Ubuntu sürümünüz Python 3.13 için ayrı paket adları kullanıyorsa uygun `python3.13-venv` ve `python3.13-dev` paketlerini kurun.

### 2. Ollama ve Gemma

Ollama'yı kendi resmî kurulum yönergeleriyle ayrıca kurun. Ardından:

```bash
ollama pull gemma4:12b-it-qat
```

Model adının Ollama dağıtımınızda mevcut olduğunu doğrulayın:

```bash
ollama list
```

### 3. Ana Python ortamı

```bash
git clone https://github.com/asunode/asunode-local-voice-assistant.git
cd asunode-local-voice-assistant
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

CPU PyTorch kurulumu ayrı yapılmalıdır:

```bash
python -m pip install \
  torch==2.12.1+cpu torchaudio==2.11.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

NeMo'nun çalışan sürümü GitHub `main` dalından gelir. Temel bağımlılıkları kurun:

```bash
python -m pip install -r requirements/main.txt
```

Tam olarak test edilen ortamı yeniden üretmek için bunun yerine kilit dosyası kullanılabilir:

```bash
python -m pip install -r requirements/main-lock.txt
```

Kilit dosyasındaki NeMo satırı doğrulanan commit'e sabitlenmiştir. Yeni `main` sürümlerinde uyumsuzluk oluşabilir.

### 4. Piper TTS ortamı

Piper, ana ASR ortamıyla bağımlılık çakışmalarını önlemek için ayrı tutulur:

```bash
python3.13 -m venv .venv-tts
.venv-tts/bin/python -m pip install --upgrade pip
.venv-tts/bin/python -m pip install -r requirements/tts.txt
mkdir -p models/piper
```

Piper'ın `tr_TR-dfki-medium` sesine ait `.onnx` ve `.onnx.json` dosyalarını Piper ses dağıtımından indirip `models/piper/` altına koyun:

```text
models/piper/tr_TR-dfki-medium.onnx
models/piper/tr_TR-dfki-medium.onnx.json
```

Model dosyaları bu repository'de yer almaz.

### 5. Yapılandırma ve mikrofon

Yerel yapılandırmayı örnekten oluşturun:

```bash
cp config/app.example.json config/app.json
```

Mikrofonları listeleyin:

```bash
arecord -l
```

`config/app.json` içindeki `audio.capture_device` değerini ayarlayın. Değişebilen kart numarası yerine kalıcı ALSA cihaz adı kullanılması önerilir:

```json
"capture_device": "hw:CARD=YOUR_MICROPHONE,DEV=0"
```

### 6. Manuel başlatma

Ollama'nın çalıştığından emin olun ve uygulamayı başlatın:

```bash
.venv/bin/python voice_assistant_ui_stage3.py
```

Arayüz:

```text
http://127.0.0.1:7860
```

İlk çalıştırmada Nemotron ve diğer model dosyalarının indirilmesi internet bağlantısı ve zaman gerektirir.

## Systemd kullanıcı servisi

Örnek dosyayı kullanıcı servis klasörüne kopyalayın:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/asunode-voice-assistant.service.example \
  ~/.config/systemd/user/asunode-voice-assistant.service
```

Kopyalanan dosyadaki bütün `__PROJECT_DIR__` yer tutucularını projenin mutlak yolu ile değiştirin. Sonra:

```bash
systemctl --user daemon-reload
systemctl --user enable --now asunode-voice-assistant.service
systemctl --user status asunode-voice-assistant.service
journalctl --user -u asunode-voice-assistant.service -f
```

Kullanıcı servisi normalde kullanıcı oturumu açıldığında başlar. Oturum açılmadan çalışması istenirse isteğe bağlı olarak:

```bash
sudo loginctl enable-linger "$USER"
```

## Sorun giderme

### İlk kelime kayboluyor

`arecord` cihazı açılırken kısa bir başlangıç gecikmesi olabilir. Kayıt düğmesine bastıktan sonra konuşmadan önce kısa süre bekleyin.

### Gerçek streaming

Modelin adında “streaming” bulunsa da mevcut uygulama gerçek zamanlı PCM akışı yapmaz. WAV tabanlı, dosya sonrasında işlenen bas-konuş yöntemidir.

### Türkçe tanıma hataları

Türkçe ASR kusursuz değildir. Yerel LLM küçük fonetik ve kelime ayrımı hatalarını bağlamdan düzeltebilir; önemli sonuçları yine de doğrulayın.

### Gemma yavaş veya tarih yanıtı yanlış

Gemma 4 12B CPU üzerinde soruya göre uzun sürebilir. Yerel model güncel tarihi kendiliğinden bilmez. Sistem tarihi henüz prompt'a otomatik enjekte edilmediği için tarih/saat cevabı yanlış olabilir.

### Gradio bağımlılık çatışması

Gradio 6.19, test ortamındaki Hugging Face Hub/Transformers zinciriyle çatıştığı için arayüz FastAPI ile yazılmıştır. Bu projeye Gradio eklenmemelidir.

### Piper neden ayrı ortamda?

Piper, NeMo ortamından farklı bağımlılık sürümleri kullanır. `.venv-tts` ayrımı çalışan ASR ortamını korur. Ayrıca TTS çalıştırıcısında `.venv-tts/bin/python` için `Path.resolve()` kullanılmamalıdır; bu işlem sanal ortam bağlantısını sistem Python'una çözebilir.

### Mikrofon meşgul

Mikrofon başka bir uygulama tarafından tutuluyorsa ALSA kayıt başlatamaz. Tarayıcı, toplantı uygulaması veya başka kayıt süreçlerini kapatın.

### Systemd ve ses cihazı

Kullanıcı servisi ses cihazına kullanıcı oturumu üzerinden erişir. Oturum, PipeWire/PulseAudio ve ALSA izinleri hazır değilse kayıt veya oynatma başarısız olabilir.

### Büyük modeller ve ilk başlangıç

Model ağırlıkları büyüktür. İlk indirme ve ilk yükleme sonraki çalıştırmalardan belirgin biçimde daha uzun sürer.

## Gizlilik

- Uygulama varsayılan olarak yalnızca `127.0.0.1` üzerinde çalışır.
- Ses ve metin yerel makinede işlenir.
- Model indirme sırasında internet bağlantısı gerekir.
- Gerekli modeller indirildikten sonra normal çıkarım tamamen yerel çalışır ve bir bulut inference API'si gerektirmez.
- Repository ses kayıtlarını, konuşma geçmişini veya yerel yapılandırmayı içermez.

## Bilinen sınırlamalar

- Dosya tabanlı push-to-talk
- Yankı engelleme yok
- VAD yok
- Wake word yok
- Gerçek zamanlı parçalı metin görünümü yok
- Gemma CPU gecikmesi yüksek
- Sistem tarih/saat entegrasyonu yok
- Tıbbi, hukuki veya güvenlik kritik kullanım için uygun değil

## Yol haritası

- RAM tabanlı PCM streaming
- VAD
- Wake word
- Yankı engelleme
- RAG ve cevap önbelleği
- Sistem tarih/saat aracı
- Daha hızlı küçük yerel model seçeneği
- TTS ses seçimi
- Tarayıcı mikrofonu veya uzak istemci desteği
- Paketleme ve kurulum betiği

## Model ve üçüncü taraf lisansları

Proje kaynak kodu [MIT Lisansı](LICENSE) ile sunulur.

- NVIDIA Nemotron model ağırlıkları bu repository'de yer almaz.
- Gemma model ağırlıkları bu repository'de yer almaz.
- Piper ses modeli bu repository'de yer almaz.
- Her modelin, sesin ve üçüncü taraf bileşeninin kendi lisans koşulları ayrıca geçerlidir.
- Kullanıcılar ilgili model kartlarını ve lisanslarını kurulumdan önce incelemelidir.
