# Aurik GPU Setup — AMD ROCm & DirectML

**Version:** 10.0.8 | **Stand:** Mai 2026

Aurik beschleunigt rechenintensive Phasen (Defect-Scanning, MERT-Embeddings, PSLA,
CLAP-Referenz-Matching, Whisper-Tiny-ONNX) optional über die GPU. Auf AMD-GPUs
werden zwei unterschiedliche APIs eingesetzt:

| Plattform | GPU-API | Framework |
| --- | --- | --- |
| **Linux** | AMD ROCm 7.2.4 | PyTorch ROCm |
| **Windows 10/11** | AMD DirectML | ONNX Runtime + torch-directml |

> **Hinweis:** GPU ist **optional** — Aurik funktioniert zu 100 % auf CPU.
> Die GPU bringt einen **2×–8× Speedup** (phase-abhängig, siehe [Performance-Erwartungen](#performance-expectations)).

---

## 1. Linux: AMD ROCm Setup

### 1.1 Voraussetzungen

- AMD GPU der Serien **Radeon VII, RX 5700+, RX 6600+, RX 7600+, Radeon Pro Wx700+, Wx800+**  
  (gfx900–gfx1102; vollständige Liste: `rocm-smi --showproductname`)
- **Ubuntu 22.04 / 24.04 LTS** (offiziell getestet; andere Distros möglich)
- **Root/sudo**-Zugriff

### 1.2 ROCm-Installation (Ubuntu)

```bash
# 1. ROCm-Repository hinzufügen
wget -q https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
  sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/7.2.4 noble main" | \
  sudo tee /etc/apt/trusted.gpg.d/rocm.list

# 2. ROCm-Pakete installieren
sudo apt update
sudo apt install rocm-hip-sdk rocm-hip-libraries

# 3. Benutzer zur render- und video-Gruppe hinzufügen
sudo usermod -a -G render,video $USER

# 4. ROCm-Pfad in ~/.bashrc eintragen
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 1.3 PyTorch mit ROCm-Unterstützung

```bash
# Innerhalb des Aurik venv / AppImage-Umgebung:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

Produktions-Stack (Rev. 2026-08-16): ROCm 7.2.4 + torch 2.11.0+rocm7.2 +
torchaudio 2.11.0+rocm7.2 + torchvision 0.26.0+rocm7.2 + onnxruntime-rocm 1.22.2.post3.

### 1.3b MIGraphX-Bridge (optionaler ONNX-GPU-Pfad)

Die Bridge `backend/core/lib/libmigraphx_bridge.so` kompiliert ONNX-Modelle
über AMDs MIGraphX-Graph-Compiler (Geschwindigkeitsvorteil für gfx1100,
siehe `.github/specs/v10.40_migraphx_gpu_integration.md`). Sie wird seit
Rev. 2026-08-16 gegen **MIGraphX 2.15 (ROCm 7.2.4)** gebaut — Quellcode und
Build-Skript liegen im Repo:

```bash
sudo apt install migraphx migraphx-dev g++
bash scripts/build_migraphx_bridge.sh
```

Verifikation (muss `True` liefern):

```bash
python -c "import sys; sys.path.insert(0,'.'); \
from backend.core.migraphx_adapter import is_migraphx_available; \
print(is_migraphx_available())"
```

Hinweis: Die Bridge kompiliert mit `offload_copy=true` — ohne diese Option
liefert die MIGraphX-2.15-C-API keine GPU-Ergebnisse auf den Host zurück.

### 1.4 Verifikation (ROCm)

```bash
# ROCm-Toolchain prüfen
rocm-smi
rocminfo | grep -E "Name:|Marketing Name:"

# PyTorch-GPU-Verfügbarkeit
python -c "import torch; print(f'ROCm={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Erwartete Ausgabe:
```
ROCm=True, GPU=AMD Radeon RX 7900 XTX
```

### 1.5 Aurik GPU-Modus aktivieren

Aurik erkennt die GPU automatisch bei Programmstart. Optional kann das Verhalten
in der `.env`-Datei gesteuert werden:

```bash
# Im Projektverzeichnis:
echo "AURIK_GPU=rocm" >> .env
echo "AURIK_GPU_DEVICE=0" >> .env    # GPU-Index (0 = erste GPU)
```

---

## 2. Windows 10/11: AMD DirectML Setup

### 2.1 Voraussetzungen

- **AMD Radeon-GPU** (RX 5000+, RX 6000+, RX 7000+, oder kompatible Radeon Pro / iGPU)
- **Windows 10 22H2** oder **Windows 11**
- Aktueller **Adrenalin Edition**-Treiber (24.x oder neuer)

### 2.2 Treiber aktualisieren

1. [AMD Adrenalin Edition](https://www.amd.com/en/support) herunterladen und installieren
2. Nach der Installation: `Win+R` → `dxdiag` → Tab "Display" → Treiberversion prüfen

### 2.3 DirectML-Pakete installieren

Aurik's Windows-Installer bringt `onnxruntime-directml` bereits mit. Falls in einer
manuellen Umgebung gearbeitet wird:

```powershell
# Innerhalb des Aurik venv:
pip install onnxruntime-directml torch-directml
```

### 2.4 Verifikation (DirectML)

```python
# In Python-Konsole:
import onnxruntime as ort

# Alle verfügbaren Execution Provider auflisten
print(ort.get_available_providers())
# Sollte 'DmlExecutionProvider' enthalten

# GPU-Test mit einem kleinen Modell
import numpy as np
session = ort.InferenceSession(
    b"dummy",  # nur Provider-Prüfung — echtes Modell siehe unten
    providers=['DmlExecutionProvider']
)
print("DirectML-Provider verfügbar: OK")
```

Für PyTorch mit DirectML:

```python
import torch
import torch_directml

dml = torch_directml.device()
t = torch.randn(1000, 1000).to(dml)
print(f"GPU-Tensor shape: {t.shape}, device: {t.device}")
```

### 2.5 Aurik GPU-Modus aktivieren

Da Windows kein ROCm unterstützt, verwendet Aurik dort standardmäßig DirectML.
Aktivierung via `.env`:

```powershell
# Im Projektverzeichnis:
echo "AURIK_GPU=directml" >> .env
```

---

## 3. GPU-Verwendung verifizieren (Laufzeit)

### 3.1 Aurik-Status beim Start

Beim Start von Aurik erscheint im Log (`logs/aurik_backend.log`) eine der folgenden Zeilen:

```
GPU: ROCm erkannt — AMD Radeon RX 7900 XTX (24 GB VRAM)
GPU: DirectML erkannt — DmlExecutionProvider aktiv
GPU: Deaktiviert / Nicht gefunden — CPU-Modus
```

### 3.2 Zur Laufzeit prüfen

```python
from backend.core.gpu_detector import is_gpu_available, get_gpu_info

info = get_gpu_info()
print(f"GPU verfügbar: {info['available']}")
print(f"Backend:       {info['backend']}")
print(f"Gerät:         {info['device_name']}")
print(f"VRAM:          {info['vram_gb']} GB")
```

### 3.3 Anzeichen für GPU-Nutzung

- **Aufgaben-Manager / radeontop:** GPU-Auslastung > 0 % während Defect-Scan oder MERT-Phase
- **Log:** Phasen mit `[GPU]`-Marker (z. B. `[GPU] DefectScanner 48/64 passes`)
- **Verarbeitungszeit:** Phase 01–08 (Defect-Scan) in < 30 s statt > 2 min

---

## 4. Troubleshooting

### 4.1 ROCm: "No HIP GPUs are available"

**Ursache:** Treiber oder Benutzer-Gruppen nicht korrekt.

```bash
# Nutzer zu render/video-Gruppen hinzufügen
sudo usermod -a -G render,video $USER
# Neu anmelden (Logout + Login)
# Prüfen:
groups | grep -E "render|video"
rocm-smi
```

### 4.2 ROCm: "torch.cuda.is_available() → False"

**Ursache:** ROCm-PyTorch nicht installiert oder Version mismatch.

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
# Prüfen:
python -c "import torch; print(torch.cuda.is_available())"
```

### 4.3 ROCm: "rocBLAS error: Could not initialize Tensile host"

**Ursache:** `LD_LIBRARY_PATH` enthält `/opt/rocm/lib` nicht.

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
# Permanent: in ~/.bashrc eintragen
```

### 4.4 DirectML: "DmlExecutionProvider not found"

**Ursache:** `onnxruntime-directml` nicht installiert oder Treiber veraltet.

```powershell
# 1. Adrenalin-Treiber aktualisieren (24.x+)
# 2. onnxruntime-directml neu installieren
pip install --force-reinstall onnxruntime-directml
# 3. Prüfen
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

### 4.5 GPU wird erkannt, aber Aurik verwendet CPU

**Ursache:** GPU-Compute nicht explizit aktiviert.

```bash
# In .env (Projekt-Root):
AURIK_GPU=rocm       # oder directml
AURIK_GPU_DEVICE=0
```

### 4.6 Out-of-Memory (OOM) auf GPU

**Ursache:** VRAM zu klein für Batch-Größe oder MERT-Modell.

**Lösungen:**
1. In der Aurik-GUI: **Einstellungen → Performance → GPU-Batch-Größe reduzieren** (Standard: 4 → 1)
2. MERT-Embedding-Größe halbieren: `AURIK_MERT_BATCH=1` in `.env`
3. Bei Radeon-Karten mit < 8 GB VRAM: CPU-Modus bevorzugen

### 4.7 Dual-GPU / iGPU + dGPU

Aurik verwendet standardmäßig GPU 0. Wenn die iGPU (z. B. AMD Radeon 780M) GPU 0 ist und
die dGPU (z. B. RX 7900 XTX) GPU 1, dann in `.env` setzen:

```bash
AURIK_GPU_DEVICE=1
```

---

## 5. Performance-Erwartungen

Die folgende Tabelle zeigt typische Verarbeitungszeiten für eine **3-Minuten-Stereodatei**
(48 kHz, 24-bit) auf verschiedenen Konfigurationen:

| Konfiguration | Phase 01–08 (Defect-Scan) | Kompletter Durchlauf (68 Phasen) | RT-Faktor |
| --- | --- | --- | --- |
| **CPU (Ryzen 9 7950X, 16 Kerne)** | ~120 s | ~18 min | ~6× |
| **CPU + ROCm (RX 7900 XTX)** | ~18 s | ~4 min | ~1.3× |
| **CPU + ROCm (RX 6800)** | ~25 s | ~5 min 30 s | ~1.8× |
| **CPU + DirectML (RX 7900 XTX, Windows)** | ~22 s | ~5 min | ~1.7× |
| **CPU + ROCm (RX 6600, 8 GB)** | ~40 s | ~8 min | ~2.7× |
| **CPU only (Apple M2 Max)** | ~95 s | ~14 min | ~4.7× |

**Faustregeln:**

- **Defect-Scan (Phase 01–08):** GPU bringt 5×–8× Speedup
- **MERT-Embeddings (Phase 04):** GPU bringt 4×–6× Speedup
- **Restaurierungs-Phasen (09–64):** Meist CPU-gebunden (2×–3× GPU-Vorteil nur bei PSLA/CLAP)
- **Gesamt-Durchlauf:** GPU bringt 3×–5× Speedup

### GPU-VRAM-Anforderungen

| Modell | Mindest-VRAM | Empfohlen |
| --- | --- | --- |
| MERT-v1-330M (Phase 04) | 4 GB | 6 GB |
| Whisper-Tiny-ONNX (Phase 58) | 1 GB | 2 GB |
| CLAP-Referenz-Matching | 2 GB | 4 GB |
| PSLA (Phase 07) | 1 GB | 2 GB |
| **Aurik gesamt (gleichzeitig)** | **6 GB** | **8+ GB** |

Mit aktiviertem GPU-Compute und einer AMD-GPU ab RX 6600 / RX 7600 (8+ GB VRAM)
läuft Aurik deutlich schneller als auf CPU — ein **3×–5× Gesamt-Speedup** ist
bei typischen Restaurierungen zu erwarten.

---

## 6. Weitere Informationen

- **[README ←](README.md)** — Zurück zur Projektübersicht
- **[Troubleshooting](guides/TROUBLESHOOTING.md)** — Allgemeine Problemlösung
- **[Konfigurations-Guide](guides/CONFIGURATION.md)** — Alle `.env`-Optionen
- **[Architektur-Übersicht](architecture/ARCHITECTURE.md)** — Technische Details zur GPU-Pipeline
