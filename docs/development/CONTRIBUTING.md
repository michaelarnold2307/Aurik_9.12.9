# Contributing to Aurik 10.0.8

**Vielen Dank für dein Interesse an Aurik!**

> **Normativer Hinweis:** Release-faehige Pfade muessen den kanonischen Vertrag
> Bridge -> `AurikDenker.denke(...)` -> `export_guard()` einhalten.
> Historische v2- und direkte Restore-Beispiele in alten Abschnitten sind `LEGACY_NON_RELEASE`.

Dieses Dokument hilft dir bei deinen ersten Beiträgen zum Projekt.

---

## Inhaltsverzeichnis

- [Code of Conduct](#code-of-conduct)
- [Wie kann ich beitragen?](#wie-kann-ich-beitragen)
- [Development Setup](#development-setup)
- [Pull Request Prozess](#pull-request-prozess)
- [Coding Guidelines](#coding-guidelines)
- [Testing](#testing)
- [Dokumentation](#dokumentation)

---

## Code of Conduct

Wir erwarten von allen Contributors:

- ✅ Respektvoller Umgang mit anderen Contributors
- ✅ Konstruktives Feedback
- ✅ Fokus auf das Projekt-Ziel
- ❌ Keine Diskriminierung, Harassment oder Trolling

Bei Verstößen kontaktiere die Projekt-Maintainer.

---

## Wie kann ich beitragen?

### 1. Bug Reports

**Problem gefunden?** Erstelle ein GitHub Issue:

```markdown
**Bug Description:**
Beschreibe das Problem kurz und präzise.

**How to Reproduce:**
1. Step 1
2. Step 2
3. Expected: ...
4. Actual: ...

**Environment:**
- OS: Linux / macOS / Windows
- Python Version: 3.11
- Aurik Version: 8.0.0
- GPU: AMD Radeon (ROCm-kompatibel) / CPU + optionale AMD-GPU (ROCM/DirectML)

**Additional Context:**
Screenshots, Error-Messages, Logs
```

### 2. Feature Requests

**Neue Idee?** Erstelle ein GitHub Issue:

```markdown
**Feature Description:**
Was soll die neue Funktion tun?

**Use Case:**
Warum ist die Funktion nützlich?

**Proposed Implementation:**
(Optional) Wie könnte man es umsetzen?

**Alternatives:**
(Optional) Andere Lösungsansätze?
```

### 3. Code Contributions

**Workflow:**

1. Fork das Repository
2. Erstelle einen Feature-Branch (`git checkout -b feature/amazing-feature`)
3. Implementiere deine Änderungen
4. Schreibe Tests
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Erstelle einen Pull Request

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- (Optional) AMD GPU mit ROCm 7.2.4

### Installation

```bash
# 1. Fork & Clone
git clone https://github.com/YOURUSERNAME/Aurik_Standalone.git
cd Aurik_Standalone

# 2. Virtual Environment
python3.11 -m venv .venv_aurik
source .venv_aurik/bin/activate  # Linux/macOS
# ODER
.venv_aurik\Scripts\activate     # Windows

# 3. Install Development Dependencies
pip install -r requirements/requirements-dev.txt

# 4. Install Pre-Commit Hooks
pre-commit install

# 5. Run Tests (verify setup)
pytest tests/ -v
```

**Expected:** `187 passed in ~5 minutes`

---

## Pull Request Prozess

### 1. Branch Naming

**Format:** `<type>/<short-description>`

**Types:**

- `feature/` - Neue Funktionen
- `bugfix/` - Bug-Fixes
- `docs/` - Dokumentation
- `refactor/` - Code-Refactoring
- `test/` - Tests hinzufügen/verbessern

**Beispiele:**

```bash
git checkout -b feature/add-neural-codec
git checkout -b bugfix/fix-vinyl-crackle-detection
git checkout -b docs/update-api-reference
```

---

### 2. Commit Messages

**Format:**

```
<type>: <subject> (max 50 chars)

<body> (optional, max 72 chars per line)

<footer> (optional: fixes #123)
```

**Types:**

- `feat:` - Neue Funktion
- `fix:` - Bug-Fix
- `docs:` - Dokumentation
- `test:` - Tests
- `refactor:` - Code-Refactoring
- `perf:` - Performance-Optimierung
- `chore:` - Build/Tooling

**Beispiele:**

```bash
git commit -m "feat: Add binaural processing (Phase 11)"

git commit -m "fix: Vinyl crackle detection false positives"

git commit -m "docs: Update API reference for ProcessingMode"
```

---

### 3. Pull Request Template

```markdown
## Description
Kurze Beschreibung der Änderungen.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring
- [ ] Performance improvement

## Testing
- [ ] All existing tests pass
- [ ] New tests added (if applicable)
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project coding standards
- [ ] Documentation updated (if needed)
- [ ] No merge conflicts
- [ ] Pre-commit hooks pass
```

---

### 4. Review Process

**Was passiert nach deinem Pull Request?**

1. **Automated Checks:** CI/CD läuft automatisch (Tests, Linting)
2. **Code Review:** Maintainer reviewt deinen Code
3. **Feedback:** Du erhältst Feedback/Verbesserungsvorschläge
4. **Approval:** Nach Review wird der PR approved
5. **Merge:** Maintainer merged deinen PR

**Timeline:** Erwarte Feedback innerhalb von 1-3 Tagen.

### 5. Push-Regeln (Release-Must)

Ein Push ist nur zulaessig, wenn der lokale Qualitaets-Gate gruen ist.

Pflicht vor jedem Push:

1. Geaenderte Unit-Tests und betroffene Integrations-Tests ausfuehren.
2. Keine neuen Lint-/Type-Fehler in geaenderten Dateien.
3. Bei normativen Verhaltensaenderungen Changelog/Spec/Audit aktualisieren.
4. Keine Sidepaths am kanonischen Bridge/Denker/Export-Vertrag vorbei.

Mindestkommando-Set:

```bash
"${workspaceFolder}/.venv_aurik/bin/python" -m pytest tests/unit -p no:xdist --override-ini="addopts=--strict-markers --import-mode=importlib" --timeout=30 --tb=short -q --disable-warnings --no-header
"${workspaceFolder}/.venv_aurik/bin/python" -m pytest tests -m "not e2e and not ml" -p no:xdist --override-ini="addopts=--strict-markers --import-mode=importlib" --timeout=30 --tb=short -q --disable-warnings --no-header --maxfail=20 --ignore=tests/unit
```

Push-Blocker:

- fehlgeschlagene Pflicht-Tests
- neue Lint-/Type-Findings in geaenderten Dateien
- normative Code-Aenderung ohne Delta in Changelog/Spec/Audit

---

## Coding Guidelines

### Python Style Guide

**We follow PEP 8** mit einigen Aurik-spezifischen Ergänzungen:

#### 1. Code Formatting

**Tool:** `black` (Auto-Formatter)

```bash
# Format all files
black .

# Check formatting
black --check .
```

**Config** (`.pyproject.toml`):

```toml
[tool.black]
line-length = 120
target-version = ['py311']
```

---

#### 2. Import Ordering

**Tool:** `isort`

```bash
# Sort imports
isort .

# Check imports
isort --check .
```

**Order:**

1. Standard Library
2. Third-Party
3. Local Imports

**Beispiel:**

```python
# Standard Library
import os
import sys
from pathlib import Path

# Third-Party
import numpy as np
import torch
from transformers import AutoModel

# Local
from backend.api.bridge import get_aurik_denker_instance
from dsp.denoiser import Denoiser
```

---

#### 3. Type Hints

**Empfohlen:** Type Hints für alle Public Functions/Classes

```python
def restore(
    audio: np.ndarray,
    sr: int,
    mode: ProcessingMode = ProcessingMode.RESTORATION,
    config: Optional[ProcessingConfig] = None,
) -> np.ndarray:
    """
    Restore audio with optional mode and config.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate (Hz)
        mode: Processing mode (default: RESTORATION)
        config: Optional custom configuration

    Returns:
        Restored audio (48 kHz, float32)
    """
    pass
```

---

#### 4. Docstrings

**Format:** Google Style

```python
def process_audio(audio: np.ndarray, sr: int, aggressive: float = 0.5) -> np.ndarray:
    """
    Process audio with adaptive restoration.

    This function applies a 14-phase restoration pipeline with automatic
    defect detection and semantic-aware processing.

    Args:
        audio: Input audio array (mono or stereo)
        sr: Sample rate in Hz
        aggressive: Restoration aggressiveness (0.0-1.0)
            - 0.0-0.3: Conservative
            - 0.3-0.7: Moderate
            - 0.7-1.0: Aggressive

    Returns:
        Restored audio array (48 kHz, float32)

    Raises:
        ValueError: If audio is empty or sr is invalid

    Example:
        >>> audio, sr = sf.read('input.wav')
        >>> restored = process_audio(audio, sr, aggressive=0.5)
        >>> sf.write('output.wav', restored, 48000)
    """
    pass
```

---

#### 5. Logging

**Use `logging` module, not `print()`**

```python
import logging

logger = logging.getLogger(__name__)

# Info
logger.info("🎵 Starting Phase 2: Mechanical Artifacts")

# Warning
logger.warning("⚠️ High noise floor detected: {:.1f} dB".format(noise_floor))

# Error
logger.error("❌ Failed to load ML model: DeepFilterNet")

# Debug (nur in Development)
logger.debug(f"Audio shape: {audio.shape}, SR: {sr}")
```

**Levels:**

- `DEBUG`: Detailed information (nur Development)
- `INFO`: Normal processing steps (default)
- `WARNING`: Potential issues
- `ERROR`: Errors that need attention
- `CRITICAL`: System failures

---

### File Structure

```
Aurik_Standalone/
├── core/                     # Core processing logic
│   ├── unified_restorer_v3.py
│   ├── processing_modes.py
│   └── ...
├── dsp/                      # DSP modules
│   ├── denoiser.py
│   ├── declicker.py
│   └── ...
├── backend/                  # Backend modules
│   ├── ml/
│   ├── adaptive_pipeline.py
│   └── ...
├── forensics/                # Forensic analysis
│   ├── unified_analyzer.py
│   └── ...
├── plugins/                  # ML model plugins
│   ├── deepfilternet_v3_ii_plugin.py
│   └── ...
├── tests/                    # Test suite
│   ├── test_unified_restorer.py
│   └── ...
├── docs/                     # Documentation
│   ├── guides/
│   ├── api/
│   └── ...
└── requirements/             # Dependencies
    ├── requirements.txt
    └── requirements-dev.txt
```

---

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/unit/test_spec_upgrade_gate.py -v

# Specific test
pytest tests/test_unified_restorer.py::test_restore_basic_noise_removal -v

# E2E tests (slow, requires audio files)
pytest tests/test_e2e_magicbutton.py -v -s

# Coverage report
pytest tests/ --cov=core --cov=dsp --cov-report=html
```

---

### Writing Tests

**Format:** `pytest` style

```python
# tests/test_my_feature.py
import pytest
import numpy as np
from backend.api.bridge import get_aurik_denker_instance

def test_restore_basic():
    """Test basic restoration."""
    # Arrange
    sr = 48000
    audio = np.random.randn(sr * 3).astype(np.float32) * 0.1  # 3s noise
    denker = get_aurik_denker_instance()

    # Act
    restored = denker.denke(audio, sr, mode="restoration").audio

    # Assert
    assert restored is not None
    assert restored.shape[0] > 0
    assert restored.dtype == np.float32
    assert -1.0 <= restored.max() <= 1.0

def test_restore_invalid_input():
    """Test error handling for invalid input."""
    denker = get_aurik_denker_instance()

    with pytest.raises(ValueError):
        denker.denke(None, 48000, mode="restoration")  # Should raise ValueError
```

**Best Practices:**

- ✅ Test one thing per test function
- ✅ Use descriptive test names (`test_feature_scenario`)
- ✅ Follow Arrange-Act-Assert pattern
- ✅ Use fixtures für repeated setup
- ✅ Test edge cases (empty input, zero SR, etc.)

---

### Test Coverage

**Ziel:** >80% Code Coverage

```bash
# Generate coverage report
pytest tests/ --cov=core --cov=dsp --cov-report=html

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## Dokumentation

### Wann Dokumentation aktualisieren?

**Ja, wenn:**

- ✅ Neue Public API hinzugefügt
- ✅ API-Signatur geändert
- ✅ Neue Features implementiert
- ✅ Bug-Fix ändert Verhalten
- ✅ Neue Dependencies hinzugefügt

**Nein, wenn:**

- ❌ Nur interne Refactorings
- ❌ Private Functions geändert
- ❌ Comments hinzugefügt

---

### Dokumentations-Struktur

```
docs/
├── guides/                   # User Guides
│   ├── USER_GUIDE.md
│   ├── INSTALLATION.md
│   ├── CONFIGURATION.md
│   └── QUICKSTART_SUPPORT.md
├── api/                      # API Reference
│   └── PYTHON_API.md
├── architecture/             # System Architecture
│   ├── ARCHITECTURE.md
│   └── PIPELINE_FLOW_ANALYSIS.md
├── development/              # Development Docs
│   ├── CONTRIBUTING.md (this file)
│   └── TESTING.md
└── INDEX.md                  # Documentation Hub
```

---

### Markdown Guidelines

**Syntax:** GitHub Flavored Markdown

**Best Practices:**

- ✅ Use headings (`##`, `###`) für Struktur
- ✅ Use code blocks (` ```python `) für Code
- ✅ Use tables für Daten
- ✅ Use lists für Aufzählungen
- ✅ Link zu anderen Docs (`[Text](path/to/doc.md)`)

**Beispiel:**

```markdown
## Feature Name

### Overview
Brief description.

### Usage
\`\`\`python
from core import Feature
feature = Feature()
result = feature.process(data)
\`\`\`

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `np.ndarray` | Required | Input data |
| `mode` | `str` | `'auto'` | Processing mode |

### See Also
- [Related Feature](./related.md)
- [Architecture](../architecture/ARCHITECTURE.md)
```

---

## Contributor Recognition

**Alle Contributors werden anerkannt!**

- GitHub Contributors Liste
- Changelog mentions
- (Optional) Contributor Badge

---

## Fragen?

**Unsicher bei etwas?**

1. Check [Existing Issues](https://github.com/yourusername/Aurik_Standalone/issues)
2. Check [Discussions](https://github.com/yourusername/Aurik_Standalone/discussions)
3. Open a new Issue mit Tag `question`

---

**Vielen Dank für deinen Beitrag zu Aurik! 🎵**

---

**© 2026 Aurik Audio Restoration System**  
**Version:** 8.0.0 | **Contributing Guide**
