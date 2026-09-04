#!/usr/bin/env python3
"""
Export Guard für die CI‑Lite‑Pipeline.
Der Guard führt:
1️⃣ Lädt Konfiguration aus metadata.yaml
2️⃣ Startet core/audio_exporter.py mit Batch‑Support
3️⃣ Loggt die Ergebnisse in export_log.txt
"""

import pathlib
import subprocess

import yaml


def run(cmd: str, cwd=None):
    """Helper to run a command and capture output."""
    result = subprocess.run(
        cmd,
        cwd=cwd or pathlib.Path.cwd(),
        shell=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    return result.returncode, result.stdout

def main():
    # 1️⃣ Load metadata
    with open("metadata.yaml") as f:
        meta = yaml.safe_load(f)
    # 2️⃣ Build command string
    batch_flag = "--batch" if meta.get("batch", True) else ""
    formats = " ".join(meta["formats"])
    cmd = (
        f"python core/audio_exporter.py "
        f"--sample-rate {meta['sample_rate']} "
        f"--bit-depth {meta['bit_depth']} "
        f"{batch_flag} "
        f"--format {formats}"
    )
    # 3️⃣ Execute
    ret, out = run(cmd)
    print(out)
    # 4️⃣ Log result
    with open("export_log.txt", "a") as log:
        log.write(f"Export finished at {pathlib.datetime.now()}\n")
        log.write(out + "\n")

if __name__ == "__main__":
    main()
