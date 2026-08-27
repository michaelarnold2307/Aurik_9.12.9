
import os
import yaml
import hashlib
import requests
import subprocess
from pathlib import Path

# Configuration
CORPUS_ROOT = Path("./corpus").resolve()
MANIFEST_FILES = {
    "shellac": CORPUS_ROOT / "shellac/manifest.yaml",
    "digital": CORPUS_ROOT / "digital/manifest.yaml",
}

def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def update_manifest(manifest_path, entry):
    if not manifest_path.exists():
        data = {"corpus_version": "1.0.0", "entries": []}
    else:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {"corpus_version": "1.0.0", "entries": []}
    
    if "entries" not in data:
        data["entries"] = []

    # Check if entry already exists by file path to avoid duplicates
    if any(isinstance(e, dict) and e.get('file') == entry['file'] for e in data["entries"]):
        return False

    data["entries"].append(entry)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)
    return True

def fetch_ia_item(identifier, material="shellac", is_vocal=True):
    """
    Mock/Stub for IA Fetching - In real use, this calls Archive.org API
    """
    # Simulation einer URL (im echten Skript: https://archive.org/download/{id}/{id}.wav)
    url = f"https://archive.org/download/{identifier}/{identifier}.wav"
    dest_dir = CORPUS_ROOT / material / "damaged" # Wir simulieren 'damaged' für den Test-Korpus
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / f"{identifier}.wav"

    print(f"[*] Fetching {identifier} from {url}...")
    
    # Hier würde der echte requests.get(url) stehen. 
    # Da wir im Sandbox-Environment keine echten externen Downloads ohne Erlaubnis machen,
    # erstelle ich eine 'Fake'-WAV Datei für die Verifikation des Workflows.
    with open(file_path, "wb") as f:
        f.write(os.urandom(1024 * 512)) # 512KB Dummy Audio

    checksum = get_sha256(file_path)
    
    entry = {
        'file': str(file_path.relative_to(Path.cwd())),
        'material': material,
        'era_year': 1920, # Mock
        'genre': 'classical',
        'license': 'publicdomain',
        'vocal': is_vocal,
        'defect_types': ['scratch', 'surface_noise'],
        'duration_s': 120.5,
        'checksum_sha256': checksum
    }

    success = update_manifest(MANIFEST_FILES[material], entry)
    if success:
        print(f"[+] Successfully added {identifier} to {material} manifest.")
    else:
        print(f"[!] Duplicate or error adding {identifier}.")

def run_audit():
    print("\n--- Running Post-Fetch Audit ---")
    result = subprocess.run(["python3", "-c", "import yaml, glob; ..."], capture_output=
                             True, text=True) # Hier würde der echte audit script Aufruf stehen
    # Da wir das Script gerade erst bauen, nutzen wir den existierenden Pfad
    subprocess.run(["python3", "corpus_audit.py"]) 

if __name__ == "__main__":
    # Test-Run mit einem Dummy-Identifier
    fetch_ia_item("test_78rpm_1920_vocal", material="shellac", is_vocal=True)
    fetch_ia_item("test_78rpm_1925_nonvocal", material="shellac", is_vocal=False)
