# Vokal-Challenger-Runde — SNR < 10 dB (MIIPHER-Stufe)

- Datum: 2026-08-15 | Task: stark degradierter Gesang (deterministisches Rauschen, SNR 5 dB)
- Incumbent: **MIIPHER-DiT** (Flow-Matching, §v10.14 — Auriks offener Ersatz des proprietären Google-MIIPHER; Checkpoint `models/miipher_dit/checkpoint_best.pt`)
- Kandidat: **SGMSE+** (Richter et al. 2022, lokal; `models/sgmse_plus/sgmse_plus_src_1.ckpt`)

## Kontext (Entscheidung a, 2026-08-15)

Spec 04 (Rev. 2026-08-15) deklariert die MIIPHER-Stufe als **Codec-Restaurierer**
(`mp3_low`/`streaming`/`aac`/`minidisc`) — nicht mehr als SNR<10-Gesang-Stufe.
Der SNR<10-Task gehört SGMSE+ v2. Die Runde wurde deshalb um die
**Codec-Aufgabe** ergänzt (`--task codec`, deklarierter mp3_low-Proxy), in der
der Incumbent (DiT) tatsächlich läuft (`model_used="miipher_dit"`).

## Ergebnis der ersten Messung (belegt)

- **DiT ist funktional** — mit korrektem Material-Kontext (`mp3_low`, rs<30)
  liefert er `model_used="miipher_dit"` (5.9–15.7 s/Item, Flow-Matching).
- **Gate-1-Hazard (§V6)**: Ohne Material-Kontext (`material="unknown"`) lehnt
  `should_apply` ab → `model_used="none"` (stiller Passthrough).
- **Spec↔Implementierungs-Mismatch**: Spec 04 weist die „MIIPHER-Stufe“ dem
  SNR<10-Vokal-Task zu; der DiT (Zielmaterialien `{mp3_low, streaming, aac,
  minidisc}`) gated nur Codec-Degradation — für den SNR<10-Rausch-Task ist er
  Passthrough. Diesen Task trägt das alte MIIPHER-Plugin (Gewichte fehlen →
  DFN-Fallback).
- **SGMSE+-Kandidat** fiel auf CPU nach 12 s internem Timeout auf WPE zurück.

Die Runde wurde deshalb um eine **Codec-Aufgabe** ergänzt (`--task codec`,
  deklarierter mp3_low-Proxy), in der der DiT tatsächlich läuft.

## Aufbau (deterministisch, §G5)

1. Task: 3 Vokal-Items (`tests/real_world_validation/test_library/vocals/`),
   48 kHz, + Rauschen bei SNR 5 dB (Seeds 2026+idx).
2. Incumbent: `miipher_dit_plugin.enhance(task, 48000)` → `incumbent_dit/`.
3. Kandidat: `sgmse_plugin.enhance(task, 48000, sigma=0.5)` → `candidate_sgmse/`.
4. Bewertung: Hörrunde (≥ 10 Hörer/Item) → `challenger_round.py decide`.

## Treiber

`scripts/prepare_vocal_snr_round.py` — erzeugt Task + beide Modell-Läufe und
schreibt `round_manifest.json` (Laufzeiten, Probleme, SNR-Ist).
