# Decrackle-Challenger-Runde — Incumbent Banquet-Vinyl vs. Kandidat RBME-Net

- Datum: 2026-08-15 | Task: Decrackle/Vinyl-Reinigung auf synthetischer Ground-Truth
- Incumbent: **Banquet-Vinyl** (ONNX, `models/banquet/banquet_vinyl_final.onnx`, 91 MB — „Blind Audio Noise Quality Enhancement“, 2023)
- Kandidat: **RBME-Net** (Bando et al. 2023 — Spec-04-Primary für Decrackle)

## Messstand Incumbent (scripts/dsp_benchmark.py, deterministisch)

| Methode | ref_snr_mean | lsd_mean | edge_ratio_max |
|---|---|---|---|
| **aurik_banquet** | **5,7 dB** | **86,6** | **1,0** |
| spectral_gating_ref | 0,6 dB | 125,2 | 1,3 |
| wiener_ref | 1,2 dB | 132,7 | 0,8 |
| aurik_omlsa | −3,4 dB | 135,6 | 393,0 |

Banquet ist auf diesem Task **objektiv der beste gemessene Pfad** (5,7 dB
Referenz-SNR, niedrigste Spektraldistanz, saubere Kanten) — der ML-Pfad ist
damit messbar gerechtfertigt, nicht nur deklariert.

## Gate-Verifikation (B6, v10.900)

`tests/unit/test_banquet_gate.py` beweist zur Laufzeit:
- Digital-Material lädt **nie** die Banquet-Session (B6: −1,3 dB auf Digital).
- Vinyl aktiviert den ML-Pfad.
- **Chain-Aware**: Vinyl irgendwo in der Kette (z. B. Vinyl→Cassette→MP3)
  aktiviert Banquet ebenfalls.

## Offene Punkte → Stand Rev. 2026-08-16

1. ~~B11 (v10.900) fehlt~~ → **umgesetzt**: HF-Rauschfloor-Check in beiden
   Banquet-Pfaden (Warnung ab +1 dB, Rollback ab +3 dB, Metadaten
   `b11_hf_floor_delta_db`) + Vertragstests.
2. ~~RBME-Net-Beschaffung offen~~ → **aufgelöst als Phantom-Zitat**: Kein
   veröffentlichtes Modell „RBME-Net (Bando et al. 2023)“ auffindbar; Spec 04
   korrigiert auf den gemessenen Primär **Banquet-Vinyl** (Rev. 2026-08-16),
   Registry-Zeile enforced.
3. **BSRNN (arXiv 2212.00406) abgelehnt**: Sprach-Enhancement (VCTK-DEMAND,
   DNS-Challenge) — Domänen- und Task-Mismatch für Musik-Decrackle; Aufnahme
   nur nach Musik-Finetuning + Challenger-Messung gegen Banquet denkbar.
4. Bewertung: Künftige Decrackle-Kandidaten entscheidet `dsp_benchmark` auf
   denselben Metriken — kein Hörtest für die Objektiv-Messung nötig.
