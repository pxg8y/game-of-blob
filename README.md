# Blob Life — Neuroevolution & Ökonomie Simulator

Eine interaktive 2D-Artificial-Life- und Wirtschaftssimulation in Python mit Echtzeit-GUI (FastAPI + WebSockets + HTML5 Canvas).

## Kern-Features

1. **Kontinuierliches 2D-Spielfeld mit Tag-/Nacht-Zyklus & Zuhause (`Home`)**
   - Blobs schwärmen tagsüber aus, suchen Nahrung (`🍏` Standard-Nahrung & `🌟` goldene Nahrungs-Cluster) und kehren in der Dämmerung zu ihrem **Zuhause** oder **Gruppen-Dorf** zurück.
   - Wer nachts zu Hause schläft, spart Energie; wer draußen übernachtet, erleidet Kälte-/Energieverlust.
2. **Überleben & Fortpflanzung durch Gene (`2x Tagesbedarf`)**
   - **Verhungern:** Isst ein Blob zu lange nichts oder fällt seine Energie am Tagesende ohne Geldreserven auf `0`, stirbt er.
   - **Reproduktion:** Erreicht ein Blob an einem Tag **mindestens das Doppelte (`2.0x`) seines Überlebensbedarfs**, produziert er am Tagesende einen neuen Blob mit seinen **trainierten neuronalen Netzgewichten** und **Körper-Genen** (`Speed`, `Sense/Sichtweite`, `Size/Größe` – inklusive echtem metabolischem Energie-Trade-off).
3. **Gruppenbildung & Handel (`Essen ↔ Geld`)**
   - **Gruppen (`Group`):** Begegnen sich Blobs mit positivem Sozial-Output, gründen sie einen gemeinsamen Clan/Stamm oder treten einem bestehenden bei, teilen sich ein Gruppendorf, genießen gemeinsame Sichtvorteile und erhalten **30 % Handelsrabatt** untereinander.
   - **Wirtschaft (`Peer-to-Peer & Dorf-Markt`):** Überschüssiges Essen verdirbt über Nacht, während **Geld (`🪙`)** dauerhaft erhalten bleibt. Blobs mit Nahrungsüberschuss verkaufen Essen gegen Geld an hungrige Blobs (oder ans Heimat-Depot). Reiche Blobs können sich an schlechten Tagen mit Geld vor dem Hungertod retten oder Nahrung zukaufen, um die `2x`-Fortpflanzungsschwelle zu knacken.
4. **Frei konfigurierbare Neuronale Netze & Lamarcksche Trainings-Präferenzen**
   - **Architektur-Editor:** Der Nutzer bestimmt jederzeit live die **Netztiefe** (`1–6` Hidden Layers), **Netzbreite** (`4–96` Neuronen oder benutzerdefinierte Layer-Liste wie `32, 16`) und **Aktivierungsfunktion** (`Tanh`, `ReLU`, `GELU`, `SiLU`).
   - **Präferenz-Parameter (Online-RL + Evolution):** Über Live-Slider steuert der Nutzer die Belohnungsgewichte (`Nahrungs-Priorität`, `Handel & Geld-Sinn`, `Gruppen-Sozialität`, `Heimkehr-Disziplin`, `Entdeckerdrang`) sowie `Online-Lernrate`, `Mutationsrate` und `Mutationsstärke`.

## Starten

```bash
python3 main.py --port 8000
```

Tests ausführen:

```bash
python3 -m unittest discover -s tests -v
```
