# Practical_Programs
Kleine Praktische Programme .EXE

## DeepFloyd-IF HUD

`deepfloyd_if_hud.py` stellt eine einfache Gradio-Oberfläche für das DeepFloyd-IF-Modell bereit. Beim Start wird überprüft, ob alle benötigten Python-Pakete in passender Version vorhanden sind. Fehlt etwas, installiert das Skript die Pakete automatisch.

Da `diffusers` 0.16.0 auf die veraltete Funktion `cached_download()` setzt, wird `huggingface-hub` strikt auf Version 0.25.2 festgenagelt. Die aktuelle 4.x-Serie von Gradio endet bei Version `4.44.1`; ältere Releases kollidieren mit `pydantic>=2.11`. Entsprechend pinnt der Installer Gradio auf genau `4.44.1` und erzwingt `pydantic<2.11`. Da PyTorch 1.13 noch gegen **NumPy 1.x** gebaut ist, bleibt `numpy` auf `<2` begrenzt.

Starten Sie das Programm innerhalb eines Python-3.10-venv mit:

```bash
python deepfloyd_if_hud.py
```

Nach dem ersten Durchlauf öffnet sich das Interface direkt unter <http://localhost:7860>.


Beim Klick auf "Generate" prüft das Programm außerdem, ob die benötigten DeepFloyd‑IF‑Modelordner (`IF-I-XL-v1.0`, `IF-II-L-v1.0`) im gewählten Checkpoint-Pfad vollständig vorliegen. Fehlende oder nur als Git‑LFS-Pointer vorhandene Dateien werden automatisch per `snapshot_download()` nachgeladen. Unterbrochene Downloads werden beim nächsten Start automatisch fortgesetzt, selbst wenn das Programm zwischendurch beendet wurde. Im Interface erscheint eine Meldung, welche Stages ggf. ergänzt wurden.

### Modellverwaltung

Der gewählte **Models Folder** kann mehrere Unterordner mit verschiedenen
Modellsets enthalten. Nach einem Klick auf „Refresh Models“ listet ein Dropdown
alle gefundenen Modelle auf. Ein farbiger Status zeigt sofort an, ob ein Modell
vollständig (`✅`), teilweise vorhanden (`🟠`) oder komplett fehlend (`❌`) ist.

Liegt ein Modell direkt im gewählten Ordner (d.h. die Stage‑Verzeichnisse
`IF-I-XL-v1.0` usw. befinden sich ohne zusätzliche Ebene im Models Folder),
wird es unter dem Namen `IF` angezeigt.
Über „Install/Update“ lassen sich fehlende Stages nachladen; "Delete Model"
entfernt das ausgewählte Modell komplett. Das zuletzt gewählte Modell und der
Pfad werden in `settings.json` gespeichert und beim nächsten Start automatisch
geladen.

Downloads erfolgen nun nacheinander (kein paralleler Fetch), wodurch
Timeout-Probleme bei instabilen Verbindungen reduziert werden. Vor jedem
Download zeigt das Log an, wie viele Dateien das jeweilige Stage-Repo enthält
und wie groß es ist.
Beim Aktualisieren listet das HUD alle gefundenen Modelle mit ihrem
Gesamtumfang (MB/GB) auf, damit man den belegten Speicher schnell überblicken
kann.

Sollte ein Modell trotz Nachladeversuchen unvollständig bleiben (etwa bei
Netzproblemen), bricht der "Generate"-Aufruf mit einer Hinweisnachricht ab,
anstatt in einem Ladefehler zu enden.

Auf Windows sollte man `HF_HUB_DISABLE_SYMLINKS_WARNING=1` setzen, damit beim
Download keine Symlink-Warnung erscheint. Wer Pakete ohne Internet aus einem
lokalen Ordner installieren möchte, kann dessen Pfad über die Umgebungsvariable
`HUD_WHEEL_DIR` angeben. Setzt man außerdem `HF_HUB_OFFLINE=1`, nutzt das HUD
ausschließlich den lokalen Hugging‑Face-Cache und setzt angefangene Downloads
mit `snapshot_download(..., resume_download=True)` fort.

Falls das Modell in einem privaten oder lizenzierten Hugging‑Face-Repository
liegt, kopieren Sie die Beispieldatei `hf_token.example.txt` zu `hf_token.txt`
und tragen dort Ihren persönlichen Token ein oder setzen Sie die
Umgebungsvariable `HF_TOKEN`. Das Skript liest diesen Token automatisch und
nutzt ihn für alle Downloads. Ist kein Token vorhanden, erscheint beim Start
eine Warnung und der Download geschützter Modelle schlägt fehl.

Das optionale Stage‑III‑Modell existiert bislang nicht öffentlich. Falls es im
Checkpoint-Ordner fehlt, nutzt das HUD automatisch den Stable-Diffusion-x4-
Upscaler für die 1024 px-Ausgabe.
