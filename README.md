# Experimental music generator

A Python script that synthesizes a dubstep-style track with generated speech, live Plotly waveform visualization served by Flask, and subtitle sync. The project generates and plays audio locally and provides a browser-based live waveform with low-opacity subtitles.

Features:
- Synthesized elements: kick, bass (wobble + sub), hats, synths, pads
- Piano + violin blended chords
- Growing arrangement complexity (arps, fills)
- gTTS-generated German speech overlaid at 10s with per-word subtitle timing
- Live waveform served at `/` (Plotly + polling `/samples`)

Requirements
-----------
- Python 3.8+
- ffmpeg (required by `pydub` and `gTTS`)

Python packages (install with `pip`):

```bash
pip install -r requirements.txt
```

Running
-------
Start the script locally and open the browser (the script attempts to open it automatically):

```bash
python3 dubstep_voz.py
```

The Flask server serves the live waveform at `http://127.0.0.1:8050/`.

Notes
-----
- Producing a highly realistic piano/violin would normally use real instrument samples; this project uses simple synthesis for portability.
- Adjust loudness and mastering parameters in `create_dubstep_track()` and `apply_mastering()`.

License
-------
This repository is released under the MIT License. See the `LICENSE` file for details.

If you publish this project, keep the `LICENSE` file in the repository root so others can reuse the code under these terms.

