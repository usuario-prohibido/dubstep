Thank you for your interest in contributing!

Guidelines
- Fork the repository and create a feature branch.
- Keep changes focused and provide a short description in your PR.
- Run the project locally and ensure it still executes.

Testing locally
- Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Run the script to check it starts and exports audio:

```bash
python3 dubstep_voz.py
# or export without playback
python3 export_mix.py
```

Reporting issues
- Use the issue templates and include logs/steps to reproduce.

Code style
- Keep code readable and avoid large unrelated changes in a single PR.
