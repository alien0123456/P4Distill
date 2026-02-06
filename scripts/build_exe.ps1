$ErrorActionPreference = "Stop"
python -m pip install --upgrade --user pyinstaller
python -m PyInstaller --onefile -n distillkit src/distillkit/cli.py
