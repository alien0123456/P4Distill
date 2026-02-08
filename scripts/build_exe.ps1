$ErrorActionPreference = "Stop"
python -m pip install --upgrade --user pyinstaller
python -m PyInstaller --clean distillkit.spec
