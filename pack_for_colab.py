"""Zip the code (not data, not checkpoints) for upload to Google Drive.

    python pack_for_colab.py   ->  bangla-mini-gpt-code.zip

Upload it to  My Drive/bangla-mini-gpt/  then open colab/train_on_colab.ipynb.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
INCLUDE = ["minigpt/*.py", "tests/*.py", "*.py", "requirements.txt", "pytest.ini"]

out = ROOT / "bangla-mini-gpt-code.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for pattern in INCLUDE:
        for f in ROOT.glob(pattern):
            z.write(f, f.relative_to(ROOT))
            print("  +", f.relative_to(ROOT))
print(f"-> {out} ({out.stat().st_size/1e3:.0f} KB)")
