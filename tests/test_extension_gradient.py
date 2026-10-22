import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_extension_gradient_prototype():
    subprocess.run([sys.executable, "scripts/extension_prototype.py"], cwd=ROOT, check=True)
    result = json.loads((ROOT / "results/extension_gradient_check.json").read_text())
    assert result["max_relative_error"] < 1e-4
