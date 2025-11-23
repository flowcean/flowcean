from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS = ROOT / "artifacts"
DATASETS = ARTIFACTS / "datasets"
MODELS = ARTIFACTS / "models"

for p in [ARTIFACTS, DATASETS, MODELS]:
    p.mkdir(parents=True, exist_ok=True)
