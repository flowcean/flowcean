from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS = ROOT / "artifacts"
DATASETS = ARTIFACTS / "datasets"
MODELS = ARTIFACTS / "models"
RESULTS = ARTIFACTS / "results"

for p in [ARTIFACTS, DATASETS, MODELS, RESULTS]:
    p.mkdir(parents=True, exist_ok=True)
