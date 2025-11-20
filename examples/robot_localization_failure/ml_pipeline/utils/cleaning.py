import shutil
from .paths import ARTIFACTS

def clean_artifacts():
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
        print("✔ Deleted artifacts/")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("✔ Recreated artifacts/")
