from pathlib import Path

ROOT = Path(__file__).parents[2]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

if __name__ == "__main__":
    print(DATA_DIR)
