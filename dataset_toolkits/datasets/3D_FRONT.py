import argparse
from pathlib import Path
import tarfile
from ..utils import load_env_variables

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "900"
# os.environ["HF_DEBUG"] = "1"

import huggingface_hub

CURRENT_DIR = Path(__file__).resolve()
ROOT = CURRENT_DIR.parent.parent.parent
RAW_DATA_DIR = ROOT / "data" / "raw" / "3D-FRONT"

def download():
    load_env_variables()
    huggingface_hub.login(token=os.environ["HF_TOKEN"])
    huggingface_hub.hf_hub_download(
        repo_id="huanngzh/3D-FRONT",
        repo_type="dataset",
        filename="3D-FRONT-SCENE.partaa",
        local_dir=RAW_DATA_DIR,
        
    )

def extract():
    with tarfile.open(RAW_DATA_DIR / "3D-FRONT-SCENE.partaa") as f:
        f.extractall(RAW_DATA_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="3D-FRONT-dataset",
        description="Download or extract 3D-FRONT dataset"
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")

    args = parser.parse_args()
    
    if not (args.download or args.extract):
        raise ValueError("Either --download or --extract flag must be set.")
    if args.download:
        download()
    if args.extract:
        extract()