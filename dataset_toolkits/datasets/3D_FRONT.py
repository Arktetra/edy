import argparse
import shutil
import subprocess
from pathlib import Path
from ..utils import load_env_variables
from ..metadata.common import RAW_DATA_DIR

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "900"
# os.environ["HF_DEBUG"] = "1"

import huggingface_hub

FILENAME = "3D-FRONT-TEST-SCENE.tar.gz"


def download(output_dir):
    load_env_variables()
    huggingface_hub.login(token=os.environ["HF_TOKEN"])
    huggingface_hub.hf_hub_download(
        repo_id="huanngzh/3D-FRONT",
        repo_type="dataset",
        filename=FILENAME,
        local_dir=output_dir,
    )


def extract(output_dir):
    print(f"Extracting {FILENAME} to {output_dir}")
    try:
        subprocess.run(f"tar -xzvf {output_dir / FILENAME} -C {output_dir}", shell=True, check=True)
        print("Extraction successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error during file extraction: {e}")
        exit()


def clean(output_dir: Path):
    """
    Cleans the unnecessary glb models from the dataset.
    """
    print("Cleaning unnecessary glb files from the given output dir.")
    required_glb_files = []
    try:
        for sub_dir in output_dir.iterdir():
            for x in Path(sub_dir).iterdir():
                if x.name.endswith(".glb") and "full" not in x.name:
                    required_glb_files.append(x)
                else:
                    if x.is_dir():
                        shutil.rmtree(x)
                    else:
                        x.unlink()

        for i, file in enumerate(required_glb_files):
            file.rename(output_dir / f"{i}.glb")

        for item in output_dir.iterdir():
            if Path(item).is_dir():
                item.rmdir()
    except NotADirectoryError:
        print("No sub-directories present to process.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="3D-FRONT-dataset", description="Download or extract 3D-FRONT dataset")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--output-dir", type=str, default="3D-FRONT")

    args = parser.parse_args()

    if not (args.download or args.extract or args.clean):
        raise ValueError("Either --download or --extract flag must be set.")
    if args.download:
        download(output_dir=RAW_DATA_DIR / args.output_dir)
    if args.extract:
        extract(output_dir=RAW_DATA_DIR / args.output_dir)
    if args.clean:
        clean(output_dir=RAW_DATA_DIR / args.output_dir)
