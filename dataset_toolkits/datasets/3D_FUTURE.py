from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

import argparse
import os
import pandas as pd
import re
from typing import Optional
import zipfile
import hashlib

def get_file_hash(file: str) -> str:
    sha256 = hashlib.sha256()
    with open(file, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)

    return sha256.hexdigest()

def download(metadata, output_dir, **kwargs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (output_dir / "raw" / "3D-FUTURE-model.zip").exists():
        raise FileNotFoundError(f"3D-FUTURE-model.zip not found. Ensure that it exists at {output_dir}/raw directory.")

    downloaded = {}
    with zipfile.ZipFile(output_dir / "raw" / "3D-FUTURE-model.zip") as zip_ref:
        all_names = zip_ref.namelist()
        instances = [instance[:-1] for instance in all_names if re.match(r"^3D-FUTURE-model/[^/]+/$", instance)]
        instances = list(filter(lambda x: x in metadata.index, instances))

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=len(instances), desc="Extracting") as pbar:
            def worker(instance: str) -> Optional[str]:
                try:
                    instance_files = list(filter(lambda x: x.startswith(f"{instance}/") and not x.endswith("/"), all_names))
                    zip_ref.extractall((output_dir / "raw"), members=instance_files)
                    sha256 = get_file_hash(output_dir/ "raw" / f"{instance}/image.jpg")
                    pbar.update()
                    return sha256

                except Exception as e:
                    pbar.update()
                    print(f"Error extracting {instance}: {e}")
                    return None
            sha256s = executor.map(worker, instances)
            executor.shutdown(wait=True)
    
    for k, sha256 in zip(instances, sha256):
        if sha256 is not None:
            if sha256 == metadata.loc[k, "sha256"]:
                downloaded[sha256] = Path("raw" / k / "raw_model.obj")
            else:
                print(f" Error downloading {k}: sha256 do no match")

    return pd.DataFrame(downloaded.items(), columns=["sha256", "local_path"])


