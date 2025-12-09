from ..metadata.front_3d import RAW_DATA_DIR

import huggingface_hub


def download(output_dir):
    huggingface_hub.snapshot_download(
        repo_id="Thaparoshan143/edy-dataset-raw",
        repo_type="dataset",
        local_dir=output_dir,
    )


if __name__ == "__main__":
    download(RAW_DATA_DIR)
