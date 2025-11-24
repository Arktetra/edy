# edy

Generalizable object aware 3D scene reconstruction from a single image.

## Setup

For setup, run the following from the project root directory:

```bash
pip install -e .
```

## Environment Variable

Create `.env` file at the project root and add the required keys shown in `.env.example` in it.

## Datasets

Download the 3D-FRONT dataset,
```bash
python -m dataset_toolkits.datasets.3D_FRONT --download
```

Extact the downloaded dataset.
```bash
python -m dataset_toolkits.datasets.3D_FRONT --extract
```

**Note:** Only the partaa of the dataset is downloaded.