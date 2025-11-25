# edy

Generalizable object aware 3D scene reconstruction from a single image.

**At high level**\
Single Image (Input) &rightarrow; Our Pipeline &rightarrow; 3D scene (Output)


> [!IMPORTANT]
> This project is actively under development. Be cautious something might break!!. Get access to the project report at [here](https://typst.app/project/wORrkYboE1ePaDPfjRUrxt)

## Setup
- Clone the repo
```bash
git clone https://github.com/Arktetra/edy # for specific branch use -b <branch_name>
```

- Install the packages (run from the project root directory)
```bash
pip install -e .
```

> [!TIP]
> If your device have lower computation power, try [google colab](https://colab.research.google.com/) for resouce hungry operartions/training and local only for any additional implementation.

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

> [!NOTE]
> Only the partaa of the dataset is downloaded.
