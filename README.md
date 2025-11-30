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

- For enforcing consistency
```bash
pre-commit install
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

Extract the downloaded dataset.
```bash
python -m dataset_toolkits.datasets.3D_FRONT --extract
```

If you go the directory of the dataset, then you will encoded directory names with structure as shown in the example below:
```
.../3D-FRONT/3D-FRONT-TEST-SCENE/
|--f7cf5008-957f-4286-8854-20f34f2a8ce3/
|  |--MasterBedroom-11613/
|  |--MasterBedroom-11613.glb
|  |--MasterBedroom-11613_full.glb
...
```
The `*_full.glb` files under these directories contain all the necessary information required in this project, so we can remove everything else. These steps can be carried out by running the following command.

```bash
python -m dataset_toolkits.datasets.3D_FRONT --clean --output-dir 3D-FRONT/3D-FRONT-TEST-SCENE
```

This command will change the directory structure to:
```
.../3D-FRONT/3D-FRONT-TEST-SCENE/
|--0.glb
|--1.glb
...
|--996.glb

```

**Note:** Only the partaa of the dataset is downloaded.