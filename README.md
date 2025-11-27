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

Extract the downloaded dataset.
```bash
python -m dataset_toolkits.datasets.3D_FRONT --extract
```

If you go the directory of the dataset, then you will encoded directory names with structure as shown in the example below:
```
/3D-FRONT/3D-FRONT-TEST-SCENE
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

**Note:** Only the partaa of the dataset is downloaded.