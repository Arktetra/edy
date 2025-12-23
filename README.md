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

Some dependencies need to be installed manually:

```bash
git clone https://github.com/JeffreyXiang/FlexGEMM.git
cd FlexGEMM
pip install . --no-build-isolation
cd ..
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=4 pip install . --no-build-isolation
cd ..
pip install ./extensions/vox2seq --no-build-isolation
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
.../edy/data/raw/3D-FRONT/3D-FRONT-TEST-SCENE/
|--f7cf5008-957f-4286-8854-20f34f2a8ce3/
|  |--MasterBedroom-11613/
|  |--MasterBedroom-11613.glb
|  |--MasterBedroom-11613_full.glb
...
```
The `*.glb` files without `full` in its name under these directories contain all the necessary information required in this project, so we can remove everything else. These steps can be carried out by running the following command.

```bash
python -m dataset_toolkits.datasets.3D_FRONT --clean --output-dir 3D-FRONT/3D-FRONT-TEST-SCENE
```

This command will change the directory structure to:
```
.../edy/data/raw/3D-FRONT/3D-FRONT-TEST-SCENE/
|--0.glb
|--1.glb
...
|--996.glb

```

The 3D models can then be rendered to 4 different views through two processes by:
```bash
python -m dataset_toolkits.render --num-views 4 --max-workers 2
```
This also performs a kind of data augmentation. The above command renders the different views of the scene and the corresponding masks:
```
.../edy/data/processed/EDY/
|--masks/
|  |--0/
|  |  |--0/
|  |  |  |--0.png
|  |  |  |--1.png
|  |  |  |--2.png
|  |  |  |--3.png
|  |  |  |--4.png
|  |  |--1/
|  |  |--2/
|  |  |--3/
|  |--1/
...
|--models/
|  |--0.ply
|  |--1.ply
...
|--renders/
|  |--0/
|  |  |--0.png
|  |  |--1.png
|  |  |--2.png
|  |  |--3.png
|  |--1/
...
```

Finally, perform post processing on the mask and apply it on the rendered views.
```bash
python -m dataset_toolkits.mask_post_processing
```
The directory structure will then become like:
```
.../edy/data/processed/EDY/
|--masked/
|  |--0/
|  |  |--0/
|  |  |  |--0.png
|  |  |  |--1.png
|  |  |  |--2.png
|  |  |  |--3.png
|  |  |  |--4.png
|  |  |--1/
|  |  |--2/
|  |  |--3/
|  |--1/
...
|--masks/
|--models/
|--processed_masks/
|  |--0/
|  |  |--0/
|  |  |  |--0.png
|  |  |  |--1.png
|  |  |  |--2.png
|  |  |  |--3.png
|  |  |  |--4.png
|  |  |--1/
|  |  |--2/
|  |  |--3/
|  |--1/
...
|--renders/
...
```

**Note:** Only the test scene of the dataset is downloaded.

<hr />

**Transform and Object (.ply) extraction**

For extraction of the transform of each object in scene and each object seperately exporting (.ply) use the script under dir `dataset_toolkits/blender_script/3D-Front` with cmd: (from project directory)

```bash
python -m dataset_toolkits.blender_script.3D-Front.extractor --at-once --dataset-dir <path-to-dataset> --output-dir <path-to-output>
# eg: python -m dataset_toolkits.blender_script.3D-Front.extractor --at-once --dataset-dir /content/edy/dataset --output-dir /content/edy/data/ 
```

all available args:
value based
- `--dataset-dir`: directory at which dataset resides
- `--output-dir`: directory where extracted files will be stored
- `--ds-filter-ext`: filter only given extension from dataset
```bash 
... --ds-filter-ext glb ply obj # this will only select files with glb, ply, obj extension from dataset folder
```

flag only
- `--objects`: when used will extract only objects
- `--transforms`: when used will extract only transforms
- `--all`: when used will extract both transform and objects (seperate load)
- `--at-once`: when used will extract both transform & objects (but at single load.)
