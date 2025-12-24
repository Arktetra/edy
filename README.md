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
cd submodules
git clone https://github.com/JeffreyXiang/FlexGEMM.git
cd FlexGEMM
pip install . --no-build-isolation
cd ..
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install . --no-build-isolation
cd ..
cd ..
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=4 pip install . --no-build-isolation
cd ../..
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