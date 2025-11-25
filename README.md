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

<hr />

### For doxygen documentation
- Verify the [doxygen](https://www.doxygen.nl/) installation on device
```bash
doxygen --version
```

> [!NOTE]
> For doxygen installation guide check [here](https://www.doxygen.nl/manual/install.html). If you are on mac with brew installed, simply run `brew install doxygen`

- Generate the Doxygen config file as:
```bash
doxygen -g config_file # config_file (optional), deafults to Doxyfile
```

- Run the config file (with all field adjusted as per required):
```bash
doxygen config_file 
```

> [!WARNING]
> Change the input (source) path to the all the source files/folders & expanding the folder option using recursive before running the config file. **Refer note below for more info**

> [!NOTE]
> Change the `INPUT`, `RECURSIVE`, `EXTRACT_ALL`, etc fields appropiately in your `config_file`. example:
> ```bash
> # inside the config_file
> INPUT = dataset_toolkits/ scene/ scripts/ src/ tests/ utils/
> 
> RECURSIVE = YES
> 
> EXTRACT_ALL = YES
> # so on, as per required
>```


After `config_file` run it will generate the `latex/` & `html/` folder with required content (docs). For html part either use `live server` (VSCode extension) or use python packages (http) as:
```bash
python -m http.server 8000 
```
