from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

def seperator(symb = "-", count = 100):
    string = f"{symb}".join(["" for _ in range(0, count)])
    print(f"{string}") 


def load_env_variables():
    current_path = Path(__file__).resolve()
    root = current_path.parent.parent

    dotenv_path = root / ".env"

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded environment variables from: {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Ensure that it is in the project root")


def get_masked_img(oimg: Image.Image, mimg: Image.Image | list, bg_color = (0, 0, 0, 0)):
    """
    args:
    - oimg: original image 
    - mimg: mask/s to use 
    - bg: mask bg color where value are non-existant (i.e 0)

    return: masked image/s after mask applied to original image
    """

    bg = Image.new("RGBA", oimg.size, bg_color)
    if isinstance(mimg, Image.Image):
        return Image.composite(oimg, bg, mimg)
    elif isinstance(mimg, list):
        res = []

        for mi in mimg:
            res.append(Image.composite(oimg, bg, mi))

        return res




