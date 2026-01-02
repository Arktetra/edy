from PIL import Image

def seperator(symb = "-", count = 100):
    string = f"{symb}".join(["" for _ in range(0, count)])
    print(f"{string}") 


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




