from transformers import Sam2Processor, Sam2Model
import torch
from PIL import Image
import numpy as np
from einops import rearrange
import os

from util import seperator

token = os.environ["HF_TOKEN"]

def_model = "facebook/sam2.1-hiera-tiny"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# model is only loaded on call.. if any opt. required.. cache the results..
def get_mask(img: Image.Image, bboxs = None, points = None, model_name = def_model):
    """
        args:
        - img: PIL image
        - bboxs: (must be ndarray) bounding boxes in the images, default None, i.e no bounding box used..
        - points: (must be ndarray) points in the images, default None, i.e no points used..
        - model_name: name of model to use for segmentation (currently only SAM2 based, with ability to choose from tiny, mid, large..)
    """

    seperator()
    print("==> SAM pipeline start")
    print(f"Running on : {device}")

    print(f"Image dim: {img.size}, bounding box count: {bboxs.shape if isinstance(bboxs, np.ndarray) else 'not found'}")

    if not token:
        return ValueError("HF token required for model access! Unable to load HF_TOKEN from .env file! Try again!")

    model = Sam2Model.from_pretrained(model_name, token=token).to(device)
    processor = Sam2Processor.from_pretrained(model_name, token=token)

    # Define bounding box as [x_min, y_min, x_max, y_max],
    inputs = None

    with torch.no_grad():
        # if bounding box is passed, make it bbox ready
        if isinstance(bboxs, np.ndarray):
            print("Shape length: ", len(bboxs.shape))
            if len(bboxs.shape) == 1:
                print("only one bounding box passed")
                bboxs = [bboxs] # converting to suitable format..

            bboxs = [bboxs]  # should be in format, [batch, number, bbox-coord], usage: [[[]]]
        # if points passed, make it points ready.
        if isinstance(points, np.ndarray):
            print("Shape length: ", len(bboxs.shape))
            if len(points.shape) == 1:
                print("only one point passed")
                points = [points]

            points = [points]

        inputs = processor(images=img, input_boxes=bboxs, input_points=points, return_tensors="pt").to(device)

        outputs = model(**inputs)

    # optional..mask post process..
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

    # The model outputs multiple mask predictions ranked by quality score
    print(f"## Generated {masks.shape[1]} masks with shape {masks.shape}")

    # flatten mask (collapse batch & channel into single..)
    fmasks = (rearrange(masks, 'b c h w -> 1 (b c) h w').squeeze(dim=0))

    # taking from tensor to numpy
    mask_np = np.array(fmasks.detach().cpu().numpy(), dtype=np.uint8)
   
    print(f"*** Masked in numpy shape: {mask_np.shape}")

    # making the order in height, width with zero filled
    combined_mask = np.zeros(shape=img.size[::-1])

    # iterating over all the object mask over bounding box..
    for ind, indm in enumerate(mask_np):
        # * 255 indicates to conversion from 0/1 binary space to 0/255 space (full 8-bit)
        muint8 = indm * 255
        combined_mask = np.clip(combined_mask + muint8, 0, 255)

    combined_mask = np.array(combined_mask, dtype=np.uint8)
    print("-> combined shape: ", combined_mask.shape)

    print("==> Completed pipeline SAM")
    seperator()

    # returning the combined mask from all bounding box...
    return Image.fromarray(combined_mask)