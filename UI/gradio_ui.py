import gradio as gr
from gradio_image_prompter import ImagePrompter
import numpy as np
from gradio_litmodel3d import LitModel3D

normalize_point = True
# for testing purpose..
include_pipeline = True
options_enabled = False

if include_pipeline:
    # later move to top, currently for testing..
    from csam2 import get_mask
    from util import get_masked_img


def segment_img(prompt, dest):
    # image is in PIL format
    img, bbox = prompt["image"], point_to_bbox(prompt["points"], None) # bounding box is not normalized
    print("recieved bbox in gradio", bbox)
    
    mask_img, masked_img = None, None

    if include_pipeline:
        mask_img = get_mask(img, np.array(bbox))
        masked_img = get_masked_img(img, mask_img)
    else:
        mask_img = img
        masked_img = img

    dest["mask"] = mask_img
    dest["masked"] = masked_img

    # returns the mask and the masked object (if pipeline enabled..)
    return (mask_img, masked_img)

def point_to_bbox(points, norm_ref = None):
    out_point = []
    for ind, point in enumerate(points):
        x1, y1, x2, y2 = point[0], point[1], point[3], point[4]
        if normalize_point and norm_ref:
            # ref order is in width, height in PIL Image
            x1, y1, x2, y2 = x1/norm_ref[0], y1/norm_ref[1], x2/norm_ref[0], y2/norm_ref[1]
        
        out_point.append(np.array([x1, y1, x2, y2], dtype=np.float16))

    return np.array(out_point)

def get_bbox_only(prompt):
    # get the bounding box coords, with actual size if normalization required.. currently no normalization
    bbox = point_to_bbox(prompt["points"])

    return bbox

def img_input(prompt, inp_state = False):
    if isinstance(prompt, dict):
        inp_state = True

    return inp_state


def run_edy_pipeline(prompt, seg_img):

    # warning: mesh must be of format glb
    mesh_path = ""
    img = prompt["image"]
    # retireve mask image and masked image (object masked) for process..
    mask_imgs, masked_imgs = seg_img["mask"], seg_img["masked"]

    # now run the edy pipeline here and forward the actual mesh file at end..
    # mesh_file = None # placeholder for running actual pipeline..

    # save the mesh file in temporary location and send the path loction of mesh ..

    return mesh_path


# some custom styling for the web
css = """
.gradio-container {
    min-width: 1400px !important;
}
footer {
    display: none !important;
}
"""

MIN_CONT_HEIGHT = 400


with gr.Blocks(title="GO3ISR Interface (edy)", theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"), css=css) as demo:
    gr.Markdown("<h1 style='text-align: center; margin: 10px !important;'>Welcome to GO3ISR (edy)</h1>")
    # keep track of if the input exist or not..
    inp_state = gr.State(False)

    # store the segmented images and mask image dict { "mask": [], "masked": [] } here..
    seg_image = gr.State({})
    # note: original image and coords are in prompt so.. let it be..
    
    with gr.Row():
        with gr.Column():
            prompt = ImagePrompter(type='pil', height=MIN_CONT_HEIGHT)
            with gr.Row():
                ext_point_btn = gr.Button("Extract Points Only", interactive=False)
                seg_btn = gr.Button("Segment Image", interactive=False)
                ext_scene_btn = gr.Button("Extract Scene", interactive=False)

            prompt.change(
                fn=img_input, 
                inputs=[prompt], 
                outputs=[inp_state]
            ).then(
                fn=lambda inps: (gr.Button(interactive=inps), gr.Button(interactive=inps)),
                inputs=[inp_state],
                outputs=[seg_btn, ext_point_btn]
            )
            
            # sn not included in this dataframe..
            ext_point_btn.click(fn=get_bbox_only, inputs=prompt, outputs=[gr.DataFrame(label="Bounding Box/Points", headers=["x1", "y1", "x2", "y2"])])
            download_btn = gr.DownloadButton(label="Download Scene File", variant="primary", interactive=False)

            # if options required..
            if options_enabled:
                import ui_options
                ui_options.get_opt_col()


        with gr.Column():
            
            with gr.Row():
                mask_img = gr.Image(label="Mask", height=MIN_CONT_HEIGHT)
                seg_img = gr.Image(label="Segmented Image", height=MIN_CONT_HEIGHT)

            seg_btn.click(
                fn=segment_img, 
                inputs=[prompt, seg_image], 
                outputs=[mask_img, seg_img]
            ).then(
                lambda: gr.Button(interactive=True), outputs=[ext_scene_btn]
            )

            # input to out pipeline will be the original images, segmented images
            ext_scene_btn.click(
                fn=run_edy_pipeline, 
                inputs=[prompt, seg_image], 
                outputs=[LitModel3D(
                    label="Scene Output",
                    exposure=10.0,
                    height=MIN_CONT_HEIGHT,
                    interactive=False,
                    clear_color=(0.25, 0.25, 0.25, 1)
                )]
            ).then(
                lambda: gr.DownloadButton(interactive=True), outputs=[download_btn]
            )

demo.launch()