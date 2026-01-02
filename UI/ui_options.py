import gradio as gr

# currently the options field are not attached due to unsure params. Just a placeholder/reference for now..

def get_opt_col():
    # parameters defined here...
    with gr.Column(scale=1) as optcol:
        # Use an Accordion (collapsed by default) and assign an id for styling
        with gr.Accordion("> Generation Controls", open=False, elem_id="generation_controls"):
            gr.Markdown("<span style='display: block; margin:10px; color: red; font-weight: bold;'>Read only value for now</span>")
            with gr.Group():
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=(1 << 16),
                        step=1,
                        value=0,
                        interactive=False
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                
                with gr.Tabs():
                    with gr.TabItem("Basic Settings"):
                        asset_order = gr.Radio(
                            label="Asset Order",
                            choices=["largest", "smallest", "order"],
                            value="largest",
                        )
                        positions_type = gr.Radio(
                            label="Positions Type",
                            choices=['last', 'avg'],
                            value='last',
                        )
                        simplify = gr.Slider(
                            label="Simplify",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.95,
                        )
                        texture_size = gr.Dropdown(
                            label="Texture Size",
                            choices=[512, 1024, 2048, 4096],
                            value=1024,
                            type="value",
                            allow_custom_value=False
                        )

                    with gr.TabItem("Structure Parameters"):
                        ss_num_inference_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=25,
                        )
                        ss_cfg_strength = gr.Slider(
                            label="CFG Scale",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=5.0,
                        )
                        with gr.Row():
                            ss_cfg_interval_start = gr.Slider(
                                label="CFG Interval Start",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.5,
                            )
                            ss_cfg_interval_end = gr.Slider(
                                label="CFG Interval End",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=1.0,
                            )
                        ss_rescale_t = gr.Slider(
                            label="Rescale Factor",
                            minimum=0.0,
                            maximum=5.0,
                            step=0.1,
                            value=3.0,
                        )
                        
                    with gr.TabItem("SLAT Parameters"):
                        slat_num_inference_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=25,
                        )
                        slat_cfg_strength = gr.Slider(
                            label="CFG Scale",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=5.0,
                        )
                        with gr.Row():
                            slat_cfg_interval_start = gr.Slider(
                                label="CFG Interval Start",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.5,
                            )
                            slat_cfg_interval_end = gr.Slider(
                                label="CFG Interval End",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=1.0,
                            )
                        slat_rescale_t = gr.Slider(
                            label="Rescale Factor",
                            minimum=0.0,
                            maximum=5.0,
                            step=0.1,
                            value=3.0,
                        )
            
    return optcol
