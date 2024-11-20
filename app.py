# Open Source Model Licensed under the Apache License Version 2.0 
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited 
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved. 
# The below software and/or models in this distribution may have been 
# modified by THL A29 Limited ("Tencent Modifications"). 
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT 
# except for the third-party components listed below. 
# Hunyuan 3D does not impose any additional limitations beyond what is outlined 
# in the repsective licenses of these third-party components. 
# Users must comply with all terms and conditions of original licenses of these third-party 
# components and must ensure that the usage of the third party components adheres to 
# all relevant laws and regulations. 

# For avoidance of doubts, Hunyuan 3D means the large language models and 
# their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import warnings
import argparse
import gradio as gr
from glob import glob
import shutil
import torch
import numpy as np
from PIL import Image
from einops import rearrange
import pandas as pd

from infer import seed_everything, save_gif
from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer
from third_party.check import check_bake_available

try:
    from third_party.mesh_baker import MeshBaker
    assert check_bake_available()
    BAKE_AVAILEBLE = True
except Exception as err:
    print(err)
    print("import baking related fail, run without baking")
    BAKE_AVAILEBLE = False


warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--use_lite", default=False, action="store_true")
parser.add_argument("--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str)
parser.add_argument("--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str)
parser.add_argument("--text2image_path", default="weights/hunyuanDiT", type=str)
parser.add_argument("--save_memory", default=False, action="store_true")
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

################################################################
# initial setting
################################################################

CONST_PORT = 8080
CONST_MAX_QUEUE = 1
CONST_SERVER = '0.0.0.0'

CONST_HEADER = '''
<h2><a href='https://github.com/tencent/Hunyuan3D-1' target='_blank'><b>Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation</b></a></h2>
⭐️Technical report: <a href='https://arxiv.org/pdf/2411.02293' target='_blank'>ArXiv</a>. ⭐️Code: <a href='https://github.com/tencent/Hunyuan3D-1' target='_blank'>GitHub</a>.
'''

CONST_NOTE = '''
❗️❗️❗️Usage❗️❗️❗️<br>

Limited by format, the model can only export *.obj mesh with vertex colors. The "texture" mod can only work on *.glb.<br>
Please click "Do Rendering" to export a GIF.<br>
You can click "Do Baking" to bake multi-view imgaes onto the shape.<br>

If the results aren't satisfactory, please try a different radnom seed (default is 0).
'''

################################################################
# prepare text examples and image examples
################################################################

def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./demos/example_*.png'))

def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list  = list()
    for line in open('./demos/example_list.txt'):
        txt_list.append(line.strip())
    return txt_list

example_is = get_example_img_list()
example_ts = get_example_txt_list()

################################################################
# initial models
################################################################

worker_xbg = Removebg()
print(f"loading {args.text2image_path}")
worker_t2i = Text2Image(
    pretrain = args.text2image_path, 
    device = args.device, 
    save_memory = args.save_memory
)
worker_i2v = Image2Views(
    use_lite = args.use_lite, 
    device = args.device,
    save_memory = args.save_memory
)
worker_v23 = Views2Mesh(
    args.mv23d_cfg_path, 
    args.mv23d_ckt_path, 
    use_lite = args.use_lite, 
    device = args.device,
    save_memory = args.save_memory
)
worker_gif = GifRenderer(args.device)

if BAKE_AVAILEBLE:
    worker_baker = MeshBaker()


### functional modules    

def gen_save_folder():
    os.makedirs('./outputs/app_output', exist_ok=True)
    exists = set(int(_) for _ in os.listdir('./outputs/app_output') if not _.startswith("."))
    if len(exists) == 30: shutil.rmtree(f"./outputs/app_output/0");cur_id = 0
    else:                 cur_id = min(set(range(30)) - exists)
    if os.path.exists(f"./outputs/app_output/{(cur_id + 1) % 30}"):
        shutil.rmtree(f"./outputs/app_output/{(cur_id + 1) % 30}")
    save_folder = f'./outputs/app_output/{cur_id}'
    os.makedirs(save_folder, exist_ok=True)
    return save_folder

def stage_0_t2i(text, seed, step, save_folder):
    dst = save_folder + '/img.png'
    image = worker_t2i(text, seed, step)
    image.save(dst)
    img_nobg = worker_xbg(image, force=True)
    dst = save_folder + '/img_nobg.png'
    img_nobg.save(dst)
    return dst

def stage_1_xbg(image, save_folder, force_remove): 
    if isinstance(image, str):
        image = Image.open(image)
    dst =  save_folder + '/img_nobg.png'
    rgba = worker_xbg(image, force=force_remove)
    rgba.save(dst)
    return dst
    
def stage_2_i2v(image, seed, step, save_folder):
    if isinstance(image, str):
        image = Image.open(image)
    gif_dst = save_folder + '/views.gif'
    res_img, pils = worker_i2v(image, seed, step)
    save_gif(pils, gif_dst)
    views_img, cond_img = res_img[0], res_img[1]
    img_array = np.asarray(views_img, dtype=np.uint8)
    show_img = rearrange(img_array, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_img = show_img[worker_i2v.order, ...]
    show_img = rearrange(show_img, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_img = Image.fromarray(show_img) 
    return views_img, cond_img, show_img

def stage_3_v23(
    views_pil, 
    cond_pil, 
    seed, 
    save_folder,
    target_face_count = 30000,
    texture_color = 'texture'
): 
    do_texture_mapping = texture_color == 'texture'
    worker_v23(
        views_pil, 
        cond_pil, 
        seed = seed, 
        save_folder = save_folder,
        target_face_count = target_face_count,
        do_texture_mapping = do_texture_mapping
    )
    glb_dst = save_folder + '/mesh.glb' if do_texture_mapping else None
    obj_dst =  save_folder + '/mesh.obj'
    obj_dst = save_folder + '/mesh_vertex_colors.obj' # gradio just only can show vertex shading
    return obj_dst, glb_dst

def stage_3p_baking(save_folder, color, bake, force, front, others, align_times):
    if color == "texture" and bake:
        obj_dst = worker_baker(save_folder, force, front, others, align_times)
        glb_dst = obj_dst.replace(".obj", ".glb")
        return glb_dst
    else:
        return None

def stage_4_gif(save_folder, color, bake, render):
    if not render: return None

    baked_fld_list = sorted(glob(save_folder + '/view_*/bake/mesh.obj'))
    obj_dst = baked_fld_list[-1] if len(baked_fld_list)>=1 else save_folder+'/mesh.obj'
    assert os.path.exists(obj_dst), f"{obj_dst} file not found"

    gif_dst = obj_dst.replace(".obj", ".gif")
    worker_gif(obj_dst, gif_dst_path=gif_dst)
    return gif_dst


def check_image_available(image):
    if image is None:
        return "Please upload image", gr.update()
    elif image.mode == "RGBA":
        data = np.array(image)
        alpha_channel = data[:, :, 3]
        unique_alpha_values = np.unique(alpha_channel)
        if len(unique_alpha_values) == 1:
            msg = "The alpha channel is missing or invalid. The background removal option is selected for you."
            return msg, gr.update(value=True, interactive=False)
        else:
            msg = "The image has four channels, and you can choose to remove the background or not."
            return msg, gr.update(value=False, interactive=True)
    elif image.mode == "RGB":
        msg = "The alpha channel is missing or invalid. The background removal option is selected for you."
        return msg, gr.update(value=True, interactive=False)
    else:
        raise Exception("Image Error")
    

def update_mode(mode):
    color_change = {
        'Quick': gr.update(value='vertex'),
        'Moderate': gr.update(value='texture'),
        'Appearance': gr.update(value='texture')
    }[mode]
    bake_change = {
        'Quick': gr.update(value=False),
        'Moderate': gr.update(value=False),
        'Appearance': gr.update(value=BAKE_AVAILEBLE)
    }[mode]
    face_change = {
        'Quick': gr.update(value=120000, maximum=300000),
        'Moderate': gr.update(value=60000, maximum=300000),
        'Appearance': gr.update(value=10000, maximum=60000)
    }[mode]
    render_change = {
        'Quick': gr.update(value=False),
        'Moderate': gr.update(value=True),
        'Appearance': gr.update(value=True)
    }[mode]
    return color_change, bake_change, face_change, render_change
    
    
# ===============================================================
# gradio display
# ===============================================================

with gr.Blocks() as demo:
    gr.Markdown(CONST_HEADER)
    with gr.Row(variant="panel"):
        
        ###### Input region
        
        with gr.Column(scale=2):
            
            ### Text iutput region
            
            with gr.Tab("Text to 3D"):
                with gr.Column():
                    text = gr.TextArea('一只黑白相间的熊猫在白色背景上居中坐着，呈现出卡通风格和可爱氛围。', 
                                       lines=3, max_lines=20, label='Input text')

                    textgen_mode = gr.Radio(
                        choices=['Quick', 'Moderate', 'Appearance'], 
                        label="Simple settings",
                        value='Appearance',
                        interactive=True
                    )
                    
                    with gr.Accordion("Custom settings", open=False):
                        textgen_color = gr.Radio(choices=["vertex", "texture"], label="Color", value="texture")
                        
                        with gr.Row():
                            textgen_render = gr.Checkbox(
                                label="Do Rendering", 
                                value=True, 
                                interactive=True
                            )
                            textgen_bake = gr.Checkbox(
                                label="Do Baking", 
                                value=True if BAKE_AVAILEBLE else False, 
                                interactive=True if BAKE_AVAILEBLE else False
                            )
                            
                        with gr.Row():
                            textgen_seed = gr.Number(value=0, label="T2I seed", precision=0, interactive=True)
                            textgen_SEED = gr.Number(value=0, label="Gen seed", precision=0, interactive=True)

                        textgen_step = gr.Slider(
                            value=25,
                            minimum=15,
                            maximum=50,
                            step=1,
                            label="T2I steps",
                            interactive=True
                        )
                        textgen_STEP = gr.Slider(
                            value=50,
                            minimum=20,
                            maximum=80,
                            step=1,
                            label="Gen steps",
                            interactive=True
                        )
                        textgen_max_faces =gr.Slider(
                            value=10000,
                            minimum=2000,
                            maximum=60000,
                            step=1000,
                            label="Face number limit",
                            interactive=True
                        )

                        with gr.Accordion("Baking Options", open=False):
                            textgen_force_bake = gr.Checkbox(
                                label="Force (Ignore the degree of matching)", 
                                value=False, 
                                interactive=True
                            )
                            textgen_front_baking = gr.Radio(
                                choices=['input image', 'multi-view front view', 'auto'], 
                                label="Front view baking",
                                value='auto',
                                interactive=True,
                                visible=True
                            )
                            textgen_other_views = gr.CheckboxGroup(
                                choices=['60°', '120°', '180°', '240°', '300°'], 
                                label="Other views Baking",
                                value=['180°'],
                                interactive=True,
                                visible=True
                            )
                            textgen_align_times =gr.Slider(
                                value=3,
                                minimum=1,
                                maximum=5,
                                step=1,
                                label="Number of alignment attempts per view",
                                interactive=True
                            )
                        
                    with gr.Row():
                        textgen_submit = gr.Button("Generate", variant="primary")

                    with gr.Row():
                        gr.Examples(examples=example_ts, inputs=[text], label="Text examples", examples_per_page=10)
                        
                        
            textgen_mode.change(
                fn=update_mode,
                inputs=textgen_mode, 
                outputs=[textgen_color, textgen_bake, textgen_max_faces, textgen_render]
            )
            textgen_color.change(
                fn=lambda x:[gr.update(value=x=='texture', interactive=x=='texture')]*2, 
                inputs=textgen_color, 
                outputs=[textgen_bake, textgen_render]
            )
            textgen_bake.change(
                fn= lambda x:[gr.update(visible=x)]*4+[gr.update(value=10000, minimum=2000, maximum=60000 if x else 300000)],
                inputs=textgen_bake, 
                outputs=[textgen_front_baking, textgen_other_views, textgen_align_times, textgen_force_bake, textgen_max_faces]
            )
            
                    
            ### Image iutput region
            
            with gr.Tab("Image to 3D"):
                with gr.Row():
                    input_image = gr.Image(label="Input image", width=256, height=256, type="pil",
                                           image_mode="RGBA", sources="upload", interactive=True)
                with gr.Row():
                    alert_message = gr.Markdown("")  # for warning 
                    
                imggen_mode = gr.Radio(
                    choices=['Quick', 'Moderate', 'Appearance'], 
                    label="Simple settings",
                    value='Appearance',
                    interactive=True
                )
                
                with gr.Accordion("Custom settings", open=False):
                    imggen_color = gr.Radio(choices=["vertex", "texture"], label="Color", value="texture")

                    with gr.Row():
                        imggen_removebg = gr.Checkbox(
                            label="Remove Background", 
                            value=True, 
                            interactive=True
                        )
                        imggen_render = gr.Checkbox(
                            label="Do Rendering", 
                            value=True, 
                            interactive=True
                        )
                        imggen_bake = gr.Checkbox(
                            label="Do Baking", 
                            value=True if BAKE_AVAILEBLE else False, 
                            interactive=True if BAKE_AVAILEBLE else False
                        )
                    imggen_SEED = gr.Number(value=0, label="Gen seed", precision=0, interactive=True)

                    imggen_STEP = gr.Slider(
                        value=50,
                        minimum=20,
                        maximum=80,
                        step=1,
                        label="Gen steps",
                        interactive=True
                    )
                    imggen_max_faces =gr.Slider(
                        value=10000,
                        minimum=2000,
                        maximum=60000,
                        step=1000,
                        label="Face number limit",
                        interactive=True
                    )

                    with gr.Accordion("Baking Options", open=False):
                        imggen_force_bake = gr.Checkbox(
                                label="Force (Ignore the degree of matching)", 
                                value=False, 
                                interactive=True
                            )
                        imggen_front_baking = gr.Radio(
                            choices=['input image', 'multi-view front view', 'auto'], 
                            label="Front view baking",
                            value='auto',
                            interactive=True,
                            visible=True
                        )
                        imggen_other_views = gr.CheckboxGroup(
                            choices=['60°', '120°', '180°', '240°', '300°'], 
                            label="Other views Baking",
                            value=['180°'],
                            interactive=True,
                            visible=True
                        )
                        imggen_align_times =gr.Slider(
                            value=3,
                            minimum=1,
                            maximum=5,
                            step=1,
                            label="Number of alignment attempts per view",
                            interactive=True
                        )

                input_image.change(
                    fn=check_image_available, 
                    inputs=input_image, 
                    outputs=[alert_message, imggen_removebg]
                )
                
                imggen_mode.change(
                    fn=update_mode,
                    inputs=imggen_mode, 
                    outputs=[imggen_color, imggen_bake, imggen_max_faces, imggen_render]
                )
                
                imggen_color.change(
                    fn=lambda x:[gr.update(value=x=='texture', interactive=x=='texture')]*2, 
                    inputs=imggen_color, 
                    outputs=[imggen_bake, imggen_render]
                )
                
                imggen_bake.change(
                    fn= lambda x:[gr.update(visible=x)]*4+[gr.update(value=10000, minimum=2000, maximum=60000 if x else 300000)],
                    inputs=imggen_bake, 
                    outputs=[imggen_front_baking, imggen_other_views, imggen_align_times, imggen_force_bake, imggen_max_faces]
                )

                with gr.Row():
                    imggen_submit = gr.Button("Generate", variant="primary")      

                with gr.Row():
                    gr.Examples(examples=example_is,  inputs=[input_image], 
                        label="Img examples", examples_per_page=10)
            
                    
            gr.Markdown(CONST_NOTE)
                    
        ###### Output region

        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(scale=2):
                    rem_bg_image = gr.Image(
                        label="Image without background", 
                        type="pil",
                        image_mode="RGBA", 
                        interactive=False
                    )
                with gr.Column(scale=3):
                    result_image = gr.Image(
                        label="Multi-view images", 
                        type="pil", 
                        interactive=False
                    )
            
            result_3dobj = gr.Model3D(
                clear_color=[0.0, 0.0, 0.0, 0.0],
                label="OBJ vertex color",
                show_label=True,
                visible=True,
                camera_position=[90, 90, None],
                interactive=False
            )
                
            result_3dglb_texture = gr.Model3D(
                clear_color=[0.0, 0.0, 0.0, 0.0],
                label="GLB texture color",
                show_label=True,
                visible=True,
                camera_position=[90, 90, None],
                interactive=False)

            result_3dglb_baked = gr.Model3D(
                clear_color=[0.0, 0.0, 0.0, 0.0],
                label="GLB baked color",
                show_label=True,
                visible=True,
                camera_position=[90, 90, None],
                interactive=False)
            
            result_gif = gr.Image(label="GIF", interactive=False)
                
            with gr.Row():
                gr.Markdown(
                    "Due to Gradio limitations, OBJ files are displayed with vertex shading only, "
                    "while GLB files can be viewed with texture shading. <br>For the best experience, "
                    "we recommend downloading the GLB files and opening them with 3D software "
                    "like Blender or MeshLab."
                )

    #===============================================================
    # gradio running code
    #===============================================================
    
    none = gr.State(None)
    save_folder = gr.State()
    cond_image = gr.State()
    views_image = gr.State()
    text_image = gr.State()
    img_dst = gr.State()
    
    
    textgen_submit.click(
        fn=gen_save_folder,
        inputs=[],
        outputs=[save_folder]
    ).success(
        fn=stage_0_t2i, 
        inputs=[text, textgen_seed, textgen_step, save_folder], 
        outputs=[rem_bg_image],
    ).success(
        fn=stage_2_i2v, 
        inputs=[rem_bg_image, textgen_SEED, textgen_STEP, save_folder], 
        outputs=[views_image, cond_image, result_image],
    ).success(
        fn=stage_3_v23, 
        inputs=[views_image, cond_image, textgen_SEED, save_folder, textgen_max_faces, textgen_color], 
        outputs=[result_3dobj, result_3dglb_texture],
    ).success(
        fn=stage_3p_baking, 
        inputs=[save_folder, textgen_color, textgen_bake,
               textgen_force_bake, textgen_front_baking, textgen_other_views, textgen_align_times], 
        outputs=[result_3dglb_baked],
    ).success(
        fn=stage_4_gif, 
        inputs=[save_folder, textgen_color, textgen_bake, textgen_render], 
        outputs=[result_gif],
    ).success(lambda: print('Text_to_3D Done ...'))

    
    imggen_submit.click(
        fn=gen_save_folder,
        inputs=[],
        outputs=[save_folder]
    ).success(
        fn=stage_1_xbg, 
        inputs=[input_image, save_folder, imggen_removebg], 
        outputs=[rem_bg_image],
    ).success(
        fn=stage_2_i2v, 
        inputs=[rem_bg_image, imggen_SEED, imggen_STEP, save_folder], 
        outputs=[views_image, cond_image, result_image],
    ).success(
        fn=stage_3_v23, 
        inputs=[views_image, cond_image, imggen_SEED, save_folder, imggen_max_faces, imggen_color],
        outputs=[result_3dobj, result_3dglb_texture],
    ).success(
        fn=stage_3p_baking, 
        inputs=[save_folder, imggen_color, imggen_bake, 
                imggen_force_bake, imggen_front_baking, imggen_other_views, imggen_align_times], 
        outputs=[result_3dglb_baked],
    ).success(
        fn=stage_4_gif, 
        inputs=[save_folder, imggen_color, imggen_bake, imggen_render], 
        outputs=[result_gif],
    ).success(lambda: print('Image_to_3D Done ...'))
    
    #===============================================================
    # start gradio server
    #===============================================================

    demo.queue(max_size=CONST_MAX_QUEUE)
    demo.launch(server_name=CONST_SERVER, server_port=CONST_PORT)

