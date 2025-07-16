# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
from PIL import Image
import subprocess

import torch
import gradio as gr
import string
import random, time, os, math   
from src.utils.gpu_momory_utils import ForwardHookManager

from src.flux.generate import generate_from_test_sample, seed_everything
from src.flux.pipeline_tools import CustomFluxPipeline, load_modulation_adapter, load_dit_lora
from src.utils.data_utils import get_train_config, image_grid, pil2tensor, json_dump, pad_to_square, cv2pil, merge_bboxes
from eval.tools.face_id import FaceID
from eval.tools.florence_sam import ObjectDetector
import shutil
import yaml, gc
import numpy as np

import argparse  # 导入 argparse 模块

# 解析命令行参数
parser = argparse.ArgumentParser(description='Run Gradio demo with configurable parameters')
parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name to bind to')
parser.add_argument('--server_port', type=int, default=7680, help='Server port to listen on')
parser.add_argument('--use_low_vram', type=bool, default=False, help='Use low vram in 24G gpu memory')
parser.add_argument('--use_lower_vram', type=bool, default=False, help='Use lower vram in 16G gpu memory')
args = parser.parse_args()

use_low_vram = args.use_low_vram
use_lower_vram = args.use_lower_vram

dtype = torch.bfloat16
num_inputs = 6 #if not use_low_vram else 2

if use_low_vram or use_lower_vram:
    init_device = torch.device("cpu")
else:
    init_device = torch.device("cuda")
do_device = torch.device("cuda")

config_path = "train/config/XVerse_config_demo.yaml"

config = config_train = get_train_config(config_path)
config["model"]["dit_quant"] = "int8-quanto"
config["model"]["use_dit_lora"] = False
model = CustomFluxPipeline(
    config, init_device, torch_dtype=dtype,
)
model.pipe.set_progress_bar_config(leave=False)

face_model = FaceID(init_device)
detector = ObjectDetector(init_device)

config = get_train_config(config_path)
model.config = config

run_mode = "mod_only" # orig_only, mod_only, both
store_attn_map = False
run_name = time.strftime("%m%d-%H%M")

ckpt_root = "./checkpoints/XVerse"
model.clear_modulation_adapters()
model.pipe.unload_lora_weights()
if not os.path.exists(ckpt_root):
    print("Checkpoint root does not exist.")

modulation_adapter = load_modulation_adapter(model, config, dtype, init_device, f"{ckpt_root}/modulation_adapter", is_training=False)
model.add_modulation_adapter(modulation_adapter)
if config["model"]["use_dit_lora"]:
    load_dit_lora(model, model.pipe, config, dtype, init_device, f"{ckpt_root}", is_training=False)

if init_device.type == 'cpu' and (args.use_low_vram or args.use_lower_vram):
        if args.use_lower_vram:
            threshold_mem = 2 * 1024 * 1024 * 1024 
        elif args.use_low_vram:
            threshold_mem = 8 * 1024 * 1024 * 1024 
        forward_hook_manager = ForwardHookManager(threshold_mem, args.use_lower_vram)
        model.pipe.transformer = forward_hook_manager.register(model.pipe.transformer)
        model.pipe.text_encoder = forward_hook_manager.register(model.pipe.text_encoder)
        model.pipe.vae = forward_hook_manager.register(model.pipe.vae)
        model.pipe.text_encoder_2 = forward_hook_manager.register(model.pipe.text_encoder_2)
        model.pipe.clip_model = forward_hook_manager.register(model.pipe.clip_model)
        for i in range(len(model.pipe.modulation_adapters)):
            model.pipe.modulation_adapters[i] = forward_hook_manager.register(model.pipe.modulation_adapters[i])
        forward_hook_manager.register(face_model.detector)
        forward_hook_manager.register(face_model.model)
        forward_hook_manager.register(detector.detector.florence2_model)
        forward_hook_manager.register(detector.detector.sam2_predictor.model)
    else:
        forward_hook_manager = None
        model.pipe=model.pipe.to("cuda")
        for i in range(len(model.pipe.modulation_adapters)):
            model.pipe.modulation_adapters[i] = model.pipe.modulation_adapters[i].to("cuda")
        model.pipe=model.pipe.to("cuda")
        face_model.detector.to("cuda")
        face_model.model.to("cuda")
        detector.detector.florence2_model.to("cuda")
        detector.detector.sam2_predictor.model.to("cuda")

vae_skip_iter = None
attn_skip_iter = 0

# 定义清空图像的函数，只返回四个 None
def clear_images():
    return [None, ]*num_inputs

def det_seg_img(image, label):
    # if forward_hook_manager is not None:
    #     detector.detector.florence2_model = forward_hook_manager.model_to_cuda(detector.detector.florence2_model)
    #     detector.detector.sam2_predictor = forward_hook_manager.model_to_cuda(detector.detector.sam2_predictor)

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    instance_result_dict = detector.get_multiple_instances(image, label, min_size=image.size[0]//20)
    indices = list(range(len(instance_result_dict["instance_images"])))
    ins, bbox = merge_instances(image, indices, instance_result_dict["instance_bboxes"], instance_result_dict["instance_images"])
    return ins

def crop_face_img(image):

    # if forward_hook_manager is not None:
    #     face_model.detector = forward_hook_manager.model_to_cuda(face_model.detector)

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    # image = resize_keep_aspect_ratio(image, 1024)
    image = pad_to_square(image).resize((2048, 2048))
    
    face_bbox = face_model.detect(
        (pil2tensor(image).unsqueeze(0) * 255).to(torch.uint8).to(do_device), 1.4
    )[0]
    face = image.crop(face_bbox)
    return face

def vlm_img_caption(image):

    # if forward_hook_manager is not None:
    #     detector.detector.florence2_model = forward_hook_manager.model_to_cuda(detector.detector.florence2_model)

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    try:
        caption = detector.detector.caption(image, "<CAPTION>").strip()
        if caption.endswith("."):
            caption = caption[:-1]

    except Exception as e:
        print(e)
        caption = ""
    
    caption = caption.lower()
    return caption


def generate_random_string(length=4):
    letters = string.ascii_letters  # 包含大小写字母的字符串
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def resize_keep_aspect_ratio(pil_image, target_size=1024):
    H, W = pil_image.height, pil_image.width
    target_area = target_size * target_size
    current_area = H * W
    scaling_factor = (target_area / current_area) ** 0.5  # sqrt(target_area / current_area)
    new_H = int(round(H * scaling_factor))
    new_W = int(round(W * scaling_factor))
    return pil_image.resize((new_W, new_H))

# 使用循环生成六个图像输入
images = []
captions = []
face_btns = []
det_btns = []
vlm_btns = []
accordions = []
idip_checkboxes = []
accordion_states = []

def open_accordion_on_example_selection(*args):
    print("enter open_accordion_on_example_selection")
    images = list(args[-18:-12])
    outputs = []
    for i, img in enumerate(images):
        if img is not None:
            print(f"open accordions {i}")
            outputs.append(True)
        else:
            print(f"close accordions {i}")
            outputs.append(False)
    print(outputs)
    return outputs

def generate_image(
    prompt, 
    cond_size, target_height, target_width, 
    seed, 
    vae_skip_iter, control_weight_lambda,
    double_attention,  # 新增参数
    single_attention,  # 新增参数
    latent_dblora_scale_str,
    latent_sblora_scale_str, vae_lora_scale,
    indexs,  # 新增参数
    *images_captions_faces,  # Combine all unpacked arguments into one tuple
):
    torch.cuda.empty_cache()
    num_images = 1

    # Determine the number of images, captions, and faces based on the indexs length
    images = list(images_captions_faces[:num_inputs])
    captions = list(images_captions_faces[num_inputs:2 * num_inputs])
    idips_checkboxes = list(images_captions_faces[2 * num_inputs:3 * num_inputs])
    images = [images[i] for i in indexs]
    captions = [captions[i] for i in indexs]
    idips_checkboxes = [idips_checkboxes[i] for i in indexs]

    print(f"Length of images: {len(images)}")
    print(f"Length of captions: {len(captions)}")
    print(f"Indexs: {indexs}")
    
    print(f"Control weight lambda: {control_weight_lambda}")
    if control_weight_lambda != "no":
        parts = control_weight_lambda.split(',')
        new_parts = []
        for part in parts:
            if ':' in part:
                left, right = part.split(':')
                values = right.split('/')
                # 保存整体值
                global_value = values[0]
                id_value = values[1]
                ip_value = values[2]
                new_values = [global_value]
                for is_id in idips_checkboxes:
                    if is_id:
                        new_values.append(id_value)
                    else:
                        new_values.append(ip_value)
                new_part = f"{left}:{('/'.join(new_values))}"
                new_parts.append(new_part)
            else:
                new_parts.append(part)
        control_weight_lambda = ','.join(new_parts)
    
    print(f"Control weight lambda: {control_weight_lambda}")

    src_inputs = []
    use_words = []
    cur_run_time = time.strftime("%m%d-%H%M%S")
    tmp_dir_root = f"tmp/gradio_demo/{run_name}"
    temp_dir = f"{tmp_dir_root}/{cur_run_time}_{generate_random_string(4)}"
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Temporary directory created: {temp_dir}")
    for i, (image_path, caption) in enumerate(zip(images, captions)):
        if image_path:
            if caption.startswith("a ") or caption.startswith("A "):
                word = caption[2:]
            else:
                word = caption
            
            if f"ENT{i+1}" in prompt:
                prompt = prompt.replace(f"ENT{i+1}", caption)
            
            image = resize_keep_aspect_ratio(Image.open(image_path), 768)
            save_path = f"{temp_dir}/tmp_resized_input_{i}.png"
            image.save(save_path)
            
            input_image_path = save_path

            src_inputs.append(
                {
                    "image_path": input_image_path,
                    "caption": caption
                }
            )
            use_words.append((i, word, word))


    test_sample = dict(
        input_images=[], position_delta=[0, -32], 
        prompt=prompt,
        target_height=target_height,
        target_width=target_width,
        seed=seed,
        cond_size=cond_size,
        vae_skip_iter=vae_skip_iter,
        lora_scale=latent_dblora_scale_str,
        control_weight_lambda=control_weight_lambda,
        latent_sblora_scale=latent_sblora_scale_str,
        condition_sblora_scale=vae_lora_scale,
        double_attention=double_attention,
        single_attention=single_attention,
    )
    if len(src_inputs) > 0:
        test_sample["modulation"] = [
            dict(
                type="adapter",
                src_inputs=src_inputs,
                use_words=use_words,
            ),
        ]
    
    json_dump(test_sample, f"{temp_dir}/test_sample.json", 'utf-8')
    assert single_attention == True
    target_size = int(round((target_width * target_height) ** 0.5) // 16 * 16)
    print(test_sample)

    model.config["train"]["dataset"]["val_condition_size"] = cond_size
    model.config["train"]["dataset"]["val_target_size"] = target_size
    
    if control_weight_lambda == "no":
        control_weight_lambda = None
    if vae_skip_iter == "no":
        vae_skip_iter = None
    use_condition_sblora_control = True
    use_latent_sblora_control = True
    image = None
    try:
        image = generate_from_test_sample(
            test_sample, model.pipe, model.config, 
            num_images=num_images, 
            target_height=target_height,
            target_width=target_width,
            seed=seed,
            store_attn_map=store_attn_map, 
            vae_skip_iter=vae_skip_iter,  # 使用新的参数
            control_weight_lambda=control_weight_lambda,  # 传递新的参数
            double_attention=double_attention,  # 新增参数
            single_attention=single_attention,  # 新增参数
            ip_scale=latent_dblora_scale_str,
            use_latent_sblora_control=use_latent_sblora_control,
            latent_sblora_scale=latent_sblora_scale_str,
            use_condition_sblora_control=use_condition_sblora_control,
            condition_sblora_scale=vae_lora_scale,
            device=do_device,
            forward_hook_manager=forward_hook_manager,
        )
        if isinstance(image, list):
            num_cols = 2
            num_rows = int(math.ceil(num_images / num_cols))
            image = image_grid(image, num_rows, num_cols)

        save_path = f"{temp_dir}/tmp_result.png"
        image.save(save_path)
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        raise torch.cuda.OutOfMemoryError
    except Exception as e:
        print(f"Error: {e}")
        raise e
    
    return image

def create_image_input(index, open=True, indexs_state=None):
    accordion_state = gr.State(open)
    with gr.Column():
        with gr.Accordion(f"Input Image {index + 1}", open=accordion_state.value) as accordion:
            image = gr.Image(type="filepath", label=f"Image {index + 1}")
            caption = gr.Textbox(label=f"Caption {index + 1}", value="")
            id_ip_checkbox = gr.Checkbox(value=False, label=f"ID or not {index + 1}", visible=True)
            with gr.Row():
                vlm_btn = gr.Button("Auto Caption")
                det_btn = gr.Button("Det & Seg")
                face_btn = gr.Button("Crop Face")
            accordion.expand(
                    inputs=[indexs_state],
                    fn = lambda x: update_inputs(True, index, x), 
                    outputs=[indexs_state, accordion_state],
                )
            accordion.collapse(
                    inputs=[indexs_state],
                    fn = lambda x: update_inputs(False, index, x), 
                    outputs=[indexs_state, accordion_state],
                )
    return image, caption, face_btn, det_btn, vlm_btn, accordion_state, accordion, id_ip_checkbox


def merge_instances(orig_img, indices, ins_bboxes, ins_images):
    orig_image_width, orig_image_height = orig_img.width, orig_img.height
    final_img = Image.new("RGB", (orig_image_width, orig_image_height), color=(255, 255, 255))
    bboxes = []
    for i in indices:
        bbox = np.array(ins_bboxes[i], dtype=int).tolist()
        bboxes.append(bbox)
        
        img = cv2pil(ins_images[i])
        mask = (np.array(img)[..., :3] != 255).any(axis=-1)
        mask = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
        final_img.paste(img, (bbox[0], bbox[1]), mask)
    
    bbox = merge_bboxes(bboxes)
    img = final_img.crop(bbox)
    return img, bbox


def change_accordion(at: bool, index: int, state: list):
    print(at, state)
    indexs = state
    if at:
        if index not in indexs:
            indexs.append(index)
    else:
        if index in indexs:
            indexs.remove(index)
    
    # 确保 indexs 是有序的
    indexs.sort()
    print(indexs)
    return gr.Accordion(open=at), indexs

def update_inputs(is_open, index, state: list):
    indexs = state
    if is_open:
        if index not in indexs:
            indexs.append(index)
    else:
        if index in indexs:
            indexs.remove(index)
    
    # 确保 indexs 是有序的
    indexs.sort()
    print(indexs)
    return indexs, is_open

with gr.Blocks() as demo:

    indexs_state = gr.State([0, 1])  # 添加状态来存储 indexs
    
    gr.Markdown("### XVerse Demo")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="")
            # 使用 Row 和 Column 来布局四个图像和描述
            with gr.Row():
                target_height = gr.Slider(512, 1024, step=128, value=768, label="Generated Height", info="")
                target_width = gr.Slider(512, 1024, step=128, value=768, label="Generated Width", info="")
                cond_size = gr.Slider(256, 384, step=128, value=256, label="Condition Size", info="")
            with gr.Row():
                # 修改 weight_id_ip_str 为两个 Slider
                weight_id = gr.Slider(0.1, 5, step=0.1, value=3, label="weight_id")
                weight_ip = gr.Slider(0.1, 5, step=0.1, value=5, label="weight_ip")
            with gr.Row():
                # 修改 ip_scale_str 为 Slider，并添加 Textbox 显示转换后的格式
                ip_scale_str = gr.Slider(0.5, 1.5, step=0.01, value=0.85, label="latent_lora_scale")
                vae_lora_scale = gr.Slider(0.5, 1.5, step=0.01, value=1.3, label="vae_lora_scale")
            with gr.Row():
                # 修改 vae_skip_iter 为两个 Slider
                vae_skip_iter_s1 = gr.Slider(0, 1, step=0.01, value=0.05, label="vae_skip_iter_before")
                vae_skip_iter_s2 = gr.Slider(0, 1, step=0.01, value=0.8, label="vae_skip_iter_after")
            
    
            with gr.Row():
                weight_id_ip_str = gr.Textbox(
                    value="0-1:1/3/5",
                    label="weight_id_ip_str",
                    interactive=False, visible=False
                )
                weight_id.change(
                    lambda s1, s2: f"0-1:1/{s1}/{s2}",
                    inputs=[weight_id, weight_ip],
                    outputs=weight_id_ip_str
                )
                weight_ip.change(
                    lambda s1, s2: f"0-1:1/{s1}/{s2}",
                    inputs=[weight_id, weight_ip],
                    outputs=weight_id_ip_str
                )
                vae_skip_iter = gr.Textbox(
                    value="0-0.05:1,0.8-1:1",
                    label="vae_skip_iter",
                    interactive=False, visible=False
                )
                vae_skip_iter_s1.change(
                    lambda s1, s2: f"0-{s1}:1,{s2}-1:1",
                    inputs=[vae_skip_iter_s1, vae_skip_iter_s2],
                    outputs=vae_skip_iter
                )
                vae_skip_iter_s2.change(
                    lambda s1, s2: f"0-{s1}:1,{s2}-1:1",
                    inputs=[vae_skip_iter_s1, vae_skip_iter_s2],
                    outputs=vae_skip_iter
                )
                
            
            with gr.Row():
                db_latent_lora_scale_str = gr.Textbox(
                    value="0-1:0.85",
                    label="db_latent_lora_scale_str",
                    interactive=False, visible=False
                )
                sb_latent_lora_scale_str = gr.Textbox(
                    value="0-1:0.85",
                    label="sb_latent_lora_scale_str",
                    interactive=False, visible=False
                )
                vae_lora_scale_str = gr.Textbox(
                    value="0-1:1.3",
                    label="vae_lora_scale_str",
                    interactive=False, visible=False
                )
                vae_lora_scale.change(
                        lambda s: f"0-1:{s}",
                        inputs=vae_lora_scale,
                        outputs=vae_lora_scale_str
                    )
                ip_scale_str.change(
                        lambda s: [f"0-1:{s}", f"0-1:{s}"],
                        inputs=ip_scale_str,
                        outputs=[db_latent_lora_scale_str, sb_latent_lora_scale_str]
                    )

            with gr.Row():
                double_attention = gr.Checkbox(value=False, label="Double Attention", visible=False)
                single_attention = gr.Checkbox(value=True, label="Single Attention", visible=False)            

            clear_btn = gr.Button("清空输入图像")
            with gr.Row():
                for i in range(num_inputs):
                    image, caption, face_btn, det_btn, vlm_btn, accordion_state, accordion, id_ip_checkbox = create_image_input(i, open=i<2, indexs_state=indexs_state)
                    images.append(image)
                    idip_checkboxes.append(id_ip_checkbox)
                    captions.append(caption)
                    face_btns.append(face_btn)
                    det_btns.append(det_btn)
                    vlm_btns.append(vlm_btn)
                    accordion_states.append(accordion_state)
                    
                    accordions.append(accordion)
        with gr.Column():
            output = gr.Image(label="生成的图像")
            seed = gr.Number(value=42, label="Seed", info="")
            gen_btn = gr.Button("生成图像")

    gr.Markdown("### Examples")
    gen_btn.click(
        generate_image, 
        inputs=[
            prompt, cond_size, target_height, target_width, seed,
            vae_skip_iter, weight_id_ip_str,
            double_attention, single_attention,
            db_latent_lora_scale_str, sb_latent_lora_scale_str, vae_lora_scale_str,
            indexs_state,  # 传递 indexs 状态
            *images,  
            *captions, 
            *idip_checkboxes,
        ], 
        outputs=output
    )

    # 修改清空函数的输出参数
    clear_btn.click(clear_images, outputs=images)

    # 循环绑定 Det & Seg 和 Auto Caption 按钮的点击事件
    for i in range(num_inputs):
        face_btns[i].click(crop_face_img, inputs=[images[i]], outputs=[images[i]])
        det_btns[i].click(det_seg_img, inputs=[images[i], captions[i]], outputs=[images[i]])
        vlm_btns[i].click(vlm_img_caption, inputs=[images[i]], outputs=[captions[i]])
        accordion_states[i].change(fn=lambda x, state, index=i: change_accordion(x, index, state), inputs=[accordion_states[i], indexs_state], outputs=[accordions[i], indexs_state])
    
    examples = gr.Examples(
        examples=[
            [
                "ENT1 wearing a tiny hat", 
                42, 256, 768, 768,
                3, 5,
                0.85, 1.3,
                0.05, 0.8,
                "sample/hamster.jpg", None, None, None, None, None,
                "a hamster", None, None, None, None, None,
                False, False, False, False, False, False
            ],
            [
                "ENT1 in a red dress is smiling", 
                42, 256, 768, 768,
                3, 5,
                0.85, 1.3,
                0.05, 0.8,
                "sample/woman.jpg", None, None, None, None, None,
                "a woman", None, None, None, None, None,
                True, False, False, False, False, False
            ],
            [
                "ENT1 and ENT2 standing together in a park.", 
                42, 256, 768, 768,
                2, 5,
                0.85, 1.3,
                0.05, 0.8,
                "sample/woman.jpg", "sample/girl.jpg", None, None, None, None,
                "a woman", "a girl", None, None, None, None,
                True, True, False, False, False, False
            ],
            [
                "ENT1, ENT2, and ENT3 standing together in a park.", 
                42, 256, 768, 768,
                2.5, 5,
                0.8, 1.2,
                0.05, 0.8,
                "sample/woman.jpg", "sample/girl.jpg", "sample/old_man.jpg", None, None, None,
                "a woman", "a girl", "an old man", None, None, None,
                True, True, True, False, False, False
            ],
        ],
        inputs=[
            prompt, seed, 
            cond_size,
            target_height,
            target_width,
            weight_id,
            weight_ip,
            ip_scale_str,
            vae_lora_scale,
            vae_skip_iter_s1,
            vae_skip_iter_s2,
            *images,
            *captions, 
            *idip_checkboxes
        ],
        outputs=accordion_states,
        fn=open_accordion_on_example_selection,
        run_on_click=True
    )

# size = 16 * 1024 * 1024 * 1024 // 4     
# big_tensor = torch.randn(size, dtype=torch.float32, device='cuda')

demo.queue().launch(share=True, inbrowser=True, server_name=args.server_name, server_port=args.server_port)