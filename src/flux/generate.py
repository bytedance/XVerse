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

import torch
import yaml, os
from PIL import Image
from diffusers.pipelines import FluxPipeline
from typing import List, Union, Optional, Dict, Any, Callable
from src.flux.transformer import transformer_forward
from src.flux.condition import Condition

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
    np,
)
from src.flux.pipeline_tools import (
    encode_prompt_with_clip_t5, tokenize_t5_prompt, clear_attn_maps, encode_vae_images
)

from src.flux.pipeline_tools import CustomFluxPipeline, load_modulation_adapter, decode_vae_images, \
    save_attention_maps, gather_attn_maps, clear_attn_maps, load_dit_lora, quantization

from src.utils.data_utils import pad_to_square, pad_to_target, pil2tensor, get_closest_ratio, get_aspect_ratios
from src.utils.modulation_utils import get_word_index, unpad_input_ids

def get_config(config_path: str = None):
    config_path = config_path or os.environ.get("XFL_CONFIG")
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_params(
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    verbose: bool = False,
    **kwargs: dict,
):
    return (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
        verbose,
    )


def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def generate(
    pipeline: FluxPipeline,
    vae_conditions: List[Condition] = None,
    config_path: str = None,
    model_config: Optional[Dict[str, Any]] = {},
    vae_condition_scale: float = 1.0,
    default_lora: bool = False,
    condition_pad_to: str = "square",
    condition_size: int = 512,
    text_cond_mask: Optional[torch.FloatTensor] = None,
    delta_emb: Optional[torch.FloatTensor] = None,
    delta_emb_pblock: Optional[torch.FloatTensor] = None,
    delta_emb_mask: Optional[torch.FloatTensor] = None,
    delta_start_ends = None,
    condition_latents = None,
    condition_ids = None,
    mod_adapter = None,
    store_attn_map: bool = False,
    vae_skip_iter: str = None,
    control_weight_lambda: str = None,
    double_attention: bool = False,
    single_attention: bool = False,
    ip_scale: str = None,
    use_latent_sblora_control: bool = False,
    latent_sblora_scale: str = None,
    use_condition_sblora_control: bool = False,
    condition_sblora_scale: str = None,
    idips = None,
    device = torch.device('cuda'),
    forward_hook_manager=None,
    **params: dict,
):
    model_config = model_config or get_config(config_path).get("model", {})

    vae_skip_iter = model_config.get("vae_skip_iter", vae_skip_iter)
    double_attention = model_config.get("double_attention", double_attention)
    single_attention = model_config.get("single_attention", single_attention)
    control_weight_lambda = model_config.get("control_weight_lambda", control_weight_lambda)
    ip_scale = model_config.get("ip_scale", ip_scale)
    use_latent_sblora_control = model_config.get("use_latent_sblora_control", use_latent_sblora_control)
    use_condition_sblora_control = model_config.get("use_condition_sblora_control", use_condition_sblora_control)

    latent_sblora_scale = model_config.get("latent_sblora_scale", latent_sblora_scale)
    condition_sblora_scale = model_config.get("condition_sblora_scale", condition_sblora_scale)

    model_config["use_attention_double"] = False
    model_config["use_attention_single"] = False
    use_attention = False
    
    if idips is not None:
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
                    for is_id in idips:
                        if is_id:
                            new_values.append(id_value)
                        else:
                            new_values.append(ip_value)
                    new_part = f"{left}:{('/'.join(new_values))}"
                    new_parts.append(new_part)
                else:
                    new_parts.append(part)
            control_weight_lambda = ','.join(new_parts)

    if vae_condition_scale != 1:
        for name, module in pipeline.transformer.named_modules():
            if not name.endswith(".attn"):
                continue
            module.c_factor = torch.ones(1, 1) * vae_condition_scale

    # self = pipeline
    (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
        verbose,
    ) = prepare_params(**params)

    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipeline.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    pipeline._guidance_scale = guidance_scale
    pipeline._joint_attention_kwargs = joint_attention_kwargs
    pipeline._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # device = self._execution_device

    lora_scale = (
        pipeline.joint_attention_kwargs.get("scale", None)
        if pipeline.joint_attention_kwargs is not None
        else None
    )
    (
        t5_prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = encode_prompt_with_clip_t5(
        self=pipeline,
        prompt="" if pipeline.text_encoder_2 is None else prompt,
        prompt_2=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # 4. Prepare latent variables
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, latent_image_ids = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        pooled_prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    latent_height = height // 16

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.base_image_seq_len,
        pipeline.scheduler.config.max_image_seq_len,
        pipeline.scheduler.config.base_shift,
        pipeline.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0
    )
    pipeline._num_timesteps = len(timesteps)

    attn_map = None

    # 6. Denoising loop
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        totalsteps = timesteps[0]
        if control_weight_lambda is not None:
            print("control_weight_lambda", control_weight_lambda)
            control_weight_lambda_schedule = []
            for scale_str in control_weight_lambda.split(','):
                time_region, scale = scale_str.split(':')
                start, end = time_region.split('-')
                scales = [float(s) for s in scale.split('/')]
                control_weight_lambda_schedule.append([(1-float(start))*totalsteps, (1-float(end))*totalsteps, scales])

        if ip_scale is not None:
            print("ip_scale", ip_scale)
            ip_scale_schedule = []
            for scale_str in ip_scale.split(','):
                time_region, scale = scale_str.split(':')
                start, end = time_region.split('-')
                ip_scale_schedule.append([(1-float(start))*totalsteps, (1-float(end))*totalsteps, float(scale)])

        if use_latent_sblora_control:
            if latent_sblora_scale is not None:
                print("latent_sblora_scale", latent_sblora_scale)
                latent_sblora_scale_schedule = []
                for scale_str in latent_sblora_scale.split(','):
                    time_region, scale = scale_str.split(':')
                    start, end = time_region.split('-')
                    latent_sblora_scale_schedule.append([(1-float(start))*totalsteps, (1-float(end))*totalsteps, float(scale)])
        
        if use_condition_sblora_control:
            if condition_sblora_scale is not None:
                print("condition_sblora_scale", condition_sblora_scale)
                condition_sblora_scale_schedule = []
                for scale_str in condition_sblora_scale.split(','):
                    time_region, scale = scale_str.split(':')
                    start, end = time_region.split('-')
                    condition_sblora_scale_schedule.append([(1-float(start))*totalsteps, (1-float(end))*totalsteps, float(scale)])


        if vae_skip_iter is not None:
            print("vae_skip_iter", vae_skip_iter)
            vae_skip_iter_schedule = []
            for scale_str in vae_skip_iter.split(','):
                time_region, scale = scale_str.split(':')
                start, end = time_region.split('-')
                vae_skip_iter_schedule.append([(1-float(start))*totalsteps, (1-float(end))*totalsteps, float(scale)])

        if control_weight_lambda is not None and attn_map is None:
            batch_size = latents.shape[0]
            latent_width = latents.shape[1]//latent_height
            attn_map = torch.ones(batch_size, latent_height, latent_width, 128, device=latents.device, dtype=torch.bfloat16)
            print("contol_weight_only", attn_map.shape)

        pipeline.scheduler.set_begin_index(0)
        pipeline.scheduler._init_step_index(0)
        for i, t in enumerate(timesteps):
            
            if control_weight_lambda is not None:
                cur_control_weight_lambda = []
                for start, end, scale in control_weight_lambda_schedule:
                    if t <= start and t >= end:
                        cur_control_weight_lambda = scale
                        break
                print(f"timestep:{t}, cur_control_weight_lambda:{cur_control_weight_lambda}")
               
                if cur_control_weight_lambda:
                    model_config["use_attention_single"] = True
                    use_attention = True
                    model_config["use_atten_lambda"] = cur_control_weight_lambda  
                else:
                    model_config["use_attention_single"] = False
                    use_attention = False
                     
            if pipeline.interrupt:
                continue

            if isinstance(delta_emb, list):
                cur_delta_emb = delta_emb[i]
                cur_delta_emb_pblock = delta_emb_pblock[i]
                cur_delta_emb_mask = delta_emb_mask[i]
            else:
                cur_delta_emb = delta_emb
                cur_delta_emb_pblock = delta_emb_pblock
                cur_delta_emb_mask = delta_emb_mask


            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000
            prompt_embeds = t5_prompt_embeds
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=prompt_embeds.dtype)

            # handle guidance
            if pipeline.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None
            pipeline.transformer.enable_lora()
            
            lora_weight  = 1
            if ip_scale is not None:
                lora_weight = 0
                for start, end, scale in ip_scale_schedule:
                    if t <= start and t >= end:
                        lora_weight = scale
                        break
                if lora_weight != 1: print(f"timestep:{t}, lora_weights:{lora_weight}")
            
            latent_sblora_weight = None
            if use_latent_sblora_control:
                if latent_sblora_scale is not None:
                    latent_sblora_weight = 0
                    for start, end, scale in latent_sblora_scale_schedule:
                        if t <= start and t >= end:
                            latent_sblora_weight = scale
                            break
                    if latent_sblora_weight != 1: print(f"timestep:{t}, latent_sblora_weight:{latent_sblora_weight}")
            
            condition_sblora_weight = None
            if use_condition_sblora_control:
                if condition_sblora_scale is not None:
                    condition_sblora_weight = 0
                    for start, end, scale in condition_sblora_scale_schedule:
                        if t <= start and t >= end:
                            condition_sblora_weight = scale
                            break
                    if condition_sblora_weight !=1: print(f"timestep:{t}, condition_sblora_weight:{condition_sblora_weight}")

            vae_skip_iter_t = False
            if vae_skip_iter is not None:
                for start, end, scale in vae_skip_iter_schedule:
                    if t <= start and t >= end:
                        vae_skip_iter_t = bool(scale)
                        break
                if vae_skip_iter_t:
                    print(f"timestep:{t}, skip vae:{vae_skip_iter_t}")               

            if forward_hook_manager is not None:
                pipeline.transformer = forward_hook_manager.model_to_cuda(pipeline.transformer)
            noise_pred = transformer_forward(
                pipeline.transformer,
                model_config=model_config,
                # Inputs of the condition (new feature)
                text_cond_mask=text_cond_mask,
                delta_emb=cur_delta_emb,
                delta_emb_pblock=cur_delta_emb_pblock,
                delta_emb_mask=cur_delta_emb_mask,
                delta_start_ends=delta_start_ends,
                condition_latents=None if vae_skip_iter_t else condition_latents,
                condition_ids=None if vae_skip_iter_t else condition_ids,
                condition_type_ids=None,
                # Inputs to the original transformer
                hidden_states=latents,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={'scale': lora_weight, "latent_sblora_weight": latent_sblora_weight, "condition_sblora_weight": condition_sblora_weight}, 
                store_attn_map=use_attention,
                last_attn_map=attn_map if cur_control_weight_lambda else None,
                use_text_mod=model_config["modulation"]["use_text_mod"],
                use_img_mod=model_config["modulation"]["use_img_mod"],
                mod_adapter=mod_adapter,
                latent_height=latent_height,
                return_dict=False,
            )[0]

            if use_attention:
                attn_maps, _ = gather_attn_maps(pipeline.transformer, clear=True)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(pipeline, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0
            ):
                progress_bar.update()

    if output_type == "latent":
        image = latents

    else:
        latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
        latents = (
            latents / pipeline.vae.config.scaling_factor
        ) + pipeline.vae.config.shift_factor
        image = pipeline.vae.decode(latents, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    pipeline.maybe_free_model_hooks()

    pipeline.transformer.enable_lora()

    if vae_condition_scale != 1:
        for name, module in pipeline.transformer.named_modules():
            if not name.endswith(".attn"):
                continue
            del module.c_factor

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)


@torch.no_grad()
def generate_from_test_sample(
    test_sample, pipe, config, 
    num_images=1, 
    vae_skip_iter: str = None, 
    target_height: int = None,
    target_width: int = None,
    seed: int = 42,
    control_weight_lambda: str = None,
    double_attention: bool = False,
    single_attention: bool = False,
    ip_scale: str = None,
    use_latent_sblora_control: bool = False,
    latent_sblora_scale: str = None,
    use_condition_sblora_control: bool = False,
    condition_sblora_scale: str = None,
    use_idip = False,
    device = torch.device('cuda'),
    forward_hook_manager=None,
    num_inference_steps: int = 28,
    **kargs
):
    target_size = config["train"]["dataset"]["val_target_size"]
    condition_size = config["train"]["dataset"].get("val_condition_size", target_size//2)
    condition_pad_to = config["train"]["dataset"]["condition_pad_to"]
    pos_offset_type = config["model"].get("pos_offset_type", "width")
    seed = config["model"].get("seed", seed)

    # device = pipe._execution_device

    condition_imgs = test_sample['input_images']
    position_delta = test_sample['position_delta']
    prompt = test_sample['prompt']
    original_image = test_sample.get('original_image', None)
    condition_type = test_sample.get('condition_type', "subject")
    modulation_input = test_sample.get('modulation', None)

    delta_start_ends = None
    condition_latents = condition_ids = None
    text_cond_mask = None
    
    delta_embs = None
    delta_embs_pblock = None
    delta_embs_mask = None

    try:
        max_length = config["model"]["modulation"]["max_text_len"]
    except Exception as e:
        print(e)
        max_length = 512

    if modulation_input is None or len(modulation_input) == 0:
        delta_emb = delta_emb_pblock = delta_emb_mask = None
    else:
        dtype = torch.bfloat16
        batch_size = 1
        N = config["model"]["modulation"].get("per_block_adapter_single_blocks", 0) + 19
        guidance = torch.tensor([3.5]).to(device).expand(batch_size)
        out_dim = config["model"]["modulation"]["out_dim"]

        tar_text_inputs = tokenize_t5_prompt(pipe, prompt, max_length)
        tar_padding_mask = tar_text_inputs.attention_mask.to(device).bool()
        tar_tokens = tar_text_inputs.input_ids.to(device)
        if config["model"]["modulation"]["eos_exclude"]:
            tar_padding_mask[tar_tokens == 1] = False

        def get_start_end_by_pompt_matching(src_prompts, tar_prompts):
            text_cond_mask = torch.zeros(batch_size, max_length, device=device, dtype=torch.bool)
            tar_prompt_input_ids = tokenize_t5_prompt(pipe, tar_prompts, max_length).input_ids
            src_prompt_count = 1
            start_ends = []
            for i, (src_prompt, tar_prompt, tar_prompt_tokens) in enumerate(zip(src_prompts, tar_prompts, tar_prompt_input_ids)):
                try:
                    tar_start, tar_end = get_word_index(pipe, tar_prompt, tar_prompt_tokens, src_prompt, src_prompt_count, max_length, verbose=False)
                    start_ends.append([tar_start, tar_end])
                    text_cond_mask[i, tar_start:tar_end] = True
                except Exception as e:
                    print(e)
            return start_ends, text_cond_mask

        def encode_mod_image(pil_images):
            if config["model"]["modulation"]["use_dit"]:
                raise NotImplementedError()
            else:
                pil_images = [pad_to_square(img).resize((224, 224)) for img in pil_images]
                if config["model"]["modulation"]["use_vae"]:
                    raise NotImplementedError()
                else:
                    clip_pixel_values = pipe.clip_processor(
                        text=None, images=pil_images, do_resize=False, do_center_crop=False, return_tensors="pt",
                    ).pixel_values.to(dtype=dtype, device=device)
                    clip_outputs = pipe.clip_model(clip_pixel_values, output_hidden_states=True, interpolate_pos_encoding=True, return_dict=True)
                    return clip_outputs

        def rgba_to_white_background(input_path, background=(255,255,255)):
            with Image.open(input_path).convert("RGBA") as img:
                img_np = np.array(img)
                alpha = img_np[:, :, 3] / 255.0  # 归一化Alpha通道[3](@ref)
                rgb = img_np[:, :, :3].astype(float)  # 提取RGB通道
                
                background_np = np.full_like(rgb, background, dtype=float)  # 根据参数生成背景[7](@ref)
                
                # 混合计算：前景色*alpha + 背景色*(1-alpha)
                result_np = rgb * alpha[..., np.newaxis] + \
                            background_np * (1 - alpha[..., np.newaxis])
                
                result = Image.fromarray(result_np.astype(np.uint8), "RGB")
                return result
        def get_mod_emb(modulation_input, timestep):
            delta_emb = torch.zeros((batch_size, max_length, out_dim), dtype=dtype, device=device)
            delta_emb_pblock = torch.zeros((batch_size, max_length, N, out_dim), dtype=dtype, device=device)
            delta_emb_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
            delta_start_ends = None
            condition_latents = condition_ids = None
            text_cond_mask = None

            if modulation_input[0]["type"] == "adapter":
                num_inputs = len(modulation_input[0]["src_inputs"])
                src_prompts = [x["caption"] for x in modulation_input[0]["src_inputs"]]
                src_text_inputs = tokenize_t5_prompt(pipe, src_prompts, max_length)
                src_input_ids = unpad_input_ids(src_text_inputs.input_ids, src_text_inputs.attention_mask)
                tar_input_ids = unpad_input_ids(tar_text_inputs.input_ids, tar_text_inputs.attention_mask)
                src_prompt_embeds = pipe._get_t5_prompt_embeds(prompt=src_prompts, max_sequence_length=max_length, device=device) # (M, 512, 4096)
                
                pil_images = [rgba_to_white_background(x["image_path"]) for x in modulation_input[0]["src_inputs"]]

                src_ds_scales = [x.get("downsample_scale", 1.0) for x in modulation_input[0]["src_inputs"]]
                resized_pil_images = []
                for img, ds_scale in zip(pil_images, src_ds_scales):
                    img = pad_to_square(img)
                    if ds_scale < 1.0:
                        assert ds_scale > 0
                        img = img.resize((int(224 * ds_scale), int(224 * ds_scale))).resize((224, 224))
                    resized_pil_images.append(img)
                pil_images = resized_pil_images
                
                img_encoded = encode_mod_image(pil_images)
                delta_start_ends = []
                text_cond_mask = torch.zeros(num_inputs, max_length, device=device, dtype=torch.bool)
                if config["model"]["modulation"]["pass_vae"]:
                    pil_images = [pad_to_square(img).resize((condition_size, condition_size)) for img in pil_images]
                    with torch.no_grad():
                        batch_tensor = torch.stack([pil2tensor(x) for x in pil_images])
                        x_0, img_ids = encode_vae_images(pipe, batch_tensor) # (N, 256, 64)

                    condition_latents = x_0.clone().detach().reshape(1, -1, 64) # (1, N256, 64)
                    condition_ids = img_ids.clone().detach()
                    condition_ids = condition_ids.unsqueeze(0).repeat_interleave(num_inputs, dim=0) # (N, 256, 3)
                    for i in range(num_inputs):
                        condition_ids[i, :, 1] += 0 if pos_offset_type == "width" else -(batch_tensor.shape[-1]//16) * (i + 1)
                        condition_ids[i, :, 2] += -(batch_tensor.shape[-1]//16) * (i + 1)
                    condition_ids = condition_ids.reshape(-1, 3) # (N256, 3)

                if config["model"]["modulation"]["use_dit"]:
                    raise NotImplementedError()
                else:
                    src_delta_embs = [] # [(512, 3072)]
                    src_delta_emb_pblock = []
                    for i in range(num_inputs):
                        if isinstance(img_encoded, dict):
                            _src_clip_outputs = {}
                            for key in img_encoded:
                                if torch.is_tensor(img_encoded[key]):
                                    _src_clip_outputs[key] = img_encoded[key][i:i+1]
                                else:
                                    _src_clip_outputs[key] = [x[i:i+1] for x in img_encoded[key]]
                            _img_encoded = _src_clip_outputs
                        else:
                            _img_encoded = img_encoded[i:i+1]
                    
                        x1, x2 = pipe.modulation_adapters[0](timestep, src_prompt_embeds[i:i+1], _img_encoded)
                        src_delta_embs.append(x1[0]) # (512, 3072)
                        src_delta_emb_pblock.append(x2[0]) # (512, N, 3072)

                for input_args in modulation_input[0]["use_words"]:
                    src_word_count = 1
                    if len(input_args) == 3:
                        src_input_index, src_word, tar_word = input_args
                        tar_word_count = 1
                    else:
                        src_input_index, src_word, tar_word, tar_word_count = input_args[:4]
                    src_prompt = src_prompts[src_input_index]
                    tar_prompt = prompt

                    src_start, src_end = get_word_index(pipe, src_prompt, src_input_ids[src_input_index], src_word, src_word_count, max_length, verbose=False)
                    tar_start, tar_end = get_word_index(pipe, tar_prompt, tar_input_ids[0], tar_word, tar_word_count, max_length, verbose=False)
                    if delta_emb is not None:
                        delta_emb[:, tar_start:tar_end] = src_delta_embs[src_input_index][src_start:src_end] # (B, 512, 3072)
                    if delta_emb_pblock is not None:
                        delta_emb_pblock[:, tar_start:tar_end] = src_delta_emb_pblock[src_input_index][src_start:src_end] # (B, 512, N, 3072)
                    delta_emb_mask[:, tar_start:tar_end] = True
                    text_cond_mask[src_input_index, tar_start:tar_end] = True
                    delta_start_ends.append([0, src_input_index, src_start, src_end, tar_start, tar_end])
                text_cond_mask = text_cond_mask.transpose(0, 1).unsqueeze(0)

            else:
                raise NotImplementedError()
            return delta_emb, delta_emb_pblock, delta_emb_mask, \
                text_cond_mask, delta_start_ends, condition_latents, condition_ids
    
    # num_inference_steps = 28 # FIXME: harcoded here
    num_channels_latents = pipe.transformer.config.in_channels // 4

    # set timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        num_channels_latents,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        None,
        sigmas,
        mu=mu,
    )

    if modulation_input is not None:
        delta_embs = []
        delta_embs_pblock = []
        delta_embs_mask = []
        for i, t in enumerate(timesteps):
            t = t.expand(1).to(torch.bfloat16) / 1000
            (
                delta_emb, delta_emb_pblock, delta_emb_mask, 
                text_cond_mask, delta_start_ends, 
                condition_latents, condition_ids
            ) = get_mod_emb(modulation_input, t)
            delta_embs.append(delta_emb)
            delta_embs_pblock.append(delta_emb_pblock)
            delta_embs_mask.append(delta_emb_mask)

    if original_image is not None:
        raise NotImplementedError()
        (target_height, target_width), closest_ratio = get_closest_ratio(original_image.height, original_image.width, train_aspect_ratios)
    elif modulation_input is None or len(modulation_input) == 0:
        delta_emb = delta_emb_pblock = delta_emb_mask = None
    else:
        for i, t in enumerate(timesteps):
            t = t.expand(1).to(torch.bfloat16) / 1000
            (
                delta_emb, delta_emb_pblock, delta_emb_mask, 
                text_cond_mask, delta_start_ends, 
                condition_latents, condition_ids
            ) = get_mod_emb(modulation_input, t)
            delta_embs.append(delta_emb)
            delta_embs_pblock.append(delta_emb_pblock)
            delta_embs_mask.append(delta_emb_mask)

    if target_height is None or target_width is None:
        target_height = target_width = target_size

    if condition_pad_to == "square":
        condition_imgs = [pad_to_square(x) for x in condition_imgs]
    elif condition_pad_to == "target":
        condition_imgs = [pad_to_target(x, (target_size, target_size)) for x in condition_imgs]
    condition_imgs = [x.resize((condition_size, condition_size)).convert("RGB") for x in condition_imgs]
    # TODO: fix position_delta
    conditions = [
        Condition(
            condition_type=condition_type,
            condition=x,
            position_delta=position_delta,
        ) for x in condition_imgs
    ]
    # vlm_images = condition_imgs if config["model"]["use_vlm"] else []

    use_perblock_adapter = False
    try:
        if config["model"]["modulation"]["use_perblock_adapter"]:
            use_perblock_adapter = True
    except Exception as e:
        pass

    results = []
    for i in range(num_images):
        clear_attn_maps(pipe.transformer)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + i)
        if modulation_input is None or len(modulation_input) == 0:
            idips = None
        else:
            idips = ["human" in p["image_path"] for p in modulation_input[0]["src_inputs"]]
            if len(modulation_input[0]["use_words"][0])==5:
                print("use idips in use_words")
                idips = [x[-1] for x in modulation_input[0]["use_words"]]
        result_img = generate(
            pipe,
            prompt=prompt,
            max_sequence_length=max_length,
            vae_conditions=conditions,
            generator=generator,
            model_config=config["model"],
            height=target_height,
            width=target_width,
            condition_pad_to=condition_pad_to,
            condition_size=condition_size,
            text_cond_mask=text_cond_mask,
            delta_emb=delta_embs,
            delta_emb_pblock=delta_embs_pblock if use_perblock_adapter else None,
            delta_emb_mask=delta_embs_mask,
            delta_start_ends=delta_start_ends,
            condition_latents=condition_latents,
            condition_ids=condition_ids,
            mod_adapter=pipe.modulation_adapters[0] if config["model"]["modulation"]["use_dit"] else None,
            vae_skip_iter=vae_skip_iter,
            control_weight_lambda=control_weight_lambda,
            double_attention=double_attention,
            single_attention=single_attention,
            ip_scale=ip_scale,
            use_latent_sblora_control=use_latent_sblora_control,
            latent_sblora_scale=latent_sblora_scale,
            use_condition_sblora_control=use_condition_sblora_control,
            condition_sblora_scale=condition_sblora_scale,
            idips=idips if use_idip else None,
            device=device,
            forward_hook_manager=forward_hook_manager,
            num_inference_steps=num_inference_steps,
            **kargs,
        ).images[0]

        final_image = result_img
        results.append(final_image)

    if num_images == 1:
        return results[0]
    return results