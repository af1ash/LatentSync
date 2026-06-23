# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import time
import os
from pathlib import Path
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
from ..utils.video_writer import VideoWriter, VideoReader
import tqdm
import soundfile as sf
import matplotlib.pyplot as plt

from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class LowPassFilter:
    """
    一阶低通滤波器
    """
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.y = None
        self.initialized = False

    def filter(self, x, alpha=None):
        if alpha is None:
            alpha = self.alpha

        if not self.initialized:
            self.y = x.copy()
            self.initialized = True
            return self.y

        self.y = alpha * x + (1 - alpha) * self.y
        return self.y


def compute_alpha(dt, cutoff):
    """
    将 cutoff frequency 转换为 alpha
    """
    if cutoff <= 0:
        return 1.0

    tau = 1.0 / (2 * math.pi * cutoff)
    te = dt

    return 1.0 / (1.0 + tau / te)


class OneEuroFilter:
    """
    One Euro Filter（支持向量输入，如 Nx2 landmarks）
    """

    def __init__(self, 
                 freq=30.0,
                 min_cutoff=1.0,
                 beta=0.007,
                 d_cutoff=1.0):

        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev = None
        self.dx_prev = None

        self.last_time = None

    def filter(self, x, timestamp=None):
        """
        x: numpy array (N, 2) 或 (N,)
        timestamp: 可选（不传则按固定帧率）
        """

        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.last_time = timestamp
            return x

        # 时间间隔
        if timestamp is None:
            dt = 1.0 / self.freq
        else:
            dt = timestamp - self.last_time
            self.last_time = timestamp

        if dt <= 0:
            dt = 1.0 / self.freq

        # --- 1. 计算速度（derivative） ---
        dx = (x - self.x_prev) / dt

        # --- 2. 平滑速度 ---
        alpha_d = compute_alpha(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        self.dx_prev = dx_hat

        # --- 3. 自适应 cutoff ---
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # --- 4. 平滑信号 ---
        alpha = np.array([compute_alpha(dt, c) for c in cutoff.flatten()])
        alpha = alpha.reshape(cutoff.shape)

        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat

        return x_hat


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        gfpgan=None,
        facehelper=None,
        rife=None,
        yolopose=None
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")
        self.gfpgan = gfpgan
        self.facehelper = facehelper
        self.rife = rife
        self.yolopose = yolopose

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (b, c, f, h, w)
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray, frame_num=None, stopat=None):
        faces = []
        boxes = []
        affine_matrices = []
        org_video_frames = []
        # print(f"Affine transforming {len(video_frames)} faces...")
        for i, frame in tqdm.tqdm(enumerate(video_frames), total=frame_num):
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            if stopat is not None and stopat > 0 and stopat == i:
                break
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)
            org_video_frames.append(frame)

        faces = torch.stack(faces)
        return org_video_frames, faces, boxes, affine_matrices

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            _, faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            _, faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices
    
    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)

        video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)

        synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            len(whisper_chunks),
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    unet_input = self.scheduler.scale_model_input(unet_input, t)

                    # concat latents, mask, masked_image_latents in the channel dimension
                    unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

                    # predict the noise residual
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)

        synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)

        # 训练过程中,为了避免内存溢出, 手动释放
        del video_frames
        del synced_video_frames
        del audio_samples
        del whisper_feature
        del whisper_chunks

    def face_enhance1(self, res_frame, fidelity_weight=1.0):
        self.facehelper.clean_all()
        self.facehelper.read_image(res_frame)
        self.facehelper.get_face_landmarks_5(
                only_center_face=True, resize=640, eye_dist_threshold=5)
        # print(f'\tdetect {num_det_faces} faces')
        # align and warp each face
        self.facehelper.align_warp_face()
        for idx, cropped_face in enumerate(self.facehelper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=False, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = self.gfpgan(cropped_face_t, w=fidelity_weight, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=False, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=False, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            self.facehelper.add_restored_face(restored_face, cropped_face)

        # upsample the background
        # if bg_upsampler is not None:
        #     # Now only support RealESRGAN for upsampling background
        #     bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
        # else:
        #     bg_img = None
        bg_img = None
        self.facehelper.get_inverse_affine(None)
        # paste each restored face to the input image
        # if args.face_upsample and face_upsampler is not None: 
        #     restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box, face_upsampler=face_upsampler)
        # else:
        restored_img = self.facehelper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=False)
        return restored_img

    def affine_transform_video1(self, video_frames: np.ndarray, frame_num=None, stopat=None):
        faces = []
        boxes = []
        affine_matrices = []
        org_video_frames = []
        filters = {}
        filters3 = {}
        landmarks_list = []
        landmarks3_list = []
        org_landmarks = []
        fbboxs = []
        # print(f"Affine transforming {len(video_frames)} faces...")
        for i, frame in tqdm.tqdm(enumerate(video_frames), total=frame_num):
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
                # alpha = frame[:, :, 3]
                # bg_r, bg_g, bg_b = 0, 255, 0
                # alpha_normal = alpha.astype(float) / 255.0
                # r_out = (frame[:, :, 0] + bg_r * (1 - alpha_normal)).astype('uint8')
                # g_out = (frame[:, :, 1] + bg_g * (1 - alpha_normal)).astype('uint8')
                # b_out = (frame[:, :, 2] + bg_b * (1 - alpha_normal)).astype('uint8')
                # frame = np.stack([r_out, g_out, b_out], axis=-1)
                # cv2.putText(frame, f"frame={i}", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0,0), 2)
            if stopat is not None and stopat > 0 and stopat == i:
                break

            # face, box, affine_matrix = self.image_processor.affine_transform(frame)
            image = frame
            bbox, landmark_2d_106 = self.image_processor.face_detector(image)
            if bbox is None:
                raise RuntimeError("Face not detected")
            fbboxs.append(bbox)
            
            current_time = time.time()
            org_landmarks.append(landmark_2d_106)
            smoothed_landmarks = np.zeros_like(landmark_2d_106)
            for j in [43, 48, 49, 51, 50, 74, 77, 83, 86, 101, 102, 103, 104, 105]:
                if i == 0:
                    curfilter = OneEuroFilter(current_time, landmark_2d_106[j], min_cutoff=0.01, beta=0.8)
                    # filters.append(curfilter)
                    filters[j] = curfilter
                    smoothed_landmarks[j] = landmark_2d_106[j]
                else:
                    smoothed_landmarks[j] = filters[j](current_time, landmark_2d_106[j])
            landmark_2d_106 = smoothed_landmarks
            landmarks_list.append(landmark_2d_106)
            # landmark_2d_106 = smoother.smooth(landmark_2d_106)
            org_video_frames.append(frame)
            pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
            pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
            pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center
            landmarks3 = np.array([pt_left_eye, pt_right_eye, pt_nose])
            smoothed_landmarks = np.zeros_like(landmarks3)
            for j in range(3):
                if i == 0:
                    curfilter = OneEuroFilter(current_time, landmarks3[j], min_cutoff=0.01, beta=0.012)
                    # filters.append(curfilter)
                    filters3[j] = curfilter
                    smoothed_landmarks[j] = landmarks3[j]
                else:
                    smoothed_landmarks[j] = filters3[j](current_time, landmarks3[j])
            landmarks3 = smoothed_landmarks
            landmarks3_list.append(landmarks3)
            face, affine_matrix = self.image_processor.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
            box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
            face = cv2.resize(face, (self.image_processor.resolution, self.image_processor.resolution), interpolation=cv2.INTER_LANCZOS4)
            face = rearrange(torch.from_numpy(face), "h w c -> c h w")
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)
        
        # self.plot_image(range(len(org_video_frames)), org_landmarks, [43, 48, 49, 51, 50, 74, 77, 83, 86, 101, 102, 103, 104, 105], filename="org.png")
        # self.plot_image(range(len(org_video_frames)), landmarks_list, [43, 48, 49, 51, 50, 74, 77, 83, 86, 101, 102, 103, 104, 105], filename="org_f12_8.png")
        # self.plot_image(range(len(org_video_frames)), landmarks3_list, list(range(3)), filename="smooth_f12_8.png")

        faces = torch.stack(faces)
       
        return org_video_frames, faces, boxes, affine_matrices, fbboxs

    def loop_video1(self, whisper_chunks: list, video_frames_gen: np.ndarray, frame_num=None):
        # If the audio is longer than the video, we need to loop the video
        print(f"{len(whisper_chunks)=},{frame_num=}")
        if len(whisper_chunks) > frame_num:
            video_frames, faces, boxes, affine_matrices, fbboxs = self.affine_transform_video1(video_frames_gen, frame_num=frame_num)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_fbboxes = []
            loop_affine_matrices = []
            for i in range(num_loops):
                # if i % 2 == 0:
                if True:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_fbboxes += fbboxs
                    loop_affine_matrices += affine_matrices
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_fbboxes += fbboxs[::-1]
                    loop_affine_matrices += affine_matrices[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            fbboxs = loop_fbboxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
        else:
            # video_frames = video_frames[: len(whisper_chunks)]
            video_frames, faces, boxes, affine_matrices, fbboxs = self.affine_transform_video1(video_frames_gen, frame_num=len(whisper_chunks), stopat=len(whisper_chunks))

        return video_frames, faces, boxes, affine_matrices, fbboxs

    def plot_image(self, x, y, indexs, filename):

        landmarks_list = np.array(y)
        plt.figure(figsize=(22,14))
        pnum = len(indexs)
        for i, pi in enumerate(indexs):
            plt.subplot(pnum,2,i*2+1)
            plt.plot(x, landmarks_list[:,pi,0], marker='.', linestyle='-', linewidth=1, markersize=2)
            plt.xlabel("帧序号")
            plt.ylabel("x")
            plt.title(pi)
            plt.subplot(pnum,2,i*2+2)
            plt.plot(x, landmarks_list[:,pi,1], marker='.', linestyle='-', linewidth=1, markersize=2, color='orange')
            plt.xlabel("帧序号")
            plt.ylabel("y")
            plt.title(pi)
        plt.savefig(filename)

        # landmarks3_list = np.array(landmarks3_list)
        # plt.figure(figsize=(22,14))
        # for pi in range(3):
        #     plt.subplot(3,2,pi*2+1)
        #     plt.plot(range(len(org_video_frames)), landmarks3_list[:,pi,0], marker='.', linestyle='-', linewidth=1, markersize=2)
        #     plt.xlabel("帧序号")
        #     plt.ylabel("x")
        #     plt.title(pi)
        #     plt.subplot(3,2,pi*2+2)
        #     plt.plot(range(len(org_video_frames)), landmarks3_list[:,pi,1], marker='.', linestyle='-', linewidth=1, markersize=2, color='orange')
        #     plt.xlabel("帧序号")
        #     plt.ylabel("y")
        #     plt.title(pi)
        # plt.savefig("filter_3p_smooth_01_both.png")
        pass

    def affine_transform_video2(self, video_frames: np.ndarray, frame_num=None, stopat=None, use_euro=False):
        faces = []
        boxes = []
        affine_matrices = []
        org_video_frames = []
        filters = {}
        filters3 = {}
        landmarks_list = []
        landmarks3_list = []
        org_landmarks = []
        fbboxs = []
        frames_data = []
        # print(f"Affine transforming {len(video_frames)} faces...")
        if stopat is not None:
            frame_num = stopat
        for i, orgframe in tqdm.tqdm(enumerate(video_frames), total=frame_num):
            frame_item = {}
            if orgframe.shape[-1] == 4:
                image = orgframe[:, :, :3]
                # alpha = orgframe[:, :, 3]
                # bg_r, bg_g, bg_b = 0, 255, 0
                # alpha_normal = alpha.astype(float) / 255.0
                # r_out = (orgframe[:, :, 0] + bg_r * (1 - alpha_normal)).astype('uint8')
                # g_out = (orgframe[:, :, 1] + bg_g * (1 - alpha_normal)).astype('uint8')
                # b_out = (orgframe[:, :, 2] + bg_b * (1 - alpha_normal)).astype('uint8')
                # image = np.stack([r_out, g_out, b_out], axis=-1)
                # orgframe = image
            else:
                image = orgframe
                # alpha = frame[:, :, 3]
                # bg_r, bg_g, bg_b = 0, 255, 0
                # alpha_normal = alpha.astype(float) / 255.0
                # r_out = (frame[:, :, 0] + bg_r * (1 - alpha_normal)).astype('uint8')
                # g_out = (frame[:, :, 1] + bg_g * (1 - alpha_normal)).astype('uint8')
                # b_out = (frame[:, :, 2] + bg_b * (1 - alpha_normal)).astype('uint8')
                # frame = np.stack([r_out, g_out, b_out], axis=-1)
                # cv2.putText(frame, f"frame={i}", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0,0), 2)
            if stopat is not None and stopat > 0 and stopat == i:
                break

            # face, box, affine_matrix = self.image_processor.affine_transform(frame)
            bbox, landmark_2d_106 = self.image_processor.face_detector(image)
            if bbox is None:
                raise RuntimeError("Face not detected")
            fbboxs.append(bbox)
            
            current_time = time.time()
            org_landmarks.append(landmark_2d_106)
            if use_euro:
                smoothed_landmarks = np.zeros_like(landmark_2d_106)
                for j in [43, 48, 49, 51, 50, 74, 77, 83, 86, 101, 102, 103, 104, 105]:
                    if i == 0:
                        curfilter = OneEuroFilter(current_time, landmark_2d_106[j], min_cutoff=0.01, beta=0.8)
                        # filters.append(curfilter)
                        filters[j] = curfilter
                        smoothed_landmarks[j] = landmark_2d_106[j]
                    else:
                        smoothed_landmarks[j] = filters[j](current_time, landmark_2d_106[j])
                landmark_2d_106 = smoothed_landmarks
            landmarks_list.append(landmark_2d_106)
            # landmark_2d_106 = smoother.smooth(landmark_2d_106)
            # org_video_frames.append(frame)
            pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
            pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
            pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center
            landmarks3 = np.array([pt_left_eye, pt_right_eye, pt_nose])
            if use_euro:
                smoothed_landmarks = np.zeros_like(landmarks3)
                for j in range(3):
                    if i == 0:
                        curfilter = OneEuroFilter(current_time, landmarks3[j], min_cutoff=0.01, beta=0.012)
                        # filters.append(curfilter)
                        filters3[j] = curfilter
                        smoothed_landmarks[j] = landmarks3[j]
                    else:
                        smoothed_landmarks[j] = filters3[j](current_time, landmarks3[j])
                landmarks3 = smoothed_landmarks
            landmarks3_list.append(landmarks3)
            face, affine_matrix = self.image_processor.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
            box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
            face = cv2.resize(face, (self.image_processor.resolution, self.image_processor.resolution), interpolation=cv2.INTER_LANCZOS4)
            face = rearrange(torch.from_numpy(face), "h w c -> c h w")
            # faces.append(face)
            # boxes.append(box)
            # affine_matrices.append(affine_matrix)
            frame_item["video"] = orgframe
            frame_item["face"] = face
            frame_item["boxes"] = box
            frame_item["affine"] = affine_matrix
            frame_item["fbbox"] = bbox
            frames_data.append(frame_item)
        
        # self.plot_image(range(len(org_video_frames)), org_landmarks, [43, 48, 49, 51, 50, 74, 77, 83, 86, 101, 102, 103, 104, 105], filename="org.png")
        # self.plot_image(range(len(org_video_frames)), landmarks_list, [43, 48, 49, 51, 50, 74, 77, 83, 86, 101, 102, 103, 104, 105], filename="org_f12_8.png")
        # self.plot_image(range(len(org_video_frames)), landmarks3_list, list(range(3)), filename="smooth_f12_8.png")

        # faces = torch.stack(faces)
       
        # return org_video_frames, faces, boxes, affine_matrices, fbboxs
        return frames_data

    def affine_transform_face(self, orgframe: np.ndarray, pti: int, one_euro=None):
        # filters = {}
        # filters3 = {}
        # print(f"Affine transforming {len(video_frames)} faces...")
        frame_item = {}
        # if orgframe.shape[-1] == 4:
        #     image = orgframe[:, :, :3]
        #     # alpha = orgframe[:, :, 3]
        #     # bg_r, bg_g, bg_b = 0, 255, 0
        #     # alpha_normal = alpha.astype(float) / 255.0
        #     # r_out = (orgframe[:, :, 0] + bg_r * (1 - alpha_normal)).astype('uint8')
        #     # g_out = (orgframe[:, :, 1] + bg_g * (1 - alpha_normal)).astype('uint8')
        #     # b_out = (orgframe[:, :, 2] + bg_b * (1 - alpha_normal)).astype('uint8')
        #     # image = np.stack([r_out, g_out, b_out], axis=-1)
        #     # orgframe = orgframe[:, :, :3]
        #     # cv2.putText(frame, f"frame={i}", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0,0), 2)
        # else:
        #     image = orgframe

        # face, box, affine_matrix = self.image_processor.affine_transform(frame)
        bbox, landmark_2d_106 = self.image_processor.face_detector(orgframe)
        if bbox is None:
            raise RuntimeError("Face not detected")
            
        current_time = time.time()
        if one_euro:
            # smoothed_landmarks = np.zeros_like(landmark_2d_106)
            # for j in [43, 48, 49, 51, 50, 74, 77, 83, 86, 101, 102, 103, 104, 105]:
            #     if pti == 0:
            #         curfilter = OneEuroFilter(current_time, landmark_2d_106[j], min_cutoff=0.01, beta=0.8)
            #         # filters.append(curfilter)
            #         filters[j] = curfilter
            #         smoothed_landmarks[j] = landmark_2d_106[j]
            #     else:
            #         smoothed_landmarks[j] = filters[j](current_time, landmark_2d_106[j])
            smoothed_landmarks = one_euro.filter(landmark_2d_106, timestamp=current_time)
            landmark_2d_106 = smoothed_landmarks
        # org_video_frames.append(frame)
        # pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        # pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
        # pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center
        # landmarks3 = np.array([pt_left_eye, pt_right_eye, pt_nose])
        pt_left_eye = np.mean(landmark_2d_106[[43, 44, 45,46, 47, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        pt_right_eye = np.mean(landmark_2d_106[[97, 98, 99, 100, 101,102, 103, 104, 105]], axis=0)  # right eyebrow center
        pt_nose = np.mean(landmark_2d_106[[74, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86]], axis=0)  # nose center
        landmarks3 = np.array([pt_left_eye, pt_right_eye, pt_nose])
        landmarks3 = landmarks3.astype(np.int32)
        if one_euro:
        #     smoothed_landmarks = np.zeros_like(landmarks3)
        #     for j in range(3):
        #         if pti == 0:
        #             curfilter = OneEuroFilter(current_time, landmarks3[j], min_cutoff=0.01, beta=0.012)
        #             # filters.append(curfilter)
        #             filters3[j] = curfilter
        #             smoothed_landmarks[j] = landmarks3[j]
        #         else:
        #             smoothed_landmarks[j] = filters3[j](current_time, landmarks3[j])
        #     landmarks3 = smoothed_landmarks
            smoothed_landmarks = one_euro.filter(landmarks3, timestamp=current_time)
            landmarks3= smoothed_landmarks
        face, affine_matrix = self.image_processor.restorer.align_warp_face(orgframe.copy(), landmarks3=landmarks3, smooth=False)
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.image_processor.resolution, self.image_processor.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        # cv2.putText(orgframe, f"{pti=},{str(landmarks3)}", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0,0), 2)
        frame_item["video"] = orgframe
        frame_item["face"] = face
        frame_item["boxes"] = box
        frame_item["affine"] = affine_matrix
        frame_item["fbbox"] = bbox
        frame_item["landmark_2d_106"] = landmark_2d_106
        frame_item["landmark3"] = landmarks3
        return frame_item

    def mirror_index(self, size, index):
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def get_frame_data_by_index(self, index, candidate_frames):
        mi = self.mirror_index( len(candidate_frames), index)
        return candidate_frames[mi]

    def get_head_bbox(self, keypoints, confidences, image_shape):
        """
        根据关键点计算肩部以上头部的裁剪区域。
        返回: (x1, y1, x2, y2) 裁剪框坐标（像素），若无法检测则返回 None
        """
        h, w = image_shape[:2]
        PADDING_RATIO = 0.2                # 裁剪框向外扩展比例（防止头部边缘被切）
        CONFIDENCE_THRESHOLD = 0.5         # 关键点置信度阈值
        SHOULDER_INDICES = [5, 6]   # 左肩、右肩
        HEAD_INDICES = [0, 1, 2, 3, 4]  # 鼻子、双眼、双耳（用于辅助定位头部中心）
        # 提取肩部关键点
        shoulder_pts = []
        for idx in SHOULDER_INDICES:
            if confidences[idx] >= CONFIDENCE_THRESHOLD:
                x, y = keypoints[idx]
                shoulder_pts.append((x, y))

        # 至少需要一个肩部点来确定水平位置和尺度
        if len(shoulder_pts) == 0:
            return None

        # 计算肩部中心
        shoulder_center_x = np.mean([p[0] for p in shoulder_pts])
        shoulder_center_y = np.mean([p[1] for p in shoulder_pts])

        # 计算肩宽（两肩距离），若只有一个肩则用固定比例估算
        if len(shoulder_pts) == 2:
            shoulder_width = np.linalg.norm(np.array(shoulder_pts[0]) - np.array(shoulder_pts[1]))
        else:
            # 单肩时，用肩到鼻子的距离估算肩宽（约为人脸宽度的 1.5 倍）
            if confidences[0] >= CONFIDENCE_THRESHOLD:
                nose = keypoints[0]
                shoulder_width = 2.0 * np.linalg.norm(np.array(shoulder_pts[0]) - np.array(nose))
            else:
                shoulder_width = 200  # 保底估算值

        # 用头部关键点辅助定位头部中心（取所有有效头部关键点的均值）
        head_pts = []
        for idx in HEAD_INDICES:
            if confidences[idx] >= CONFIDENCE_THRESHOLD:
                x, y = keypoints[idx]
                head_pts.append((x, y))

        if len(head_pts) > 0:
            head_center_x = np.mean([p[0] for p in head_pts])
            head_center_y = np.mean([p[1] for p in head_pts])
        else:
            # 若头部关键点全部丢失，用肩部中心向上偏移估算
            head_center_x = shoulder_center_x
            head_center_y = shoulder_center_y - shoulder_width * 0.5

        # 计算裁剪框大小：以肩宽为基准，宽高比 0.8~1.0（头部通常是偏正方形区域）
        crop_size = shoulder_width * 1.2  # 稍大于肩宽确保头部完整

        # 以头部中心为中心，构建正方形裁剪框
        half_size = crop_size / 2
        x1 = int(head_center_x - half_size)
        y1 = int(head_center_y - half_size)
        x2 = int(head_center_x + half_size)
        y2 = int(head_center_y + half_size)

        # 添加 padding 向外扩展
        pad = int(crop_size * PADDING_RATIO)
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad

        # 边界裁剪，防止超出图像范围
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 确保裁剪框有效
        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def datagen_whisper_frames(
        self,
        inf_step,
        audio_array,
        whisper_chunks,
        candidate_frames,
        audio_batch_size,
        audio_channel,
        batch_size=8,
    ):
        whisper_batch = []
        audio_batchs = []
        face_batch = []
        vframe_batch = []
        gi = 0
        candidate_geneter = candidate_frames.read_iter()
        # one_euro_filter = OneEuroFilter(freq=25, min_cutoff=0.01, beta=0.012, d_cutoff=1.0)
        one_euro_filter = None
        for i in range(inf_step):

            # whisper_batch.append(whisper_chunks[i])
            whisper_batch = whisper_chunks[i*batch_size: (i+1)*batch_size]
            # frame_batch.append(candidate_frame["video"])
            batch_len = len(whisper_batch)

            for j in range(batch_len):
                # candidate_frame = self.get_frame_data_by_index(gi, candidate_frames)
                try:
                    org_frame = next(candidate_geneter)
                except StopIteration:
                    candidate_geneter = candidate_frames.read_iter()
                    org_frame = next(candidate_geneter)
                if org_frame.shape[-1] == 4:
                    org_frame = org_frame[:, :, :3]
                    # alpha = orgframe[:, :, 3]
                    # bg_r, bg_g, bg_b = 0, 255, 0
                    # alpha_normal = alpha.astype(float) / 255.0
                    # r_out = (orgframe[:, :, 0] + bg_r * (1 - alpha_normal)).astype('uint8')
                    # g_out = (orgframe[:, :, 1] + bg_g * (1 - alpha_normal)).astype('uint8')
                    # b_out = (orgframe[:, :, 2] + bg_b * (1 - alpha_normal)).astype('uint8')
                    # image = np.stack([r_out, g_out, b_out], axis=-1)
                    # orgframe = orgframe[:, :, :3]
                    # cv2.putText(frame, f"frame={i}", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0,0), 2)

                results = self.yolopose(org_frame, verbose=False)

                keypoints = results[0].keypoints.xy[0].cpu().numpy()  # (17, 2)
                confidences = results[0].keypoints.conf[0].cpu().numpy()  # (17,)
                hbbox = self.get_head_bbox(keypoints, confidences, org_frame.shape)
                x1, y1, x2, y2 = hbbox
                crop_frame = org_frame[y1:y2, x1:x2]

                candidate_frame = self.affine_transform_face(crop_frame, gi, one_euro_filter)
                candidate_frame["input_img"] = org_frame
                candidate_frame["headbbox"] = hbbox

                face_batch.append(candidate_frame["face"])
                # alpha_img = candidate_frame["alpha_img"]
                # if alpha_img is not None:
                #     alpha_batch.append(alpha_img)
                vframe_batch.append(candidate_frame)

                audio_batch = audio_array[
                    gi * audio_batch_size : (gi + 1)
                    * audio_batch_size
                ]
                if len(audio_batch) < audio_batch_size:
                    # 长度不够的时候, 补充静音
                    audio_batch = np.pad(
                        audio_batch,
                        (0, audio_batch_size - len(audio_batch)),
                        "constant",
                        constant_values=0,
                    )
                    # audio_batch = torch.from_numpy(audio_batch)

                audio_batch = np.reshape(
                    audio_batch, (audio_channel, len(audio_batch))
                )
                # audio_batchs.append(audio_batch)
                # if i < len(whisper_chunks):
                #     whisper_batch.append(whisper_chunks[i])
                # else:
                #     # 通过补充空白音频推理恢复静态口型
                #     empty_batch  = self.audio_processor.audio2chunks(
                #         audio_batch[0], fps=self.fps, audio_feat_length=[1, 1]
                #     )
                #     whisper_chunks.extend(empty_batch)
                #     whisper_batch.append(whisper_chunks[i])
                gi += 1
            yield (
                whisper_batch,
                face_batch,
                vframe_batch,
                audio_batchs,
                    
            )
            whisper_batch = []
            face_batch = []
            vframe_batch = []
            audio_batchs = []

    def boxblur(self, image, radius=1):
        height, width, _ = image.shape
        # radius = int(min(height, width) * 0.001)
        # radius = 1
        ksize = (2*radius+1, 2*radius+1)
        blurred = cv2.blur(image, ksize)
        return blurred

    @torch.no_grad()
    def stream(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # video_frames = read_video(video_path, use_decord=False)
        vr = VideoReader(video_path)
        audio_samples = vr.read(type_="audio")
        audio_samples = audio_samples.astype(np.float32)[0]
        # audio_samples = read_audio(audio_path)
        # audio_samples = audio_samples.numpy().astype(np.float32)
        # audio_samples = audio_samples * 0.00001

        # video_frame_generater = vr.read_iter()
        # video_frames, faces, boxes, affine_matrices, fbboxs = self.loop_video1(whisper_chunks, video_frame_generater, vr.frames)
        # source_frames = self.affine_transform_video2(video_frame_generater, vr.frames, stopat=len(whisper_chunks))
        # source_frames = self.affine_transform_video2(video_frame_generater, vr.frames, stopat=None)
        
        whisper_feature = self.audio_encoder.audio2feat(audio_samples)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        synced_video_frames = []

        # audio_samples_remain_length = int(len(video_frames) / video_fps * audio_sample_rate)
        # audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()
        audio_chunk_size = int(audio_sample_rate / video_fps)
        audio_channel = 1
        video_out_path = Path(video_out_path)
        vwriter = VideoWriter(str(video_out_path), outformat=video_out_path.suffix)

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        # all_latents = self.prepare_latents(
        #     len(whisper_chunks),
        #     num_channels_latents,
        #     height,
        #     width,
        #     weight_dtype,
        #     device,
        #     generator,
        # )
        strict = kwargs.get("strict", "video")
        if strict == "video":
            target_frame_num = vr.frames
            whisper_chunks = whisper_chunks[:vr.frames]
        else:
            target_frame_num = len(whisper_chunks)
        
        print(f"{target_frame_num=},{len(whisper_chunks)=}")

        num_inferences = math.ceil(target_frame_num / num_frames)
        data_gen = self.datagen_whisper_frames(num_inferences, audio_samples, whisper_chunks, vr, audio_chunk_size, audio_channel, batch_size=num_frames)

        for i, batch_data in tqdm.tqdm(enumerate(data_gen), total=num_inferences,desc="Doing inference..."):
        # for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):

            whisper_batch, face_batch, vframe_batch, audio_batch = batch_data
            inference_faces = np.array(face_batch)

            latents = self.prepare_latents(len(whisper_batch), num_channels_latents, height, width, weight_dtype, device, generator)

            if self.unet.add_audio_layer:
                # audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = torch.stack(whisper_batch)
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            # inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            # latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    unet_input = self.scheduler.scale_model_input(unet_input, t)

                    # concat latents, mask, masked_image_latents in the channel dimension
                    # print(f"{unet_input.shape=},{mask_latents.shape=},{masked_image_latents.shape=},{ref_latents.shape}")
                    unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

                    # predict the noise residual
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            # synced_video_frames.append(decoded_latents)
            alpha = None 
            # decoded_latents = inference_faces
            for index, new_face in enumerate(decoded_latents):

                ni = i * num_frames + index
                # restore face
                boxes = vframe_batch[index]["boxes"]
                x1, y1, x2, y2 = boxes
                org_height = int(y2 - y1)
                org_width = int(x2 - x1)
                # nor_face = new_face.cpu().numpy().astype(np.float16)
                # nor_face = nor_face / 127.5 - 1.0
                # new_face = torch.from_numpy(nor_face).to(self.device)
                face = torchvision.transforms.functional.resize(
                    new_face, size=(org_height, org_width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
                )
                # org_frame = org_frames[index]
                org_frame = vframe_batch[index]["video"]

                if org_frame.shape[-1] == 4:
                    if not (
                        np.all(org_frame[:, :, 3] == 255) or np.all(org_frame[:, :, 3] == 0)
                    ):
                        alpha = org_frame[:, :, 3]
                    oframe = org_frame[:, :, :3]
                else:
                    oframe = org_frame
                affine_matrice = vframe_batch[index]["affine"]
                out_frame = self.image_processor.restorer.restore_img(oframe, face, affine_matrice)

                out_frame = self.boxblur(out_frame, radius=1)
                if self.gfpgan:
                    # fbbox = fbboxs[index]
                    # fbbox = vframe_batch[index]["fbbox"]
                    # x1, y1, x2, y2 = fbbox
                    out_frame = self.face_enhance1(out_frame.copy())
                    # out_frame[y1:y2, x1:x2] = gan_face
                # else:
                #     out_frame = self.boxblur(out_frame)

                if alpha is not None:
                    out_frame = cv2.merge([
                        out_frame[:, :, 0],
                        out_frame[:, :, 1],
                        out_frame[:, :, 2],
                        alpha,
                    ])
                # audio_data = audio_samples[ni*audio_chunk_size:(ni+1) *audio_chunk_size]
                # audio_data = audio_data.numpy()
                # if len(audio_data) < audio_chunk_size:
                #     # 长度不够的时候, 补充静音
                #     audio_data = np.pad(
                #         audio_data,
                #         (0, audio_chunk_size - len(audio_data)),
                #         "constant",
                #         constant_values=0,
                #     )
                # audio_data = np.reshape(
                #     audio_data, (audio_channel, len(audio_data))
                # )
                audio_data = audio_batch[index]
                # audio_data = audio_data.numpy()
                # audio_data = audio_data.astype(np.float32)
                if ni == 0:
                    vwriter.init_stream(out_frame, audio_data, fps=video_fps, sample_rate=audio_sample_rate)
                vwriter.encode_frame(out_frame, audio_data, video_pts=ni)
        vwriter.close()
        vr.close()

        # synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)

        # audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        # audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # os.makedirs(temp_dir, exist_ok=True)

        # write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)

        # sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        # command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        # subprocess.run(command, shell=True)

        # 训练过程中,为了避免内存溢出, 手动释放
        # del source_frames
        del synced_video_frames
        del audio_samples
        del whisper_feature
        del whisper_chunks

    @torch.no_grad()
    def lipsync(
        self,
        num_inferences,
        audio_samples,
        whisper_chunks,
        source_frames,
        audio_chunk_size,
        audio_channel,
        num_frames: int = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # is_train = self.unet.training
        # self.unet.eval()

        # check_ffmpeg_installed()

        # 0. Define call parameters
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        # all_latents = self.prepare_latents(
        #     len(whisper_chunks),
        #     num_channels_latents,
        #     height,
        #     width,
        #     weight_dtype,
        #     device,
        #     generator,
        # )
        # strict = kwargs.get("strict", "video")
        # if strict == "video":
        #     target_frame_num = len(source_frames)
        #     whisper_chunks = whisper_chunks[:len(source_frames)]
        # else:
        #     target_frame_num = len(whisper_chunks)

        # num_inferences = math.ceil(target_frame_num / num_frames)
        data_gen = self.datagen_whisper_frames(num_inferences, audio_samples, whisper_chunks, source_frames, audio_chunk_size, audio_channel, batch_size=num_frames)

        for i, batch_data in tqdm.tqdm(enumerate(data_gen), total=num_inferences,desc="Doing inference..."):
        # for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):

            whisper_batch, face_batch, vframe_batch, audio_batch = batch_data
            inference_faces = np.array(face_batch)

            latents = self.prepare_latents(len(whisper_batch), num_channels_latents, height, width, weight_dtype, device, generator)

            if self.unet.add_audio_layer:
                # audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = torch.stack(whisper_batch)
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            # inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            # latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    unet_input = self.scheduler.scale_model_input(unet_input, t)

                    # concat latents, mask, masked_image_latents in the channel dimension
                    # print(f"{unet_input.shape=},{mask_latents.shape=},{masked_image_latents.shape=},{ref_latents.shape}")
                    unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

                    # predict the noise residual
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            # synced_video_frames.append(decoded_latents)
            alpha = None 
            # decoded_latents = inference_faces
            for index, new_face in enumerate(decoded_latents):

                ni = i * num_frames + index
                # restore face
                boxes = vframe_batch[index]["boxes"]
                x1, y1, x2, y2 = boxes
                org_height = int(y2 - y1)
                org_width = int(x2 - x1)
                # nor_face = new_face.cpu().numpy().astype(np.float16)
                # nor_face = new_face / 127.5 - 1.0
                # new_face = torch.from_numpy(nor_face).to(self.device)
                # new_face = new_face / 255
                # new_face = torch.from_numpy(new_face).to(self.device)
                face = torchvision.transforms.functional.resize(
                    new_face, size=(org_height, org_width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
                )
                # org_frame = org_frames[index]
                oframe = vframe_batch[index]["video"]
                affine_matrice = vframe_batch[index]["affine"]
                out_frame = self.image_processor.restorer.restore_img(oframe, face, affine_matrice)

                out_frame = self.boxblur(out_frame, radius=1)
                if self.gfpgan:
                    # fbbox = fbboxs[index]
                    # fbbox = vframe_batch[index]["fbbox"]
                    # x1, y1, x2, y2 = fbbox
                    out_frame = self.face_enhance1(out_frame.copy())
                    # out_frame[y1:y2, x1:x2] = gan_face
                # else:
                #     out_frame = self.boxblur(out_frame)
               
                # audio_data = audio_batch[index]
                # landmarks3 = vframe_batch[index]["landmark3"]
                # for point in landmarks3:
                #     x, y = point
                #     cv2.circle(out_frame, (x, y), 1, (0, 0, 255), -1)  # 使用红色标记点
                # landmarks106 = vframe_batch[index]["landmark_2d_106"]
                # landmarks106 = landmarks106.astype(np.int32)
                # for point in landmarks106:
                #     x, y = point
                #     cv2.circle(out_frame, (x, y), 1, (0, 255, 0), -1)  # 使用红色标记点
                # x1, y1, x2, y2 = vframe_batch[index]['fbbox']
                # cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                input_img = vframe_batch[index]["input_img"]
                if input_img.shape[-1] == 4:
                    if not (
                        np.all(input_img[:, :, 3] == 255) or np.all(input_img[:, :, 3] == 0)
                    ):
                        alpha = input_img[:, :, 3]
                    frame_rgb = input_img[:, :, :3]
                else:
                    frame_rgb = input_img
                headbbox = vframe_batch[index]["headbbox"]
                x1, y1, x2, y2 = headbbox
                frame_rgb[y1:y2, x1:x2] = out_frame
                if alpha is not None:
                    frame_rgb = cv2.merge([
                        frame_rgb[:, :, 0],
                        frame_rgb[:, :, 1],
                        frame_rgb[:, :, 2],
                        alpha,
                    ])

                yield frame_rgb


    @torch.no_grad()
    def stream1(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        audio_samples = read_audio(audio_path)
        audio_samples = audio_samples.numpy().astype(np.float32)
        # audio_samples = audio_samples * 0.00001
        # video_frames = read_video(video_path, use_decord=False)
        vr = VideoReader(video_path)
        # audio_samples = vr.read(type_="audio")
        # audio_samples = audio_samples.astype(np.float32)[0]
        # video_frame_generater = vr.read_iter()
        # video_frames, faces, boxes, affine_matrices, fbboxs = self.loop_video1(whisper_chunks, video_frame_generater, vr.frames)
        # source_frames = self.affine_transform_video2(video_frame_generater, vr.frames, stopat=len(whisper_chunks))
        # source_frames = self.affine_transform_video2(video_frame_generater, vr.frames, stopat=None)
        # audio_samples = vr.read(type_="audio")
        # audio_samples = audio_samples.astype(np.float32)[0]

        strict = kwargs.get("strict", "audio")
        if strict == "video":
            target_frame_num = vr.frames
            audio_samples = vr.read(type_="audio")
            audio_samples = audio_samples.astype(np.float32)[0]
            whisper_feature = self.audio_encoder.audio2feat(audio_samples)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
        else:
            audio_samples = read_audio(audio_path)
            audio_samples = audio_samples.numpy().astype(np.float32)
            whisper_feature = self.audio_encoder.audio2feat(audio_samples)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
            target_frame_num = len(whisper_chunks)

        if self.rife: 
            video_fps = video_fps * self.rife.multi
        synced_video_frames = []

        # audio_samples_remain_length = int(len(video_frames) / video_fps * audio_sample_rate)
        # audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()
        audio_chunk_size = int(audio_sample_rate / video_fps)
        audio_channel = 1
        video_out_path = Path(video_out_path)
        vformat = kwargs.get("vformat", "mp4")
        vwriter = VideoWriter(str(video_out_path), codec_info=vformat)

        num_inferences = math.ceil(target_frame_num / num_frames)

        frames_gen = self.lipsync(
            num_inferences, audio_samples, whisper_chunks, vr, 
            audio_chunk_size, audio_channel,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=weight_dtype,
            eta=eta,
            mask_image_path=mask_image_path,
            temp_dir=temp_dir,
            generator=generator,
            callback=callback,
            callback_steps=callback_steps, **kwargs,
        )
        if self.rife:
            frames_gen = self.rife.inference(frames_gen)
        for ni, out_frame in enumerate(frames_gen):
            # out_frame, audio_data = frame_data

            audio_batch = audio_samples[
                ni * audio_chunk_size: (ni + 1)
                * audio_chunk_size
            ]
            if len(audio_batch) < audio_chunk_size:
                # 长度不够的时候, 补充静音
                audio_batch = np.pad(
                    audio_batch,
                    (0, audio_chunk_size - len(audio_batch)),
                    "constant",
                    constant_values=0,
                )
                # audio_batch = torch.from_numpy(audio_batch)

            audio_data = np.reshape(
                audio_batch, (audio_channel, len(audio_batch))
            )
            if ni == 0:
                vwriter.init_stream(out_frame, audio_data, fps=video_fps, sample_rate=audio_sample_rate)
            vwriter.encode_frame(out_frame, audio_data, video_pts=ni)
        vwriter.close()

        # synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)

        # audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        # audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # os.makedirs(temp_dir, exist_ok=True)

        # write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)

        # sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        # command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        # subprocess.run(command, shell=True)

        # 训练过程中,为了避免内存溢出, 手动释放
        del synced_video_frames
        del audio_samples
        del whisper_feature
        del whisper_chunks
