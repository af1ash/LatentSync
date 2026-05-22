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
from ..gfpgan.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class OneEuroFiler:

    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = np.array(x0, dtype=np.float32)
        self.dx_prev = np.zeros_like(self.x_prev, dtype=np.float32)
        self.t_prev = t0
    
    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)
    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev
    
    def __call__(self, t, x):
        t_e = t - self.t_prev

        d_cutoff = self.d_cutoff
        alpha_d = self.smoothing_factor(t_e, d_cutoff)

        dx = (np.array(x, dtype=np.float32) - self.x_prev) / t_e

        dx_hat = self.exponential_smoothing(alpha_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(alpha, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
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
        gfpgan=None
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

    def face_enhance(self, res_frame, face_restore_visibility=0.5):
        assert self.gfpgan, f"{self.gfpgan=}"
        height, width, _ = res_frame.shape
        res_frame = cv2.resize(
                            res_frame.astype(np.uint8), (512, 512)
                        )
        cropped_face_t = img2tensor(
            res_frame / 255.0, bgr2rgb=True, float32=True
        )
        normalize(
            cropped_face_t,
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            inplace=True,
        )
        cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
        output = self.gfpgan(cropped_face_t)[0]
        cropped_face = tensor2img(
            output, rgb2bgr=True, min_max=(-1, 1)
        )
        res_frame = (
            res_frame * (1 - face_restore_visibility)
            + cropped_face * face_restore_visibility
        )
        res_frame = cv2.resize(
                    res_frame.astype(np.uint8), (width, height)
                )
        return res_frame

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
            if stopat is not None and stopat > 0 and stopat == i:
                break

            # face, box, affine_matrix = self.image_processor.affine_transform(frame)
            image = frame
            bbox, landmark_2d_106 = self.image_processor.face_detector(image)
            if bbox is None:
                raise RuntimeError("Face not detected")
            fbboxs.append(bbox)
          
            current_time = time.time()
            # org_landmarks.append(landmark_2d_106)
            smoothed_landmarks = np.zeros_like(landmark_2d_106)
            for j in [43, 48, 49, 51, 50, 74, 77, 83, 86, 101, 102, 103, 104, 105]:
                if i == 0:
                    curfilter = OneEuroFiler(current_time, landmark_2d_106[j], min_cutoff=0.01, beta=0.5)
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
                    curfilter = OneEuroFiler(current_time, landmarks3[j], min_cutoff=0.01, beta=0.012)
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
        # self.plot_image(range(len(org_video_frames)), landmarks_list, [43, 48, 49, 51, 50, 74, 77, 83, 86, 101, 102, 103, 104, 105], filename="org_f12.png")
        # self.plot_image(range(len(org_video_frames)), landmarks3_list, list(range(3)), filename="smooth_f12.png")

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
                if i % 2 == 0:
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

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        # video_frames = read_video(video_path, use_decord=False)
        with VideoReader(video_path) as vr:
            video_frame_generater = vr.read_iter()
            video_frames, faces, boxes, affine_matrices, fbboxs = self.loop_video1(whisper_chunks, video_frame_generater, vr.frames)

        synced_video_frames = []

        # audio_samples_remain_length = int(len(video_frames) / video_fps * audio_sample_rate)
        # audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()
        audio_chunk_size = int(audio_sample_rate / video_fps)
        audio_channel = 1
        video_out_path = Path(video_out_path)
        vwriter = VideoWriter(str(video_out_path), outformat=video_out_path.suffix)

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
                x1, y1, x2, y2 = boxes[ni]
                org_height = int(y2 - y1)
                org_width = int(x2 - x1)
                # nor_face = new_face.cpu().numpy().astype(np.float16)
                # nor_face = nor_face / 127.5 - 1.0
                # new_face = torch.from_numpy(nor_face).to(self.device)
                face = torchvision.transforms.functional.resize(
                    new_face, size=(org_height, org_width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
                )
                org_frame = video_frames[ni]

                if org_frame.shape[-1] == 4:
                    if not (
                        np.all(org_frame[:, :, 3] == 255) or np.all(org_frame[:, :, 3] == 0)
                    ):
                        alpha = org_frame[:, :, 3]
                    oframe = org_frame[:, :, :3]
                else:
                    oframe = org_frame
                out_frame = self.image_processor.restorer.restore_img(oframe, face, affine_matrices[ni])

                if self.gfpgan:
                    fbbox = fbboxs[ni]
                    x1, y1, x2, y2 = fbbox
                    gan_face = self.face_enhance(out_frame[y1:y2, x1:x2].copy())
                    out_frame[y1:y2, x1:x2] = gan_face

                if alpha is not None:
                    out_frame = cv2.merge([
                        out_frame[:, :, 0],
                        out_frame[:, :, 1],
                        out_frame[:, :, 2],
                        alpha,
                    ])
                audio_data = audio_samples[ni*audio_chunk_size:(ni+1) *audio_chunk_size]
                audio_data = audio_data.numpy()
                if len(audio_data) < audio_chunk_size:
                    # 长度不够的时候, 补充静音
                    audio_data = np.pad(
                        audio_data,
                        (0, audio_chunk_size - len(audio_data)),
                        "constant",
                        constant_values=0,
                    )
                audio_data = np.reshape(
                    audio_data, (audio_channel, len(audio_data))
                )
                audio_data = audio_data.astype(np.float32)
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
        del video_frames
        del synced_video_frames
        del audio_samples
        del whisper_feature
        del whisper_chunks

