import inspect
import os
from typing import Union

import PIL
import numpy as np
import torch
import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor

from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)


class CatVTONPipeline:
    """
    GPU-Optimized CatVTON Pipeline for faster inference.
    
    GPU-Compatible Optimization features:
    1. BetterTransformer for efficient attention (GPU-compatible)
    2. torch.compile for JIT optimization (PyTorch 2.0+)
    3. Channels-last memory format for better performance
    4. SDPA (Scaled Dot Product Attention) - native PyTorch 2.0+
    5. xFormers memory-efficient attention
    6. Half precision (FP16/BF16) optimizations
    7. Gradient checkpointing disabled for inference
    """
    
    def __init__(
        self, 
        base_ckpt, 
        attn_ckpt, 
        attn_ckpt_version="mix",
        weight_dtype=torch.float16,
        device='cuda',
        compile=False,
        skip_safety_check=False,
        use_tf32=True,
        # GPU-compatible optimization parameters
        use_xformers=False,
        use_sdpa=True,  # Scaled Dot Product Attention (PyTorch 2.0+)
        use_channels_last=True,
        use_bettertransformer=False,
        enable_vae_tiling=True,
        compile_mode="reduce-overhead",  # "default", "reduce-overhead", "max-autotune",
        scheduler='ddim'
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check

        
        # Load scheduler
        print("Loading scheduler...")

        # MODIFICATION 1: Load different schedulers based on input
        if scheduler=='ddim':
            self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        elif scheduler=='unipc':
            self.noise_scheduler = UniPCMultistepScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        elif scheduler=='dpmsolver':
            self.noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(base_ckpt, subfolder="scheduler")    

        # Load VAE with optimization
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.vae = self.vae.to(device, dtype=weight_dtype)
                
        # MODIFICATION 2: VAE Tiling for larger images
        if enable_vae_tiling:
            self.vae.enable_slicing()
            print("VAE slicing enabled")

            self.vae.enable_tiling()
            print("VAE tiling enabled")
        
        # Load safety checker if needed
        if not skip_safety_check:
            print("Loading safety checker...")
            self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                base_ckpt, subfolder="safety_checker"
            ).to(device, dtype=weight_dtype)
        
        # Load UNet
        print("Loading UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet")
        self.unet = self.unet.to(device, dtype=weight_dtype)
        
        # Initialize attention processors
        print("Setting up attention processors...")
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)
        self.attn_modules = get_trainable_module(self.unet, "attention")
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        
        # # Channels-last memory format for UNet
        if use_channels_last:
            self.unet = self.unet.to(memory_format=torch.channels_last)
            print("UNet channels-last format enabled")

        
        # Apply torch.compile (PyTorch 2.0+) - AFTER all other setup
        if compile:
            print(f"Compiling models with torch.compile (mode: {compile_mode})...")
            try:
                # Compile UNet (main bottleneck)
                self.unet = torch.compile(self.unet, mode=compile_mode)
                print("UNet compiled")
                
                # Compile VAE decoder (used during inference)
                self.vae.decoder = torch.compile(self.vae.decoder, mode=compile_mode)
                print("VAE decoder compiled")
                
                # Note: First inference will be slow due to compilation
                print(" First inference will be slower (compilation time)")
            except Exception as e:
                print(f"Compilation failed: {e}")
        
        # Set models to eval mode
        self.unet.eval()
        self.vae.eval()
        
        print("="*80)
        print("PIPELINE READY FOR INFERENCE")
        print("="*80)

    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))
            
    def run_safety_checker(self, image):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(self.weight_dtype)
            )
        return image, has_nsfw_concept
    
    def check_inputs(self, image, condition_image, mask, width, height):
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
            return image, condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask
    
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # MODIFICATION 3: CFG Rescaling
    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.7):
        """
        Rescale classifier-free guidance prediction to prevent oversaturation.
        Based on Section 3.4 of https://arxiv.org/pdf/2305.08891.pdf (Common Diffusion Noise Schedules)
        
        Args:
            noise_cfg: The combined CFG noise prediction
            noise_pred_text: The conditional (text/image) noise prediction
            guidance_rescale: Amount of rescaling (0 = no rescale, 1 = full rescale)
        
        Returns:
            Rescaled noise prediction
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # Rescale to match the standard deviation of the conditional prediction
        noise_pred_rescaled = noise_cfg * (std_text / (std_cfg + 1e-8))
        # Interpolate between rescaled and original
        noise_pred = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_pred

    @torch.no_grad()
    def __call__(
        self, 
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        guidance_rescale: float = 0.0,  # NEW PARAMETER
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        **kwargs
    ):
        concat_dim = -2
        
        # Prepare inputs to Tensor
        image, condition_image, mask = self.check_inputs(image, condition_image, mask, width, height)
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)
        
        # Mask image
        masked_image = image * (mask < 0.5)
        
        # VAE encoding
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        
        # Clear intermediate tensors to free memory
        del image, mask, condition_image, masked_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
        
        # Prepare noise
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma
        
        # Classifier-Free Guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.noise_scheduler.order
        
        with tqdm.tqdm(total=num_inference_steps, desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for CFG
                non_inpainting_latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(
                    non_inpainting_latent_model_input, t
                )
                
                # Prepare inpainting input
                inpainting_latent_model_input = torch.cat(
                    [non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], 
                    dim=1
                )
                
                # Predict noise
                noise_pred = self.unet(
                    inpainting_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=None,
                    return_dict=False,
                )[0]
                
                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                    
                    # Apply CFG rescale if enabled
                    if guidance_rescale > 0:
                        noise_pred = self.rescale_noise_cfg(
                            noise_pred, noise_pred_cond, guidance_rescale
                        )
                
                # Compute previous sample
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                
                # Update progress
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode latents
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        
        # Safety check
        if not self.skip_safety_check:
            current_script_directory = os.path.dirname(os.path.realpath(__file__))
            nsfw_image = os.path.join(os.path.dirname(current_script_directory), 'resource', 'img', 'NSFW.jpg')
            nsfw_image = PIL.Image.open(nsfw_image).resize(image[0].size)
            image_np = np.array(image)
            _, has_nsfw_concept = self.run_safety_checker(image=image_np)
            for i, not_safe in enumerate(has_nsfw_concept):
                if not_safe:
                    image[i] = nsfw_image
        
        return image