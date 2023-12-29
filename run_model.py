import os

from diffusers import (
    KandinskyImg2ImgPipeline,
    KandinskyInpaintPipeline,
    KandinskyPipeline,
    KandinskyPriorPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from fastapi import UploadFile
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


def run_stable_diffusion_model(
    prompt: str, lora_model_file: UploadFile | None, device: str
) -> Image.Image:
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

    if lora_model_file:
        pipe.load_lora_weights(f"lora_models/{lora_model_file.filename}")
    pipe.to(device)

    image = pipe(
        prompt, num_inference_steps=25, guidance_scale=7.5, width=512, height=720
    ).images[0]
    return image


def run_stable_diffusion_xl_model(
    prompt: str, lora_model_file: UploadFile | None, device: str
) -> Image.Image:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0"
    )
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    elif device == "mps":
        pipe.enable_attention_slicing()
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    if lora_model_file:
        pipe.load_lora_weights(f"lora_models/{lora_model_file.filename}")
    pipe.to(device)
    image = pipe(prompt, num_inference_steps=25, width=768, height=768).images[0]
    return image


def run_blip(image: Image.Image, text: str | None = None, device: str = "cpu") -> str:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    image_rgb = image.convert("RGB")
    if text:
        # conditional image captioning
        inputs = processor(image_rgb, text, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    else:
        # unconditional image captioning
        inputs = processor(image_rgb, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)


def run_stable_diffusion_inpaint_model(
    prompt: str, lora_model_file: UploadFile | None, device: str
) -> Image.Image:
    pass


def run_stable_diffusion_img2img_model(
    init_image: Image.Image,
    prompt: str,
    lora_model_file: UploadFile | None,
    device: str,
) -> Image.Image:
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0"
    )
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    elif device == "mps":
        pipe.enable_attention_slicing()
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    if lora_model_file:
        pipe.load_lora_weights(f"lora_models/{lora_model_file.filename}")
    pipe.to(device)
    init_image = init_image.resize((768, 512))
    image = pipe(
        prompt,
        image=init_image,
        num_inference_steps=10,
        strength=0.75,
        guidance_scale=7.5,
    ).images[0]
    return image


def run_kandinsky_model(
    prompt: str, lora_model_file: UploadFile | None, device: str
) -> Image.Image:
    pipe_prior = KandinskyPriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-1-prior"
    )
    if device == "cuda":
        pipe_prior.enable_xformers_memory_efficient_attention()
    elif device == "mps":
        pipe_prior.enable_attention_slicing()
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    pipe_prior.to(device)

    prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
    negative_prompt = "low quality, bad quality"

    image_emb = pipe_prior(
        prompt,
        guidance_scale=1.0,
        num_inference_steps=25,
        negative_prompt=negative_prompt,
    )

    images_emb = image_emb.image_embeds

    zero_image_emb = pipe_prior(
        negative_prompt,
        guidance_scale=1.0,
        num_inference_steps=25,
        negative_prompt=negative_prompt,
    )

    zero_images_emb = zero_image_emb.negative_image_embeds

    pipe = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1")
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    elif device == "mps":
        pipe.enable_attention_slicing()
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    pipe.to(device)

    images = pipe(
        prompt,
        image_embeds=images_emb,
        negative_image_embeds=zero_images_emb,
        num_images_per_prompt=2,
        height=768,
        width=768,
        num_inference_steps=25,
        guidance_scale=4.0,
    )
    return images.images[0]
