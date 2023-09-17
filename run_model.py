from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from fastapi import UploadFile


def run_stable_diffusion_model(prompt: str, lora_model_file: UploadFile | None, device: str) -> Image.Image:
    pipe = StableDiffusionPipeline.from_single_file('main_models/sd_1_5.safetensors')

    if lora_model_file:
        pipe.load_lora_weights(f'lora_models/{lora_model_file.filename}')
    pipe.to(device)

    image = pipe(prompt,
                 num_inference_steps=25,
                 guidance_scale=7.5,
                 width=512,
                 height=720).images[0]
    return image


def run_stable_diffusion_xl_model(prompt: str, lora_model_file: UploadFile | None, device: str) -> Image.Image:
    vae = AutoencoderKL.from_single_file('main_models/sd_xl_vae.safetensors')
    pipe = StableDiffusionPipeline.from_single_file('main_models/sd_xl.safetensors', vae=vae)

    if lora_model_file:
        pipe.load_lora_weights(f'lora_models/{lora_model_file.filename}')
    pipe.to(device)
    image = pipe(prompt,
                 num_inference_steps=25,
                 width=768,
                 height=768).images[0]
    return image


def run_blip(image: Image.Image, text: str | None = None, device: str = 'cpu') -> str:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    image_rgb = image.convert('RGB')
    if text:
        # conditional image captioning
        inputs = processor(image_rgb, text, return_tensors="pt")

        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    else:
        # unconditional image captioning
        inputs = processor(image_rgb, return_tensors="pt")

        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
