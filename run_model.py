from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image


def run_stable_diffusion_model(prompt: str, lora_model_path: str | None, device: str) -> Image.Image:
    pipe = StableDiffusionPipeline.from_single_file('main_models/dreamshaper_8.safetensors')
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if lora_model_path:
        pipe.load_lora_weights(lora_model_path)
    pipe.to(device)

    image = pipe(prompt,
                 num_inference_steps=25,
                 guidance_scale=7.5,
                 width=512,
                 height=720).images[0]
    return image
