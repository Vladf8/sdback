import base64
import io
import logging
import os

import uvicorn
from fastapi import Body, FastAPI, Query, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from run_model import (
    run_blip,
    run_kandinsky_model,
    run_stable_diffusion_img2img_model,
    run_stable_diffusion_model,
    run_stable_diffusion_xl_model,
)

app = FastAPI(title="SD API")
logger = logging.RootLogger(logging.INFO)


@app.get(
    "/",
    summary="Index page",
    response_class=HTMLResponse,
    description="Base index page - just links to docs/schemas",
    include_in_schema=False,
)
async def root() -> str:
    """Index page - show a link to Swagger for the API"""
    logger.info("Get index page")
    return """<a href="/docs">Swagger documentation</a>"""


@app.post(
    "/text2image",
    summary="Generate text to image",
    description="Generate text to image",
    tags=["generate_text2image"],
)
async def generate_text2image(
    prompt: str = Query(),
    main_model: str = Query(default="stable_diffusion"),
    device: str = Query(default="cpu"),
    lora_model: UploadFile = None,
) -> Response:
    logger.info(f"Generate text to image with prompt: {prompt}")
    if lora_model:
        logger.info(f"lora_model: {lora_model.filename}")
        if not os.path.exists(f"lora_models/{lora_model.filename}"):
            with open(f"lora_models/{lora_model.filename}", "wb") as f:
                f.write(lora_model.file.read())
    if main_model == "stable_diffusion":
        image = run_stable_diffusion_model(prompt, lora_model, device)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()
        return Response(content=image_bytes, media_type="image/png")
    elif main_model == "stable_diffusion_xl":
        image = run_stable_diffusion_xl_model(prompt, lora_model, device)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()
        return Response(content=image_bytes, media_type="image/png")
    elif main_model == "kandinsky":
        image = run_kandinsky_model(prompt, lora_model, device)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()
        return Response(content=image_bytes, media_type="image/png")
    else:
        raise HTTPException(
            status_code=400, detail="main_model must be 'stable_diffusion'"
        )


@app.post(
    "/img2text",
    summary="Generate image to text",
    description="Generate image to text",
    tags=["generate_img2text"],
)
async def generate_img2text(
    image: str = Body(),
    main_model: str = Query(default="blip"),
    device: str = Query(default="cpu"),
    conditional_text: str = Query(default=""),
) -> str:
    # logger.info(f"Generate image to text with image: {image.filename}")
    if main_model == "blip":
        image = io.BytesIO(base64.b64decode(image))
        image = Image.open(image)
        text = run_blip(image, conditional_text, device)
        return text


@app.post(
    "/img2img",
    summary="Generate image to image",
    description="Generate image to image",
    tags=["generate_img2img"],
)
async def generate_img2img(
    init_image: UploadFile = None,
    prompt: str = Query(),
    main_model: str = Query(default="stable_diffusion"),
    device: str = Query(default="cpu"),
    lora_model: UploadFile = None,
) -> Response:
    logger.info(f"Generate image to image with prompt: {prompt}")
    if lora_model:
        logger.info(f"lora_model: {lora_model.filename}")
        if not os.path.exists(f"lora_models/{lora_model.filename}"):
            with open(f"lora_models/{lora_model.filename}", "wb") as f:
                f.write(lora_model.file.read())
    if main_model == "stable_diffusion":
        init_image = io.BytesIO(init_image.file.read())
        init_image = Image.open(init_image)
        image = run_stable_diffusion_img2img_model(
            init_image, prompt, lora_model, device
        )
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()
        return Response(content=image_bytes, media_type="image/png")


# def main() -> None:
#     logger.info(f"Start fashion_api_backend service with host and port: ")
#     uvicorn.run("main:app", log_level="debug", reload=True, port=8000)
#
#
# if __name__ == "__main__":
#     main()
