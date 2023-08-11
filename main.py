import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import HTTPException
from fastapi import Query, UploadFile
import logging
from run_model import run_stable_diffusion_model
import os
import io

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
    "/generate_text2image",
    summary="Generate text to image",
    description="Generate text to image",
    tags=["generate_text2image"],
)
async def generate_text2image(prompt: str = Query(),
                              main_model: str = Query(default='stable_diffusion'),
                              device: str = Query(default='cpu'),
                              lora_model: UploadFile = None) -> JSONResponse:
    logger.info(f"Generate text to image with prompt: {prompt}")
    if lora_model:
        logger.info(f"lora_model: {lora_model.filename}")
        if not os.path.exists(f'lora_models/{lora_model.filename}'):
            with open(f'lora_models/{lora_model.filename}', 'wb') as f:
                f.write(lora_model.file.read())
    if main_model == 'stable_diffusion':
        image = run_stable_diffusion_model(prompt, f'lora_models/{lora_model.filename}', device)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = str(image_bytes.getvalue())
        return JSONResponse(content=image_bytes)
    else:
        raise HTTPException(status_code=400, detail="main_model must be 'stable_diffusion'")



def main() -> None:
    logger.info(f"Start fashion_api_backend service with host and port: ")
    uvicorn.run("main:app", log_level="debug", reload=True)


if __name__ == "__main__":
    main()
