FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED True

WORKDIR /app

COPY ./stable-diffusion /app/stable-diffusion
COPY ./requirements.txt /app/requirements.txt
COPY ./main.py /app/main.py
RUN apt-get update && apt-get install -y gcc
RUN apt-get install git -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install cuda-drivers
## Installing the list of resolved dependencies from requirements.txt
RUN pip install -r /app/requirements.txt
#
## Run the service
CMD echo "Starting the service..."
CMD python stable-diffusion/scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms --outdir /app/output --config stable-diffusion/configs/stable-diffusion/v1-inference.yaml --ckpt stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt