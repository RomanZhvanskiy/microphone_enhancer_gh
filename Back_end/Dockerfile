FROM python:3.10.6-slim-bullseye

WORKDIR /prod/Back_end

# upgrade pip
RUN pip install --upgrade pip

# install requirements
COPY requirements_prod.txt requirements_prod.txt
RUN pip install -r requirements_prod.txt

# install our package
COPY api api
COPY audio_preprocessing audio_preprocessing
COPY hugging_models hugging_models
COPY image_metrics image_metrics
COPY interface interface
COPY pretrained_models pretrained_models
COPY ml_logic ml_logic
COPY params.py params.py
COPY setup.py setup.py
RUN pip install .

# create needed directories
RUN mkdir -p ../Data/raw_data
RUN mkdir -p ../Data/audio_data/audio_in
RUN mkdir ../Data/audio_data/audio_out

# set in a docker variable
ENV AM_I_IN_A_DOCKER_CONTAINER Yes

# run api
CMD uvicorn api.enhancer_api_connected_to_backend2:app --host 0.0.0.0 --port $PORT
