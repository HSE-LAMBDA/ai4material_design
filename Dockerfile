FROM nvcr.io/nvidia/tensorflow:21.08-tf2-py3

RUN apt-get update
ENV CUDA=9.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    git

WORKDIR /ai4material_design

COPY . /ai4material_design

RUN pip install poetry==1.1.8
RUN poetry export --without-hashes -o requirements.txt && pip install -r requirements.txt
# RUN python megnet_graphs_train.py
