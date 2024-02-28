FROM nvidia/cuda:12.2.0-devel-ubuntu20.04 AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN useradd --create-home genius

RUN apt-get update \
 && apt-get install -y software-properties-common build-essential curl wget vim git libpq-dev pkg-config \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y python3.10 python3.10-dev python3.10-distutils \
 && apt-get clean
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
 && python3.10 get-pip.py

RUN apt-get update && apt-get install -y git && apt-get clean

RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir jupyterlab
RUN pip install --no-cache-dir transformers
RUN pip install --no-cache-dir datasets
RUN pip install --no-cache-dir diffusers
RUN pip install --no-cache-dir --upgrade geniusrise

ENV AWS_DEFAULT_REGION=ap-south-1
ENV AWS_SECRET_ACCESS_KEY=
ENV AWS_ACCESS_KEY_ID=
ENV HUGGINGFACE_ACCESS_TOKEN=
ENV GENIUS=/home/genius/.local/bin/genius

COPY --chown=genius:genius . /app/

RUN pip3.10 install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt
RUN pip install --no-cache-dir numpy==1.26.3
USER genius

CMD ["genius", "--help"]
