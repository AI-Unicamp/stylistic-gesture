FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y wget git nano ffmpeg

RUN conda --version

WORKDIR /root
COPY environment.yml /root

RUN conda install tqdm -f
RUN conda update conda
RUN conda install pip
RUN conda --version
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "stylistic-env", "/bin/bash", "-c"]
RUN python -m spacy download en_core_web_sm
RUN pip install blobfile
RUN pip install PyYAML
RUN pip install librosa
RUN pip install python_speech_features
RUN pip install einops
RUN pip install wandb
