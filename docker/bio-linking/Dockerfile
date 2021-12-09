FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DOCKER_UID=1234 # <- change here
ARG DOCKER_USERNAME=experiment
ARG DOCKER_PASSWORD=passwd

RUN apt update && apt install -y less sudo
RUN apt install -y language-pack-ja && localedef -i ja_JP -c -f UTF-8 -A /usr/share/locale/locale.alias ja_JP.UTF-8 && ldconfig
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP.UTF-8

RUN apt install -y python3 python3-pip python3-protobuf && ldconfig && \
    pip3 install -U pip && \
    pip3 install tqdm numpy pandas && \
    pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install torch_optimizer sentencepiece transformers==4.5.1 && \
    pip3 install boto3 requests regex && \
    ldconfig
ENV CUDA_VISIBLE_DEVICES 0

RUN useradd --uid ${DOCKER_UID} -m ${DOCKER_USERNAME} && usermod -aG sudo ${DOCKER_USERNAME} && echo "${DOCKER_USERNAME}:${DOCKER_PASSWORD}" | chpasswd && su ${DOCKER_USERNAME} && exit
USER ${DOCKER_USERNAME}
WORKDIR /home/${DOCKER_USERNAME}/

CMD ["bash"]

