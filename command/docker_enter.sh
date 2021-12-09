#!/bin/bash
DOCKER_USER_NAME=experiment
REPO_ROOT_DIR=`pwd`
DOCKER_IMAGE_NAME=bio-linking
CUDA_VDEVICE="CUDA_VISIBLE_DEVICES=0"


docker run -it --rm --gpus all \
    --env ${CUDA_VDEVICE} \
    --env TOKENIZERS_PARALLELISM=false \
    -v ${REPO_ROOT_DIR}:${REPO_ROOT_DIR} \
    -v ${REPO_ROOT_DIR}/data/dockercache:/home/${DOCKER_USER_NAME}/.cache \
    --workdir ${REPO_ROOT_DIR} \
    ${DOCKER_IMAGE_NAME}
