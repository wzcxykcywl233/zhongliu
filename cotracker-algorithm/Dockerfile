FROM --platform=linux/amd64 pytorch/pytorch
# Use an appropriate base image for your algorithm.
# As an example, we use the official pytorch image.

# In reality this baseline algorithm only needs numpy so we could use a smaller image:
#FROM --platform=linux/amd64 python:3.11-slim


# Ensure that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV CONDA_OVERRIDE_CUDA=12.4

RUN apt-get update && apt-get install -y \
    curl \
    git

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

ENV TORCH_HOME=/opt/app/torch/

RUN curl -fsSL https://pixi.sh/install.sh | bash
RUN . ~/.bashrc

COPY --chown=user:user . /opt/app/

# RUN git submodule init
# RUN git submodule update

RUN ~/.pixi/bin/pixi install -e cuda

# The evaluator starts the container with --network none. Calling `pixi run`
# here can revalidate local PyPI packages and attempt a network request even
# though the environment was fully installed during the image build.
ENTRYPOINT ["/opt/app/.pixi/envs/cuda/bin/python", "inference.py"]
