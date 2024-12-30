FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install \
    python3-pip \
    xvfb \
    ffmpeg \
    git \
    build-essential \
    python-opengl
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get update && apt-get install -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    git \
    libpq-dev \
    protobuf-compiler \
    gnupg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# Install pyenv
RUN curl https://pyenv.run | bash

# Set environment variables for pyenv
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/bin:$PATH

# Initialize pyenv
RUN echo 'eval "$(pyenv init --path)"' >> /root/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> /root/.bashrc
RUN /bin/bash -c "source /root/.bashrc"

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH=/root/.local/bin:$PATH

# install mujoco_py
RUN apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    swig

# Create a working directory
RUN git clone https://github.com/goncamateus/dylam.git /dylam
WORKDIR /dylam
RUN pyenv install 3.10.15
RUN pyenv local 3.10.15
RUN poetry env use /root/.pyenv/shims/python
RUN poetry install

RUN cp entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]