# 1. Test setup:
# docker run -it --rm --gpus all nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
#
# If the above does not work, try adding the --privileged flag
# and changing the command to `sh -c 'ldconfig -v && nvidia-smi'`.
#
# 2. Start training:
# docker build -f  dreamerv3/Dockerfile -t img . && \
# docker run -it --rm --gpus all -v ~/logdir:/logdir img \
#   sh scripts/xvfb_run.sh python3 dreamerv3/train.py \
#   --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs dmc_vision --task dmc_walker_walk
#
# 3. See results:
# tensorboard --logdir ~/logdir

# System
FROM nvcr.io/nvidia/jax:24.04-py3
USER root

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cmake \
    curl \
    ffmpeg \
    freeglut3-dev \
    gcc \
    git \
    unzip \
    vim \
    wget \
    zip

# Install python packages
RUN pip install pip==24.0 setuptools==59.5.0 wheel==0.34.2
RUN pip install \
    gym==0.21.0 \
    autorom==0.4.2 \
    gym[accept-rom-license] \
    atari_py==0.2.9 \
    minedojo==0.1 \
    carla==0.9.15 \
    crafter==1.8.2

RUN pip install setuptools==69.5.1 wheel==0.43.0
RUN pip install \
    distro==1.9.0 \
    dotmap==1.3.30 \
    easydict==1.11 \
    gdown==4.7.1 \
    gradio==4.10.0 \
    h5py==3.10.0 \
    imageio==2.33.1 \
    importlib_resources==5.13.0 \
    ipdb==0.13.13 \
    kornia==0.6.5 \
    matplotlib==3.7.4 \
    moviepy==1.0.3 \
    opencv-python==4.8.1.78 \
    packaging==23.2 \
    Pillow==9.5.0 \
    protobuf==3.20.3 \
    pygame==2.5.2 \
    pyglet==2.0.10 \
    PyYAML==6.0.1 \
    av==12.0.0 \
    pyrender==0.1.45 \
    requests==2.31.0 \
    ruamel.yaml==0.17.31 \
    shapely==2.0.2 \
    sk-video==1.1.10 \
    tensorboard==2.14.0 \
    termcolor==2.4.0 \
    tqdm==4.66.1 \
    typing==3.7.4.3 \
    beartype==0.1.1 \
    omegaconf==2.3.0


RUN pip install setuptools==69.5.1 wheel==0.43.0
RUN pip install --upgrade pip

RUN pip install ale_py==0.8.1 autorom[accept-rom-license]
RUN pip install memory_maze==1.0.3

# Install MuJoCo
RUN mkdir /root/.mujoco/
RUN pip install cython
COPY mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
ENV MUJOCO_PY_MJKEY_PATH=/root/.mujoco/mjkey.txt
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco200
RUN cd /root/ && \
    wget https://www.roboti.us/download/mujoco200_linux.zip && \
    unzip mujoco200_linux.zip && \
    mv mujoco200_linux /root/.mujoco/mujoco200 && \
    pip install mujoco_py==2.0.2.8 && \
    rm mujoco200_linux.zip
RUN pip install "cython<3"

# Install metaworld
RUN pip install git+https://github.com/rlworkgroup/metaworld.git@v2.0.0

# Install dm_control
RUN pip install git+https://github.com/google-deepmind/dm_control.git@1.0.16
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_GL=egl

# Install mjrl
RUN pip install git+https://github.com/aravindr93/mjrl.git


# Agent
# RUN pip3 install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip3 install jaxlib
RUN pip install numpy==1.24.1
RUN pip install tensorflow_probability==0.24.0
RUN pip install optax==0.2.2
RUN pip install tensorflow-cpu
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# Google Cloud DNS cache (optional)
ENV GCS_RESOLVE_REFRESH_SECS=60
ENV GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
ENV GCS_METADATA_REQUEST_TIMEOUT_SECS=300
ENV GCS_READ_REQUEST_TIMEOUT_SECS=300
ENV GCS_WRITE_REQUEST_TIMEOUT_SECS=600

# Embodied
RUN pip3 install cloudpickle rich zmq msgpack