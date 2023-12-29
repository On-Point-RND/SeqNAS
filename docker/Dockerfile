FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV DEBIAN_FRONTEND noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV _JAVA_OPTIONS="-Xmx20g"

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    zip \
    git \
    ninja-build \
    libglib2.0 \
    ca-certificates \
    wget \
    sudo \
    graphviz \
    libgraphviz-dev \
    mysql-server \
    default-libmysqlclient-dev \
    && service mysql start \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PySpark dependencies
RUN apt-get update && apt-get install -y \
    openjdk-8-jdk-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install other requirements
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -U \
    'jupyterlab' \
    'ipywidgets' \
    'seaborn' \
    'kaleido'

# Install SeqNAS
COPY pyproject.toml setup.py README.md /naslib_proj/
COPY src /naslib_proj/src
RUN pip install --no-cache-dir -U pip \
    && pip install /naslib_proj \
    && rm -rf /naslib_proj

WORKDIR /
