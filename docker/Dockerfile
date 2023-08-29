# start with cupy docker https://github.com/cupy/cupy/blob/master/docker/python3/Dockerfile
# but change the image tag https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
#FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
# for error E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Release' does not have a Release file.
# on some machines https://github.com/NVIDIA/nvidia-docker/issues/1009
RUN rm /etc/apt/sources.list.d/cuda.list 
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir cupy-cuda11x
#RUN pip3 install --no-cache-dir cupy-cuda116
# use MKL instead of openBLAS since eigh is quite a bit faster on cpu
RUN pip3 install mkl mkl_fft mkl-service mkl_random mkl_umath tbb4py
RUN pip3 install -i https://pypi.anaconda.org/intel/simple numpy==1.22.3

RUN pip3 install h5py \
                 ipympl \
                 jupyterlab \
                 jupytext \
                 matplotlib \
                 pydicom \
                 sigpy \
                 scikit-image \
                 opencv-python

# libraries for opencv, noninteractive gets rid of some interactive prompts...
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6

# Use old version of ipywidgets for VS Code functionality
RUN pip3 install ipywidgets==7.7.1

USER root

WORKDIR /root
	
# add a user avoiding interactive questions https://stackoverflow.com/questions/27701930/how-to-add-users-to-docker-container
RUN adduser --disabled-password --gecos '' sdkuser
RUN mkdir /home/sdkuser/workspace
	
# install and link newer version of openssl
RUN curl -S --insecure https://www.openssl.org/source/old/1.0.1/openssl-1.0.1e.tar.gz | tar xvz -C . && \
    cd openssl-1.0.1e && \
    ./config shared && \
    make && \
    make install_sw && \
    cd .. && \
    rm -rf openssl-1.0.1e && \
    ldconfig
RUN ln -s /usr/local/ssl/lib/libssl.so.1.0.0 /usr/local/ssl/lib/libssl.so.10
RUN ln -s /usr/local/ssl/lib/libcrypto.so.1.0.0 /usr/local/ssl/lib/libcrypto.so.10
RUN cp /usr/local/ssl/lib/libssl.so.10 /lib/x86_64-linux-gnu/libssl.so.10
RUN cp /usr/local/ssl/lib/libcrypto.so.10 /lib/x86_64-linux-gnu/libcrypto.so.10

# setup environment
RUN apt-get update && apt-get -qq -y install curl
RUN apt-get install -y vim wget

# switch user
USER sdkuser

# open a command line
CMD /bin/bash

