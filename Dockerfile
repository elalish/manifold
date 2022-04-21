FROM nvidia/cuda:11.0-devel-ubuntu20.04
LABEL Name=manifold Version=0.0.2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y install \
    cmake \
    libglm-dev \
    libassimp-dev \
    libcgal-dev
# RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cuda-drivers
COPY . /usr/src
WORKDIR /usr/src
RUN mkdir buildCPP && cd buildCPP && \
    cmake -DCMAKE_BUILD_TYPE=Release -DMANIFOLD_USE_CPP=ON .. && make
RUN mkdir buildCUDA && cd buildCUDA && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && make
CMD cd buildCPP/test && \
    ./manifold_test
