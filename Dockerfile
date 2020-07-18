FROM nvidia/cuda:9.2-devel-ubuntu16.04
LABEL Name=manifold Version=0.0.1
RUN apt-get -y update && apt-get -y install \
    wget \
    libssl-dev
RUN version=3.17 && build=3 && \
    mkdir ~/temp && cd ~/temp && \
    wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz && \
    tar -xzvf cmake-$version.$build.tar.gz && cd cmake-$version.$build/ && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install
RUN apt-get -y install \
    libglm-dev \
    libassimp-dev \
    libboost-graph-dev
# RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cuda-drivers
COPY . /usr/src
WORKDIR /usr/src
RUN mkdir buildOMP && cd buildOMP && \
    cmake -DCMAKE_BUILD_TYPE=Release -DMANIFOLD_USE_OMP=ON .. && make
RUN mkdir buildCUDA && cd buildCUDA && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && make
CMD cd buildOMP/test && \
    ./manifold_test