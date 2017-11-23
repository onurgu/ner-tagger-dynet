FROM ubuntu:16.04

# Build dependencies
RUN echo "deb-src http://archive.ubuntu.com/ubuntu/ xenial main restricted" >> /etc/apt/sources.list && apt-get update && apt-get install -y wget git mercurial libgoogle-perftools-dev libsparsehash-dev libboost-all-dev autoconf libtool
RUN apt-get build-dep -y cmake

# CMake 3.8.2
RUN cd /opt && \
        wget "https://cmake.org/files/v3.8/cmake-3.8.2.tar.gz" && \
        tar xf cmake-3.8.2.tar.gz && \
        cd cmake-3.8.2 && \
        ./configure && \
        make -j2 install

# Eigen, version 3.3.4
ENV EIGEN_VERSION 3.3.4
RUN cd /opt && \
        hg clone https://bitbucket.org/eigen/eigen/ && \
        cd eigen && \
        hg update -r ${EIGEN_VERSION}

# DyNet, version 4234759
ENV DYNET_VERSION v2.0
RUN cd /opt && \
        git clone https://github.com/clab/dynet.git && \
        cd dynet && \
        git checkout ${DYNET_VERSION} && \
        mkdir build && \
        cd build && \
        cmake .. -DEIGEN3_INCLUDE_DIR=/opt/eigen && \
        make -j2 install

RUN mkdir /opt/ner-tagger-dynet

WORKDIR /opt/ner-tagger-dynet

COPY *.py /opt/ner-tagger-dynet/

RUN mkdir dataset

COPY evaluation/conlleval evaluation/

RUN mkdir models/

