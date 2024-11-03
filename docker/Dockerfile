FROM ubuntu:24.04

WORKDIR /code

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y \
    build-essential \
    cmake \
    git-all \
    libomp-dev \
    libvtk9-qt-dev \
    nlohmann-json3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y  \
    libgtest-dev \
 && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=dialog

