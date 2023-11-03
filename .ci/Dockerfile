FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/GMT
RUN apt-get update --quiet --yes && \
    apt-get install --quiet --yes --no-install-recommends \
        ca-certificates \
        curl \
        gnupg2 && \
    . /etc/os-release && \
    echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/deadsnakes.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 && \
    apt-get update --quiet --yes && \
    apt-get install --quiet --yes --no-install-recommends \
        git \
        openssh-client \
        libenchant-2-2 \
        python3.9 \
        python3.9-dev \
        python3.9-distutils \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3.11 \
        python3.11-dev \
        python3.11-distutils && \
    curl -fsSo /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.9 /tmp/get-pip.py && \
    python3.9 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 /tmp/get-pip.py && \
    python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 /tmp/get-pip.py && \
    python3.11 -m pip install --no-cache-dir --upgrade pip && \
    rm /tmp/get-pip.py && \
    python3 -m pip install --no-cache-dir nox==2023.4.22 && \
    rm -rf /var/cache/apt/lists
