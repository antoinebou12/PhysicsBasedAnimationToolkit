# Use NVIDIA CUDA base image with development tools
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV VCPKG_ROOT=/opt/vcpkg
ENV PATH="$VCPKG_ROOT:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libtool \
    autoconf \
    unzip \
    wget \
    curl \
    zip \
    tar \
    python3 \
    python3-pip \
    clang-format \
    # Dependencies for PBAT
    libeigen3-dev \
    doctest-dev \
    libfmt-dev \
    librange-v3-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Install vcpkg
RUN git clone https://github.com/microsoft/vcpkg.git $VCPKG_ROOT
RUN $VCPKG_ROOT/bootstrap-vcpkg.sh -disableMetrics

# Set VCPKG environment variables
ENV VCPKG_DEFAULT_TRIPLET=x64-linux
ENV VCPKG_FEATURE_FLAGS=manifests

# Clone the PBAT repository
WORKDIR /opt
RUN git clone https://github.com/Q-Minh/PhysicsBasedAnimationToolkit.git
WORKDIR /opt/PhysicsBasedAnimationToolkit

# Install PBAT dependencies via vcpkg
RUN $VCPKG_ROOT/vcpkg install --triplet=$VCPKG_DEFAULT_TRIPLET

# Build PBAT with Python bindings and CUDA support
RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPBAT_BUILD_PYTHON_BINDINGS=ON \
    -DPBAT_USE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=all \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_ROOT=$VCPKG_ROOT

RUN cmake --build build --config Release --parallel $(nproc)

# Install PBAT
RUN cmake --install build --config Release

# Install PBAT Python package
RUN pip3 install .

# Set the working directory
WORKDIR /workspace
