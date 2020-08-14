#!/bin/bash

# This script installs all necessary apt, llvm, and cmake dependencies required for any daemon/non-daemon x86/aarch64 build configuration

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root to enable package installation"
  exit 0
fi

install_apt_dependencies_prellvm() {
  echo "Installing pre-llvm apt dependencies"
  apt update 
  apt install -y \
    wget \
    lsb-release \
    apt-transport-https \
    gnupg2 \
    software-properties-common \
    curl \
    unzip \
    tar
}

install_llvm() {
  echo "Installing LLVM 9"
  wget https://apt.llvm.org/llvm.sh && \
    chmod u+x llvm.sh && \
    ./llvm.sh 9
  apt install -y clang-9 clang-tidy-9 clang-format-9
  ln -s /usr/bin/clang++-9 /usr/bin/clang++
  ln -s /usr/bin/clang-9 /usr/bin/clang
  ln -s /usr/bin/clang-tidy-9 /usr/bin/clang-tidy
  ln -s /usr/bin/clang-format-9 /usr/bin/clang-format
  rm llvm.sh
}

install_apt_dependencies_postllvm() {
  echo "Installing post-llvm apt dependencies"
  apt install -y \
    git \
    g++-7-aarch64-linux-gnu \
    gcc-7-arm-linux-gnueabi \
    libstdc++6-7-dbg-arm64-cross \
    zlib1g-dev
}

install_updated_cmake() {
  echo "Installing updated cmake"
  wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null 
  apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
  apt update
  apt install -y cmake
}

install_apt_dependencies_prellvm
install_llvm
install_apt_dependencies_postllvm
install_updated_cmake
