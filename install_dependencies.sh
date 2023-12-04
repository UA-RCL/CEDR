#!/bin/bash

# This script installs all necessary apt, llvm, and cmake dependencies required for any daemon/non-daemon x86/aarch64 build configuration
# Lots of neat version checking tricks stolen from llvm.sh

# -e: exit early on any failure
# -u: treat unset shell variables as an error when substituting
# -x: echo commands as they're executed
set -eux

# Funny enough we have the same required binaries as LLVM
needed_binaries=(lsb_release wget add-apt-repository)
missing_binaries=()
for binary in "${needed_binaries[@]}"; do
    if ! which $binary &>/dev/null ; then
        missing_binaries+=($binary)
    fi
done
if [[ ${#missing_binaries[@]} -gt 0 ]] ; then
    echo "You are missing some tools this script requires: ${missing_binaries[@]}"
    echo "I'm going to install them now..."
    apt update
    DEBIAN_FRONTEND=noninteractive apt install -y lsb-release wget software-properties-common
fi

DISTRO=$(lsb_release -is)
VERSION=$(lsb_release -sr)
DIST_VERSION="${DISTRO}_${VERSION}"
DIST_CODENAME=$(lsb_release -sc)

GCC_VERSION=7
GSL_VERSION=23

case "${DIST_VERSION}" in
  Ubuntu_16.04 ) GCC_VERSION=5; GSL_VERSION=23 ;;
  Ubuntu_18.04 ) GCC_VERSION=7; GSL_VERSION=23 ;;
  Ubuntu_20.04 ) GCC_VERSION=10; GSL_VERSION=23 ;;
  Ubuntu_22.04 ) GCC_VERSION=11; GSL_VERSION=27 ;;
  * )
    echo "[WARNING] - this installation script has not been tested with this particular distro/version (${DIST_VERSION})"
    # https://stackoverflow.com/a/1885534
    read -p "Do you want to continue? (y/n) " -n 1 -r REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
      exit 0
    fi
  ;;
esac

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root to enable package installation"
  exit 1
fi

install_apt_development_dependencies() {
  echo "Installing general development dependencies"
  echo "Using GCC version: ${GCC_VERSION}"

  apt update
  DEBIAN_FRONTEND=noninteractive \
    apt install -y \
    apt-transport-https \
    gnupg2 \
    curl \
    unzip \
    build-essential \
    libpapi-dev \
    tar \
    vim \
    gdb \
    git \
    g++-${GCC_VERSION} \
    g++-${GCC_VERSION}-aarch64-linux-gnu \
    gcc-${GCC_VERSION}-arm-linux-gnueabi \
    libstdc++6-${GCC_VERSION}-dbg-arm64-cross \
    zlib1g-dev \
    python3-tk \
    python3-pip

  update-alternatives --install /usr/bin/aarch64-linux-gnu-g++ aarch64-linux-gnu-g++ /usr/bin/aarch64-linux-gnu-g++-${GCC_VERSION} 10
  update-alternatives --install /usr/bin/aarch64-linux-gnu-gcc aarch64-linux-gnu-gcc /usr/bin/aarch64-linux-gnu-gcc-${GCC_VERSION} 10
}

install_updated_cmake() {
  echo "Installing updated cmake"
  wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null 
  chmod 644 /etc/apt/trusted.gpg.d/kitware.gpg
  apt-add-repository "deb https://apt.kitware.com/ubuntu/ ${DIST_CODENAME} main"
  apt update
  DEBIAN_FRONTEND=noninteractive apt install -y cmake
}

install_libgsl() {
  echo "Installing libgsl-dev"
  DEBIAN_FRONTEND=noninteractive apt install -y libgsl-dev libgsl${GSL_VERSION} libgslcblas0
}

install_apt_development_dependencies
install_updated_cmake
install_libgsl
