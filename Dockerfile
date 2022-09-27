FROM ubuntu:20.04

# Make files built with this have the permissions of the host user instead of root (since sudo runs the container)
# https://vsupalov.com/docker-shared-permissions/
#ARG USERID
#ARG GROUPID
#RUN addgroup --gid $GROUPID user
#RUN adduser --disabled-password --gecos '' --uid $USERID --gid $GROUPID user
#USER user

#WORKDIR /home/user
WORKDIR /root

# Baseline stuff that would basically be on any actual user's machine but not many barebones containers
RUN apt update && apt install -y software-properties-common lsb-release wget
COPY install_dependencies.sh /
RUN chmod u+x /install_dependencies.sh && \
  /install_dependencies.sh && \
  rm /install_dependencies.sh

RUN add-apt-repository ppa:gnuradio/gnuradio-releases -y && apt update && DEBIAN_FRONTEND=noninteractive TZ=America/Phoenix apt install -y gnuradio gnuradio-dev python3-gi gobject-introspection gir1.2-gtk-3.0
RUN apt install language-pack-en-base -y

RUN apt install -y build-essential cmake libfftw3-dev libmbedtls-dev libboost-program-options-dev libconfig++-dev libsctp-dev

RUN apt install -y cython3 libczmq-dev libspdlog-dev

WORKDIR /root/repository

#VOLUME /home/user/repository
