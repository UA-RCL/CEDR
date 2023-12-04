FROM ubuntu:18.04

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

COPY requirements.txt /
RUN pip3 install -r /requirements.txt && \
  rm /requirements.txt

WORKDIR /root/repository

#VOLUME /home/user/repository
