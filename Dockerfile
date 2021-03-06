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

COPY install_dependencies.sh /
RUN chmod u+x /install_dependencies.sh && \
  /install_dependencies.sh && \
  rm /install_dependencies.sh
WORKDIR /root/repository

#VOLUME /home/user/repository
