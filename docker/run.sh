#!/usr/bin/env bash

ROOTFS=$(pwd)/docker/rootfs
[ ! -f $ROOTFS/etc/passwd ] && echo $(getent passwd $(id -un)) > $ROOTFS/etc/passwd
[ ! -f $ROOTFS/etc/group ] && echo $(getent group $(id -un)) > $ROOTFS/etc/group

[ -z $1 ] && GPU="all" || GPU=$1

xhost +local:

docker run -it --rm \
    --gpus '"device='$GPU'"' \
    --name demo-cnn-$(openssl rand -hex 4) \
    --hostname $(hostname) \
    --shm-size 4g \
    --ipc host \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e HOME \
    -e XDG_RUNTIME_DIR=/run/user/$(id -u) \
    -u $(id -u):$(id -g) \
    -v /run/user/$(id -u):/run/user/$(id -u) \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $ROOTFS/etc/passwd:/etc/passwd:ro \
    -v $ROOTFS/etc/group:/etc/group:ro \
    -v $ROOTFS/home/user:$HOME \
    -v $(pwd):/my_workspace \
    -w /my_workspace \
    demo-cnn

xhost -local:
