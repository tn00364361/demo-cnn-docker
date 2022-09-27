#!/usr/bin/env bash
############################################################
# Help                                                     #
############################################################
Help()
{
    # Display Help
    echo "Launch a container."
    echo "Syntax: ./docker/run.sh [-g DEVICES|-s SHM_SIZE|-i IPC_MODE|-n NAME|-h]"
    echo
    echo "options:"
    echo "g     Specify the GPU ID(s).      [0]"
    echo "s     SHM size.                   [4g]"
    echo "n     Name of the container       [demo-cnn-\$(openssl rand -hex 4)]"
    echo "i     IPC mode.                   [host]"
    echo "h     Print this Help."
}

while getopts ":g:s:i:n:h" arg; do
    case $arg in
        g)  GPU=$OPTARG;;
        s)  SHM=$OPTARG;;
        i)  IPC=$OPTARG;;
        n)  NAME=$OPTARG;;
        h)  # display Help
            Help
            exit;;
        *)  # invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done

############################################################
# Default values                                           #
############################################################
[ -z $GPU ] && GPU=0
[ -z $SHM ] && SHM=4g
[ -z $IPC ] && IPC=host
[ -z $NAME ] && NAME=demo-cnn-$(openssl rand -hex 4)

############################################################
# Main program                                             #
############################################################
ROOTFS=$(pwd)/docker/rootfs
[ ! -f $ROOTFS/etc/passwd ] && echo $(getent passwd $(id -un)) > $ROOTFS/etc/passwd
[ ! -f $ROOTFS/etc/group ] && echo $(getent group $(id -un)) > $ROOTFS/etc/group

xhost +local:

docker run -it --rm \
    --gpus '"device='$GPU'"' \
    --name $NAME \
    --hostname $(hostname) \
    --shm-size $SHM \
    --ipc $IPC \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e HOME \
    -e XDG_RUNTIME_DIR=/run/user/$(id -u) \
    -u $(id -u):$(id -g) \
    -v /run/user/$(id -u):/run/user/$(id -u) \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /usr/share/zoneinfo:/usr/share/zoneinfo:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $ROOTFS/etc/passwd:/etc/passwd:ro \
    -v $ROOTFS/etc/group:/etc/group:ro \
    -v $ROOTFS/home/user:$HOME \
    -v $(pwd):/my_workspace \
    -w /my_workspace \
    demo-cnn

xhost -local:
