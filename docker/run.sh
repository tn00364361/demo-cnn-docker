#!/usr/bin/env bash
############################################################
# Help                                                     #
############################################################
Help()
{
    # Display Help
    echo "Launch a container."
    echo "Syntax: ./docker/run.sh [-option VALUE|-h]"
    echo
    echo "options:"
    echo "d     Specify the GPU ID(s).      [0]"
    echo "n     Name of the container       [demo-cnn-\$(openssl rand -hex 4)]"
    echo "h     Print this Help."
}

############################################################
# Default values                                           #
############################################################
DEVICES=0
CONTAINER_NAME=demo-cnn-$(openssl rand -hex 4)

while getopts ":d:n:h" arg; do
    case $arg in
        d)  DEVICES=$OPTARG;;
        n)  CONTAINER_NAME=$OPTARG;;
        h)  # display Help
            Help
            exit;;
        *)  # invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done

############################################################
# Main program                                             #
############################################################
USER_ID=$(id -u)
GROUP_ID=$(id -g)
PASSWD_FILE=$(mktemp) && echo $(getent passwd $USER_ID) > $PASSWD_FILE
GROUP_FILE=$(mktemp) && echo $(getent group $GROUP_ID) > $GROUP_FILE

xhost +local:

docker run -it --rm \
    --gpus '"device='$DEVICES'"' \
    --name $CONTAINER_NAME \
    --hostname $(hostname) \
    --ipc host \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e HOME \
    -e XDG_RUNTIME_DIR=/run/user/$USER_ID \
    -u $USER_ID:$GROUP_ID \
    -v /run/user/$USER_ID:/run/user/$USER_ID \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /usr/share/zoneinfo:/usr/share/zoneinfo:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PASSWD_FILE:/etc/passwd:ro \
    -v $GROUP_FILE:/etc/group:ro \
    -v $(pwd)/docker/home:$HOME \
    -v $(pwd):/my_workspace \
    -w /my_workspace \
    demo-cnn

xhost -local:

rm -f $PASSWD_FILE $GROUP_FILE
