#!/bin/env bash
############################################################
# Help                                                     #
############################################################
Help()
{
    # Display Help
    echo "Build the image."
    echo "Syntax: ./docker/build.sh [-t TAG|-h]"
    echo
    echo "options:"
    echo "t     Tag of the image.       [demo-cnn]"
    echo "h     Print this Help."
}

while getopts ":t:h" arg; do
    case $arg in
        t)  TAG=$OPTARG;;
        h)  # display Help
            Help
            exit;;
        *)  # invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done

[ -z $TAG ] && TAG=demo-cnn

docker build \
    -t $TAG \
    -f docker/Dockerfile \
    $(pwd)
