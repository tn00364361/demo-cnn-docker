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
    echo "t     Tag of the image.       [demo-cnn:latest]"
    echo "h     Print this Help."
}

############################################################
# Default values                                           #
############################################################
TAG=demo-cnn:latest

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

docker build \
    -t $TAG \
    -f docker/Dockerfile \
    $(pwd)
