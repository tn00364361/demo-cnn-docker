#!/bin/env bash

docker build \
    -t demo-cnn \
    -f docker/Dockerfile \
    .
