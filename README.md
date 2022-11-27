# Demo: Train a small neural network in a Docker container

Docker enables portability and reproducibility of a codebase. This repository serves as an example for training neural networks in Docker containers, as well as running GUI applications.

Notes:

- Instead of developing the ultimate neural network, the focus of this repository is to show how one can train a neural network in Docker containers. In other words, contents in `train.py` and `utils.py` are out of scope.

- Although this repository only uses Python, one can extend the idea to arbitrary languages/tools.

- In addition to [Docker Hub](https://hub.docker.com/), [NGC](https://catalog.ngc.nvidia.com/containers) is also a popular place (i.e. registry) for pulling prebuilt images. One can also host private registries using [Red Hat Quay](https://quay.io/) or [Google Cloud Container Registry](https://cloud.google.com/container-registry).

- Any feedback is welcome! Feel free to start a [discussion](https://github.com/tn00364361/demo-cnn-docker/discussions).

## Prerequisites

### Hardware

- `amd64` CPU(s)
- NVIDIA GPU(s) with Architecture >= Kepler (or compute capability 3.0)

### Software

- Ubuntu 18.04, 20.04, or 22.04 (either bare metal or WSL2)
- NVIDIA driver >= 450
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- git


## Instructions

1. Clone the repository

    ```bash
    $ git clone https://github.com/tn00364361/demo-cnn-docker.git
    $ cd demo-cnn-docker
    ```

2. Build the Docker image:

    ```bash
    $ ./docker/build.sh
    ```

    This step can take a couple of minutes, depending on your internet connection. Once it finishes, run `docker image ls` to verify the existence of the image.

3. Launch a container

    ```bash
    $ ./docker/run.sh           # use GPU0 only (default)
    $ ./docker/run.sh -d all    # use all GPU(s)
    $ ./docker/run.sh -d 2,3    # use GPU2 and GPU3
    ```

    See [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration) for valid values for GPU ID(s).

    The username and hostname should become yellow and the working directory should be `/my_workspace`, indicating that you are now in a container. Run `ll` to verify that you can see the host files.

    In addition, run `nvidia-smi` in the container to veirfy that the GPU is visible in the container.

## Details


- Differences between images and containers

    An image can be seen as a snapshot of an environment. It encapsulates all dependencies, including software packages and environmental variables, in a static state. On can pull prebuilt images from [Docker Hub](https://hub.docker.com/). For example:

    ```bash
    $ docker pull ubuntu:22.04
    ```

    By contrast, a container is (usually) interactive and can be launched given the name of an image. For example:

    ```bash
    $ docker run -it --rm ubuntu:22.04
    ```

    A more sophisticated example can be found in the [run script](docker/run.sh) which will be explained later.

- Running as the `root` user or using `sudo` in containers

    **TL;DR: Don't**.

    Once the image is built, one should launch containers using the `-u` flag, as shown in the [run script](docker/run.sh). This allows the written files to have the correct ownership, instead of to be owned by the `root` user.

    If you want to install packages or change system cofiguration files, do them in the Dockerfile.

- `docker/home`

    This is the home directory to be mounted to a container. Mounting it allows the user in the container to keep files like Bash/Python history and some common configuration files.

- [`docker/Dockerfile`](docker/Dockerfile)

    This file containes instructions to build the desired image. Starting from a base image, environmental variables are set, and packages are installed.

    When developing custom/new algorithms, one **almost always** wants to write their own Dockerfile instead of using a prebuilt image from Dockerh Hub. A typical pipeline may look like this:

    1. Figure out dependencies of the project.
    2. Write a [Dockerfile](docker/Dockerfile) with all dependencies.
    3. [Build](docker/build.sh) the image.
    4. [Launch](docker/run.sh) a container.
    5. Develop and test your algorithm in the container.
    6. If new dependency appears during development, go back to 1 and reiterate.

    In addition, it is a good pratice to explicitly specify the software versions. For example, when providing a [`requirements.txt`](requirements.txt) for PIP, one should prefer `==` or `>=X,<Y` over `>=` or no versioning at all.

    Another thing to pay attention to is that when cloning a thirdparty repository as a dependency, one should checkout to a particular tag or commit in the Dockerfile.

- [`docker/build.sh`](docker/build.sh)

    This script builds the image with a predefined tag (i.e. name of the image). See the [official documentation](https://docs.docker.com/engine/reference/commandline/build/) for more details.

- [`docker/run.sh`](docker/run.sh)

    This script launchs a container from the built image. In short, this script handles GUI- and user-related stuff, in addition to accessing GPU(s) for training neural networks.

    - GUI (requires `x11-xserver-utils` installed on the host)

        ```bash
        #!/usr/bin/env bash
        ...
        xhost +local:

        docker run -it --rm \
            ...
            -e DISPLAY \
            -e QT_X11_NO_MITSHM=1 \
            -e XDG_RUNTIME_DIR=/run/user/$USER_ID
            -v /run/user/$USER_ID:/run/user/$USER_ID
            -v /tmp/.X11-unix:/tmp/.X11-unix
            ...
            demo-cnn

        xhost -local:
        ```

    - User & group passthrough

        ```bash
        #!/usr/bin/env bash
        ...
        USER_ID=$(id -u)
        GROUP_ID=$(id -g)
        PASSWD_FILE=$(mktemp) && echo $(getent passwd $USER_ID) > $PASSWD_FILE
        GROUP_FILE=$(mktemp) && echo $(getent group $GROUP_ID) > $GROUP_FILE
        ...
        docker run -it --rm \
            ...
            -e HOME \
            -u $USER_ID:$GROUP_ID \
            -v $PASSWD_FILE:/etc/passwd:ro \
            -v $GROUP_FILE:/etc/group:ro \
            -v $ROOTFS/home:$HOME \
            ...
            demo-cnn
        ...
        ```

    - GPU access

        ```bash
        #!/usr/bin/env bash
        ...
        DEVICES=0
        ...
        docker run -it --rm \
            --gpus '"device='$DEVICES'"' \
            ...
            demo-cnn
        ...
        ```
