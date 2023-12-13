
FROM ubuntu:latest

LABEL AUTHOR="simone.balducci00@gmail.com"
LABEL VERSION="1.0"

SHELL ["/bin/bash", "-c"]

# update
RUN apt-get update && apt upgrade -y

# install vim
RUN apt-get install -y vim

# install dependencies
## python3 and pip
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
## CLUEstering python dependencies
RUN pip3 install numpy matplotlib scikit-learn pandas
## boost
RUN apt-get install -y libboost-all-dev
## tbb
RUN apt-get install -y libtbb-dev
## cuda
RUN apt-get install -y nvidia-cuda-toolkit

# copy extern folders
COPY ./extern/alpaka /app/extern/alpaka
COPY ./extern/pybind11 /app/extern/pybind11

# copy library source files
COPY ./CLUEstering/alpaka/ /app/CLUEstering/alpaka/

WORKDIR /app
