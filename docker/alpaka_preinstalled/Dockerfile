
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
		python3-dev \
		python3-pip \
		cmake \
		g++ \
		libboost-all-dev \
		git

RUN git clone https://github.com/alpaka-group/alpaka.git \
	&& cd alpaka \
	&& cmake -B build -S . \
	&& cmake --build build \
	&& cmake --install build

CMD ["/bin/bash"]
