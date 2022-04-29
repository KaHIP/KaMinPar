FROM ubuntu:21.10 

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing -y && apt-get upgrade -y

# Benchmark script 
RUN apt-get install zsh git gawk -y

# Build dependencies
RUN apt-get install gcc-10 g++-10 libtbb-dev cmake libhwloc-dev clang build-essential libopenmpi-dev -y 

# Utilities
RUN apt-get install wget curl ca-certificates wget -y

ARG RUNNER_VERSION=2.288.1
RUN useradd -m docker
RUN cd /home/docker && mkdir actions-runner && cd actions-runner \
	&& curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
	&& tar -xzf actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
RUN chown -R docker ~docker && /home/docker/actions-runner/bin/installdependencies.sh

RUN wget https://algo2.iti.kit.edu/seemaier/data/kaminpar_ci.tar.gz && tar -xzf kaminpar_ci.tar.gz 

COPY start.sh start.sh
RUN chmod +x start.sh
USER docker 
ENTRYPOINT ["./start.sh"]
