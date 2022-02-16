FROM public.ecr.aws/docker/library/ubuntu:latest

ARG RUNNER_VERSION="2.287.1"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install zsh git gawk gcc-10 g++-10 libtbb-dev cmake libhwloc-dev clang curl wget ca-certificates wget build-essential libopenmpi-dev -y

RUN useradd -m docker
RUN cd /home/docker && mkdir actions-runner && cd actions-runner \
	&& curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L https://github.com/actions/runner/releases/download/v2.287.1/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
	&& tar -xzf actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
RUN chown -R docker ~docker && /home/docker/actions-runner/bin/installdependencies.sh

RUN wget https://algo2.iti.kit.edu/seemaier/data/kaminpar_ci.tar.gz && tar -xzf kaminpar_ci.tar.gz 

COPY start.sh start.sh
RUN chmod +x start.sh
USER docker 
ENTRYPOINT ["./start.sh"]

