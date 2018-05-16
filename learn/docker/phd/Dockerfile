FROM ubuntu:16.04
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Install essential packages.
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  software-properties-common \
  git curl sudo python python-pip

# TODO: Determine if these need installing.
# python3-distutils python3-venv python3-dev

# Instal Beignet for OpenCL CPU support.
RUN apt-get install -y --no-install-recommends beignet-opencl-icd clinfo

# Install the OpenCL headers from Khronos.
WORKDIR /tmp
RUN git clone https://github.com/KhronosGroup/OpenCL-Headers.git
RUN mv OpenCL-Headers/CL /usr/local/include/CL

# Create the phd repository.
WORKDIR /
RUN git clone https://github.com/ChrisCummins/phd.git
WORKDIR /phd

# Bootstrap the PhD repo.
RUN ./tools/bootstrap | bash

# Install and configure my preferred shell.
RUN apt-get install -y --no-install-recommends zsh
ENV HOME /root
ENV USER docker
ENV SHELL zsh
WORKDIR /phd/system/dotfiles
RUN ./run -v Zsh

# Set and configure the locale. This is necessary for CLgen, which uses en_GB.
# RUN apt-get install -y --no-install-recommends language-pack-en
# RUN locale-gen --purge en_GB.UTF-8

# Clean up.
RUN apt autoremove -y
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD ["zsh"]
