FROM ubuntu:16.04
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Install essential packages.
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  software-properties-common git curl sudo python

# TODO: Determine if these need installing.
# python3-distutils python3-venv python3-dev

# Instal Beignet for OpenCL CPU support.
RUN apt-get install -y --no-install-recommends beignet-opencl-icd clinfo

# Add Beignet's libOpenCL to the default LD_LIBRARY_PATH. This is needed
# because pyopencl uses -lOpenCL during build.
RUN ln -s /usr/lib/x86_64-linux-gnu/beignet/libcl.so /usr/local/lib/libOpenCL.so

# Install the OpenCL headers from Khronos.
WORKDIR /tmp
RUN git clone https://github.com/KhronosGroup/OpenCL-Headers.git
RUN mv OpenCL-Headers/CL /usr/include/CL

# Set and configure the locale. This is necessary for CLgen, which uses en_GB,
# and Linuxbrew, which uses en_US.
RUN apt-get install -y --no-install-recommends language-pack-en
RUN locale-gen --purge en_GB.UTF-8

# Create the phd repository.
WORKDIR /root
RUN git clone https://github.com/ChrisCummins/phd.git

# Bootstrap the PhD repo.
ENV HOME /root
ENV USER docker
WORKDIR /root/phd
RUN /root/phd/tools/bootstrap.sh

# Install and configure my preferred shell.
RUN apt-get install -y --no-install-recommends zsh
ENV SHELL zsh
WORKDIR /root/phd/system/dotfiles
RUN ./run -v Zsh

# Install and configure my preferred editor.
ENV ./run -v Vim

# Clean up.
RUN apt-get autoremove -y
RUN apt-get clean
RUN rm -rf \
    /root/phd/.git \
    /home/linuxbrew/.cache/Homebrew/* \
    /tmp/* \
    /var/lib/apt/lists/* \
    /var/tmp/*

WORKDIR /root/phd
CMD ["zsh"]
