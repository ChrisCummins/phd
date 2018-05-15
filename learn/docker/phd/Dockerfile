FROM ubuntu:16.04
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Install essential packages.
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  software-properties-common \
  git curl sudo python python-pip

# TODO: Determine if these need installing.
# python3-distutils python3-venv python3-dev

###############################################################################
# Install Intel OpenCL. Taken from a Dockerfile written by Rex Tsai:
# https://github.com/chihchun/opencl-docker/blob/a8ed4c94fd929b6d3c506f128b8f0d9a4fbdca4f/intel/2016R2/Dockerfile
WORKDIR /tmp

# OpenCL™ 2.0 Driver for Intel® HD, Iris™, and Iris™ Pro Graphics for Linux* (64-bit)
ENV INTEL_DRIVER_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/9418/intel-opencl-2.0-2.0-54425.tar.gz
# Intel® SDK for OpenCL™ Applications 2016 R2 for Linux* (64 bit)
ENV INTEL_SDK_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/9553/intel_sdk_for_opencl_2016_ubuntu_6.2.0.1760_x64.tgz

# Build dependencies for Intel SDK.
RUN apt-get install -y --no-install-recommends \
  alien wget tar libxcb-dri3-0 libxcb-dri2-0

# Install driver.
RUN wget ${INTEL_DRIVER_URL}
RUN TARBALL=$(basename ${INTEL_DRIVER_URL}) \
  && DIR=$(basename ${INTEL_DRIVER_URL} .tar.gz) \
  && tar zxvf ${TARBALL} \
  && for i in ${DIR}/*rpm ; do alien --to-deb $i ; done

# Install SDK.
RUN wget ${INTEL_SDK_URL}
RUN TARBALL=$(basename ${INTEL_SDK_URL}) \
  && DIR=$(basename ${INTEL_SDK_URL} .tar.gz) \
  && tar zxvf ${TARBALL} \
  && for i in ${DIR}/*rpm ; do alien --to-deb $i ; done

# Install the debian packages.
RUN dpkg -i *.deb

# End of Rex Tsai's Intel OpenCL install commands.
###############################################################################

# Install the OpenCL headers from Khronos.
WORKDIR /tmp
RUN git clone https://github.com/KhronosGroup/OpenCL-Headers.git
RUN mv OpenCL-Headers/CL /usr/local/include/CL

# Add Intel libraries to link path for -lOpenCL. This is needed for building
# the Python pyopencl package.
RUN ln -s /opt/intel/opencl/libOpenCL.so /usr/local/lib/libOpenCL.so

# Create the phd repository.
RUN git clone https://github.com/ChrisCummins/phd.git
WORKDIR /phd

# Bootstrap the PhD repo.
RUN ./tools/bootstrap | bash

# Install my preferred shell.
RUN apt-get install -y --no-install-recommends zsh

# Prepare the user shell environment.
ENV HOME /root
ENV USER docker
ENV SHELL zsh

# Install my Zsh configuration.
RUN cd phd/system/dotfiles && ./run -v Zsh

# Clean up.
RUN apt autoremove -y
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD ["zsh"]
