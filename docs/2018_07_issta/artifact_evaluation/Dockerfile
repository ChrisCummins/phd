FROM ubuntu:16.04
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Install essential packages.
RUN apt-get update
RUN apt-get install --no-install-recommends -y \
    alien \
    apt-utils \
    clinfo \
    curl \
    git \
    python \
    software-properties-common \
    sudo \
    tar \
    unzip \
    xz-utils

# Install OpenCL support for Intel CPU.
# Based on Paul Kienzle's work in: https://github.com/pkienzle/opencl_docker
ARG INTEL_DRIVER=opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz
ARG INTEL_DRIVER_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/9019
RUN mkdir -p /tmp/opencl-driver-intel
WORKDIR /tmp/opencl-driver-intel
RUN echo INTEL_DRIVER is $INTEL_DRIVER; \
    curl -O $INTEL_DRIVER_URL/$INTEL_DRIVER; \
    if echo $INTEL_DRIVER | grep -q "[.]zip$"; then \
        unzip $INTEL_DRIVER; \
        mkdir fakeroot; \
        for f in intel-opencl-*.tar.xz; do tar -C fakeroot -Jxvf $f; done; \
        cp -R fakeroot/* /; \
        ldconfig; \
    else \
        tar -xzf $INTEL_DRIVER; \
        for i in $(basename $INTEL_DRIVER .tgz)/rpm/*.rpm; do alien --to-deb $i; done; \
        dpkg -i *.deb; \
        rm -rf $INTEL_DRIVER $(basename $INTEL_DRIVER .tgz) *.deb; \
        mkdir -p /etc/OpenCL/vendors; \
        echo /opt/intel/*/lib64/libintelocl.so > /etc/OpenCL/vendors/intel.icd; \
    fi; \
    rm -rf /tmp/opencl-driver-intel;

# Set and configure the locale. This is necessary for CLgen, which uses en_GB,
# and Linuxbrew, which uses en_US.
RUN apt-get install -y --no-install-recommends language-pack-en
RUN locale-gen --purge en_GB.UTF-8

# Setup the environment.
ENV HOME /root
ENV USER docker
ENV PHD /root/phd

# Git commit IDs of our dependencies.
ENV PHD_VERSION fb0fcf59d3c6c8451ba44b1bb2b2db41ddaf9af7
ENV CLGEN_VERSION 1c126b5bb9ace265fe9bc25f8beef6702a169fec

# Clone the source code for this project.
RUN git clone https://github.com/ChrisCummins/phd.git $PHD
RUN git -C $PHD reset --hard $PHD_VERSION
RUN rm -rf $PHD/deeplearning/clgen
RUN git clone https://github.com/ChrisCummins/clgen.git $PHD/deeplearning/clgen
RUN git -C $PHD/deeplearning/clgen reset --hard $CLGEN_VERSION

# Install OpenCL 1.2 headers, required by pyopencl.
ENV OPENCL_HEADERS_VERSION e986688daf750633898dfd3994e14a9e618f2aa5
RUN git clone https://github.com/KhronosGroup/OpenCL-Headers.git /tmp/opencl-headers
RUN git -C /tmp/opencl-headers reset --hard $OPENCL_HEADERS_VERSION
RUN mv /tmp/opencl-headers/opencl12/CL /usr/include/CL

# Link libOpenCL.so into /usr/lib64 so that pyopencl can find it.
RUN ln -s /opt/intel/opencl-*/lib64/libOpenCL.so /usr/lib/libOpenCL.so

# Build the project.
RUN $PHD/tools/bootstrap.sh

# Install and configure my preferred shell.
RUN apt-get install -y --no-install-recommends zsh
ENV SHELL zsh
RUN /root/phd/system/dotfiles/run -v Zsh

# Install and configure my preferred editor.
ENV /root/phd/system/dotfiles/run -v Vim

# Add out newly-minted python3 to the PATH.
ENV PATH /home/linuxbrew/.linuxbrew/bin:$PATH

# Build the artifact.
RUN cd $PHD/docs/2018_07_issta/artifact_evaluation && ./install.sh

# TODO(cec): See if we can still eval artifact after this.
# # Clean up.
RUN sudo -H -u linuxbrew bash -c \
    '/home/linuxbrew/.linuxbrew/bin/brew remove buildifier llvm'
RUN python3 -m pip uninstall -y virtualenv
RUN apt-get remove -y openjdk-8-jdk bazel texlive-full build-essential
RUN apt-get autoremove -y
RUN apt-get clean
RUN find / -name '.git' -type d -exec rm -rfv {} \; || true
RUN rm -rf \
    $HOME/.cache \
    $PHD/config \
    $PHD/datasets \
    $PHD/docs/2015_08_msc_thesis \
    $PHD/docs/2015_08_msc_thesis.pdf \
    $PHD/docs/2015_09_progression_review \
    $PHD/docs/2015_09_progression_review.pdf \
    $PHD/docs/2016_01_adapt \
    $PHD/docs/2016_01_adapt.pdf \
    $PHD/docs/2016_01_hlpgpu \
    $PHD/docs/2016_01_hlpgpu.pdf \
    $PHD/docs/2016_06_pldi \
    $PHD/docs/2016_06_pldi.pdf \
    $PHD/docs/2016_07_acaces \
    $PHD/docs/2016_07_acaces.pdf \
    $PHD/docs/2016_07_pact \
    $PHD/docs/2016_11_first_year_review \
    $PHD/docs/2016_11_first_year_review.pdf \
    $PHD/docs/2016_12_wip_taco \
    $PHD/docs/2017_02_cgo \
    $PHD/docs/2017_02_cgo.pdf \
    $PHD/docs/2017_09_pact \
    $PHD/docs/2017_09_pact.pdf \
    $PHD/experimental \
    $PHD/http \
    $PHD/learn \
    $PHD/talks \
    $PHD/third_party \
    $PHD/util \
    /home/linuxbrew/.cache \
    /tmp \
    /var/lib/apt/lists \
    /var/tmp
RUN mkdir $HOME/.cache /var/tmp /var/lib/apt/lists

WORKDIR $PHD/docs/2018_07_issta/artifact_evaluation
CMD [clinfo]
