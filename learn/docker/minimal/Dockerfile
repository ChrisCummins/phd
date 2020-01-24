FROM ubuntu:16.04
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Install OpenCL support for Intel CPU.
# Based on Paul Kienzle's work in: https://github.com/pkienzle/opencl_docker
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        alien \
        curl \
        tar \
        unzip
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
RUN apt-get remove -y alien && apt-get autoremove -y

# Set and configure the locale. This is necessary for CLgen, which uses en_GB,
# and Linuxbrew, which uses en_US.
RUN apt-get install -y --no-install-recommends language-pack-en
RUN locale-gen --purge en_GB.UTF-8

# Setup the environment.
ENV HOME /root
ENV USER docker
ENV PHD /root/phd

# Download the phd sources.
WORKDIR /root
RUN apt-get install -y --no-install-recommends ca-certificates
RUN curl -o phd.zip -L http://github.com/ChrisCummins/phd/archive/master.zip
RUN unzip phd.zip && mv phd-master phd && rm phd.zip
RUN apt-get remove -y ca-certificates

# Build project.
WORKDIR /root/phd
RUN apt-get install --no-install-recommends -y python

# WORKDIR $PHD
# CMD [clinfo]
