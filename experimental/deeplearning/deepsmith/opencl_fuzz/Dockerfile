# Create a minimal Ubuntu environment which is capable of running
# //experimental/deeplearning/deepsmith/opencl_fuzz.
FROM chriscummins/phd_base
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Install OpenCL.
# Based on Paul Kienzle's work in: https://github.com/pkienzle/opencl_docker
# TODO(cec): This is a work around for the fact that @CLSmith//:cl_launcher
# crashes at runtime when built hemetically using @libopencl//:libOpenCL.
# Because of this, we link against the system libOpenCL.so, so we must install
# an OpenCL implementation in the docker environment. Once the problem with
# using the repo-local libOpenCL has been fixed, we can remove this.
RUN apt-get update && apt-get install --no-install-recommends -y \
    alien \
    curl
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
WORKDIR /

# Link libOpenCL.so into /usr/lib so that it can be found using the default
# LD_LIBRARY_PATH.
RUN ln -s /opt/intel/opencl-*/lib64/libOpenCL.so /usr/lib/libOpenCL.so
RUN ln -s /usr/lib/libOpenCL.so /usr/lib/libOpenCL.so.1

# Add and unpack the pre-trained corpus and config.
RUN mkdir -p /data/
ADD clgen.pbtxt /data
ADD clsmith.pbtxt /data
ADD model /data/model

# Add and unpack the CLgen build archive.
ADD opencl_fuzz_image-layer.tar /

ENTRYPOINT ["python", "/app/experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz_image.binary", "--generator=clgen", "--generator_config=/data/clgen.pbtxt", "--interesting_results_dir=/out"]
