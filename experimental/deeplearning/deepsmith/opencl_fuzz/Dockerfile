FROM python:3.6-slim
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Install python3-pyopencl, not because we need it, but because pyopencl has
# dependencies not described by it's python setup.py file, and I haven't been
# able to figure out how to install all of them on my own yet. So far I know
# that ocl-icd-libopencl1 package is needed for libOpenCL.so.1, but that alone
# is not enough.
# Also install libreadline6-dev, because libreadline.so is needed by oclgrind.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pyopencl libreadline6-dev

# Link the libOpenCL installed by ocl-icd-libopencl1 into the default library
# search path, so that cldrive harness can compile binaries using -lOpenCL.
RUN ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/local/lib/libOpenCL.so

# Dirty hack to workaround the fact that oclgrind demands libreadline.so.6, but
# the current Ubuntu package provides .so.7.
RUN ln -s /lib/x86_64-linux-gnu/libreadline.so.7 \
    /lib/x86_64-linux-gnu/libreadline.so.6

# Add and unpack the pre-trained corpus and config.
RUN mkdir -p /data/
ADD generator.pbtxt /data
ADD model /data/model

# Add and unpack the CLgen build archive.
ADD opencl_fuzz_image-layer.tar /
