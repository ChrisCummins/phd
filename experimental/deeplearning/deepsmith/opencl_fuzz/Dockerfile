FROM python:3.6-slim
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Install python3-pyopencl, not because we need it, but because pyopencl has
# dependencies not described by it's python setup.py file, and I haven't been
# able to figure out how to install all of them on my own yet. So far I know
# that ocl-icd-libopencl1 package is needed for libOpenCL.so.1, but that alone
# is not enough.
# Also install gcc so that we can compile cldrive harnesses.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-pyopencl

# Add and unpack the tiny corpus and config.
RUN mkdir -p /datasets/
ADD corpus.tar.bz2 /datasets
ADD generator.pbtxt /datasets
ADD cache /clgen/cache

# Add and unpack the CLgen build archive.
ADD opencl_fuzz_image-layer.tar /
