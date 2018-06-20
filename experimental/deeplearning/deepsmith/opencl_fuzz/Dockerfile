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

# Link the libOpenCL installed by ocl-icd-libopencl1 into the default library
# search path, so that cldrive harness can compile binaries using -lOpenCL.
RUN ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/local/lib/libOpenCL.so

# Install OpenCL 1.2 headers, required by cldrive harness to compiler headers.
RUN apt-get update && apt-get install -y git
ENV OPENCL_HEADERS_VERSION e986688daf750633898dfd3994e14a9e618f2aa5
RUN git clone https://github.com/KhronosGroup/OpenCL-Headers.git /tmp/opencl-headers
RUN git -C /tmp/opencl-headers reset --hard $OPENCL_HEADERS_VERSION
RUN mv /tmp/opencl-headers/opencl12/CL /usr/include/CL

# Add and unpack the pre-trained corpus and config.
RUN mkdir -p /datasets/
ADD corpus.tar.bz2 /datasets
ADD generator.pbtxt /datasets
ADD cache /clgen/cache

# Add and unpack the CLgen build archive.
ADD opencl_fuzz_image-layer.tar /
