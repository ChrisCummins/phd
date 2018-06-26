# Create a minimal Ubuntu environment which is capable of running
# //experimental/deeplearning/deepsmith/opencl_fuzz.
FROM python:3.6-slim
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Install a standard compiler toolchain which is needed for cldrive harnesses
# (because the current clang compilation is not yet entirely hemetic and relies
# on system linker and C headers), and libreadline6-dev to provide
# libreadline.so which is needed by oclgrind.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc libc6-dev libreadline6-dev

# Dirty hack to workaround the fact that oclgrind demands libreadline.so.6, but
# the current Ubuntu package provides libreadline.so.7.
RUN ln -s /lib/x86_64-linux-gnu/libreadline.so.7 \
    /lib/x86_64-linux-gnu/libreadline.so.6

# Add and unpack the pre-trained corpus and config.
RUN mkdir -p /data/
ADD generator.pbtxt /data
ADD model /data/model

# Add and unpack the CLgen build archive.
ADD opencl_fuzz_image-layer.tar /

ENTRYPOINT ["python3.6", "/app/experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz_image.binary", "--generator=/data/generator.pbtxt", "--interesting_results_dir=/out"]
