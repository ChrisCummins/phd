FROM python:3.6-slim
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Add and unpack the corpus and cache dirs.
RUN mkdir -p /datasets/
ADD corpus /clgen/corpus/
ADD cache /clgen/cache/
ADD config.pbtxt /datasets/

# Add and unpack the CLgen build archive.
RUN mkdir -p /clgen/bin
ADD clgen.tar.bz2 /clgen/bin/

# Add the CLgen binaries to $PATH.
ENV PATH="/clgen/bin:${PATH}"

WORKDIR /
