FROM python:3.6-slim
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Add and unpack the tiny corpus and config.
RUN mkdir -p /datasets/tiny
ADD corpus.tar.bz2 /datasets/tiny/
ADD config.pbtxt /datasets/tiny/

# Add and unpack the CLgen build archive.
RUN mkdir -p /clgen/bin
ADD clgen.tar.bz2 /clgen/bin/

# Add the CLgen binaries to $PATH.
ENV PATH="/clgen/bin:${PATH}"

WORKDIR /
